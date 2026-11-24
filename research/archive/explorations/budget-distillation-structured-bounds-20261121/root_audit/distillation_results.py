"""Independent CNN, calibration, bootstrap, provenance and dense-work audit."""
from pathlib import Path
import csv,datetime,gzip,hashlib,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image,ImageOps
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent;B=ROOT/'distillation'
read=lambda p:list(csv.DictReader(p.open(newline='')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ash=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def ce(z,y):
    z=np.asarray(z,np.float64)
    return np.logaddexp.reduce(z,axis=1)-z[np.arange(len(y)),y]

def beta(z,y):
    z=np.asarray(z,np.float64);chosen=z[np.arange(len(y)),y]
    def grad(b):
        a=b*z;a-=a.max(1,keepdims=True);p=np.exp(a);p/=p.sum(1,keepdims=True)
        return float(np.mean(np.sum(p*z,axis=1)-chosen))
    lo,hi=.25,4.
    if grad(lo)>=0:return lo
    if grad(hi)<=0:return hi
    for _ in range(90):
        mid=(lo+hi)/2
        if grad(mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2

@torch.inference_mode()
def forward(images,state):
    outputs=[]
    for j in range(0,len(images),512):
        x=images[j:j+512]
        for i in (0,3,6):
            x=F.relu(F.conv2d(x,state[f'features.{i}.weight'],state[f'features.{i}.bias'],padding=1))
            if i!=6:x=F.max_pool2d(x,2)
        x=F.relu(F.linear(x.flatten(1),state['head.1.weight'],state['head.1.bias']))
        outputs.append(F.linear(x,state['head.3.weight'],state['head.3.bias']))
    return torch.cat(outputs).cpu().numpy()

def main():
    assert (B/'SUCCESS_CRITERIA.json').exists()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    images=np.frombuffer(gzip.decompress((B/'data/train-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,1,28,28).copy()
    labels=np.frombuffer(gzip.decompress((B/'data/train-labels-idx1-ubyte.gz').read_bytes()),np.uint8,offset=8).astype(np.int64)
    splits=dict(np.load(B/'splits.npz'));ext=dict(np.load(B/'external/population.npz'))
    assert len(np.unique(np.concatenate(list(splits.values()))))==60000
    assert sorted(map(len,splits.values()))==[5000,5000,10000,40000]
    for key,n in [('train',4000),('validation',500),('temperature',500),('holdout',1000)]:
        assert np.array_equal(np.bincount(labels[splits[key]]),np.full(10,n))
    provenance=json.loads((B/'external/population_manifest.json').read_text());exhash=[]
    for i,r in enumerate(provenance):
        path=B/'external/selected_raw'/f"{r['id']}.jpg";assert sha(path)==r['raw_sha256']
        im=Image.open(path).convert('L');im=ImageOps.invert(im);im.thumbnail((24,24),Image.Resampling.LANCZOS)
        frame=Image.new('L',(28,28),0);frame.paste(im,((28-im.width)//2,(28-im.height)//2));a=np.asarray(frame)
        assert np.array_equal(a,ext['images'][i,0]);assert ash(a)==r['pixels_sha256'];exhash.append(ash(a))
        assert str(ext['ids'][i])==str(r['id']) and ext['labels'][i]==r['label']
    oldtest=np.frombuffer(gzip.decompress((B/'data/t10k-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,1,28,28)
    fmhash={ash(a) for a in np.concatenate([images,oldtest])}
    assert len(set(exhash))==1500 and not (set(exhash)&fmhash)
    assert np.array_equal(np.unique(ext['labels'],return_counts=True)[1],np.full(6,250))
    frozen=json.loads((B/'SOURCE_FREEZE.json').read_text());committed=0
    for rel,h in frozen['hashes'].items():assert sha(B/rel)==h;committed+=1
    model_commit=json.loads((B/'MODEL_COMMIT.json').read_text())
    assert sha(B/'temperatures.json')==model_commit['temperatures_sha256']
    for rel,h in model_commit['manifest'].items():assert sha(B/'models'/rel)==h;committed+=1
    for rel,h in json.loads((B/'SCORE_COMPLETE.json').read_text())['manifest'].items():assert sha(B/'predictions'/rel)==h;committed+=1
    rows=json.loads((B/'temperatures.json').read_text());traces=read(B/'training_trace.csv');train=read(B/'training.csv')
    pairing=json.loads((B/'PAIRING_AUDIT.json').read_text());qrows=read(B/'quality.csv');classes=read(B/'classwise.csv')
    max_beta=max_pred=max_metric=max_target=max_val=0.;extra=0;checkpoint_count=0;maps={}
    for row in rows:
        pair,arm=row['pair'],row['arm'];tag=f'{pair}_{arm}'
        payload=torch.load(B/'models'/f'{tag}.pt',map_location='cpu',weights_only=True)
        state={k:v.cuda() for k,v in payload['state_dict'].items()}
        # Recompute all four validation losses and independently select a checkpoint.
        tv=sorted([r for r in traces if int(r['pair'])==pair and r['arm']==arm],key=lambda r:int(r['checkpoint']))
        valx=torch.as_tensor(images[splits['validation']],device='cuda',dtype=torch.float32)/255
        valy=labels[splits['validation']];losses=[]
        for r in tv:
            cp=torch.load(B/'models'/f"{tag}_checkpoint{r['checkpoint']}.pt",map_location='cpu',weights_only=True)
            zs=forward(valx,{k:v.cuda() for k,v in cp['state_dict'].items()});extra+=len(valx)
            value=float(F.cross_entropy(torch.tensor(zs,device='cuda'),torch.tensor(valy,device='cuda')))
            max_val=max(max_val,abs(value-float(r['validation_ce'])));assert abs(value-float(r['validation_ce']))<1e-6
            losses.append(value)
        selected=int(np.argmin(losses));cp=torch.load(B/'models'/f'{tag}_checkpoint{selected+1}.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v,cp['state_dict'][k]) for k,v in payload['state_dict'].items());checkpoint_count+=1
        record=next(r for r in pairing if r['pair']==pair and r['arm']==arm)
        tr=next(r for r in train if int(r['pair'])==pair and r['arm']==arm);budget=int(tr['training_examples'])
        rr=np.random.default_rng(2323000+pair);cr=np.random.default_rng(2324000+pair);orders=[];coins=[];flip=0;seen=0
        while seen<budget:
            order=rr.permutation(40000);coin=cr.random(40000)<.5;orders.append(ash(order));coins.append(ash(coin));take=min(40000,budget-seen);flip+=int(coin[:take].sum());seen+=take
        assert orders==record['epoch_orders'] and coins==record['flip_coins'] and flip==record['flipped_examples']
        assert len({r['initial_sha256'] for r in pairing if r['pair']==pair})==1
        cal=dict(np.load(B/'models'/f'{tag}_calibration.npz'));calx=torch.as_tensor(images[splits['temperature']],device='cuda',dtype=torch.float32)/255
        assert np.array_equal(cal['indices'],splits['temperature'])
        assert np.array_equal(cal['labels'],labels[splits['temperature']])
        h=forward(calx,state);extra+=len(calx);assert np.array_equal(h,cal['raw'])
        for variant in ['raw']+(['full'] if arm=='teacher' else []):
            z=cal['raw'].astype(float) if variant=='raw' else (cal['raw'].astype(float)+cal['reflected'])/2
            err=abs(beta(z,cal['labels'])-row[variant]['beta']);max_beta=max(max_beta,err);assert err<1e-12
        if arm=='teacher':
            ht=forward(calx.flip(-1),state);extra+=len(calx);assert np.array_equal(ht,cal['reflected'])
            target=dict(np.load(B/'models'/f'{pair}_targets.npz'));xx=torch.as_tensor(images[splits['train']],device='cuda',dtype=torch.float32)/255
            a=forward(xx,state);b=forward(xx.flip(-1),state);extra+=2*len(xx)
            assert np.array_equal(a,target['raw']) and np.array_equal(b,target['reflected'])
            qq=torch.softmax((torch.tensor(a,device='cuda')+torch.tensor(b,device='cuda'))/4,dim=1).cpu().numpy()
            err=float(np.max(abs(qq-target['targets'])));max_target=max(max_target,err);assert err<1e-7
        for dataset in ['holdout','external']:
            pix=images[splits['holdout']] if dataset=='holdout' else ext['images'];yy=labels[splits['holdout']] if dataset=='holdout' else ext['labels']
            p=dict(np.load(B/'predictions'/f'{dataset}_{tag}.npz'));assert np.array_equal(p['labels'],yy)
            xx=torch.as_tensor(pix,device='cuda',dtype=torch.float32)/255
            h=forward(xx,state);extra+=len(xx);err=float(np.max(abs(h-p['raw'])));max_pred=max(max_pred,err);assert err<1e-6
            assert np.array_equal(h.argmax(1),p['raw'].argmax(1))
            h=h.astype(float)
            if arm=='teacher':
                ht=forward(xx.flip(-1),state);extra+=len(xx);assert np.array_equal(ht,p['reflected'])
                outputs={'B':h*row['raw']['beta'],'T':(h+ht)/2*row['full']['beta']}
            else:outputs={('S' if arm=='student' else 'C'):h*row['raw']['beta']}
            for name,z in outputs.items():
                loss=ce(z,yy);error=(z.argmax(1)!=yy).astype(float);maps[dataset,pair,name]=np.stack([loss,error])
                r=next(r for r in qrows if r['dataset']==dataset and int(r['pair'])==pair and r['method']==name)
                for key,val in [('ce',loss.mean()),('error',error.mean())]:
                    e=abs(float(r[key])-val);max_metric=max(max_metric,e);assert e<2e-12
                for c in np.unique(yy):
                    r=next(r for r in classes if r['dataset']==dataset and int(r['pair'])==pair and r['method']==name and int(r['label'])==c)
                    for key,v in [('ce',loss[yy==c].mean()),('error',error[yy==c].mean())]:assert abs(float(r[key])-v)<2e-12
    rng=np.random.default_rng(2328000);contrasts=read(B/'contrasts.csv');max_boot=0.
    for dataset in ['holdout','external']:
        yy=labels[splits['holdout']] if dataset=='holdout' else ext['labels']
        mean={m:np.mean([maps[dataset,p,m] for p in (1,2,3)],axis=0) for m in 'BTSC'}
        specs=[('S-B','S','B'),('S-C','S','C'),('S-T','S','T'),('T-B','T','B')]
        matrix=np.concatenate([mean[a]-mean[b] for _,a,b in specs]);boot=[]
        for _ in range(200):
            vals=np.zeros((50,8))
            for c in np.unique(yy):
                ids=np.flatnonzero(yy==c);counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=50)
                vals+=counts@matrix[:,ids].T/len(yy)
            boot.append(vals)
        boot=np.concatenate(boot);saved=np.load(B/f'{dataset}_bootstrap.npz')['values'];e=float(np.max(abs(saved-boot)));max_boot=max(max_boot,e);assert e<1e-12
        for j,(contrast,_,_) in enumerate(specs):
            for k,metric in enumerate(['ce','error']):
                r=next(r for r in contrasts if r['dataset']==dataset and r['contrast']==contrast and r['metric']==metric)
                lo,hi=np.quantile(boot[:,2*j+k],[.05/32,1-.05/32])
                assert max(abs(lo-float(r['lower'])),abs(hi-float(r['upper'])),abs(matrix[2*j+k].mean()-float(r['mean'])))<1e-12
    conv=[28*28*16*1*3*3,14*14*32*16*3*3,7*7*64*32*3*3];linear=[3136*128,128*10]
    f=2*sum(conv+linear);g=3*f-2*conv[0];budget=960000*g+135000*f;control=(budget-25000*f)//g
    assert f==4643840 and g==13705728 and control==997270
    for r in train:
        n=control if r['arm']=='control' else 480000
        assert int(r['training_examples'])==n
        expected=n*g+(110000 if r['arm']=='teacher' else 25000)*f
        assert int(r['matrix_flops'])==expected
    assert 0<=budget-(control*g+25000*f)<g
    out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        evaluated_external_raw_images_reconstructed=1500,exact_external_fashion_overlap=0,
        split_ids_checked=60000,selected_checkpoints_replayed=checkpoint_count,checkpoint_validation_forwards=180000,
        max_validation_loss_difference=max_val,temperature_fits=12,max_beta_error=max_beta,
        all_target_inputs_replayed=120000,max_target_error=max_target,all_holdout_external_views_replayed=True,
        quality_rows=len(qrows),classwise_rows=len(classes),max_metric_error=max_metric,max_direct_logit_error=max_pred,
        class_discrepancies=0,bootstrap_intervals=len(contrasts),max_bootstrap_error=max_boot,
        additional_audit_forward_examples=extra,committed_files_checked=committed,
        forward_matrix_flops=f,training_matrix_flops=g,control_training_examples=control,
        distillation_network_budget_per_pair=budget,control_network_budget_per_pair=control*g+25000*f,
        network_budget_relative_gap=(budget-control*g-25000*f)/budget,
        limitations='No independent full retraining; conditional image bootstrap does not cover training-set or product-family variation. Exact duplicate audit does not rule out near duplicates.')
    (ROOT/'root_audit/DISTILLATION_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
