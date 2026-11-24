"""Independent functional CNN replay, seed inference and crossed bootstrap."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gzip,hashlib,math
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from common import ROOT,read,get,sha,record,multiclass_loss,multiclass_beta
B=ROOT/'distillation'
ash=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
@torch.inference_mode()
def forward(x,state):
    out=[]
    for begin in range(0,len(x),512):
        z=x[begin:begin+512]
        for i in (0,3,6):
            z=F.relu(F.conv2d(z,state[f'features.{i}.weight'],state[f'features.{i}.bias'],padding=1))
            if i!=6:z=F.max_pool2d(z,2)
        z=F.relu(F.linear(z.flatten(1),state['head.1.weight'],state['head.1.bias']))
        out.append(F.linear(z,state['head.3.weight'],state['head.3.bias']))
    return torch.cat(out).cpu().numpy()
def t_quantile(prob):
    # Independently integrate Student(5) density with Gauss-Legendre quadrature.
    nodes,weights=np.polynomial.legendre.leggauss(96)
    def cdf(t):
        u=(nodes+1)*t/2
        return .5+t/2*np.sum(weights*(8/(3*np.pi*np.sqrt(5)))*(1+u*u/5)**-3)
    lo,hi=0.,20.
    for _ in range(80):
        m=(lo+hi)/2
        if cdf(m)<prob:lo=m
        else:hi=m
    return (lo+hi)/2
def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    images=np.frombuffer(gzip.decompress((B/'data/train-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,1,28,28).copy()
    labels=np.frombuffer(gzip.decompress((B/'data/train-labels-idx1-ubyte.gz').read_bytes()),np.uint8,offset=8).astype(np.int64)
    split=dict(np.load(B/'splits.npz'));old=dict(np.load(B/'legacy_external/population.npz'));new=dict(np.load(B/'external/population.npz'))
    assert len(np.unique(np.concatenate(list(split.values()))))==60000
    for key,n in [('train',4000),('validation',500),('temperature',500),('holdout',1000)]:assert np.array_equal(np.bincount(labels[split[key]]),np.full(10,n))
    populations={'holdout':(images[split['holdout']],labels[split['holdout']],split['holdout'],split['holdout']),
        'myntra_seen':(old['images'],old['labels'],old['ids'],old['ids']),
        'zappos_new':(new['images'],new['labels'],new['ids'],new['groups'])}
    hashes=0
    for rel,h in get(B/'SOURCE_FREEZE.json')['hashes'].items():assert sha(B/rel)==h;hashes+=1
    for rel,h in get(B/'MODEL_COMMIT.json')['manifest'].items():assert sha(B/'models'/rel)==h;hashes+=1
    assert sha(B/'temperatures.json')==get(B/'MODEL_COMMIT.json')['temperatures_sha256']
    for rel,h in get(B/'SCORE_COMPLETE.json')['manifest'].items():assert sha(B/'predictions'/rel)==h;hashes+=1
    assert get(B/'MODEL_COMMIT.json')['utc']<get(B/'SCORE_START.json')['utc']
    temps=get(B/'temperatures.json');traces=read(B/'training_trace.csv');training=read(B/'training.csv');pairing=get(B/'PAIRING_AUDIT.json')
    quality=read(B/'quality.csv');classes=read(B/'classwise.csv');maps={};extra=checkpoints=fits=0
    max_pred=max_beta=max_target=max_val=max_metric=0.
    for row in temps:
        pair,arm=row['pair'],row['arm'];tag=f'{pair}_{arm}'
        payload=torch.load(B/'models'/f'{tag}.pt',map_location='cpu',weights_only=True)
        state={k:v.cuda() for k,v in payload['state_dict'].items()}
        rows=sorted((r for r in traces if int(r['pair'])==pair and r['arm']==arm),key=lambda r:int(r['checkpoint']))
        xv=torch.as_tensor(images[split['validation']],device='cuda',dtype=torch.float32)/255;yv=labels[split['validation']];vals=[]
        for r in rows:
            ck=torch.load(B/'models'/f"{tag}_checkpoint{r['checkpoint']}.pt",map_location='cpu',weights_only=True)
            z=forward(xv,{k:v.cuda() for k,v in ck['state_dict'].items()});extra+=len(xv)
            value=float(F.cross_entropy(torch.tensor(z,device='cuda'),torch.tensor(yv,device='cuda')))
            err=abs(value-float(r['validation_ce']));max_val=max(max_val,err);assert err<1e-6;vals.append(value)
        ck=torch.load(B/'models'/f'{tag}_checkpoint{np.argmin(vals)+1}.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v,ck['state_dict'][k]) for k,v in payload['state_dict'].items());checkpoints+=1
        tr=next(r for r in training if int(r['pair'])==pair and r['arm']==arm);budget=int(tr['training_examples'])
        pr=next(r for r in pairing if r['pair']==pair and r['arm']==arm)
        rr=np.random.default_rng(2423000+pair);cr=np.random.default_rng(2424000+pair);orders=[];coins=[];seen=flips=0
        while seen<budget:
            order=rr.permutation(40000);coin=cr.random(40000)<.5;orders.append(ash(order));coins.append(ash(coin));take=min(40000,budget-seen);flips+=int(coin[:take].sum());seen+=take
        assert orders==pr['epoch_orders'] and coins==pr['flip_coins'] and flips==pr['flipped_examples']
        assert len({r['initial_sha256'] for r in pairing if r['pair']==pair})==1
        cal=dict(np.load(B/'models'/f'{tag}_calibration.npz'));xc=torch.as_tensor(images[split['temperature']],device='cuda',dtype=torch.float32)/255
        assert np.array_equal(cal['indices'],split['temperature']) and np.array_equal(cal['labels'],labels[split['temperature']])
        h=forward(xc,state);extra+=len(xc);assert np.array_equal(h,cal['raw'])
        for variant in ['raw']+(['full'] if arm=='teacher' else []):
            z=cal['raw'].astype(float) if variant=='raw' else (cal['raw'].astype(float)+cal['reflected'])/2
            err=abs(multiclass_beta(z,cal['labels'])-row[variant]['beta']);max_beta=max(max_beta,err);assert err<1e-12;fits+=1
        if arm=='teacher':
            ht=forward(xc.flip(-1),state);extra+=len(xc);assert np.array_equal(ht,cal['reflected'])
            target=dict(np.load(B/'models'/f'{pair}_targets.npz'));xx=torch.as_tensor(images[split['train']],device='cuda',dtype=torch.float32)/255
            assert np.array_equal(target['indices'],split['train'])
            a=forward(xx,state);b=forward(xx.flip(-1),state);extra+=2*len(xx)
            assert np.array_equal(a,target['raw']) and np.array_equal(b,target['reflected'])
            q=torch.softmax((torch.tensor(a,device='cuda')+torch.tensor(b,device='cuda'))/4,dim=1).cpu().numpy()
            err=float(np.max(abs(q-target['targets'])));max_target=max(max_target,err);assert err<1e-7
        for dataset,(pixels,y,ids,groups) in populations.items():
            saved=dict(np.load(B/'predictions'/f'{dataset}_{tag}.npz'))
            for key,v in [('labels',y),('indices',ids),('groups',groups)]:assert np.array_equal(saved[key],v)
            xx=torch.as_tensor(pixels,device='cuda',dtype=torch.float32)/255;h=forward(xx,state);extra+=len(xx)
            err=float(np.max(abs(h-saved['raw'])));max_pred=max(max_pred,err);assert err<1e-6
            assert np.array_equal(h.argmax(1),saved['raw'].argmax(1));h=h.astype(float)
            if arm=='teacher':
                ht=forward(xx.flip(-1),state);extra+=len(xx);assert np.array_equal(ht,saved['reflected'])
                outputs={'B':h*row['raw']['beta'],'T':(h+ht)/2*row['full']['beta']}
            else:outputs={('S' if arm=='student' else 'C'):h*row['raw']['beta']}
            for method,z in outputs.items():
                loss=multiclass_loss(z,y);err=(z.argmax(1)!=y).astype(float);maps[dataset,pair,method]=np.stack([loss,err])
                r=next(r for r in quality if r['dataset']==dataset and int(r['pair'])==pair and r['method']==method)
                for key,v in [('ce',loss.mean()),('error',err.mean())]:
                    discrepancy=abs(v-float(r[key]));max_metric=max(max_metric,discrepancy);assert discrepancy<2e-12
                for label in np.unique(y):
                    r=next(r for r in classes if r['dataset']==dataset and int(r['pair'])==pair and r['method']==method and int(r['label'])==label)
                    assert abs(loss[y==label].mean()-float(r['ce']))<2e-12 and abs(err[y==label].mean()-float(r['error']))<2e-12
    specs=[('S-B','S','B'),('S-C','S','C'),('S-T','S','T'),('T-B','T','B')]
    critical=[t_quantile(.975),t_quantile(.9875)];contrasts=read(B/'seed_contrasts.csv');max_ci=0.;decisions={}
    for r in contrasts:
        first,second=next((a,b) for name,a,b in specs if name==r['contrast']);idx=0 if r['metric']=='ce' else 1
        diffs=np.array([(maps[r['dataset'],pair,first][idx]-maps[r['dataset'],pair,second][idx]).mean() for pair in range(1,7)])
        primary=r['primary']=='True';t=critical[int(primary)];se=diffs.std(ddof=1)/np.sqrt(6);lo=diffs.mean()-t*se;hi=diffs.mean()+t*se
        err=max(abs(float(r['mean'])-diffs.mean()),abs(float(r['lower'])-lo),abs(float(r['upper'])-hi),abs(float(r['critical_t'])-t));max_ci=max(max_ci,err);assert err<2e-12
        if primary:decisions[r['contrast']]=bool(diffs.mean()<=-.01 and hi<0 and (diffs<0).sum()>=5)
    matrix=np.stack([np.concatenate([maps['zappos_new',p,a]-maps['zappos_new',p,b] for _,a,b in specs]) for p in range(1,7)])
    rng=np.random.default_rng(2428000);strata=[np.flatnonzero(new['labels']==c) for c in np.unique(new['labels'])];boots=[];seedcounts=[]
    for _ in range(400):
        sc=rng.multinomial(6,np.full(6,1/6),size=50);seedcounts.append(sc)
        counts=np.zeros((50,1500),np.int16)
        for indices in strata:counts[:,indices]=rng.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=50)
        weighted=(sc@matrix.reshape(6,-1)).reshape(50,8,1500)/6
        boots.append((weighted*counts[:,None,:]).sum(-1)/1500)
    boot=np.concatenate(boots);saved=dict(np.load(B/'new_source_crossed_bootstrap.npz'));max_boot=float(np.max(abs(boot-saved['values'])))
    assert max_boot<2e-12 and np.array_equal(np.concatenate(seedcounts),saved['seed_counts'])
    for r in read(B/'crossed_bootstrap_contrasts.csv'):
        j=next(i for i,v in enumerate(specs) if v[0]==r['contrast'])*2+(r['metric']=='error');alpha=1-float(r['nominal_interval']);lo,hi=np.quantile(boot[:,j],[alpha/2,1-alpha/2])
        assert max(abs(lo-float(r['lower'])),abs(hi-float(r['upper'])))<2e-12
    success=get(B/'SUCCESS_CRITERIA.json');assert all(decisions.values())==success['new_source_error_ordering_replication_success']
    conv=[28*28*16*9,14*14*32*16*9,7*7*64*32*9];linear=[3136*128,128*10];f=2*sum(conv+linear);g=3*f-2*conv[0];budget=960000*g+135000*f;control=(budget-25000*f)//g
    for r in training:
        n=control if r['arm']=='control' else 480000;assert int(r['training_examples'])==n
        assert int(r['matrix_flops'])==n*g+(110000 if r['arm']=='teacher' else 25000)*f
    assert 0<=budget-control*g-25000*f<g
    record('DISTILLATION_RESULTS.json',dict(selected_checkpoints_replayed=checkpoints,validation_forward_examples=360000,
        temperature_fits=fits,quality_rows=len(quality),classwise_rows=len(classes),max_metric_error=max_metric,
        max_direct_logit_error=max_pred,max_temperature_error=max_beta,max_target_error=max_target,max_validation_error=max_val,
        seed_intervals=len(contrasts),independent_t_critical=critical,max_interval_error=max_ci,
        crossed_bootstrap_replicates=20000,max_crossed_bootstrap_error=max_boot,primary_success=all(decisions.values()),
        additional_audit_forward_examples=extra,committed_files_checked=hashes,full_distillation_matrix_work_per_pair=budget,
        full_control_matrix_work_per_pair=control*g+25000*f,limits='Full forward replay, not independent full retraining. Six seeds condition on a fixed source training dataset; brand dependence remains unresolved.'))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
