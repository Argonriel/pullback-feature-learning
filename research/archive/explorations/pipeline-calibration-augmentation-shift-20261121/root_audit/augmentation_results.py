"""Independent split, fit, quality, bootstrap and complete holdout-CNN audit."""
from pathlib import Path
import csv, datetime, gzip, hashlib, json, os, struct, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from threadpoolctl import threadpool_limits
import torch
from cnn_range_bounds import IndependentCNN

ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'augmentation'
read=lambda p:list(csv.DictReader(p.open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ash=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def ce(z,y):
    z=np.asarray(z,dtype=np.float64)
    return np.logaddexp.reduce(z,axis=1)-z[np.arange(len(y)),y]

def arrays(prefix):
    images=gzip.decompress((HERE/'data'/f'{prefix}-images-idx3-ubyte.gz').read_bytes())
    labels=gzip.decompress((HERE/'data'/f'{prefix}-labels-idx1-ubyte.gz').read_bytes())
    magic,n,h,w=struct.unpack('>4I',images[:16]);lm,ln=struct.unpack('>2I',labels[:8])
    assert (magic,h,w,lm,n)==(2051,28,28,2049,ln)
    return np.frombuffer(images,np.uint8,offset=16).reshape(n,1,h,w).copy(),np.frombuffer(labels,np.uint8,offset=8).astype(np.int64)

def fit_beta(z,y):
    z=np.asarray(z,np.float64);chosen=z[np.arange(len(y)),y]
    def derivative(beta):
        v=beta*z-np.max(beta*z,axis=1,keepdims=True)
        prob=np.exp(v);prob/=prob.sum(1,keepdims=True)
        return float(np.mean(np.sum(prob*z,axis=1)-chosen))
    lo,hi=.25,4.
    if derivative(lo)>=0:return lo
    if derivative(hi)<=0:return hi
    for _ in range(90):
        mid=(lo+hi)/2
        if derivative(mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2

def variants(raw,reflected,row):
    h=np.asarray(raw,np.float64);ht=np.asarray(reflected,np.float64);m=(h+ht)/2
    return dict(raw=h,raw_Traw=h*row['raw']['beta'],full=m,full_Traw=m*row['raw']['beta'],full_Tfull=m*row['full']['beta'])

def main():
    assert (HERE/'TIMING_COMPLETE.json').exists() and (HERE/'ANALYSIS_COMPLETE.json').exists()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    images,labels=arrays('train');test_images,test_labels=arrays('t10k')
    splits=dict(np.load(HERE/'splits.npz'));prior=dict(np.load(HERE/'prior_splits.npz'))
    rng=np.random.default_rng(2161000);held=[]
    for c in range(10):
        old=prior['train'][labels[prior['train']]==c];held.append(rng.permutation(old)[:1000])
    held=rng.permutation(np.concatenate(held));remaining=prior['train'][~np.isin(prior['train'],held)]
    train=rng.permutation(np.concatenate([remaining,prior['selector_fit'],prior['selector_validation']]))
    assert np.array_equal(held,splits['holdout']) and np.array_equal(train,splits['train'])
    assert len(np.unique(np.concatenate(list(splits.values()))))==60000
    assert np.array_equal(splits['validation'],prior['validation']) and np.array_equal(splits['temperature'],prior['temperature'])
    frozen=json.loads((HERE/'SOURCE_FREEZE.json').read_text())
    for relative,expected in frozen['hashes'].items():assert sha(HERE/relative)==expected
    commit=json.loads((HERE/'MODEL_COMMIT.json').read_text());hashes=0
    for relative,expected in commit['manifest'].items():assert sha(HERE/'models'/relative)==expected;hashes+=1
    for relative,expected in json.loads((HERE/'SCORE_COMPLETE.json').read_text())['manifest'].items():
        assert sha(HERE/'predictions'/relative)==expected;hashes+=1
    rows=json.loads((HERE/'temperatures.json').read_text())['rows']
    model_rows={(r['pair'],r['arm']):r for r in rows}
    trace=read(HERE/'training_trace.csv');pairing=json.loads((HERE/'PAIRING_AUDIT.json').read_text())['rows']
    max_beta=0.;checkpoint_checks=0
    for row in rows:
        pair,arm=row['pair'],row['arm'];tag=f'{pair}_{arm}'
        traces=[r for r in trace if int(r['pair'])==pair and r['arm']==arm and r['validation_ce']!='']
        best=min(traces,key=lambda r:(float(r['validation_ce']),int(r['epoch'])))
        selected=torch.load(HERE/'models'/f'{tag}.pt',map_location='cpu',weights_only=True)
        checkpoint=torch.load(HERE/'models'/f"{tag}_epoch{best['epoch']}.pt",map_location='cpu',weights_only=True)
        assert selected['epoch']==int(best['epoch'])
        assert all(torch.equal(value,checkpoint['state_dict'][name]) for name,value in selected['state_dict'].items());checkpoint_checks+=1
        record=next(r for r in pairing if r['pair']==pair and r['arm']==arm)
        rng=np.random.default_rng(2163000+pair);aug_rng=np.random.default_rng(2164000+pair)
        expected_orders=[];expected_coins=[];flip_count=0
        for _ in range(12):
            expected_orders.append(ash(rng.permutation(40000)));coin=aug_rng.random(40000)<.5
            expected_coins.append(ash(coin));flip_count+=int(coin.sum())
        assert expected_orders==record['epoch_order_sha256'] and expected_coins==record['augmentation_coin_sha256']
        assert record['actual_flipped_examples']==(flip_count if arm=='aug' else 0)
        paired=next(r for r in pairing if r['pair']==pair and r['arm']!=arm)
        assert record['initial_sha256']==paired['initial_sha256']
        cal=dict(np.load(HERE/'models'/f'{tag}_calibration.npz'))
        assert np.array_equal(cal['indices'],splits['temperature']) and np.array_equal(cal['labels'],labels[splits['temperature']])
        for name,z in [('raw',cal['raw']),('full',(cal['raw'].astype(float)+cal['reflected'].astype(float))/2)]:
            error=abs(fit_beta(z,cal['labels'])-row[name]['beta']);max_beta=max(max_beta,error);assert error<1e-12
    quality={(r['dataset'],int(r['pair']),r['arm'],r['pipeline']):r for r in read(HERE/'quality.csv')}
    classes={(r['dataset'],int(r['pair']),r['arm'],int(r['class_id'])):r for r in read(HERE/'classwise.csv')}
    max_metric=0.;matrices={};max_direct=0.;extra_forwards=0
    for dataset in ['holdout','official_test']:
        expected_ids=splits['holdout'] if dataset=='holdout' else np.arange(10000)
        expected_labels=labels[expected_ids] if dataset=='holdout' else test_labels
        data={}
        for row in rows:
            pair,arm=row['pair'],row['arm'];pred=dict(np.load(HERE/'predictions'/f'{dataset}_{pair}_{arm}.npz'))
            assert np.array_equal(pred['indices'],expected_ids) and np.array_equal(pred['labels'],expected_labels)
            outputs=variants(pred['raw'],pred['reflected'],row)
            assert np.array_equal(outputs['raw'].argmax(1),outputs['raw_Traw'].argmax(1))
            assert np.array_equal(outputs['full'].argmax(1),outputs['full_Tfull'].argmax(1))
            for name,z in outputs.items():
                loss=ce(z,expected_labels);err=(z.argmax(1)!=expected_labels).astype(float)
                data[pair,arm,name]=(loss,err)
                report=quality[dataset,pair,arm,name]
                for key,value in [('ce',loss.mean()),('error',err.mean())]:
                    error=abs(float(report[key])-value);max_metric=max(max_metric,error);assert error<2e-12
            for c in range(10):
                mask=expected_labels==c;report=classes[dataset,pair,arm,c]
                for key,z in [('original_error',pred['raw']),('reflected_error',pred['reflected']),('full_error',outputs['full'])]:
                    assert abs(np.mean(z.argmax(1)[mask]!=expected_labels[mask])-float(report[key]))<1e-14
            if dataset=='holdout':
                net=IndependentCNN();payload=torch.load(HERE/'models'/f'{pair}_{arm}.pt',map_location='cpu',weights_only=True)
                net.load_state_dict(payload['state_dict']);net.cuda().eval()
                x=torch.as_tensor(images[expected_ids],device='cuda',dtype=torch.float32)/255
                with torch.inference_mode():
                    for key,xx in [('raw',x),('reflected',x[:,:,:,torch.arange(27,-1,-1,device='cuda')])]:
                        direct=torch.cat([net(xx[j:j+512]) for j in range(0,len(xx),512)]).cpu().numpy();extra_forwards+=len(xx)
                        error=float(np.max(abs(direct-pred[key])));max_direct=max(max_direct,error)
                        assert error<1e-5 and np.array_equal(direct.argmax(1),pred[key].argmax(1))
        specs=[('augmentation_raw',('aug','raw_Traw'),('noaug','raw_Traw')),
               ('augmented_full_vs_raw',('aug','full_Tfull'),('aug','raw_Traw')),
               ('noaug_full_vs_raw',('noaug','full_Tfull'),('noaug','raw_Traw')),
               ('aug_full_own_vs_raw_temperature',('aug','full_Tfull'),('aug','full_Traw')),
               ('noaug_full_own_vs_raw_temperature',('noaug','full_Tfull'),('noaug','full_Traw'))]
        matrix=[]
        for _,a,b in specs:
            for metric in (0,1):matrix.append(np.mean([data[p,*a][metric]-data[p,*b][metric] for p in (1,2,3)],axis=0))
        matrices[dataset]=np.array(matrix)
    # Regenerate all bootstrap draws once, shared across the two evaluation datasets.
    combined=np.concatenate(list(matrices.values()),axis=0);rng=np.random.default_rng(2168000);boot=[]
    for _ in range(500):
        counts=rng.multinomial(10000,np.full(10000,.0001),size=20)
        boot.append(counts@combined.T/10000)
    boot=np.concatenate(boot);max_boot=0.;contrast_rows=read(HERE/'contrasts.csv')
    for di,dataset in enumerate(matrices):
        saved=np.load(HERE/f'{dataset}_bootstrap.npz')['values'];actual=boot[:,di*10:(di+1)*10]
        max_boot=max(max_boot,float(np.max(abs(saved-actual))));assert np.max(abs(saved-actual))<1e-12
        rows_for_dataset=[r for r in contrast_rows if r['dataset']==dataset]
        for j,row in enumerate(rows_for_dataset):
            alpha=.0125 if dataset=='holdout' and j<4 else .05
            lo,hi=np.quantile(actual[:,j],[alpha/2,1-alpha/2])
            assert max(abs(lo-float(row['lower'])),abs(hi-float(row['upper'])),abs(matrices[dataset][j].mean()-float(row['mean'])))<1e-12
    timings=read(HERE/'latency.csv');costs=read(HERE/'timing_calls.csv')
    assert len(timings)==378
    assert sum(int(r['actual_forward_examples']) for r in costs if r['warmup']=='False')==645120
    assert sum(int(r['actual_forward_examples']) for r in costs if r['warmup']=='True')==276480
    for row in read(HERE/'serving_summary.csv'):
        medians={method:np.median([float(r['ms_per_batch']) for r in timings if r['pair']==row['pair'] and r['arm']==row['arm'] and r['mode']==row['mode'] and r['method']==method]) for method in ['raw','full_fused','full_sequential']}
        for method,value in medians.items():assert abs(float(row[method+'_ms'])-value)<1e-14
        assert abs(float(row['strongest_full_ms'])-min(medians['full_fused'],medians['full_sequential']))<1e-14
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',
        split_ids_reconstructed=60000,checkpoint_choices=checkpoint_checks,temperature_fits=12,
        max_temperature_error=max_beta,quality_rows=60,classwise_rows=120,max_metric_error=max_metric,
        direct_holdout_model_view_replays=12,max_direct_logit_error=max_direct,direct_class_discrepancies=0,
        bootstrap_intervals=20,max_bootstrap_replay_error=max_boot,latency_rows=378,
        measured_forward_examples=645120,warmup_forward_examples=276480,quality_forward_examples=240000,
        additional_audit_forward_examples=extra_forwards,committed_files=hashes,
        limitation='Holdout excluded from this round, but images were used in historical studies; image bootstrap conditions on these six models.')
    (ROOT/'root_audit/AUGMENTATION_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
