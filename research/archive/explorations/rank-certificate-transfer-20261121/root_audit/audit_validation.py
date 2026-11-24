"""Independent reconstruction of risk rows, ledgers and selected training states."""
from pathlib import Path
import csv,gzip,hashlib,json,sys,time
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'validation'))
import adapters

def read_csv(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def sha(b):return hashlib.sha256(b).hexdigest()
def posterior(cohort,x):
    if cohort=='transfer':
        u=np.clip(np.sqrt(x[:,2]**2+x[:,3]**2)-.4,0,1)
        f0=30*u*(1-u)**4;f1=30*u**4*(1-u)
        return np.divide(f1,f0+f1,out=np.where(u>=.5,1.,0.),where=(f0+f1)>0)
    r=np.sqrt((x*x).sum(1));v=.15**2
    f0=np.exp(-.5**2/(2*v))*np.i0(.5*r/v)
    f1=np.exp(-1.2**2/(2*v))*np.i0(1.2*r/v)
    return f1/(f0+f1)

def run():
    out=HERE/'validation_checks';out.mkdir(exist_ok=False)
    rows=read_csv(ROOT/'validation/per_model.csv');samples=read_csv(ROOT/'validation/samples.csv')
    sample_index={(r['cohort'],int(r['pair'])):r for r in samples}
    row_index={(r['cohort'],int(r['pair']),r['method'],r['policy']):r for r in rows}
    metric_errors={};prediction_records={};input_checks=[]
    for cohort in ('rings','transfer'):
        adapter=adapters.get_adapter(cohort)
        for pair in range(1,17):
            rec=sample_index[cohort,pair];x=adapter.sample_inputs(int(rec['test_seed']),65536)
            assert sha(x.tobytes())==rec['x_sha256'];p=posterior(cohort,x)
            assert np.max(abs(p-adapter.posterior(x)))<1e-12
            y=(np.random.default_rng(int(rec['label_seed'])).random(len(x))<p).astype(float)
            assert sha(y.tobytes())==rec['labels_sha256']
            position=np.clip((np.linalg.norm(x,axis=1)-.25)/1.25,0,1) if cohort=='rings' else np.clip(np.linalg.norm(x[:,2:],axis=1)-.4,0,1)
            worlds={'half':np.full(len(x),.5),'increasing':position,'decreasing':1-position,'threshold':(position>=.5).astype(float)}
            for method in adapter.methods:
                pred=dict(np.load(ROOT/'validation/predictions'/f'{cohort}_{pair}_{method}.npz'))
                for j,name in enumerate(pred['names']):
                    h=pred['outputs'][j];mask=np.unpackbits(pred['masks'][j]).astype(bool);key=(cohort,pair,method,str(name))
                    row=row_index[key];assert sha(h.tobytes())==row['output_sha256']
                    assert sha(np.packbits(mask).tobytes())==row['mask_sha256']
                    values=dict(expected_risk=float((np.logaddexp(0,h)-p*h).mean()),
                        sampled_label_risk=float((np.logaddexp(0,h)-y*h).mean()),
                        expected_error=float((p+(h>=0)*(1-2*p)).mean()),
                        sampled_error=float(((h>=0)!=y).mean()),second_forwards=float(mask.sum()))
                    for world,pp in worlds.items():values[world+'_risk']=float((np.logaddexp(0,h)-pp*h).mean())
                    for field,value in values.items():
                        err=abs(value-float(row[field]));metric_errors[field]=max(metric_errors.get(field,0),err)
                        assert err<1e-11,(key,field,err)
                    quota=0 if name=='raw' else (256 if name in ('full','full_radial') else 64)
                    assert np.all(mask.reshape(-1,256).sum(1)==quota)
                    prediction_records[key]=mask
            input_checks.append(dict(cohort=cohort,pair=pair,test_sha256=sha(x.tobytes()),posterior_max_error=float(np.max(abs(p-adapter.posterior(x))))))
    call_count=0;total_forwards=0
    with gzip.open(ROOT/'validation/deployment_audit.csv.gz','rt',newline='',encoding='utf8') as f:
        for row in csv.DictReader(f):
            key=(row['cohort'],int(row['pair']),row['method'],row['policy']);batch=int(row['batch'])
            name=row['policy'];calls=json.loads(row['calls']);expected=[256] if name=='raw' else ([256,256] if name in ('full','full_radial') else [256,64])
            assert calls==expected;mask=prediction_records[key][batch*256:(batch+1)*256]
            assert mask.sum()==int(row['selected']);assert sha(np.packbits(mask).tobytes())==row['mask_sha256']
            call_count+=1;total_forwards+=sum(calls)
    assert len(rows)==320 and call_count==81920
    train_records=[]
    for cohort in ('rings','transfer'):
        adapter=adapters.get_adapter(cohort);base=1000000 if cohort=='rings' else 1040000
        for pair in (1,8,16):
            saved=dict(np.load(ROOT/'validation/states'/f'{cohort}_{pair}.npz'))
            x,y=adapter.sample_train(base+1000+pair);initial=adapter.initialize(base+2000+pair)
            assert np.array_equal(x,saved['x']) and np.array_equal(y,saved['y']) and np.array_equal(initial,saved['initial'])
            for method in adapter.methods:
                start=time.perf_counter();theta=initial.copy();cfg=adapter.training[method]
                for step in range(1,501):
                    theta+=cfg['rate']*adapter.direction(theta,x,y,method,cfg['damping'])
                    if step==100:assert np.array_equal(theta,saved[method+'100'])
                error=float(np.max(abs(theta-saved[method])));assert error==0
                train_records.append(dict(cohort=cohort,pair=pair,method=method,max_error=error,seconds=time.perf_counter()-start))
    # Selected deployment paths are recalculated on every observation of six
    # models. All methods use the same fixed masks and allowed partner calls.
    deployment=[]
    for cohort in ('rings','transfer'):
        adapter=adapters.get_adapter(cohort)
        for pair in (1,8,16):
            sample=sample_index[cohort,pair];x=adapter.sample_inputs(int(sample['test_seed']),65536)
            ties=np.random.default_rng(int(sample['ties_seed'])).random(len(x));keys=np.random.default_rng(int(sample['actions_seed'])).random(len(x))
            state=dict(np.load(ROOT/'validation/states'/f'{cohort}_{pair}.npz'))
            tables={k:dict(np.load(ROOT/'validation/policies'/f'{cohort}_{pair}_pb_{k}.npz')) for k in ('rank','folded','angles')}
            pred=dict(np.load(ROOT/'validation/predictions'/f'{cohort}_{pair}_pb.npz'))
            for j,name in enumerate(pred['names']):
                result,mask=adapters.deploy(adapter,state['pb'],x,str(name),tables,ties,keys)
                error=float(np.max(abs(result-pred['outputs'][j])))
                assert error<1e-12 and np.array_equal(mask,np.unpackbits(pred['masks'][j]).astype(bool))
                deployment.append(dict(cohort=cohort,pair=pair,policy=str(name),max_output_error=error,mask_changes=0,
                    replay_forward_examples=len(x)+int(mask.sum())))
    summary=dict(outcome_rows=len(rows),batch_ledgers=call_count,scored_forwards=total_forwards,
        maximum_metric_errors=metric_errors,input_checks=input_checks,training_replays=train_records,
        deployment_replays=deployment,all_passed=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('input_checks','training_replays','deployment_replays')},indent=2))

if __name__=='__main__':
    with threadpool_limits(1):run()
