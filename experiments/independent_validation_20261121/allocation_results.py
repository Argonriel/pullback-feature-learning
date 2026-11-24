"""Independently reconstruct allocation outcomes, fits, streams and work."""
from pathlib import Path
import csv,datetime,hashlib,importlib.util,json
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent;B=ROOT/'allocation'
spec=importlib.util.spec_from_file_location('root_allocation_task',ROOT/'root_audit/reference/task.py')
task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
read=lambda p:list(csv.DictReader(p.open(newline='')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((B/'SOURCE_FREEZE.json').read_text())['config']
rows=read(B/'per_model.csv');rowmap={(int(r['pair']),r['procedure'],r['arm']):r for r in rows}

def forward(theta,x):
    a=theta[:32].reshape(8,4);b=theta[32:64].reshape(4,8);bias=theta[64:72]
    z=x+np.tanh(x@a.T+bias)@b.T/2
    a=theta[72:104].reshape(8,4);b=theta[104:136].reshape(4,8);bias=theta[136:]
    z=z+np.tanh(z@a.T+bias)@b.T/2
    return z[:,0]

def scale(h,y):
    lo,hi=np.exp(-3),np.exp(3)
    derivative=lambda b:float(np.mean(-y*h/(1+np.exp(np.clip(y*b*h,-700,700)))))
    if derivative(lo)>=0:return lo
    if derivative(hi)<=0:return hi
    for _ in range(90):
        mid=(lo+hi)/2
        if derivative(mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2

max_metric=max_beta=max_pred=0.;additional=0;committed=0
assert (B/'ANALYSIS_COMPLETE.json').exists()
with threadpool_limits(limits=1):
    for name in ('MODEL_COMMIT.json','FIT_COMMIT.json','SCORE_COMPLETE.json'):
        for rel,h in json.loads((B/name).read_text())['manifest'].items():
            assert sha(B/rel)==h,rel;committed+=1
    for pair in range(1,17):
        split=dict(np.load(B/'splits'/f'{pair}.npz'));state=dict(np.load(B/'states'/f'{pair}.npz'))
        pred=dict(np.load(B/'predictions'/f'{pair}.npz'))
        chunks=[];ys=[]
        for offset,size in zip([0,100,200,300],[256,512,512,1024]):
            x,y=task.sample_train(2311000+offset+pair,size);chunks.append(x);ys.append(y)
        assert np.array_equal(split['initial'],task.initialize(2312000+pair))
        for i,n in enumerate((256,768,1280,2304)):
            order=np.random.default_rng(2311800+pair).permutation(n)
            assert np.array_equal(split[f'x_{n}'],np.concatenate(chunks[:i+1])[order])
            assert np.array_equal(split[f'y_{n}'],np.concatenate(ys[:i+1])[order])
        for prefix,seed,n in [('validation',2313000,1024),('calibration',2314000,4096)]:
            x,y,_=task.sample_test(seed+pair,n)
            assert np.array_equal(split[prefix+'_x'],x) and np.array_equal(split[prefix+'_y'],y)
        assert np.array_equal(pred['x'],task.sample_inputs(2316000+pair,65536))
        assert np.array_equal(pred['p'],task.posterior(pred['x']))
        assert np.array_equal(pred['y'],np.where(np.random.default_rng(2317000+pair).random(65536)<pred['p'],1.,-1.))
        xv=split['validation_x'];yv=split['validation_y']
        vv=[np.logaddexp(0,-yv*forward(state[f'checkpoint_1280_{s}'],xv)).mean() for s in [100,250,500,896]]
        additional+=4096
        selected=[100,250,500,896][int(np.argmin(vv))]
        assert selected==int(state['validation_selected'])
        assert np.array_equal(state['validation_1280'],state[f'checkpoint_1280_{selected}'])
        for proc in cfg['procedures']:
            name=proc['name'];n=proc['calibration'];theta=state[name]
            fits=json.loads((B/'temperatures'/f'{pair}_{name}.json').read_text())['fits']
            cal=dict(np.load(B/'temperatures'/f'{pair}_{name}_outputs.npz'))
            xc=split['calibration_x'][:n];yc=split['calibration_y'][:n]
            hc=forward(theta,xc);xt=xc.copy();xt[:,:2]*=-1;ht=forward(theta,xt);additional+=2*n
            for key,h in [('raw',hc),('partner',ht),('full',(hc+ht)/2)]:
                assert np.max(abs(h-cal[key]))<2e-12
            assert np.array_equal(cal['labels'],yc)
            for key in ('raw','full'):
                err=abs(scale(cal[key],yc)-fits[key]['beta']);max_beta=max(max_beta,err);assert err<2e-12
            # Complete independent forward replay, not only saved-output arithmetic.
            h=forward(theta,pred['x']);xt=pred['x'].copy();xt[:,:2]*=-1;ht=forward(theta,xt);additional+=2*len(h)
            for key,expected in [('raw',h),('full',(h+ht)/2)]:
                e=float(np.max(abs(expected-pred[name+'_'+key])));max_pred=max(max_pred,e);assert e<2e-12
            for arm,pipeline,temp in cfg['arms']:
                base=pred[name+'_'+pipeline];out=base*(1 if temp=='none' else fits[temp]['beta'])
                assert np.array_equal(base>=0,out>=0)
                p=pred['p'];y=pred['y'];q=1/(1+np.exp(-out));bins=np.minimum((q*10).astype(int),9)
                values=dict(expected_nll=np.mean((1-p)*np.logaddexp(0,out)+p*np.logaddexp(0,-out)),
                    expected_error=np.where(out>=0,1-p,p).mean(),sampled_nll=np.logaddexp(0,-y*out).mean(),
                    sampled_error=np.mean((out>=0)!=(y==1)),expected_brier=np.mean((q-p)**2+p*(1-p)),
                    posterior_binned_ece=sum(abs(np.sum(q[bins==i]-p[bins==i])) for i in range(10))/len(q))
                row=rowmap[(pair,name,arm)]
                for key,val in values.items():
                    err=abs(float(val)-float(row[key]));max_metric=max(max_metric,err);assert err<2e-12
                count=proc['n']+proc['validation']+(0 if temp=='none' else n)
                assert int(row['actual_unique_fit_labels'])==count
    training=read(B/'training.csv');ledger=read(B/'deployment_ledger.csv');actual=read(B/'actual_training_work.csv')
    # Literal scalar-operation derivation reassembled without importing runner.work.
    p=144
    lu=sum(m+2*m*m for m in range(1,p));substitutions=2*p*(p-1)+p
    coefficient=2*136+2*(64+16+400+4032+512+384+96+72)+5+8*p*p+4*p+8*p
    constant=-p*p+p-p+lu+substitutions+2*p
    assert (coefficient,constant)==(179045,2001144)
    for r in training+actual:
        n=int(r['n_train']);s=int(r['updates_executed'])
        for key,val in [('training_example_updates',n*s),('jacobian_rows',4*n*s),('lu_solves_144',s),('nominal_arithmetic_flops',(coefficient*n+constant)*s),('tanh_scalar_evaluations',32*n*s)]:
            assert int(r[key])==val
        if r.get('regime')=='nominal_flops':assert s==(500*(coefficient*2304+constant))//(coefficient*n+constant)
    for r in ledger:
        mult=1 if r['arm'].startswith('raw') else 2
        assert int(r['inference_forward_examples'])==65536*mult
        assert int(r['inference_invocations'])==256*mult
    draws=np.random.default_rng(2319000).integers(0,16,(20000,16))
    paired=read(B/'paired_contrasts.csv');contrasts=read(B/'contrasts.csv');max_ci=0.;prim=[]
    for r in contrasts:
        # Independently rebuild all differences from procedure rows, using arm definitions.
        label=r['contrast'];metric=r['metric']
        if label.endswith('_full_own_vs_rawT'):
            first=second=label[:-len('_full_own_vs_rawT')];aa='full_T';ab='full_rawT'
        else:
            aa=ab='raw_T' if label.endswith('_raw_T') else 'full_T'
            stem=label[:-len('_'+aa)]
            if stem=='validation_vs_flop1280':first,second='validation_1280','flop_1280'
            else:
                first,second=stem.split('_vs_');second='fixed_2304' if second=='baseline' else second
        d=np.array([float(rowmap[(i,first,aa)][metric])-float(rowmap[(i,second,ab)][metric]) for i in range(1,17)])
        pr=sorted([v for v in paired if v['contrast']==label and v['metric']==metric],key=lambda v:int(v['pair']))
        assert np.max(abs(d-np.array([float(v['difference']) for v in pr])))<1e-14
        alpha=float(r['alpha']);low,high=np.quantile(d[draws].mean(1),[alpha/2,1-alpha/2])
        err=max(abs(d.mean()-float(r['mean_difference'])),abs(low-float(r['lower'])),abs(high-float(r['upper'])));max_ci=max(max_ci,err);assert err<2e-14
        if r['primary']=='True':prim.append(r)
    assert len(prim)==8

out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    metric_rows=len(rows),replayed_full_test_models=144,temperature_fits=288,checkpoint_selections=16,
    max_metric_error=max_metric,max_forward_error=max_pred,max_beta_error=max_beta,
    contrasts=len(contrasts),max_interval_error=max_ci,committed_files_checked=committed,
    additional_audit_forward_examples=additional,training_procedure_ledger_rows=len(training),
    actual_training_trajectories=len(actual),nominal_flops_formula=[coefficient,constant],
    actual_deployment_forwards=sum(int(r['inference_forward_examples']) for r in ledger),
    actual_training_nominal_flops=sum(int(r['nominal_arithmetic_flops']) for r in actual),
    primary_contrasts=prim,
    limits='All saved model outputs replayed; full training trajectories are not independently retrained.')
(ROOT/'root_audit/ALLOCATION_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k!='primary_contrasts'},indent=2))
