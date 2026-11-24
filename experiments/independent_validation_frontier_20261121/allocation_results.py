"""Reconstruct both phases, recipe selection, temperatures, predictions and costs."""
import importlib.util
import numpy as np
from threadpoolctl import threadpool_limits
from common import ROOT,read,get,sha,record,synthetic_forward,binary_beta
B=ROOT/'allocation'
spec=importlib.util.spec_from_file_location('root_frozen_task',ROOT/'root_audit/reference/task.py')
task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)

def main():
    cfg=get(B/'SOURCE_FREEZE.json')['config'];selected=get(B/'SELECTION_COMMIT.json')['selected_procedure']
    phases={};max_metric=max_pred=max_beta=0.;extra=models=fits=hashes=0;actual_ops=training_union=0
    p=144;lu=sum(m+2*m*m for m in range(1,p));sub=2*p*(p-1)+p
    slope=2*136+2*(64+16+400+4032+512+384+96+72)+5+8*p*p+4*p+8*p
    constant=-p*p+p-p+lu+sub+2*p
    assert (slope,constant)==(179045,2001144)
    budget=500*(slope*2304+constant)
    for phase,count,offset in [('development',8,0),('confirmation',16,50)]:
        folder=B/phase;rows=read(folder/'per_model.csv');phases[phase]=rows
        mapping={(int(r['pair']),r['procedure'],r['arm']):r for r in rows}
        procs=cfg['procedures'] if phase=='development' else [v for v in cfg['procedures'] if v['name'] in ('train_2304',selected)]
        for name in ('MODEL_COMMIT.json','FIT_COMMIT.json','SCORE_COMPLETE.json'):
            for rel,h in get(folder/name)['manifest'].items():assert sha(B/rel)==h;hashes+=1
        for pair in range(1,count+1):
            sid=pair+offset;split=dict(np.load(folder/'splits'/f'{pair}.npz'));state=dict(np.load(folder/'states'/f'{pair}.npz'));pred=dict(np.load(folder/'predictions'/f'{pair}.npz'))
            chunks=[];ys=[];maxn=max(v['n'] for v in procs);total=0
            for seedoff,size in zip([0,100,200,300],[2304,768,512,256]):
                x,y=task.sample_train(2411000+sid+seedoff,size);chunks.append(x);ys.append(y);total+=size
                if total==maxn:break
            assert np.array_equal(split['initial'],task.initialize(2412000+sid))
            cx,cy,_=task.sample_test(2414000+sid,2048)
            assert np.array_equal(split['calibration_x'],cx) and np.array_equal(split['calibration_y'],cy)
            assert np.array_equal(pred['x'],task.sample_inputs(2416000+sid,65536))
            posterior=task.posterior(pred['x'])
            assert np.array_equal(pred['y'],np.where(np.random.default_rng(2417000+sid).random(65536)<posterior,1.,-1.))
            if phase=='confirmation':assert np.array_equal(pred['p'],posterior)
            else:assert 'p' not in pred
            training_union+=maxn+2048
            for proc in procs:
                n=proc['n'];name=proc['name'];take=[2304,3072,3584,3840].index(n)+1;order=np.random.default_rng(2411800+sid).permutation(n)
                assert np.array_equal(split[f'x_{n}'],np.concatenate(chunks[:take])[order])
                assert np.array_equal(split[f'y_{n}'],np.concatenate(ys[:take])[order])
                assert proc['steps']==budget//(slope*n+constant)
                theta=state[name];cal=dict(np.load(folder/'temperatures'/f'{pair}_{name}_outputs.npz'))
                temperatures=get(folder/'temperatures'/f'{pair}_{name}.json')['fits'];nc=proc['calibration']
                x=split['calibration_x'][:nc];y=split['calibration_y'][:nc];tx=x.copy();tx[:,:2]*=-1
                h=synthetic_forward(theta,x);ht=synthetic_forward(theta,tx);extra+=2*nc
                for key,v in [('raw',h),('partner',ht),('full',(h+ht)/2)]:assert np.max(abs(v-cal[key]))<2e-12
                assert np.array_equal(cal['labels'],y)
                for key in ('raw','full'):
                    err=abs(binary_beta(cal[key],y)-temperatures[key]['beta']);max_beta=max(max_beta,err);assert err<2e-12;fits+=1
                x=pred['x'];tx=x.copy();tx[:,:2]*=-1;h=synthetic_forward(theta,x);ht=synthetic_forward(theta,tx);extra+=2*len(x);models+=1
                for key,v in [('raw',h),('full',(h+ht)/2)]:
                    err=float(np.max(abs(v-pred[name+'_'+key])));max_pred=max(max_pred,err);assert err<2e-12
                for arm,pipeline,temp in cfg['arms']:
                    base=pred[name+'_'+pipeline];out=base*(1 if temp=='none' else temperatures[temp]['beta'])
                    assert np.array_equal(base>=0,out>=0)
                    y=pred['y'];q=1/(1+np.exp(-out));target=(y==1).astype(float);bins=np.minimum((q*10).astype(int),9)
                    values=dict(sampled_nll=np.logaddexp(0,-y*out).mean(),sampled_error=np.mean((out>=0)!=(y==1)),sampled_brier=np.mean((q-target)**2))
                    if phase=='confirmation':
                        pp=posterior
                        values.update(expected_nll=np.mean((1-pp)*np.logaddexp(0,out)+pp*np.logaddexp(0,-out)),expected_error=np.where(out>=0,1-pp,pp).mean(),expected_brier=np.mean((q-pp)**2+pp*(1-pp)))
                    ecetarget=posterior if phase=='confirmation' else target
                    values['posterior_binned_ece' if phase=='confirmation' else 'sampled_binned_ece']=sum(abs(np.sum(q[bins==i]-ecetarget[bins==i])) for i in range(10))/len(q)
                    row=mapping[pair,name,arm]
                    for key,value in values.items():
                        err=abs(float(value)-float(row[key]));max_metric=max(max_metric,err);assert err<2e-12,(phase,pair,name,arm,key)
                    assert int(row['actual_unique_fit_labels'])==n+(0 if temp=='none' else nc)
        for r in read(folder/'training.csv'):
            n=int(r['n_train']);s=int(r['updates_executed'])
            for key,value in [('training_example_updates',n*s),('jacobian_rows',4*n*s),('lu_144_solves',s),('nominal_arithmetic_flops',(slope*n+constant)*s)]:assert int(r[key])==value
            actual_ops+=int(r['nominal_arithmetic_flops'])
        for r in read(folder/'deployment_ledger.csv'):
            assert int(r['inference_forward_examples'])==65536*(1 if r['arm'].startswith('raw') else 2)
    scores={v['name']:np.mean([float(r['sampled_nll']) for r in phases['development'] if r['procedure']==v['name'] and r['arm']=='raw_T']) for v in cfg['procedures']}
    minimum=min(scores.values());chosen=min((v for v in cfg['procedures'] if scores[v['name']]<=minimum+1e-12),key=lambda v:v['n'])['name']
    assert chosen==selected
    assert get(B/'confirmation/TRAIN_START.json')['selection_commit_sha256']==sha(B/'SELECTION_COMMIT.json')
    assert get(B/'SELECTION_COMMIT.json')['utc']<get(B/'confirmation/TRAIN_START.json')['utc']
    draw=np.random.default_rng(2419050).integers(0,16,(20000,16));rows=phases['confirmation'];max_ci=0.
    for r in read(B/'confirmation/contrasts.csv'):
        def values(name):return np.array([float(z[r['metric']]) for z in sorted((z for z in rows if z['procedure']==name and z['arm']==r['arm']),key=lambda z:int(z['pair']))])
        diff=values(selected)-values('train_2304');assert abs(diff.mean()-float(r['mean_difference']))<2e-14
        if selected!='train_2304':
            alpha=float(r['alpha']);lo,hi=np.quantile(diff[draw].mean(1),[alpha/2,1-alpha/2]);err=max(abs(lo-float(r['lower'])),abs(hi-float(r['upper'])));max_ci=max(max_ci,err);assert err<2e-14
        else:assert not r['lower'] and not r['upper']
    record('ALLOCATION_RESULTS.json',dict(replayed_models=models,temperature_fits=fits,metric_rows=sum(map(len,phases.values())),
        selected_procedure=selected,development_selection_reconstructed=True,development_selection_labels=8*65536,
        confirmation_pairs=16,max_metric_error=max_metric,max_forward_error=max_pred,max_temperature_error=max_beta,
        max_interval_error=max_ci,committed_files_checked=hashes,additional_audit_forward_examples=extra,
        actual_fit_label_union=training_union,actual_nominal_training_arithmetic=actual_ops,
        limits='Independent complete final-output replay and arithmetic reconstruction; full training is not independently repeated.'))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
