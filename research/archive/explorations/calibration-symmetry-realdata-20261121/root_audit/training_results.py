"""Independent reconstruction of confirmation risks, splits, fits and bootstrap."""
from pathlib import Path
import csv,gzip,json,hashlib,datetime,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'training'
sys.path.insert(0,str(HERE));import model
from adapters import load_variant,policy_paths,VARIANTS,FAMILY_FOR_VARIANT
sys.path.insert(0,str(HERE/'reference'));import baseline
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
assert (HERE/'ANALYSIS_COMPLETE.json').exists()
rowmap={(int(r['pair']),r['variant'],r['policy']):r for r in read(HERE/'per_model.csv')}
max_metric=max_prediction=max_temperature=0.;mask_changes=0;manifest_checks=0;extra_forwards=0
with threadpool_limits(limits=1):
 for name in ('MODEL_COMMIT.json','POLICY_COMMIT.json','SCORE_COMPLETE.json'):
    for p,h in json.loads((HERE/name).read_text())['manifest'].items():assert sha(HERE/p)==h,p;manifest_checks+=1
 for pair in range(1,17):
    d=dict(np.load(HERE/'predictions'/f'{pair}.npz'));state=dict(np.load(HERE/'states'/f'{pair}.npz'));split=dict(np.load(HERE/'splits'/f'{pair}.npz'))
    tx,ty=model.task.sample_train(2101000+pair);assert np.array_equal(tx,state['x']) and np.array_equal(ty,state['y'])
    assert np.array_equal(model.task.initialize(2102000+pair),state['initial'])
    assert np.array_equal(d['x'],model.task.sample_inputs(2106000+pair,65536))
    assert np.array_equal(d['ties'],np.random.default_rng(2107000+pair).random(65536))
    posterior=model.task.posterior(d['x']);assert np.array_equal(d['p'],posterior)
    labels=np.where(np.random.default_rng(2108000+pair).random(65536)<posterior,1.,-1.);assert np.array_equal(d['y'],labels)
    for prefix,seed in [('validation',2103000+pair),('temperature',2104000+pair)]:
        x,y=model.labeled_inputs(seed,2048);assert np.array_equal(x,split[prefix+'_x']) and np.array_equal(y,split[prefix+'_y'])
    temps=json.loads((HERE/'temperatures'/f'{pair}.json').read_text())['families']
    for family in ('legacy_pb','selected_pb','selected_gd'):
        h=state[family+'_calibration_logits'];y=split['temperature_y'];lo,hi=np.exp(-3),np.exp(3)
        derivative=lambda b:float(np.mean(-y*h/(1+np.exp(np.clip(y*b*h,-700,700)))))
        if derivative(lo)>=0:beta=lo
        elif derivative(hi)<=0:beta=hi
        else:
            for _ in range(90):
                mid=(lo+hi)/2
                if derivative(mid)>0:hi=mid
                else:lo=mid
            beta=(lo+hi)/2
        error=abs(beta-temps[family]['beta']);max_temperature=max(max_temperature,error);assert error<1e-12
        if family!='legacy_pb':
            checkpoint=state[family+'_checkpoints'];losses=[model.task.loss(th,split['validation_x'],split['validation_y']) for th in checkpoint];extra_forwards+=4*2048
            chosen=int(np.argmin(losses));assert chosen==int(state[family+'_selected_index'])
            assert np.array_equal(state[family],checkpoint[chosen])
    masks=np.unpackbits(d['masks'],axis=-1)[...,:65536]
    for vi,variant in enumerate(d['variants']):
        adapter,theta,meta=load_variant(pair,str(variant));paths=policy_paths(pair,str(variant));grid=dict(np.load(paths['grid']));fold=dict(np.load(paths['folded']))
        independent_masks=[np.zeros(65536,bool),np.ones(65536,bool),baseline.select_folded(adapter,d['x'],fold,d['ties']),baseline.select(adapter,d['x'],grid,d['ties'])]
        raw=d['outputs'][vi,0];partner=2*d['outputs'][vi,1]-raw
        for pi,policy in enumerate(d['policies']):
            mask=independent_masks[pi];mask_changes+=int(np.sum(mask!=masks[vi,pi]));assert np.array_equal(mask,masks[vi,pi])
            expected=np.where(mask,(raw+partner)/2,raw);h=d['outputs'][vi,pi]
            error=float(np.max(abs(expected-h)));max_prediction=max(max_prediction,error);assert error<1e-12
            q=1/(1+np.exp(-h));p=posterior;nll=(1-p)*np.logaddexp(0,h)+p*np.logaddexp(0,-h)
            bin_ids=np.minimum((q*10).astype(int),9);ece=sum(abs(float(np.sum(q[bin_ids==b]-p[bin_ids==b]))) for b in range(10))/len(q)
            metrics=dict(expected_nll=nll.mean(),expected_error=np.where(h>=0,1-p,p).mean(),sampled_nll=np.logaddexp(0,-labels*h).mean(),
                sampled_error=np.mean((h>=0)!=(labels==1)),expected_brier=np.mean((q-p)**2+p*(1-p)),
                squared_posterior_error=np.mean((q-p)**2),confidence=np.maximum(q,1-q).mean(),posterior_binned_ece=ece)
            row=rowmap[(pair,str(variant),str(policy))]
            for key,val in metrics.items():
                err=abs(float(val)-float(row[key]));max_metric=max(max_metric,err);assert err<2e-12,(pair,variant,policy,key,err)
        if str(variant).endswith('_temp'):
            assert np.array_equal(d['outputs'][vi,0]>=0,d['outputs'][vi-1,0]>=0)
        if pair in (1,16) and variant in ('legacy_raw','pb_temp','gd_temp'):
            first=adapter.predict(theta,d['x'][:256]);second=adapter.predict(theta,adapter.partner(d['x'][:256]));extra_forwards+=512
            assert np.max(abs(first-raw[:256]))<1e-12 and np.max(abs(second-partner[:256]))<1e-12
 with gzip.open(HERE/'deployment_ledger.csv.gz','rt') as f:
    ledger=list(csv.DictReader(f))
 for row in ledger:
    name=row['policy'];expected=[256] if name=='raw' else [256,256] if name=='full' else [256,64]
    assert list(map(int,row['calls'].split('|')))==expected
    assert int(row['forward_examples'])==sum(expected)
 total=sum(int(r['forward_examples']) for r in ledger);assert len(ledger)==98304 and total==34603008
 draws=np.random.default_rng(2109001).integers(0,16,(20000,16));bootcheck=[]
 for row in read(HERE/'contrasts.csv'):
    metric=row['metric'];a=(row['a_variant'],row['a_policy']);b=(row['b_variant'],row['b_policy'])
    delta=np.array([float(rowmap[(i,*a)][metric])-float(rowmap[(i,*b)][metric]) for i in range(1,17)])
    alpha=float(row['alpha']);lo,hi=np.quantile(delta[draws].mean(1),[alpha/2,1-alpha/2])
    assert max(abs(lo-float(row['lower'])),abs(hi-float(row['upper'])))<1e-14
    if row['primary']=='True':bootcheck.append(dict(contrast=row['contrast'],mean=float(delta.mean()),lower=float(lo),upper=float(hi)))
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',models=48,variants=96,metric_rows=384,
 max_metric_error=max_metric,max_output_error=max_prediction,mask_discrepancies=mask_changes,max_temperature_error=max_temperature,
 committed_files=manifest_checks,seed_replays=16,validation_checkpoint_choices=32,temperature_fits=48,
 deployment_ledger_rows=len(ledger),actual_deployment_forwards=total,additional_audit_forward_examples=extra_forwards,primary_bootstraps=bootcheck)
(ROOT/'root_audit/TRAINING_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
