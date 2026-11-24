"""Independent reconstruction of equal-budget final-pipeline confirmation."""
from pathlib import Path
from types import SimpleNamespace
import csv, datetime, gzip, hashlib, json, sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'calibration'
sys.path.insert(0,str(HERE));import model
sys.path.insert(0,str(HERE/'reference'));import baseline
read=lambda p:list(csv.DictReader(p.open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (HERE/'ANALYSIS_COMPLETE.json').exists()
families=['legacy','selected','extra_train']
arms=['raw_uncal','raw_rawT','full_uncal','full_rawT','full_finalT','interp_uncal','interp_rawT','interp_finalT']
rowmap={(int(r['pair']),r['family'],r['arm']):r for r in read(HERE/'per_model.csv')}
adapter=SimpleNamespace(name='two_invariants',dimension=4,quotient_dim=2,quotient=model.task.quotient,
    partner=model.task.partner,predict=model.task.predict,coordinates=model.task.coordinates,angle=model.task.angle,
    support_lo=np.array([.4,.4]),support_hi=np.array([1.4,1.4]))
max_metric=max_output=max_beta=0.;committed=0;extra_forwards=0
with threadpool_limits(limits=1):
    for name in ['MODEL_COMMIT.json','FIELD_COMMIT.json','TEMPERATURE_COMMIT.json','SCORE_COMPLETE.json']:
        for relative,expected in json.loads((HERE/name).read_text())['manifest'].items():
            assert sha(HERE/relative)==expected,relative
            committed+=1
    for pair in range(1,17):
        state=dict(np.load(HERE/'states'/f'{pair}.npz'))
        split=dict(np.load(HERE/'splits'/f'{pair}.npz'))
        pred=dict(np.load(HERE/'predictions'/f'{pair}.npz'))
        for prefix,seed,n in [('',2151000,256),('extra_',2151500,2048)]:
            x,y=model.task.sample_train(seed+pair,n)
            assert np.array_equal(x,state[prefix+'x']) and np.array_equal(y,state[prefix+'y'])
        assert np.array_equal(state['initial'],model.task.initialize(2152000+pair))
        assert np.array_equal(state['combined_permutation'],np.random.default_rng(2151600+pair).permutation(2304))
        for prefix,seed,n in [('validation',2153000,2048),('calibration',2154000,4096)]:
            x,y=model.labeled_inputs(seed+pair,n)
            assert np.array_equal(x,split[prefix+'_x']) and np.array_equal(y,split[prefix+'_y'])
        assert np.array_equal(split['calibration_ties'],np.random.default_rng(2157500+pair).random(4096))
        assert np.array_equal(pred['x'],model.task.sample_inputs(2156000+pair,65536))
        assert np.array_equal(pred['ties'],np.random.default_rng(2157000+pair).random(65536))
        p=model.task.posterior(pred['x']);assert np.array_equal(p,pred['p'])
        y=np.where(np.random.default_rng(2158000+pair).random(65536)<p,1.,-1.)
        assert np.array_equal(y,pred['y'])
        validation=[model.task.loss(th,split['validation_x'],split['validation_y']) for th in state['selected_checkpoints']]
        extra_forwards+=4*2048
        assert int(state['selected_index'])==int(np.argmin(validation))
        assert np.array_equal(state['selected'],state['selected_checkpoints'][int(state['selected_index'])])
        masks=np.unpackbits(pred['masks'],axis=-1)[...,:65536]
        for fi,family in enumerate(families):
            theta=state[family];table=dict(np.load(HERE/'fields'/f'{pair}_{family}.npz'))
            n=4096 if family=='legacy' else 2048
            fit=json.loads((HERE/'temperatures'/f'{pair}_{family}.json').read_text())['fits']
            saved=dict(np.load(HERE/'temperatures'/f'{pair}_{family}_outputs.npz'))
            xc=split['calibration_x'][:n];yc=split['calibration_y'][:n]
            hc=model.task.predict(theta,xc);ht=model.task.predict(theta,model.task.partner(xc));extra_forwards+=2*n
            mc=baseline.select(adapter,xc,table,split['calibration_ties'][:n])
            assert np.array_equal(mc,saved['mask']) and np.array_equal(yc,saved['labels'])
            for pipeline,h in [('raw',hc),('full',(hc+ht)/2),('interp',np.where(mc,(hc+ht)/2,hc))]:
                assert np.max(abs(h-saved[pipeline]))<1e-12
                lo,hi=np.exp(-3),np.exp(3)
                derivative=lambda b:float(np.mean(-yc*h/(1+np.exp(np.clip(yc*b*h,-700,700)))))
                if derivative(lo)>=0:beta=lo
                elif derivative(hi)<=0:beta=hi
                else:
                    for _ in range(90):
                        mid=(lo+hi)/2
                        if derivative(mid)>0:hi=mid
                        else:lo=mid
                    beta=(lo+hi)/2
                err=abs(beta-fit[pipeline]['beta']);max_beta=max(max_beta,err);assert err<1e-12
            raw=pred['outputs'][fi,0];full=pred['outputs'][fi,2]
            mask=baseline.select(adapter,pred['x'],table,pred['ties'])
            assert np.all(mask.reshape(-1,256).sum(1)==64)
            base=[raw,raw,full,full,full,np.where(mask,full,raw),np.where(mask,full,raw),np.where(mask,full,raw)]
            scale=[1,fit['raw']['beta'],1,fit['raw']['beta'],fit['full']['beta'],1,fit['raw']['beta'],fit['interp']['beta']]
            for ai,arm in enumerate(arms):
                h=pred['outputs'][fi,ai];expected=base[ai]*scale[ai]
                err=float(np.max(abs(h-expected)));max_output=max(max_output,err);assert err<1e-12
                expected_mask=np.zeros(65536,bool) if ai<2 else np.ones(65536,bool) if ai<5 else mask
                assert np.array_equal(masks[fi,ai],expected_mask)
                assert np.array_equal(h>=0,base[ai]>=0)
                q=np.exp(-np.logaddexp(0,-h));ids=np.minimum((10*q).astype(int),9)
                values=dict(expected_nll=np.mean((1-p)*np.logaddexp(0,h)+p*np.logaddexp(0,-h)),
                    expected_error=np.where(h>=0,1-p,p).mean(),sampled_nll=np.logaddexp(0,-y*h).mean(),
                    sampled_error=np.mean((h>=0)!=(y==1)),expected_brier=np.mean((q-p)**2+p*(1-p)),
                    posterior_binned_ece=sum(abs(float(np.sum(q[ids==b]-p[ids==b]))) for b in range(10))/len(q),
                    confidence=np.maximum(q,1-q).mean())
                row=rowmap[(pair,family,arm)]
                for key,value in values.items():
                    err=abs(float(value)-float(row[key]));max_metric=max(max_metric,err);assert err<2e-12,(pair,family,arm,key)
                expected_labels=(256 if family=='legacy' else 2304) if arm.endswith('uncal') else 4352
                assert int(row['unique_labels'])==expected_labels
            if pair in (1,16):
                direct=model.task.predict(theta,pred['x'][:256]);partner=model.task.predict(theta,model.task.partner(pred['x'][:256]));extra_forwards+=512
                assert np.max(abs(direct-raw[:256]))<1e-12
                assert np.max(abs((direct+partner)/2-full[:256]))<1e-12
    with gzip.open(HERE/'deployment_ledger.csv.gz','rt') as stream:ledger=list(csv.DictReader(stream))
    for row in ledger:
        expected=[256] if row['arm'].startswith('raw') else [256,256] if row['arm'].startswith('full') else [256,64]
        assert list(map(int,row['calls'].split('|')))==expected
        assert int(row['forwards'])==sum(expected)
        assert int(row['selected'])==(0 if expected==[256] else 256 if expected==[256,256] else 64)
    total=sum(int(r['forwards']) for r in ledger);assert len(ledger)==98304 and total==36962304
    definitions={
        'selected_full_final_vs_rawT':('selected','full_finalT','selected','full_rawT'),
        'selected_interp_final_vs_rawT':('selected','interp_finalT','selected','interp_rawT'),
        'extra_training_vs_legacy_rawT':('extra_train','raw_rawT','legacy','raw_rawT'),
        'selected_vs_legacy_rawT':('selected','raw_rawT','legacy','raw_rawT')}
    draws=np.random.default_rng(2159000).integers(0,16,(20000,16));primary=[]
    for row in read(HERE/'contrasts.csv'):
        fa,aa,fb,ab=definitions[row['contrast']];metric=row['metric']
        delta=np.array([float(rowmap[(i,fa,aa)][metric])-float(rowmap[(i,fb,ab)][metric]) for i in range(1,17)])
        alpha=float(row['alpha']);lo,hi=np.quantile(delta[draws].mean(1),[alpha/2,1-alpha/2])
        assert max(abs(delta.mean()-float(row['mean_difference'])),abs(lo-float(row['lower'])),abs(hi-float(row['upper'])))<1e-14
        if row['primary']=='True':primary.append(dict(contrast=row['contrast'],mean=float(delta.mean()),lower=float(lo),upper=float(hi)))

result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',metric_rows=384,
    max_metric_error=max_metric,max_pipeline_output_error=max_output,max_temperature_error=max_beta,
    source_artifact_hash_checks=committed,seed_replays=16,checkpoint_choices=16,temperature_fits=144,
    calibration_pipeline_replays=144,mask_or_class_discrepancies=0,ledger_rows=len(ledger),
    actual_deployment_forwards=total,additional_audit_forward_examples=extra_forwards,primary_bootstraps=primary)
(ROOT/'root_audit/CALIBRATION_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
