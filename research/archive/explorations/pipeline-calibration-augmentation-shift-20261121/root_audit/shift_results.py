"""Independently reconstruct every same-law TV bound and stress result."""
from pathlib import Path
import csv, datetime, hashlib, json, math, sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'shift'
sys.path.insert(0,str(HERE));from common import components
from math_utils import sample
read=lambda p:list(csv.DictReader((HERE/p).open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ash=lambda x:hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
task,policy,adapter=components()
shifts=[0.,.01,.1,.6]
bounds=read('certificate_bounds.csv');stress=read('stress_results.csv');costs=read('certificate_costs.csv')
stress_costs=read('stress_costs.csv');max_bound=max_stress=0.;hash_checks=0;extra_forwards=0;replays=[]
with threadpool_limits(limits=1):
    for commit,folder in [('CERTIFICATE_COMMIT.json','certificate'),('STRESS_COMMIT.json','stress')]:
        for file,expected in json.loads((HERE/commit).read_text())['raw_files'].items():
            assert sha(HERE/folder/file)==expected;hash_checks+=1
    for cost in costs:
        pair=int(cost['pair']);a=float(cost['a']);j=shifts.index(a)
        theta=np.load(HERE/'historical'/f'{pair}_theta.npy')
        grid=dict(np.load(HERE/'historical'/f'{pair}_grid.npz'));fold=dict(np.load(HERE/'historical'/f'{pair}_folded.npz'))
        c=float(np.clip(np.quantile(grid['folded_jensen'],.95),1e-6,.5))
        L=1.;C=0.
        for A,B,b in task.unpack(theta):L*=1+.5*np.linalg.norm(A,2)*np.linalg.norm(B,2);C+=.5*abs(B[0]).sum()
        D=min(1.4+C,1.4*L);eta=2*a/np.pi
        saved=dict(np.load(HERE/'certificate'/f'{pair}_{j}.npz'))
        for row in [r for r in bounds if int(r['pair'])==pair and float(r['a'])==a]:
            n=int(row['n_batches']);name=row['policy'];A=saved[name+'_A'][:n];B=saved[name+'_B'][:n]
            delta=.05/512;z=math.log(2/delta);scale=math.sqrt(math.log(1/delta)/(2*n*256))
            ea=math.sqrt(2*A.var(ddof=1)*z/n)+7*(c/4)*z/(3*(n-1))
            eb=math.sqrt(2*B.var(ddof=1)*z/n)+7*(D/4)*z/(3*(n-1))
            lo=max(0.,A.mean()-ea,A.mean()-c*scale);up=min(D/4,B.mean()+eb,B.mean()+D*scale)
            values=dict(c=c,D=D,eta=eta,A_eb_lower=A.mean()-ea,A_mcd_lower=A.mean()-c*scale,
                B_eb_upper=B.mean()+eb,B_mcd_upper=B.mean()+D*scale,A_lower=lo,B_upper=up,
                lower_epsilon0=lo-D*eta,eta_tolerance_epsilon0=lo/D,
                epsilon_tolerance=max(0.,min(1.,2*(lo-D*eta)/up)))
            for key,value in values.items():
                error=abs(float(value)-float(row[key]));max_bound=max(max_bound,error);assert error<2e-13,(pair,a,key)
        if pair in (1,8) and j in (0,1,3):
            x,_=sample(task,int(cost['input_seed']),2048*256,a);ties=np.random.default_rng(int(cost['tie_seed'])).random(len(x))
            assert ash(x)==cost['input_sha256'] and ash(ties)==cost['tie_sha256']
            x=x[:128*256];ties=ties[:len(x)]
            masks=[policy.select(adapter,x,grid,ties),policy.select_folded(adapter,x,fold,ties)]
            union=masks[0]|masks[1];h=np.zeros(len(x));ht=h.copy()
            h[union]=task.predict(theta,x[union]);ht[union]=task.predict(theta,task.partner(x[union]))
            forwards=2*int(union.sum());extra_forwards+=forwards
            J=np.maximum(0.,.5*(np.logaddexp(0,h)+np.logaddexp(0,ht))-np.logaddexp(0,.5*(h+ht)))
            d=abs(h-ht)/2
            for name,mask in zip(['symmetric_jensen','folded'],masks):
                assert np.max(abs((mask*np.minimum(J,c)).reshape(-1,256).mean(1)-saved[name+'_A'][:128]))<1e-13
                assert np.max(abs((mask*d).reshape(-1,256).mean(1)-saved[name+'_B'][:128]))<1e-13
            assert forwards==int(saved['forwards'][:128].sum())
            replays.append(dict(pair=pair,a=a,batches=128,additional_forwards=forwards))
        s=dict(np.load(HERE/'stress'/f'{pair}_{j}.npz'))
        sc=next(r for r in stress_costs if int(r['pair'])==pair and float(r['a'])==a)
        x,_=sample(task,int(sc['input_seed']),65536,a);ties=np.random.default_rng(int(sc['tie_seed'])).random(len(x))
        assert ash(x)==sc['input_sha256'] and ash(ties)==sc['tie_sha256']
        p0=task.posterior(x);cos=x[:,0]/np.linalg.norm(x[:,:2],axis=1)
        assert np.array_equal(p0,s['p0']) and np.array_equal(cos,s['cosphi'])
        h=s['first'];ht=s['partner'];m=(h+ht)/2;d=(h-ht)/2
        masks=np.unpackbits(s['masks'],axis=1)[:,:65536]
        expected_masks=np.stack([np.zeros(len(x),bool),np.ones(len(x),bool),policy.select(adapter,x,grid,ties),
            policy.select_folded(adapter,x,fold,ties),policy.allocate(-abs(h),ties)])
        assert np.array_equal(masks,expected_masks)
        output={str(name):np.where(mask,m,h) for name,mask in zip(s['policy_names'],masks)}
        partner={str(name):np.where(mask,m,ht) for name,mask in zip(s['policy_names'],masks)}
        for row in [r for r in stress if int(r['pair'])==pair and float(r['a'])==a]:
            eps=float(row['epsilon']);q=cos if row['world']=='cosine' else np.sign(d)
            amplitude=np.minimum(eps/2,np.minimum(p0,1-p0));p=p0+amplitude*q;pt=p0-amplitude*q
            z=output[row['policy']];zt=partner[row['policy']]
            loss=np.logaddexp(0,z)-p*z;gain=(np.logaddexp(0,h)-p*h)-loss
            batch=gain.reshape(-1,256).mean(1)
            values=dict(expected_nll=loss.mean(),expected_error=np.where(z>=0,1-p,p).mean(),
                direct_gain=batch.mean(),direct_batch_se=batch.std(ddof=1)/16,
                observed_posterior_tv=np.max(abs(p-pt)))
            if row['policy']!='confidence':
                ft=(np.logaddexp(0,ht)-pt*ht)-(np.logaddexp(0,zt)-pt*zt)
                paired=(gain+ft)/2;correction=.5*a*cos*(gain-ft);corrected=paired+correction
                cb=corrected.reshape(-1,256).mean(1)
                values.update(known_density_gain=cb.mean(),known_density_batch_se=cb.std(ddof=1)/16,
                    unweighted_paired_gain=paired.mean(),density_correction=correction.mean())
            for key,value in values.items():
                error=abs(float(value)-float(row[key]));max_stress=max(max_stress,error);assert error<2e-12,(pair,a,row['policy'],key)
    ledger=read('certificate_ledger.csv');assert len(ledger)==65536
    for row in ledger:
        assert 64<=int(row['union_selected'])<=128
        assert int(row['forward_examples'])==2*int(row['union_selected'])
    total=sum(int(r['forward_examples']) for r in ledger)
    assert total==sum(int(r['forward_examples']) for r in costs)
    counts=[]
    for a in shifts:
        selected=[r for r in bounds if float(r['a'])==a and int(r['n_batches'])==2048]
        counts.append(dict(a=a,known_eta=2*a/np.pi,positive_epsilon0=sum(float(r['lower_epsilon0'])>0 for r in selected),
            lines=len(selected),minimum_conditional_eta_tolerance=min(float(r['eta_tolerance_epsilon0']) for r in selected),
            maximum_conditional_eta_tolerance=max(float(r['eta_tolerance_epsilon0']) for r in selected)))
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',
    certificate_rows=len(bounds),elementary_bounds=512,max_bound_error=max_bound,
    stress_rows=len(stress),max_stress_metric_error=max_stress,committed_arrays=hash_checks,
    certificate_records=len(ledger),actual_certificate_forwards=total,
    certificate_replays=replays,additional_audit_forward_examples=extra_forwards,
    primary_counts=counts,mask_discrepancies=0)
(ROOT/'root_audit/SHIFT_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
