"""Frozen same-input-law verification and separately seeded posterior stress."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,now,sha,ash,jo,co,components,predict
from math_utils import BATCH,QUOTA,LOOKS,SHIFTS,EPS,FAMILY,DELTA,lse,ce,multi_gap,span,gap,nll,clip,bound,sample

CONFIG=dict(pairs=list(range(1,9)),shifts=list(SHIFTS),epsilons=list(EPS),looks=list(LOOKS),primary_batches=2048,
    policies=['symmetric_jensen','folded'],stress_n=65536,family_events=FAMILY,delta=DELTA,
    certificate_inputs=2171000,certificate_ties=2172000,stress_inputs=2173000,stress_ties=2174000,
    sampler_inputs=2178000,identity_seed=2179000,clip_rule='min(.5,max(1e-6,q95(historical grid Jensen)))')
def records():return json.loads((HERE/'HISTORICAL_INPUTS.json').read_text())['models']
def load_model(r):return np.load(HERE/r['theta_path']),dict(np.load(HERE/r['grid_path'])),dict(np.load(HERE/r['folded_path']))
def freeze():
    files=[HERE/n for n in ['run.py','math_utils.py','common.py','assets.py','PROTOCOL.md','THEORY.md','HISTORICAL_INPUTS.json']]
    files+=list((HERE/'reference').glob('*.py'))+list((HERE/'historical').glob('*'))
    jo(HERE/'FREEZE.json',dict(timestamp_utc=now(),config=CONFIG,fresh_inputs_generated=False,
        hashes={p.relative_to(HERE).as_posix():sha(p) for p in files}))
def verify():
    f=json.loads((HERE/'FREEZE.json').read_text());assert f['config']==CONFIG
    for p,h in f['hashes'].items():assert sha(HERE/p)==h,p
    a=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
    assert a['shift_freeze_sha256']==sha(HERE/'FREEZE.json') and a['fresh_execution_authorized'] is True
def prepare():
    verify();task,_,_=components();out=[]
    for r in records():
        theta,grid,_=load_model(r);L,C=task.network_constants(theta)
        out.append(dict(pair=r['pair'],c=clip(grid),L=L,C=C,D=min(1.4+C,1.4*L)))
    jo(HERE/'CLIP_COMMIT.json',dict(timestamp_utc=now(),freeze_sha256=sha(HERE/'FREEZE.json'),records=out,
        fresh_verification_accessed=False))
def ready():
    verify();c=json.loads((HERE/'CLIP_COMMIT.json').read_text());assert c['freeze_sha256']==sha(HERE/'FREEZE.json')
    return records(),{r['pair']:r for r in c['records']}
def masks_for(policy,adapter,x,grid,folded,ties):
    return [policy.select(adapter,x,grid,ties),policy.select_folded(adapter,x,folded,ties)]
def checks():
    verify();task,_,_=components();rng=np.random.default_rng(CONFIG['identity_seed']);rows=[]
    for K in [2,3,10]:
        for scale in [.1,1.,5.,20.]:
            z=rng.normal(size=(4096,K))*scale;zt=rng.normal(size=z.shape)*scale
            p=rng.dirichlet(np.ones(K),len(z));pt=rng.dirichlet(np.ones(K),len(z))
            d=(z-zt)/2;m=(z+zt)/2;J=multi_gap(z,zt);S=span(d);eps=.5*np.abs(p-pt).sum(1)
            f=ce(z,p)-ce(m,p);ft=ce(zt,pt)-ce(m,pt)
            rhs=J-.5*np.sum((p-pt)*d,axis=1);err=float(np.max(abs(.5*(f+ft)-rhs)))
            off=rng.normal(size=(len(z),1))*100;offt=rng.normal(size=(len(z),1))*100
            z2=z+off;zt2=zt+offt;m2=(z2+zt2)/2;d2=(z2-zt2)/2
            offset_error=max(float(np.max(abs(multi_gap(z2,zt2)-J))),float(np.max(abs(span(d2)-S))),
                float(np.max(abs((ce(z2,p)-ce(m2,p))-f))))
            label_violation=float(np.max(np.abs(np.sum((p-pt)*d,axis=1))-eps*S))
            loss_violation=float(np.max(abs(f)-S))
            jbound=np.logaddexp(S/2,-S/2)-np.log(2)
            jviolation=float(np.max(J-jbound))
            assert err<1e-11 and offset_error<1e-10 and max(label_violation,loss_violation,jviolation)<1e-10
            rows.append(dict(classes=K,scale=scale,n=4096,identity_error=err,offset_error=offset_error,
                posterior_tv_violation=label_violation,loss_range_violation=loss_violation,jensen_range_violation=jviolation))
    co(HERE/'multiclass_checks.csv',rows);srows=[];phi=(np.arange(1048576)+.5)*2*np.pi/1048576
    for j,a in enumerate(SHIFTS):
        x,counts=sample(task,CONFIG['sampler_inputs']+j,65536,a);angle=task.angle(x)
        numerical_tv=float(np.mean(abs(a*np.cos(phi))));exact_tv=2*abs(a)/np.pi
        assert abs(numerical_tv-exact_tv)<1e-10
        # Moment deviations are diagnostics, not outcome-dependent rejection rules.
        srows.append(dict(a=a,known_tv=exact_tv,numerical_tv=numerical_tv,mean_cos=float(np.cos(angle).mean()),
            expected_cos=a/2,mean_sin=float(np.sin(angle).mean()),mean_cos2=float(np.cos(2*angle).mean()),
            input_seed=CONFIG['sampler_inputs']+j,input_sha256=ash(x),**counts))
    co(HERE/'sampler_checks.csv',srows)
    jo(HERE/'CHECKS.json',dict(timestamp_utc=now(),passed=True,multiclass_cases=12,multiclass_records=49152,
        coordinate_only_sampler_inputs=262144,model_forward_examples=0))
def certificate():
    entries,constants=ready();task,policy,adapter=components();folder=HERE/'certificate';folder.mkdir(exist_ok=False)
    jo(HERE/'certificate_start.json',dict(timestamp_utc=now(),clip_commit=sha(HERE/'CLIP_COMMIT.json')))
    out=[];curves=[];costs=[];ledger=[];files={};start_all=time.perf_counter()
    for entry in entries:
        i=entry['pair'];theta,grid,folded=load_model(entry);c=constants[i]['c'];D=constants[i]['D']
        for j,a in enumerate(SHIFTS):
            start=time.perf_counter();seed=CONFIG['certificate_inputs']+100*j+i;tseed=CONFIG['certificate_ties']+100*j+i
            x,reject=sample(task,seed,max(LOOKS)*BATCH,a);ties=np.random.default_rng(tseed).random(len(x));tx=task.partner(x)
            masks=masks_for(policy,adapter,x,grid,folded,ties);tmasks=masks_for(policy,adapter,tx,grid,folded,ties)
            assert all(np.array_equal(m,t) for m,t in zip(masks,tmasks))
            # Single-coordinate transformation audit, holding all partners fixed.
            mixed=x[:BATCH].copy();mixed[17]=task.partner(mixed[17:18])[0]
            mm=masks_for(policy,adapter,mixed,grid,folded,ties[:BATCH])
            assert all(np.array_equal(m[:BATCH],v) for m,v in zip(masks,mm))
            union=masks[0]|masks[1];ids=np.flatnonzero(union)
            h,calls=predict(task,theta,x[ids]);ht,other=predict(task,theta,tx[ids]);calls+=other
            J=np.zeros(len(x));d=np.zeros(len(x));J[ids]=gap(h,ht);d[ids]=abs(h-ht)/2
            assert d.max()<=D+1e-10
            data={};eta=2*a/np.pi
            for name,mask in zip(CONFIG['policies'],masks):
                assert np.all(mask.reshape(-1,BATCH).sum(1)==QUOTA)
                A=(mask*np.minimum(J,c)).reshape(-1,BATCH).mean(1);B=(mask*d).reshape(-1,BATCH).mean(1)
                data[name+'_A']=A;data[name+'_B']=B
                data[name+'_unclipped']=(mask*J).reshape(-1,BATCH).mean(1)
                for n in LOOKS:
                    row=bound(A[:n],B[:n],c,D);row.update(pair=i,a=a,eta=eta,policy=name,
                        empirical_unclipped_gain=float(data[name+'_unclipped'][:n].mean()),
                        input_tv_penalty=D*eta)
                    row['lower_epsilon0']=row['A_lower']-D*eta
                    row['eta_tolerance_epsilon0']=row['A_lower']/D
                    row['epsilon_tolerance']=max(0.,min(1.,2*row['lower_epsilon0']/row['B_upper']))
                    out.append(row)
                    for eps in EPS:
                        curves.append(dict(pair=i,a=a,eta=eta,policy=name,n_batches=n,epsilon=eps,
                            lower=row['A_lower']-.5*eps*row['B_upper']-D*eta))
            data['union_count']=union.reshape(-1,BATCH).sum(1);data['forwards']=2*data['union_count']
            assert sum(calls)==int(data['forwards'].sum())
            for b,u in enumerate(data['union_count']):ledger.append(dict(pair=i,a=a,batch=b,inputs=BATCH,
                candidate_selected=64,folded_selected=64,union_selected=int(u),forward_examples=2*int(u)))
            path=folder/f'{i}_{j}.npz';np.savez_compressed(path,**data);files[path.name]=sha(path)
            costs.append(dict(pair=i,a=a,input_seed=seed,tie_seed=tseed,coordinate_inputs=len(x),
                forward_examples=sum(calls),invocations=len(calls),seconds=time.perf_counter()-start,
                input_sha256=ash(x),tie_sha256=ash(ties),candidate_mask_sha256=ash(masks[0]),folded_mask_sha256=ash(masks[1]),
                exact_T_invariance=True,**reject))
            print(f'Certificate pair {i}, a={a}',flush=True)
    co(HERE/'certificate_bounds.csv',out);co(HERE/'certificate_curves.csv',curves)
    co(HERE/'certificate_costs.csv',costs);co(HERE/'certificate_ledger.csv',ledger)
    jo(HERE/'CERTIFICATE_COMMIT.json',dict(timestamp_utc=now(),seconds=time.perf_counter()-start_all,
        family_events=FAMILY,rows=len(out),raw_files=files))
def stress():
    entries,constants=ready();task,policy,adapter=components();folder=HERE/'stress';folder.mkdir(exist_ok=False)
    jo(HERE/'stress_start.json',dict(timestamp_utc=now(),clip_commit=sha(HERE/'CLIP_COMMIT.json')))
    rows=[];costs=[];audits=[];files={};start_all=time.perf_counter()
    names=['raw','full','symmetric_jensen','folded','confidence']
    for entry in entries:
        i=entry['pair'];theta,grid,folded=load_model(entry)
        for j,a in enumerate(SHIFTS):
            start=time.perf_counter();seed=CONFIG['stress_inputs']+100*j+i;tseed=CONFIG['stress_ties']+100*j+i
            x,reject=sample(task,seed,CONFIG['stress_n'],a);ties=np.random.default_rng(tseed).random(len(x));tx=task.partner(x)
            h,calls=predict(task,theta,x);ht,other=predict(task,theta,tx);calls+=other
            d=(h-ht)/2;m=(h+ht)/2;J=gap(h,ht);p0=task.posterior(x);cosphi=x[:,0]/np.linalg.norm(x[:,:2],axis=1)
            safe=masks_for(policy,adapter,x,grid,folded,ties)
            masks=[np.zeros(len(x),bool),np.ones(len(x),bool),*safe,policy.allocate(-abs(h),ties)]
            maxerr=0.;audit_calls=0
            for index in [2,3,4]:
                first=task.predict(theta,x[:BATCH]);mask=masks[index][:BATCH]
                second=task.predict(theta,task.partner(x[:BATCH][mask]));out=first.copy();out[mask]=(out[mask]+second)/2
                err=float(np.max(abs(out-np.where(mask,m[:BATCH],h[:BATCH]))));assert err<1e-11 and mask.sum()==64
                audit_calls+=320;audits.append(dict(pair=i,a=a,policy=names[index],first=256,second=64,output_error=err))
            for world,q in [('cosine',cosphi),('adversarial',np.sign(d))]:
                for eps in EPS:
                    amplitude=np.minimum(eps/2,np.minimum(p0,1-p0));p=p0+amplitude*q;pt=p0-amplitude*q
                    assert p.min()>=0 and p.max()<=1 and np.max(abs(p-pt))<=eps+1e-12
                    raw=nll(h,p);rawt=nll(ht,pt)
                    for name,mask in zip(names,masks):
                        out=np.where(mask,m,h);loss=nll(out,p);gain=raw-loss
                        error=(out>=0)*(1-p)+(out<0)*p;gb=gain.reshape(-1,BATCH).mean(1)
                        corrected='';cse='';paired='';corrterm=''
                        if name!='confidence':
                            ft=rawt-nll(np.where(mask,m,ht),pt);H=(gain+ft)/2
                            expected=mask*(J-.5*(p-pt)*d);err=float(np.max(abs(H-expected)));maxerr=max(maxerr,err)
                            assert err<1e-11
                            correction=.5*a*cosphi*(gain-ft);C=H+correction;cb=C.reshape(-1,BATCH).mean(1)
                            corrected=float(cb.mean());cse=float(cb.std(ddof=1)/np.sqrt(len(cb)))
                            paired=float(H.mean());corrterm=float(correction.mean())
                        rows.append(dict(pair=i,a=a,eta=2*a/np.pi,world=world,epsilon=eps,policy=name,
                            expected_nll=float(loss.mean()),expected_error=float(error.mean()),direct_gain=float(gb.mean()),
                            direct_batch_se=float(gb.std(ddof=1)/np.sqrt(len(gb))),known_density_gain=corrected,
                            known_density_batch_se=cse,unweighted_paired_gain=paired,density_correction=corrterm,
                            observed_posterior_tv=float(np.max(abs(p-pt))),known_uniform_posterior_tv=eps))
            path=folder/f'{i}_{j}.npz';np.savez_compressed(path,first=h,partner=ht,p0=p0,cosphi=cosphi,
                policy_names=np.array(names),masks=np.packbits(np.stack(masks),axis=1));files[path.name]=sha(path)
            costs.append(dict(pair=i,a=a,input_seed=seed,tie_seed=tseed,inputs=len(x),shared_forward_examples=sum(calls),
                extra_deployment_audit_forwards=audit_calls,seconds=time.perf_counter()-start,input_sha256=ash(x),tie_sha256=ash(ties),
                maximum_binary_identity_error=maxerr,**reject))
            print(f'Stress pair {i}, a={a}',flush=True)
    co(HERE/'stress_results.csv',rows);co(HERE/'stress_costs.csv',costs);co(HERE/'deployment_audits.csv',audits)
    jo(HERE/'STRESS_COMMIT.json',dict(timestamp_utc=now(),seconds=time.perf_counter()-start_all,rows=len(rows),raw_files=files))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','prepare','checks','certificate','stress']);args=p.parse_args()
    try:
        with threadpool_limits(limits=1):globals()[args.stage]()
    except Exception:
        path=HERE/('FAILURE_'+args.stage+'_'+str(time.time_ns())+'.json')
        jo(path,dict(timestamp_utc=now(),stage=args.stage,traceback=traceback.format_exc()));raise
