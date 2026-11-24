"""Prospectively frozen structured input-law certificate and independent truth stress."""
import argparse,json,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,now,sha,ash,jo,co,components,predict
from math_utils import BATCH,QUOTA,LOOKS,SHIFTS,EPS,MOMENTS,FAMILY,DELTA,lse,ce,gap,nll,clip,local,interval,parameter_interval,sample,posteriors

CONFIG=dict(pairs=list(range(1,9)),laws=list(SHIFTS),epsilons=list(EPS),looks=list(LOOKS),primary_batches=2048,
    policies=['symmetric_jensen','folded'],moments=list(MOMENTS),stress_n=65536,parameter_n=65536,
    family_events=FAMILY,delta=DELTA,certificate_inputs=2331000,certificate_ties=2332000,
    stress_inputs=2333000,stress_ties=2334000,parameter_inputs=2338000,identity_seed=2339000,
    worlds=['cosine','adversarial','orbit_worst'],clip_rule='min(.5,max(1e-6,q95(historical grid Jensen)))',
    primary_criterion='a=.01 epsilon=0 n=2048; positive new bound with paired nonpositive old bound',
    misspecification=dict(b=.6,k=.2,quadrature_points=1048576))
def records():return json.loads((HERE/'HISTORICAL_INPUTS.json').read_text())['models']
def model(r):return np.load(HERE/r['theta_path']),dict(np.load(HERE/r['grid_path'])),dict(np.load(HERE/r['folded_path']))
def freeze():
    files=[HERE/n for n in ['run.py','math_utils.py','common.py','assets.py','PROTOCOL.md','THEORY.md','HISTORICAL_INPUTS.json','PROVENANCE.json']]
    files+=list((HERE/'reference').glob('*.py'))+list((HERE/'historical').glob('*'))
    jo(HERE/'FREEZE.json',dict(timestamp_utc=now(),config=CONFIG,fresh_inputs_generated=False,
        hashes={p.relative_to(HERE).as_posix():sha(p) for p in files}))
def verify():
    f=json.loads((HERE/'FREEZE.json').read_text());assert f['config']==CONFIG
    for p,h in f['hashes'].items():assert sha(HERE/p)==h,p
    approval=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
    assert approval['structured_freeze_sha256']==sha(HERE/'FREEZE.json') and approval['fresh_execution_authorized'] is True
def prepare():
    verify();task,_,_=components();out=[]
    for r in records():
        theta,grid,_=model(r);L,C=task.network_constants(theta)
        out.append(dict(pair=r['pair'],c=clip(grid),L=L,C=C,D=min(1.4+C,1.4*L)))
    jo(HERE/'CONSTANTS.json',dict(timestamp_utc=now(),freeze_sha256=sha(HERE/'FREEZE.json'),records=out,fresh_verification_accessed=False))
def ready():
    verify();c=json.loads((HERE/'CONSTANTS.json').read_text());assert c['freeze_sha256']==sha(HERE/'FREEZE.json')
    return records(),{r['pair']:r for r in c['records']}
def masks(policy,adapter,x,grid,fold,ties):return [policy.select(adapter,x,grid,ties),policy.select_folded(adapter,x,fold,ties)]
def checks():
    verify();rng=np.random.default_rng(CONFIG['identity_seed']);rows=[]
    for k in [2,3,10]:
        for scale in [.1,1.,5.,20.]:
            n=4096;z=rng.normal(size=(n,k))*scale;zt=rng.normal(size=z.shape)*scale;m=(z+zt)/2;d=(z-zt)/2
            r=rng.uniform(-1,1,n);p=rng.dirichlet(np.ones(k),n);pt=rng.dirichlet(np.ones(k),n)
            eps=abs(p-pt).sum(1)/2;S=np.ptp(d,axis=1);center=(d.max(1)+d.min(1))/2
            J=(lse(z)+lse(zt))/2-lse(m);K=(lse(z)-lse(zt))/2;t=K-center
            f=ce(z,p)-ce(m,p);ft=ce(zt,pt)-ce(m,pt)
            Q=np.sum(((1+r[:,None])*p-(1-r[:,None])*pt)*d,axis=1)/2
            actual=(f+ft)/2+r*(f-ft)/2;identity=float(np.max(abs(actual-(J+r*K-Q))))
            lower=J+r*t-(abs(r)+(1-abs(r))*eps)*S/2
            violation=float(np.max(lower-actual))
            e=rng.random(n);hi=np.eye(k)[d.argmax(1)];lo=np.eye(k)[d.argmin(1)]
            pe=np.where((r>=0)[:,None],hi,(1-e[:,None])*lo+e[:,None]*hi)
            pte=np.where((r>=0)[:,None],(1-e[:,None])*hi+e[:,None]*lo,lo)
            qe=np.sum(((1+r[:,None])*pe-(1-r[:,None])*pte)*d,axis=1)/2
            extremal=float(np.max(abs(qe-(r*center+(abs(r)+(1-abs(r))*e)*S/2))))
            oz=rng.normal(size=(n,1))*100;ot=rng.normal(size=(n,1))*100
            z2=z+oz;zt2=zt+ot;d2=(z2-zt2)/2;K2=(lse(z2)-lse(zt2))/2
            t2=K2-(d2.max(1)+d2.min(1))/2
            offset=float(np.max(abs(t2-t)));range_violation=float(np.max(abs(t)-S/2))
            assert max(identity,extremal,offset)<1e-10 and max(violation,range_violation)<1e-10
            rows.append(dict(classes=k,scale=scale,n=n,identity_error=identity,posterior_bound_violation=violation,
                extremal_error=extremal,common_offset_error=offset,t_range_violation=range_violation))
    co(HERE/'algebra_checks.csv',rows)
    cfg=CONFIG['misspecification'];phi=(np.arange(cfg['quadrature_points'])+.5)*2*np.pi/cfg['quadrature_points']
    q=np.cos(3*phi);h=cfg['k']*q;J=np.logaddexp(h/2,-h/2)-np.log(2);r=cfg['b']*q
    true_gain=float(np.mean(J-r*h/2)/4);false_bound=float(np.mean(J)/4)
    assert true_gain<0<false_bound
    jo(HERE/'MISSPECIFICATION.json',dict(config=cfg,analytic_true_gain_upper=(cfg['k']**2/16-cfg['b']*cfg['k']/4)/4,
        quadrature_true_gain=true_gain,incorrect_family_a0_lower=false_bound,
        first_harmonic_moment=float(np.mean(np.cos(phi)*(1+cfg['b']*q))),
        actual_tv=2*cfg['b']/np.pi,posterior_tv=0,quota_fraction=.25,
        statement='The first-harmonic parameter is zero; a narrow parameter CI cannot validate the one-harmonic family.'))
    jo(HERE/'CHECKS.json',dict(timestamp_utc=now(),passed=True,random_algebra_records=49152,
        angular_quadrature_points=cfg['quadrature_points'],model_forward_examples=0))
def parameter():
    ready();task,_,_=components();rows=[];raw={}
    for j,a in enumerate(SHIFTS):
        seed=CONFIG['parameter_inputs']+j;x,reject=sample(task,seed,CONFIG['parameter_n'],a)
        u=x[:,0]/np.linalg.norm(x[:,:2],axis=1);record=parameter_interval(u)
        rows.append(dict(a=a,input_seed=seed,input_sha256=ash(x),cosine_sha256=ash(u),coordinate_inputs=len(x),
            forward_examples=0,**record,**reject));raw[f'cosine_{j}']=u
    np.savez_compressed(HERE/'parameter_raw.npz',**raw);co(HERE/'parameter_intervals.csv',rows)
    jo(HERE/'PARAMETER_COMMIT.json',dict(timestamp_utc=now(),rows=rows,raw_sha256=sha(HERE/'parameter_raw.npz')))
def certificate():
    entries,constants=ready();task,policy,adapter=components()
    estimates=json.loads((HERE/'PARAMETER_COMMIT.json').read_text())['rows'];folder=HERE/'certificate';folder.mkdir(exist_ok=False)
    jo(HERE/'certificate_start.json',dict(timestamp_utc=now(),constants_sha256=sha(HERE/'CONSTANTS.json'),parameter_commit=sha(HERE/'PARAMETER_COMMIT.json')))
    moments=[];curves=[];costs=[];ledger=[];files={};started=time.perf_counter()
    for entry in entries:
        i=entry['pair'];theta,grid,fold=model(entry);c=constants[i]['c'];D=constants[i]['D']
        for j,a in enumerate(SHIFTS):
            start=time.perf_counter();seed=CONFIG['certificate_inputs']+100*j+i;tseed=CONFIG['certificate_ties']+100*j+i
            estimate=estimates[j];assert estimate['a']==a;alo=estimate['lower'];ahi=estimate['upper']
            x,reject=sample(task,seed,max(LOOKS)*BATCH,a);ties=np.random.default_rng(tseed).random(len(x));tx=task.partner(x)
            mm=masks(policy,adapter,x,grid,fold,ties);mt=masks(policy,adapter,tx,grid,fold,ties)
            assert all(np.array_equal(v,w) for v,w in zip(mm,mt))
            for coordinate in [0,17,63,255]:
                mixed=x[:BATCH].copy();mixed[coordinate]=task.partner(mixed[coordinate:coordinate+1])[0]
                mixed_masks=masks(policy,adapter,mixed,grid,fold,ties[:BATCH])
                assert all(np.array_equal(v[:BATCH],w) for v,w in zip(mm,mixed_masks))
            union=mm[0]|mm[1];ids=np.flatnonzero(union);h,calls=predict(task,theta,x[ids]);ht,more=predict(task,theta,tx[ids]);calls+=more
            u=x[:,0]/np.linalg.norm(x[:,:2],axis=1);jj,ss,ww,vv,tt=local(h,ht,u[ids]);assert max(ss)<=D+1e-10
            weights={key:np.zeros(len(x)) for key in MOMENTS}
            selected=dict(A=np.minimum(jj,c),W=ww,B=ss/2,R_known=(1-a*abs(u[ids]))*ss/2,R_unknown=(1-alo*abs(u[ids]))*ss/2)
            for key in MOMENTS:weights[key][ids]=selected[key]
            data=dict(union_ids=ids,first=h,partner=ht,cosphi=u,packed_masks=np.packbits(np.stack(mm),axis=1),
                policy_names=np.array(CONFIG['policies']),union_count=union.reshape(-1,BATCH).sum(1))
            ranges=dict(A=c,W=D,B=D/2,R_known=D/2,R_unknown=D/2)
            for name,mask in zip(CONFIG['policies'],mm):
                assert np.all(mask.reshape(-1,BATCH).sum(1)==QUOTA)
                for key in MOMENTS:data[name+'_'+key]=(mask*weights[key]).reshape(-1,BATCH).mean(1)
                for n in LOOKS:
                    bounds={}
                    for key in MOMENTS:
                        result=interval(data[name+'_'+key][:n],ranges[key],'lower' if key=='A' else 'upper')
                        bounds[key]=result['bound'];moments.append(dict(pair=i,a=a,policy=name,moment=key,**result))
                    for eps in EPS:
                        curves.append(dict(pair=i,a=a,eta=2*a/np.pi,policy=name,n_batches=n,epsilon=eps,
                            known_lower=bounds['A']-a*bounds['W']-eps*bounds['R_known'],
                            estimated_lower=bounds['A']-ahi*bounds['W']-eps*bounds['R_unknown'],
                            old_lower=bounds['A']-eps*bounds['B']-D*2*a/np.pi,
                            A_lower=bounds['A'],W_upper=bounds['W'],B_upper=bounds['B'],R_known_upper=bounds['R_known'],
                            R_unknown_upper=bounds['R_unknown'],a_lower=alo,a_upper=ahi,c=c,D=D,
                            known_shift_penalty=a*bounds['W'],estimated_shift_penalty=ahi*bounds['W'],old_shift_penalty=D*2*a/np.pi))
            data['forwards']=2*data['union_count'];assert sum(calls)==int(data['forwards'].sum())
            for b,count in enumerate(data['union_count']):ledger.append(dict(pair=i,a=a,batch=b,inputs=BATCH,quota=QUOTA,union_selected=int(count),forward_examples=2*int(count)))
            path=folder/f'{i}_{j}.npz';np.savez_compressed(path,**data);files[path.name]=sha(path)
            costs.append(dict(pair=i,a=a,input_seed=seed,tie_seed=tseed,coordinate_inputs=len(x),forward_examples=sum(calls),
                invocations=len(calls),seconds=time.perf_counter()-start,input_sha256=ash(x),tie_sha256=ash(ties),
                candidate_mask_sha256=ash(mm[0]),folded_mask_sha256=ash(mm[1]),exact_single_coordinate_invariance=True,**reject))
            print(f'Certificate model {i}, a={a}',flush=True)
    co(HERE/'moment_bounds.csv',moments);co(HERE/'certificate_curves.csv',curves);co(HERE/'certificate_costs.csv',costs);co(HERE/'certificate_ledger.csv',ledger)
    jo(HERE/'CERTIFICATE_COMMIT.json',dict(timestamp_utc=now(),seconds=time.perf_counter()-started,family_events=FAMILY,rows=len(curves),raw_files=files))
def stress():
    entries,constants=ready();task,policy,adapter=components();folder=HERE/'stress';folder.mkdir(exist_ok=False)
    jo(HERE/'stress_start.json',dict(timestamp_utc=now(),constants_sha256=sha(HERE/'CONSTANTS.json')))
    rows=[];costs=[];audits=[];files={};started=time.perf_counter();names=['raw','full',*CONFIG['policies']]
    for entry in entries:
        i=entry['pair'];theta,grid,fold=model(entry)
        for j,a in enumerate(SHIFTS):
            start=time.perf_counter();seed=CONFIG['stress_inputs']+100*j+i;tseed=CONFIG['stress_ties']+100*j+i
            x,reject=sample(task,seed,CONFIG['stress_n'],a);ties=np.random.default_rng(tseed).random(len(x));tx=task.partner(x)
            h,calls=predict(task,theta,x);ht,more=predict(task,theta,tx);calls+=more
            u=x[:,0]/np.linalg.norm(x[:,:2],axis=1);J,S,W,V,t=local(h,ht,u);d=(h-ht)/2;m=(h+ht)/2;p0=task.posterior(x)
            mm=[np.zeros(len(x),bool),np.ones(len(x),bool),*masks(policy,adapter,x,grid,fold,ties)]
            extra=0
            for name,mask in zip(names[2:],mm[2:]):
                first=task.predict(theta,x[:BATCH]);second=task.predict(theta,task.partner(x[:BATCH][mask[:BATCH]]))
                out=first.copy();out[mask[:BATCH]]=(out[mask[:BATCH]]+second)/2
                error=float(np.max(abs(out-np.where(mask[:BATCH],m[:BATCH],h[:BATCH]))));assert error<1e-11
                extra+=320;audits.append(dict(pair=i,a=a,policy=name,first=256,second=64,output_error=error))
            max_identity=max_violation=max_extremal=max_partner=0.
            for world in CONFIG['worlds']:
                for eps in EPS:
                    p,pt=posteriors(world,p0,d,u,eps);pt_check,p_check=posteriors(world,p0,-d,-u,eps)
                    partner_error=float(max(np.max(abs(p-p_check)),np.max(abs(pt-pt_check))));max_partner=max(max_partner,partner_error)
                    assert p.min()>=-1e-12 and p.max()<=1+1e-12 and np.max(abs(p-pt))<=eps+1e-12 and partner_error<1e-12
                    for name,mask in zip(names,mm):
                        out=np.where(mask,m,h);loss=nll(out,p);gain=nll(h,p)-loss
                        ft=nll(ht,pt)-nll(np.where(mask,m,ht),pt);H=(gain+ft)/2;C=H+a*u*(gain-ft)/2
                        identity=float(np.max(abs(H-mask*(J-(p-pt)*d/2))));max_identity=max(max_identity,identity)
                        sharp=mask*(J-a*W-eps*(1-a*abs(u))*S/2);violation=float(np.max(sharp-C));max_violation=max(max_violation,violation)
                        assert identity<1e-11 and violation<1e-11
                        if world=='orbit_worst':
                            extrema=float(np.max(abs(sharp-C)));max_extremal=max(max_extremal,extrema);assert extrema<1e-11
                        gb=gain.reshape(-1,BATCH).mean(1);cb=C.reshape(-1,BATCH).mean(1)
                        rows.append(dict(pair=i,a=a,world=world,epsilon=eps,policy=name,expected_nll=float(loss.mean()),
                            expected_error=float(np.where(out>=0,1-p,p).mean()),direct_gain=float(gb.mean()),
                            direct_batch_se=float(gb.std(ddof=1)/np.sqrt(len(gb))),known_density_gain=float(cb.mean()),
                            known_density_batch_se=float(cb.std(ddof=1)/np.sqrt(len(cb))),unweighted_paired_gain=float(H.mean()),
                            density_correction=float((C-H).mean()),sharp_unclipped_mean=float(sharp.mean()),
                            observed_posterior_tv=float(np.max(abs(p-pt)))))
            path=folder/f'{i}_{j}.npz';np.savez_compressed(path,first=h,partner=ht,p0=p0,cosphi=u,
                policy_names=np.array(names),packed_masks=np.packbits(np.stack(mm),axis=1));files[path.name]=sha(path)
            costs.append(dict(pair=i,a=a,input_seed=seed,tie_seed=tseed,coordinate_inputs=len(x),shared_forward_examples=sum(calls),
                invocations=len(calls),extra_audit_forwards=extra,seconds=time.perf_counter()-start,input_sha256=ash(x),tie_sha256=ash(ties),
                max_identity_error=max_identity,max_lower_violation=max_violation,max_extremal_error=max_extremal,max_partner_error=max_partner,**reject))
            print(f'Stress model {i}, a={a}',flush=True)
    co(HERE/'stress_results.csv',rows);co(HERE/'stress_costs.csv',costs);co(HERE/'deployment_audits.csv',audits)
    jo(HERE/'STRESS_COMMIT.json',dict(timestamp_utc=now(),seconds=time.perf_counter()-started,rows=len(rows),raw_files=files))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','prepare','checks','parameter','certificate','stress']);args=parser.parse_args()
    try:
        with threadpool_limits(limits=1):globals()[args.stage]()
    except Exception:
        jo(HERE/('FAILURE_'+args.stage+'_'+str(time.time_ns())+'.json'),dict(timestamp_utc=now(),stage=args.stage,traceback=traceback.format_exc()));raise
