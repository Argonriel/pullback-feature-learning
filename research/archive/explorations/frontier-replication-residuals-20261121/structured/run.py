"""Frozen two-harmonic/residual certificate with independent finite-cell labels."""
import argparse,json,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,now,sha,ash,jo,co,components,predict
from math_utils import *

CONFIG=dict(models=list(range(1,9)),laws=LAWS,looks=list(LOOKS),eps=list(EPS),policies=list(POLICIES),
    parameter_n=262144,label_n=131072,truth_n=65536,label_laws=list(LABEL_LAWS),cell_defects=list(DEFECTS),
    parameter_seed=2431000,verification_seed=2432000,verification_ties=2433000,truth_seed=2434000,
    truth_ties=2435000,label_input_seed=2436000,label_response_seed=2437000,algebra_seed=2438000,
    family_events=FAMILY,moment_events=MOMENT_EVENTS,delta=DELTA,
    primary_law='covered_third',primary_batches=2048,primary_epsilon=0.,primary_label_defect=0.,
    primary='Separate positive counts for estimated-coefficient epsilon0 and independent-label invariant-cell certificates; at least one is useful.',
    worlds=['cosine','adversarial','orbit_worst'],quadrature_points=1048576)
def records():return json.loads((HERE/'HISTORICAL_INPUTS.json').read_text())['models']
def model(r):return np.load(HERE/r['theta_path']),dict(np.load(HERE/r['grid_path'])),dict(np.load(HERE/r['folded_path']))
def freeze():
    files=[HERE/n for n in ['run.py','math_utils.py','common.py','assets.py','PROTOCOL.md','THEORY.md','HISTORICAL_INPUTS.json','PROVENANCE.json']]
    files+=list((HERE/'reference').glob('*.py'))+list((HERE/'historical').glob('*'))
    jo(HERE/'FREEZE.json',dict(utc=now(),config=CONFIG,fresh_samples_generated=False,hashes={p.relative_to(HERE).as_posix():sha(p) for p in files}))
def verify():
    frozen=json.loads((HERE/'FREEZE.json').read_text());assert frozen['config']==CONFIG
    for name,value in frozen['hashes'].items():assert sha(HERE/name)==value,name
    release=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
    assert release['residual_freeze_sha256']==sha(HERE/'FREEZE.json') and release['fresh_execution_authorized']
def prepare():
    verify();task,_,_=components();rows=[]
    for record in records():
        theta,grid,_=model(record);L,C=task.network_constants(theta)
        rows.append(dict(pair=record['pair'],L=L,C=C,D=min(1.4+C,1.4*L),c=clip(grid)))
    jo(HERE/'CONSTANTS.json',dict(utc=now(),freeze_sha256=sha(HERE/'FREEZE.json'),records=rows,fresh_verification_accessed=False))
def ready():
    verify();out=json.loads((HERE/'CONSTANTS.json').read_text());assert out['freeze_sha256']==sha(HERE/'FREEZE.json')
    return records(),{r['pair']:r for r in out['records']}
def get_masks(policy,adapter,x,grid,folded,ties):return [policy.select(adapter,x,grid,ties),policy.select_folded(adapter,x,folded,ties)]
def checks():
    verify();rng=np.random.default_rng(CONFIG['algebra_seed']);rows=[]
    for classes in [2,3,10]:
        for scale in [.1,1.,5.,20.]:
            n=4096;z=rng.normal(size=(n,classes))*scale;zt=rng.normal(size=z.shape)*scale;d=(z-zt)/2;m=(z+zt)/2
            S=np.ptp(d,axis=1);center=(d.max(1)+d.min(1))/2;t=(lse(z)-lse(zt))/2-center
            J=(lse(z)+lse(zt))/2-lse(m);lower=rng.uniform(-1,1,n);upper=lower+(1-lower)*rng.random(n);eps=rng.random(n)
            H=penalty(S,t,lower,upper,eps);gridmax=np.full(n,-np.inf)
            for fraction in np.linspace(0,1,101):
                r=lower+fraction*(upper-lower);gridmax=np.maximum(gridmax,(abs(r)+(1-abs(r))*eps)*S/2-r*t)
            endpoint=float(np.max(abs(H-gridmax)));even=float(np.max(abs(H-penalty(S,-t,-upper,-lower,eps))))
            rbar=np.maximum(abs(lower),abs(upper));range_violation=float(np.max(H-S*(eps/2+(1-eps/2)*rbar)))
            nonnegative=float(np.min(H));r=lower+(upper-lower)*rng.random(n)
            hi=np.eye(classes)[d.argmax(1)];lo=np.eye(classes)[d.argmin(1)]
            p=np.where((r>=0)[:,None],hi,(1-eps[:,None])*lo+eps[:,None]*hi)
            pt=np.where((r>=0)[:,None],(1-eps[:,None])*hi+eps[:,None]*lo,lo)
            f=lse(z)-np.sum(p*z,1)-lse(m)+np.sum(p*m,1)
            ft=lse(zt)-np.sum(pt*zt,1)-lse(m)+np.sum(pt*m,1);C=(f+ft)/2+r*(f-ft)/2
            equality=float(np.max(abs(C-(J-(abs(r)+(1-abs(r))*eps)*S/2+r*t))))
            violation=float(np.max(J-H-C))
            oz=rng.normal(size=(n,1))*50;ot=rng.normal(size=(n,1))*50;dz=(z+oz-zt-ot)/2
            t2=(lse(z+oz)-lse(zt+ot))/2-(dz.max(1)+dz.min(1))/2
            offset=float(np.max(abs(t-t2)))
            assert max(endpoint,even,equality,offset,range_violation,violation)<1e-10 and nonnegative>-1e-10
            rows.append(dict(classes=classes,scale=scale,n=n,endpoint_error=endpoint,even_error=even,
                range_violation=range_violation,minimum_penalty=nonnegative,extremal_error=equality,
                bound_violation=violation,offset_error=offset))
    co(HERE/'algebra_checks.csv',rows)
    phi=(np.arange(CONFIG['quadrature_points'])+.5)*2*np.pi/CONFIG['quadrature_points'];examples=[]
    for harmonic in [3,5]:
        u=np.cos(harmonic*phi);h=.2*u;r=.6*u;J=np.logaddexp(h/2,-h/2)-np.log(2)
        true=float(np.mean(J-r*h/2)/4);false=float(J.mean()/4);enveloped=float(np.mean(J-.6*abs(h)/2)/4)
        examples.append(dict(harmonic=harmonic,true_gain=true,analytic_true_upper=-.006875,
            invalid_zero_imbalance_gain=false,known_harmonic_lower=true,residual_rho_pointwise_lower=enveloped,
            needed_rho_if_harmonic_omitted=.6,first_moment=float(np.mean(np.cos(phi)*(1+r))),third_moment=float(np.mean(np.cos(3*phi)*(1+r)))))
        assert true<-.006875 and false>0 and enveloped<=true
    p=(1+np.sin(3*phi))/2
    cell_means=[float(p[np.cos(phi)<0].mean()),float(p[np.cos(phi)>0].mean())]
    assert max(abs(v-.5) for v in cell_means)<1e-10
    jo(HERE/'COUNTEREXAMPLES.json',dict(utc=now(),density_examples=examples,
        label_cell_counterexample=dict(half_circle_means=cell_means,pointwise_posterior_tv_supremum=1.,
            explanation='Equal cell means do not establish within-cell constancy or pointwise invariance.'),
        quadrature_points=CONFIG['quadrature_points'],neural_forward_examples=0))
    jo(HERE/'CHECKS.json',dict(utc=now(),passed=True,algebra_records=49152,neural_forward_examples=0))
def evidence():
    ready();task,_,_=components();directory=HERE/'evidence';directory.mkdir(exist_ok=False)
    parameters=[];parameter_rows=[];labels=[];cell_rows=[];costs=[];files={}
    for j,law in enumerate(LAWS):
        seed=CONFIG['parameter_seed']+j;x,reject=sample(task,seed,CONFIG['parameter_n'],law);feature=features(x);estimate=coefficient_ci(feature)
        estimate.update(law=law['name'],law_index=j,coverage=bool(np.all(abs(np.asarray(estimate['estimate'])-law['theta'])<=estimate['radius'])))
        parameters.append(estimate)
        for k,harmonic in enumerate([1,3]):parameter_rows.append(dict(law=law['name'],harmonic=harmonic,true_coefficient=law['theta'][k],
            estimate=estimate['estimate'][k],radius=estimate['radius'][k],lower=estimate['estimate'][k]-estimate['radius'][k],
            upper=estimate['estimate'][k]+estimate['radius'][k],n=len(x),delta=DELTA,covered=estimate['coverage']))
        path=directory/f'parameter_{j}.npz';np.savez_compressed(path,features=feature);files[path.name]=sha(path)
        costs.append(dict(kind='parameter',law=law['name'],defect='',input_seed=seed,label_seed='',inputs=len(x),sampled_labels=0,
            model_forwards=0,input_sha256=ash(x),feature_or_cell_sha256=ash(feature),label_sha256='',**reject))
        if law['name'] not in LABEL_LAWS:continue
        for k,defect in enumerate(DEFECTS):
            seed=CONFIG['label_input_seed']+100*j+k;yseed=CONFIG['label_response_seed']+100*j+k
            x,reject=sample(task,seed,CONFIG['label_n'],law);cell=cells(x);probability=cell_posterior(cell,defect)
            y=np.random.default_rng(yseed).random(len(x))<probability;rows,epsilon=label_intervals(cell,y)
            coverage=all(r['lower']<=cell_posterior(np.array([r['cell']]),defect)[0]<=r['upper'] for r in rows)
            labels.append(dict(law=law['name'],law_index=j,defect=defect,audit_index=k,epsilon_upper=epsilon,
                cell_interval_coverage=bool(coverage),minimum_count=min(r['count'] for r in rows),n=len(x)))
            for row in rows:cell_rows.append(dict(law=law['name'],true_defect=defect,true_cell_probability=float(cell_posterior(np.array([row['cell']]),defect)[0]),**row))
            path=directory/f'labels_{j}_{k}.npz';np.savez_compressed(path,cell=cell.astype(np.uint8),y=y,probability=probability);files[path.name]=sha(path)
            costs.append(dict(kind='labels',law=law['name'],defect=defect,input_seed=seed,label_seed=yseed,inputs=len(x),sampled_labels=len(y),
                model_forwards=0,input_sha256=ash(x),feature_or_cell_sha256=ash(cell),label_sha256=ash(y),**reject))
    co(HERE/'parameter_intervals.csv',parameter_rows);co(HERE/'label_intervals.csv',cell_rows);co(HERE/'evidence_costs.csv',costs)
    jo(HERE/'EVIDENCE_COMMIT.json',dict(utc=now(),parameters=parameters,labels=labels,raw_files=files))
def scenarios(law,evidence):
    out=[dict(kind='grid',key=f'e{k}',epsilon=eps,true_defect=eps) for k,eps in enumerate(EPS)]
    out += [dict(kind='labeled',key=f'label{k}',epsilon=row['epsilon_upper'],true_defect=row['defect'])
            for k,row in enumerate([r for r in evidence['labels'] if r['law']==law['name']])]
    return out
def certificate():
    entries,constants=ready();task,policy,adapter=components();evidence=json.loads((HERE/'EVIDENCE_COMMIT.json').read_text())
    directory=HERE/'certificate';directory.mkdir(exist_ok=False);jo(HERE/'CERTIFICATE_START.json',dict(utc=now(),evidence_sha256=sha(HERE/'EVIDENCE_COMMIT.json')))
    moments=[];curves=[];costs=[];ledger=[];files={};started=time.perf_counter()
    for record in entries:
        i=record['pair'];theta,grid,folded=model(record);D=constants[i]['D'];c=constants[i]['c']
        for j,law in enumerate(LAWS):
            start=time.perf_counter();seed=CONFIG['verification_seed']+100*j+i;tseed=CONFIG['verification_ties']+100*j+i
            x,reject=sample(task,seed,max(LOOKS)*BATCH,law);ties=np.random.default_rng(tseed).random(len(x));tx=task.partner(x)
            masks=get_masks(policy,adapter,x,grid,folded,ties);tm=get_masks(policy,adapter,tx,grid,folded,ties)
            assert all(np.array_equal(a,b) for a,b in zip(masks,tm))
            for coordinate in [0,17,63,255]:
                mixed=x[:BATCH].copy();mixed[coordinate]=task.partner(mixed[coordinate:coordinate+1])[0]
                mm=get_masks(policy,adapter,mixed,grid,folded,ties[:BATCH]);assert all(np.array_equal(a[:BATCH],b) for a,b in zip(masks,mm))
            union=masks[0]|masks[1];ids=np.flatnonzero(union);batch_ids=ids//BATCH;feature=features(x[ids]);feature_t=features(tx[ids]);assert np.array_equal(feature_t,-feature)
            h,calls=predict(task,theta,x[ids]);ht,more=predict(task,theta,tx[ids]);calls+=more;J,S,t=geometry(h,ht);assert S.max()<=D+1e-10
            parameter=evidence['parameters'][j];env={}
            for mode in MODES:
                center=law['theta'] if mode=='known' else parameter['estimate'];radius=[0,0] if mode=='known' else parameter['radius']
                env[mode]=envelope(feature,center,radius,law['rho']);et=envelope(feature_t,center,radius,law['rho'])
                assert np.array_equal(et[0],-env[mode][1]) and np.array_equal(et[1],-env[mode][0])
            data=dict(union_ids=ids,first=h,partner=ht,features=feature,cell=cells(x).astype(np.uint8),
                packed_masks=np.packbits(np.stack(masks),axis=1),policy_names=np.array(POLICIES),union_count=union.reshape(-1,BATCH).sum(1))
            scenario_list=scenarios(law,evidence);max_interval_miss=0.
            for name,mask in zip(POLICIES,masks):
                assert np.all(mask.reshape(-1,BATCH).sum(1)==QUOTA);chosen=mask[ids]
                reduce=lambda weight:np.bincount(batch_ids,weights=chosen*weight,minlength=max(LOOKS))/BATCH
                data[name+'_A']=reduce(np.minimum(J,c));data[name+'_B']=reduce(S/2)
                for scenario in scenario_list:
                    for mode in MODES:
                        low,up,rbar=env[mode];H=penalty(S,t,low,up,scenario['epsilon']);item_range=penalty_range(D,rbar,scenario['epsilon'])
                        assert H.min()>-1e-10 and H.max()<=item_range+1e-10
                        data[name+'_'+mode+'_'+scenario['key']]=reduce(H)
                for n in LOOKS:
                    base={}
                    for key,L,side in [('A',c,'lower'),('B',D/2,'upper')]:
                        result=interval(data[name+'_'+key][:n],L,side);base[key]=result['bound']
                        moments.append(dict(pair=i,law=law['name'],policy=name,moment=key,mode='shared',scenario='shared',epsilon='',**result))
                    for scenario in scenario_list:
                        bounds={}
                        for mode in MODES:
                            rbar=env[mode][2];result=interval(data[name+'_'+mode+'_'+scenario['key']][:n],penalty_range(D,rbar,scenario['epsilon']),'upper')
                            bounds[mode]=result['bound'];moments.append(dict(pair=i,law=law['name'],policy=name,moment='H',mode=mode,scenario=scenario['key'],epsilon=scenario['epsilon'],**result))
                        eta=min(1.,2*sum(abs(v) for v in law['theta'])/np.pi+law['rho'])
                        curves.append(dict(pair=i,law=law['name'],law_index=j,assumptions_covered=law['covered'],policy=name,n_batches=n,
                            kind=scenario['kind'],scenario=scenario['key'],epsilon=scenario['epsilon'],true_defect=scenario['true_defect'],
                            known_lower=base['A']-bounds['known'],estimated_lower=base['A']-bounds['estimated'],
                            old_global_lower=base['A']-scenario['epsilon']*base['B']-D*eta,A_lower=base['A'],B_upper=base['B'],
                            H_known_upper=bounds['known'],H_estimated_upper=bounds['estimated'],D=D,c=c,rho=law['rho'],
                            known_rbar=env['known'][2],estimated_rbar=env['estimated'][2],assumed_tv_upper=eta))
            data['forwards']=2*data['union_count'];assert sum(calls)==int(data['forwards'].sum())
            true_r=imbalance(feature,law)
            for mode in MODES:max_interval_miss=max(max_interval_miss,float(np.max(np.maximum(env[mode][0]-true_r,true_r-env[mode][1]))))
            path=directory/f'{i}_{j}.npz';np.savez_compressed(path,**data);files[path.name]=sha(path)
            for b,count in enumerate(data['union_count']):ledger.append(dict(pair=i,law=law['name'],batch=b,coordinate_inputs=BATCH,union_selected=int(count),forward_examples=2*int(count)))
            costs.append(dict(pair=i,law=law['name'],input_seed=seed,tie_seed=tseed,coordinate_inputs=len(x),forward_examples=sum(calls),
                invocations=len(calls),seconds=time.perf_counter()-start,input_sha256=ash(x),ties_sha256=ash(ties),
                candidate_mask_sha256=ash(masks[0]),folded_mask_sha256=ash(masks[1]),maximum_interval_miss=max_interval_miss,**reject))
            print(f'Certificate {i} {law["name"]}',flush=True)
    co(HERE/'moment_bounds.csv',moments);co(HERE/'certificate_curves.csv',curves);co(HERE/'certificate_costs.csv',costs);co(HERE/'certificate_ledger.csv',ledger)
    assert 2*len(moments)==MOMENT_EVENTS
    jo(HERE/'CERTIFICATE_COMMIT.json',dict(utc=now(),seconds=time.perf_counter()-started,moment_rows=len(moments),curve_rows=len(curves),raw_files=files))
def stress():
    entries,constants=ready();task,policy,adapter=components();directory=HERE/'stress';directory.mkdir(exist_ok=False)
    evidence=json.loads((HERE/'EVIDENCE_COMMIT.json').read_text());jo(HERE/'STRESS_START.json',dict(utc=now(),evidence_sha256=sha(HERE/'EVIDENCE_COMMIT.json')))
    rows=[];costs=[];audits=[];files={};started=time.perf_counter();names=['raw','full',*POLICIES]
    for record in entries:
        i=record['pair'];theta,grid,folded=model(record)
        for j,law in enumerate(LAWS):
            start=time.perf_counter();seed=CONFIG['truth_seed']+100*j+i;tseed=CONFIG['truth_ties']+100*j+i
            x,reject=sample(task,seed,CONFIG['truth_n'],law);ties=np.random.default_rng(tseed).random(len(x));tx=task.partner(x)
            h,calls=predict(task,theta,x);ht,more=predict(task,theta,tx);calls+=more;feature=features(x);r=imbalance(feature,law)
            J,S,t=geometry(h,ht);d=(h-ht)/2;m=(h+ht)/2;p0=task.posterior(x);cell=cells(x)
            masks=[np.zeros(len(x),bool),np.ones(len(x),bool),*get_masks(policy,adapter,x,grid,folded,ties)];extra=0
            for name,mask in zip(names[2:],masks[2:]):
                first=task.predict(theta,x[:BATCH]);second=task.predict(theta,task.partner(x[:BATCH][mask[:BATCH]]));out=first.copy()
                out[mask[:BATCH]]=(out[mask[:BATCH]]+second)/2;error=float(np.max(abs(out-np.where(mask[:BATCH],m[:BATCH],h[:BATCH]))));assert error<1e-11
                extra+=320;audits.append(dict(pair=i,law=law['name'],policy=name,forwards=320,max_error=error))
            posterior_cases=[]
            for world in CONFIG['worlds']:
                for eps in EPS:
                    p,pt=posterior(world,p0,d,feature,r,eps);other_p,other_pt=posterior(world,p0,-d,-feature,-r,eps)
                    assert np.max(abs(p-other_pt))<1e-12 and np.max(abs(pt-other_p))<1e-12
                    posterior_cases.append((world,eps,p,pt))
            if law['name'] in LABEL_LAWS:
                assert np.array_equal(cells(tx),cell^1)
                for defect in DEFECTS:posterior_cases.append(('cell',defect,cell_posterior(cell,defect),cell_posterior(cell^1,defect)))
            max_identity=max_extremal=0.
            for world,eps,p,pt in posterior_cases:
                assert p.min()>=0 and p.max()<=1 and np.max(abs(p-pt))<=eps+1e-12
                for name,mask in zip(names,masks):
                    output=np.where(mask,m,h);loss=nll(output,p);gain=nll(h,p)-loss;ft=nll(ht,pt)-nll(np.where(mask,m,ht),pt)
                    paired=(gain+ft)/2;C=paired+r*(gain-ft)/2
                    sharp=mask*(J-(abs(r)+(1-abs(r))*eps)*S/2+r*t)
                    identity=float(np.max(abs(paired-mask*(J-(p-pt)*d/2))));max_identity=max(max_identity,identity)
                    assert identity<1e-11 and np.max(sharp-C)<1e-11
                    if world=='orbit_worst':
                        extremal=float(np.max(abs(sharp-C)));max_extremal=max(max_extremal,extremal);assert extremal<1e-11
                    g=gain.reshape(-1,BATCH).mean(1);corrected=C.reshape(-1,BATCH).mean(1)
                    rows.append(dict(pair=i,law=law['name'],assumptions_covered=law['covered'],world=world,epsilon=eps,policy=name,
                        expected_nll=float(loss.mean()),expected_error=float(np.where(output>=0,1-p,p).mean()),direct_gain=float(g.mean()),
                        direct_batch_se=float(g.std(ddof=1)/np.sqrt(len(g))),known_density_gain=float(corrected.mean()),
                        known_density_batch_se=float(corrected.std(ddof=1)/np.sqrt(len(corrected))),unweighted_paired_gain=float(paired.mean()),
                        density_correction=float((C-paired).mean()),sharp_unclipped_mean=float(sharp.mean()),observed_posterior_tv=float(np.max(abs(p-pt)))))
            path=directory/f'{i}_{j}.npz';np.savez_compressed(path,first=h,partner=ht,p0=p0,features=feature,true_r=r,cell=cell.astype(np.uint8),
                policy_names=np.array(names),packed_masks=np.packbits(np.stack(masks),axis=1));files[path.name]=sha(path)
            costs.append(dict(pair=i,law=law['name'],input_seed=seed,tie_seed=tseed,coordinate_inputs=len(x),forward_examples=sum(calls),
                invocations=len(calls),extra_audit_forwards=extra,seconds=time.perf_counter()-start,input_sha256=ash(x),ties_sha256=ash(ties),
                max_identity_error=max_identity,max_extremal_error=max_extremal,**reject));print(f'Truth {i} {law["name"]}',flush=True)
    co(HERE/'stress_results.csv',rows);co(HERE/'stress_costs.csv',costs);co(HERE/'deployment_audits.csv',audits)
    jo(HERE/'STRESS_COMMIT.json',dict(utc=now(),seconds=time.perf_counter()-started,rows=len(rows),raw_files=files))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','prepare','checks','evidence','certificate','stress']);args=parser.parse_args()
    try:
        with threadpool_limits(limits=1):globals()[args.stage]()
    except Exception:
        jo(HERE/f'FAILURE_{args.stage}_{time.time_ns()}.json',dict(utc=now(),stage=args.stage,traceback=traceback.format_exc()));raise
