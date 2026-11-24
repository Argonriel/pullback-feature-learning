"""Roundoff-only recovery of the frozen truth stage; original run.py is unchanged."""
from run import *
from math_utils import posterior as frozen_posterior
ROUNDING=dict(calls=0,clipped_values=0,maximum_absolute_excursion=0.)

def posterior(*args):
    p,pt=frozen_posterior(*args)
    for value in [p,pt]:
        excursion=float(max(0.,-value.min(),value.max()-1.))
        assert excursion<=1e-12, 'Recovery is limited to probability endpoint roundoff'
        ROUNDING['maximum_absolute_excursion']=max(ROUNDING['maximum_absolute_excursion'],excursion)
        ROUNDING['clipped_values']+=int(np.sum((value<0)|(value>1)))
    ROUNDING['calls']+=1
    return np.clip(p,0.,1.),np.clip(pt,0.,1.)

def stress():
    entries,constants=ready();task,policy,adapter=components();directory=HERE/'stress_recovery';directory.mkdir(exist_ok=False)
    evidence=json.loads((HERE/'EVIDENCE_COMMIT.json').read_text());jo(HERE/'STRESS_RECOVERY_START.json',dict(utc=now(),evidence_sha256=sha(HERE/'EVIDENCE_COMMIT.json')))
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
    jo(HERE/'STRESS_RECOVERY_COMMIT.json',dict(utc=now(),seconds=time.perf_counter()-started,rows=len(rows),raw_files=files,raw_directory='stress_recovery',recovery_freeze_sha256=sha(HERE/'STRESS_RECOVERY_FREEZE.json'),probability_roundoff=ROUNDING))

if __name__=='__main__':
    recovery=json.loads((HERE/'STRESS_RECOVERY_FREEZE.json').read_text())
    assert recovery['original_freeze_sha256']==sha(HERE/'FREEZE.json')
    for name,value in recovery['hashes'].items():assert sha(HERE/name)==value,name
    approval=json.loads((HERE/'STRESS_RECOVERY_APPROVAL.json').read_text())
    assert approval['recovery_freeze_sha256']==sha(HERE/'STRESS_RECOVERY_FREEZE.json') and approval['execution_authorized']
    try:
        with threadpool_limits(limits=1):stress()
    except Exception:
        jo(HERE/f'FAILURE_stress_recovery_{time.time_ns()}.json',dict(utc=now(),stage='stress_recovery',traceback=traceback.format_exc()));raise
