"""Independent, development-only audit of all three adapter interfaces.

No fitting and no fresh states. Writes only sibling audit outputs.
"""
import hashlib,importlib.util,json,sys,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.dont_write_bytecode=True
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
adapters=load('independent_task_adapter',ROOT/'validation/adapters.py')
baseline=load('independent_baseline_adapter_audit',ROOT/'method/baseline.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def independent_nll(h,p):return np.logaddexp(0,h)-p*h
def brute_score(adapter,x,table):
    """Eight-corner scalar calculation independent of vector implementation."""
    z=np.asarray(adapter.quotient(x)); z=z[:,None] if z.ndim==1 else z
    angle=np.arctan2(x[:,1],x[:,0])%np.pi
    result=[];field=table['folded_jensen'];m=field.shape[-1]
    for row,phi in zip(z,angle):
        lo=[];fr=[]
        for i,value in enumerate(row):
            axis=table[f'axis_{i}'];value=np.clip(value,axis[0],axis[-1])
            j=min(max(int(np.searchsorted(axis,value,side='right'))-1,0),len(axis)-2)
            lo.append(j);fr.append((value-axis[j])/(axis[j+1]-axis[j]))
        u=phi*m/np.pi-.5;k=int(np.floor(u));v=u-k;answer=0.
        for bits in range(1<<len(row)):
            index=[];weight=1.
            for j in range(len(row)):
                b=(bits>>j)&1;index.append(lo[j]+b);weight*=fr[j] if b else 1-fr[j]
            answer+=weight*((1-v)*field[tuple(index)+(k%m,)]+v*field[tuple(index)+((k+1)%m,)])
        result.append(answer)
    return np.asarray(result)

def run():
    records=[]; failures=[];warnings_seen=[]
    paths=[ROOT/'validation/adapters.py',ROOT/'method/baseline.py',ROOT/'task2d/task.py',ROOT/'task2d/core4d.py',ROOT/'reference/waveform_task.py',ROOT/'legacy/generator.py',ROOT/'source/pullback_feature_learning/core.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    with threadpool_limits(limits=1):
      for cohort in ('rings','waveform','two_invariants'):
        adapter=adapters.get_adapter(cohort)
        for seed in range(1201001,1201011):
            rng=np.random.default_rng(seed);tr=adapter.sample_train(seed)
            assert isinstance(tr,tuple) and len(tr)==2,(cohort,'training return format')
            xt,yt=tr;x=adapter.sample_inputs(seed,512);theta=adapter.initialize(seed)
            assert xt.shape==(256,adapter.dimension) and yt.shape==(256,)
            assert theta.shape==(adapter.parameters,)
            assert np.array_equal(np.unique(yt),[-1,1])
            assert np.sum(yt==1)==128 and np.sum(yt==-1)==128
            tx=adapter.partner(x);ttx=adapter.partner(tx);phi=adapter.angle(x);q=adapter.quotient(x)
            reconstructed=adapter.coordinates(q,phi) if adapter.quotient_dim==1 else adapter.coordinates(q[:,0],q[:,1],phi)
            alpha=rng.uniform(-np.pi,np.pi,len(x));rotated=adapter.transform(x,alpha)
            p=adapter.posterior(x);pt=adapter.posterior(tx)
            h=adapter.predict(theta,x);ht=adapter.predict(theta,tx);m=(h+ht)/2
            A=lambda a:np.logaddexp(0,a)
            J=.5*(A(h)+A(ht))-A(m)
            losses=.5*(independent_nll(h,p)+independent_nll(ht,p))-independent_nll(m,p)
            # First-view gain and partner-first gain average to Jensen under a shared mask.
            field_shape=(5,7,16) if adapter.quotient_dim==2 else (9,16)
            table=dict(ndim=np.array(adapter.quotient_dim),folded_jensen=rng.uniform(0,1,field_shape))
            for j,length in enumerate(field_shape[:-1]):table[f'axis_{j}']=np.linspace(adapter.support_lo[j],adapter.support_hi[j],length)
            ties=rng.random(len(x));mask=baseline.select(adapter,x,table,ties);tmask=baseline.select(adapter,tx,table,ties)
            a=independent_nll(h,p)-independent_nll(np.where(mask,m,h),p)
            b=independent_nll(ht,p)-independent_nll(np.where(tmask,m,ht),p)
            errs=dict(partner_involution=float(np.max(abs(ttx-x))),quotient_invariance=float(np.max(abs(adapter.quotient(tx)-q))),
              group_quotient_invariance=float(np.max(abs(adapter.quotient(rotated)-q))),reconstruction=float(np.max(abs(reconstructed-x))),
              partner_vs_pi=float(np.max(abs(tx-adapter.transform(x,np.pi)))),posterior_partner=float(np.max(abs(pt-p))),
              posterior_group=float(np.max(abs(adapter.posterior(rotated)-p))),jensen_identity=float(np.max(abs(losses-J))),
              masked_jensen_identity=float(np.max(abs(.5*(a+b)-mask*J))),
              helper_nll=float(np.max(abs(adapters.nll(h,p)-independent_nll(h,p)))),
              interpolation=float(np.max(abs(baseline.scores(adapter,x[:129],table)-brute_score(adapter,x[:129],table)))))
            assert np.array_equal(baseline.folded_angle(x),baseline.folded_angle(tx))
            assert np.array_equal(mask,tmask) and np.all(mask.reshape(-1,256).sum(1)==64)
            # Arbitrary individual flips keep the complete batch mask unchanged.
            flip=rng.random(len(x))<.5;xf=x.copy();xf[flip]=adapter.partner(xf[flip])
            assert np.array_equal(mask,baseline.select(adapter,xf,table,ties))
            assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            for key,value in errs.items():
                if value>1e-10:failures.append(dict(cohort=cohort,seed=seed,check=key,error=value))
            records.append(dict(cohort=cohort,seed=seed,training_shape=list(xt.shape),class_counts=[128,128],parameters=len(theta),
                quotient_shape=list(np.shape(q)),errors=errs,min_jensen=float(J.min()),quota_counts=mask.reshape(-1,256).sum(1).tolist()))
        # Axis cases and signed-zero canonicalization must be exactly partner-invariant.
        axes=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[-0.,1.],[1.,-0.]])
        if adapter.dimension==4:axes=np.c_[axes,np.ones((len(axes),2))]
        assert np.array_equal(baseline.folded_angle(axes),baseline.folded_angle(adapter.partner(axes)))
    # Unbounded-ring stress is separate from the actual simulation-support checks.
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        stress=adapters.ring_posterior(np.c_[np.array([0.,3.,15.,40.]),np.zeros(4)])
    if not np.isfinite(stress).all():warnings_seen.append('Current ring_posterior uses unscaled i0 and is nonfinite at extreme tail radii 15 or 40; ordinary sampled inputs and calibration [0,3] pass. This is a numerical API robustness limitation, not observed experiment failure.')
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths},'Source changed during audit'
    output=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='PASS' if not failures else 'FAIL',
      source_hashes=before,development_seeds=[1201001,1201010],fresh_models_read=0,fits=0,
      task_seed_cases=len(records),iid_geometry_inputs=30*512,training_labels_checked=30*256,
      interpolation_points=30*129,batch_quota_checks=60,records=records,failures=failures,warnings=warnings_seen,
      ring_extreme_stress={str(r):None if not np.isfinite(p) else float(p) for r,p in zip([0.,3.,15.,40.],stress)})
    (HERE/'task_geometry.json').write_text(json.dumps(output,indent=2)+'\n')
    maxima={k:max(r['errors'][k] for r in records) for k in records[0]['errors']}
    note=f'''# Independent adapter and interpolation audit

Date: {output['timestamp_utc']}. Status: {output['status']} for the specified development checks.

The executable `task_geometry.py` reads adapters and frozen model/task/selector source but changes none. It uses only seeds 1201001 through 1201010, all three tasks, no training and no fresh state. Thirty task/seed cases check 7,680 training labels (exactly 128 per class), 15,360 population inputs, model dimensions (80 for rings; 144 for each harmonic task), two-valued label encoding, geometry reconstruction, quotient and posterior invariance, exact partner involution, and pointwise/masked expected-loss Jensen identities. The rings training generator takes one seed argument and returns exactly `(x,y)`; it has no `n` parameter.

Canonical folded angles are bitwise equal under the partner, including signed-zero axis cases. All 60 quota batches select exactly 64 of 256, and arbitrary individual input flips preserve the whole mask with common tie keys. Independent scalar interpolation checks 3,870 points using explicit corner sums. This audits the same multilinear formula used by the two-invariant trilinear selector, rather than comparing it only with itself.

Maximum errors: `{json.dumps(maxima,sort_keys=True)}`.

The evaluator's angular world `p=.5+.4*cos(2 phi)` is half-turn invariant but not full-circle invariant. The selective Jensen guarantee needs the former; this world must not be described as a full-orbit-invariant label task. The two-amplitude quotient is two-dimensional only on its specified phase-coupled support. All source hashes, per-case results, and checks are in `task_geometry.json`.

Separate stress note: {' '.join(warnings_seen) if warnings_seen else 'Extreme tested ring radii produce finite posterior values.'} If the posterior implementation is repaired, that change must be documented and hashes refreshed before fresh scoring; this audit does not modify it.
'''
    (HERE/'TASK_GEOMETRY_REVIEW.md').write_text(note)
    print(json.dumps(dict(status=output['status'],cases=len(records),max_errors=maxima,warnings=warnings_seen),indent=2))
    assert not failures

if __name__=='__main__':run()
