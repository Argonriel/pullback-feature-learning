"""Frozen local-bound confirmation; field fitting and prediction policies are unchanged."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,shutil,sys,time
import numpy as np
from threadpoolctl import threadpool_limits
import local,checks
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'reference'))
import continuous_policy as reference
CONFIG=dict(sub=4,na=256,old_models=list(range(1,17)),fresh_models=list(range(1,17)),diagnostic_nr=2049,diagnostic_folded_na=512,old_check_seed=947000,fresh_check_seed=948000,chunk=4096,primary='Mean input-mass integrated local bound at most 1% of the mean previous global bound on 16 fresh models; every failed check retained.',pointwise_required=True,finite_precision='float64 with padding, not formally directed-rounded',fresh_access_before_freeze=False)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,v):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def csvout(p,rows):
    with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def density(r):return sum(.5*r/.15**2*np.exp(-(r*r+a*a)/(2*.15**2))*np.i0(r*a/.15**2) for a in [.5,1.2])
def trap(x,r):return float(np.sum((x[:-1]+x[1:])*np.diff(r)/2))
def previous_uniform(table):
    E=table['anchor_residual']+table['angular_bound'];g=table['gamma'];width=np.diff(table['radii']);v=float(table['L'])*width*(g[:-1]+g[1:]);t=np.clip(np.divide(E[1:]-E[:-1]+v,2*v,out=np.zeros_like(v),where=v>0),0,1)
    return float(np.max((E[:-1]+(E[1:]-E[:-1])*t+v*t*(1-t))/8))

def freeze():
    inputs=ROOT/'development_inputs';inputs.mkdir(exist_ok=False)
    for seed in range(1,17):shutil.copy2(local.PRIOR/f'continuous/fresh/policies/{seed}_policy.npz',inputs/f'{seed}_policy.npz')
    shutil.copy2(local.PRIOR/'continuous/fresh/balance_bounds.csv',inputs/'prior_balance_bounds.csv')
    files=['local.py','checks.py','experiment.py','THEORY.md','PROTOCOL.md']
    dump(ROOT/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,source_hashes={p:sha(ROOT/p) for p in files},core_sha256=sha(local.core.__file__),reference_policy_sha256=sha(reference.__file__),pilot_summary_sha256=sha(ROOT/'pilot/summary.csv'),development_input_sha256={p.name:sha(p) for p in inputs.iterdir()},fresh_models_or_outcomes_accessed=False))

def run(cohort):
    freeze=json.loads((ROOT/'freeze.json').read_text());assert CONFIG==freeze['config']
    assert all(sha(ROOT/k)==v for k,v in freeze['source_hashes'].items())
    assert sha(local.core.__file__)==freeze['core_sha256'] and sha(reference.__file__)==freeze['reference_policy_sha256']
    out=ROOT/cohort;out.mkdir(exist_ok=False);dump(out/'start.json',dict(timestamp_utc=now(),freeze_sha256=sha(ROOT/'freeze.json')))
    rows=[];audits=[];manifest={}
    for seed in range(1,17):
        source=ROOT/'development_inputs'/f'{seed}_policy.npz' if cohort=='old' else ROOT.parent/'validation/policies'/f'{seed}_continuous.npz'
        table=dict(np.load(source));b=local.construct(table,CONFIG['sub'],CONFIG['na'],CONFIG['chunk']);dest=out/f'{seed}_local.npz';np.savez_compressed(dest,**b);manifest[dest.name]=sha(dest)
        audit=checks.verify(table,b,CONFIG[cohort+'_check_seed']+seed);audits.append(dict(seed=seed,**audit))
        r=np.linspace(0,3,CONFIG['diagnostic_nr']);c=np.column_stack([np.interp(r,table['radii'],table['balanced'][:,k]) for k in range(16)]);observed=local.exact_moments(table['theta'],r,c,CONFIG['diagnostic_folded_na']);lower,upper=local.pointwise_bound(b,r);violation=float(max(np.max(lower-observed),np.max(observed-upper)));assert violation<1e-11
        rho=density(r);oldangular,oldradial=reference.bound_at(table,r);oldintegrated=trap(rho*(oldangular+oldradial),r);observedintegral=trap(rho*abs(observed),r)
        np.savez_compressed(out/f'{seed}_diagnostics.npz',r=r,density=rho,observed_moment=observed,pointwise_lower=lower,pointwise_upper=upper,old_angular_bound=oldangular,old_radial_bound=oldradial)
        row=dict(seed=seed,cohort=cohort,source_sha256=sha(source),theta_sha256=hashlib.sha256(table['theta'].tobytes()).hexdigest(),**{k:float(b[k]) for k in ['uniform_bound','integrated_bound','angular_integral','radial_integral','endpoint_integral','point_seconds','interval_seconds','poisson_cdf_tail_bound']},old_uniform_bound=previous_uniform(table),old_integrated_bound=oldintegrated,observed_integrated_abs_b=observedintegral,observed_max_abs_b=float(abs(observed).max()),maximum_moment_bound_violation=violation,interval_jet_propagations=int(b['interval_jet_propagations']),exact_point_forwards=int(b['exact_point_forwards']),diagnostic_moment_point_forwards=2*len(r)*CONFIG['diagnostic_folded_na'],spotcheck_moment_point_forwards=audit['diagnostic_point_forwards'],finite_difference_diagnostic_pass=audit['finite_difference_diagnostic_pass'])
        rows.append(row);print(json.dumps(row),flush=True)
    csvout(out/'summary.csv',rows);dump(out/'numerical_checks.json',audits);dump(out/'commit.json',dict(timestamp_utc=now(),freeze_sha256=sha(ROOT/'freeze.json'),bound_file_sha256=manifest,summary_sha256=sha(out/'summary.csv'),all_models_retained=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','old','fresh']);args=ap.parse_args()
    with threadpool_limits(1):freeze() if args.action=='freeze' else run(args.action)
