"""Independent prevalence, sharp-attainment, and immutable-mask repair audit."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'continuous';FIX=BASE/'diagnostic_repair';sys.path.insert(0,str(BASE))
import run as original
import policy
N,W=np.polynomial.legendre.leggauss(24)

def rows(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def density(r):
    # Independent direct Rice-mixture formula, without the study density helper.
    sigma=.15
    return sum(.5*r/sigma**2*np.exp(-(r*r+a*a)/(2*sigma**2))*np.i0(r*a/sigma**2) for a in [.5,1.2])
def probs(world,r,knots,curves):
    b=np.interp(r,knots,curves[world['field']+'_b'],left=0,right=0)
    t=world['threshold'];ret=np.where(b>t,1.,0.) if world['high'] else np.where(b<t,1.,0.)
    return np.where(b==t,world['atom_probability'],ret)
def quadrature(knots,curves,world=None):
    edges=list(knots)
    if world is not None:
        b=curves[world['field']+'_b'];t=world['threshold']
        # Crossing cuts are found independently and added to the complete grid.
        for i in range(len(knots)-1):
            if min(b[i],b[i+1])<t<max(b[i],b[i+1]):
                edges.append(float(knots[i]+(knots[i+1]-knots[i])*(t-b[i])/(b[i+1]-b[i])))
    edges=np.unique(edges);l=edges[:-1,None];u=edges[1:,None]
    r=((l+u)/2+(u-l)/2*N).ravel();weight=((u-l)/2*W).ravel()*density(r)
    pp=np.ones_like(r) if world is None else probs(world,r,knots,curves)
    return r,weight,pp

def main():
    frozen=json.loads((BASE/'freeze.json').read_text());repairfreeze=json.loads((FIX/'freeze.json').read_text());commit=json.loads((FIX/'diagnostic_commit.json').read_text());start=json.loads((FIX/'replay_start.json').read_text());complete=json.loads((FIX/'replay_complete.json').read_text())
    assert all(sha(BASE/k)==v for k,v in frozen['hashes'].items())
    assert sha(FIX/'repair.py')==repairfreeze['source_sha256'] and sha(FIX/'SPEC.md')==repairfreeze['spec_sha256']
    assert sha(BASE/'fresh/per_seed.csv')==repairfreeze['original_outcomes_sha256']
    assert repairfreeze['timestamp_utc']<commit['timestamp_utc']<start['timestamp_utc']<complete['timestamp_utc']
    assert all(sha(FIX/'prepared'/k)==v for k,v in commit['manifest'].items())
    before={(int(r['seed']),r['policy'],r['world']):r for r in rows(BASE/'fresh/per_seed.csv')}
    after={(int(r['seed']),r['policy'],r['world']):r for r in rows(FIX/'per_seed.csv')}
    gainrows={(int(r['seed']),r['field'],r['world']):r for r in rows(FIX/'continuous_gains.csv')}
    intervals={(int(r['seed']),r['field'],r['family']):r for r in rows(FIX/'intervals.csv')}
    primary_hashes=json.loads((BASE/'fresh/policy_commit.json').read_text())['manifest']
    assert all(sha(BASE/'fresh'/k)==v for k,v in primary_hashes.items())
    report=[];max_gain_error=0.;max_probability_error=0.;max_replayed_risk_error=0.;max_prevalence=.0;max_mass_error=0.;max_parity=.0;replay_checks=[];forwards=0
    for seed in range(1,17):
        data=dict(np.load(FIX/'prepared'/f'{seed}_curves.npz'));olddata=dict(np.load(BASE/'fresh/diagnostics'/f'{seed}_moments.npz'));r=data['r'];worlds=json.loads((FIX/'prepared'/f'{seed}_worlds.json').read_text())
        table=dict(np.load(BASE/'fresh/policies'/f'{seed}_policy.npz'));odd=np.tile(np.arange(1,9)%2==1,2)
        for field in policy.FIELDS:
            # Compute coefficient interpolation ourselves from the saved anchor table.
            cc=np.column_stack([np.interp(r,table['radii'],table[field][:,k],left=0,right=0) for k in range(16)])
            structural_zero=np.all(cc[:,odd]==0,axis=1);changed=data[field+'_b']!=olddata[field+'_b']
            assert np.all(~changed|structural_zero) and np.all(data[field+'_b'][structural_zero]==0)
            assert np.array_equal(data[field+'_C'],olddata[field+'_C'])
            max_parity=max(max_parity,float(np.max(abs(data[field+'_b']-olddata[field+'_b']))))
        rr,ww,_=quadrature(r,data);total=float(ww.sum());max_mass_error=max(max_mass_error,abs(total-1))
        C={f:float(ww@np.interp(rr,r,data[f+'_C'])) for f in policy.FIELDS}
        true_inputs=original.sample(original.CONFIG['test_seed_base']+seed,65536);rtest=np.linalg.norm(true_inputs,axis=1);saved=np.load(FIX/'prepared'/f'{seed}_test_posteriors.npz');pp={}
        for world in worlds:
            qx,qw,qprob=quadrature(r,data,world);prevalence=float(qw@qprob)
            assert 0<=world['atom_probability']<=1 and np.all((qprob>=0)&(qprob<=1))
            half='prevalence_half' in world['name']
            if half:assert abs(prevalence-.5)<2e-12;max_prevalence=max(max_prevalence,abs(prevalence-.5))
            pp[world['name']]=probs(world,rtest,r,data)
            max_probability_error=max(max_probability_error,float(np.max(abs(pp[world['name']]-saved[world['name']]))))
            assert np.array_equal(pp[world['name']],saved[world['name']])
            for field in policy.FIELDS:
                independent_gain=C[field]-float(qw@(qprob*np.interp(qx,r,data[field+'_b'])))
                reported=float(gainrows[seed,field,world['name']]['continuous_gain']);err=abs(independent_gain-reported);max_gain_error=max(max_gain_error,err);assert err<1e-12
                if world['name'].startswith(field+'_'):
                    family='prevalence_half' if half else 'unrestricted';side='lower' if world['high'] else 'upper'
                    assert abs(float(intervals[seed,field,family][side])-independent_gain)<1e-12
            report.append(dict(seed=seed,world=world['name'],independent_prevalence=prevalence,half=half,threshold=world['threshold'],atom_probability=world['atom_probability']))
        for field in policy.FIELDS:
            unrestricted=intervals[seed,field,'unrestricted'];half=intervals[seed,field,'prevalence_half']
            assert float(unrestricted['lower'])<=float(half['lower'])+1e-12<=float(half['upper'])+2e-12<=float(unrestricted['upper'])+3e-12
            if field=='even':assert float(unrestricted['width'])==float(half['width'])==0
        for name in original.POLICIES:
            for world in worlds:assert after[seed,name,world['name']]['mask_sha256']==before[seed,name,'rings']['mask_sha256']
        if seed in [1,8,16]:
            gt=dict(np.load(BASE/'fresh/policies'/f'{seed}_gate.npz'));gate=policy.Gate(gt['edges'],gt['radius_scores'],gt['folded_scores'])
            ties=np.random.default_rng(original.CONFIG['tie_seed_base']+seed).random(65536);uniform=np.random.default_rng(original.CONFIG['group_seed_base']+seed).random(16384)
            for name in original.POLICIES:
                calls=[]
                def predict(x):calls.append(len(x));return policy.core.representation(table['theta'],x)[:,0]
                output,mask=policy.deploy(true_inputs,predict,table,gate,name,ties,uniform);forwards+=sum(calls)
                mask_hash=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest();assert mask_hash==before[seed,name,'rings']['mask_sha256']
                assert calls==([65536] if name=='raw' else [65536,65536] if name=='full' else [65536,16384])
                rrisk=float(np.mean(np.logaddexp(0,output)-original.post(rtest)*output));assert abs(rrisk-float(before[seed,name,'rings']['expected_risk']))<1e-13
                for world in worlds:
                    risk=float(np.mean(np.logaddexp(0,output)-pp[world['name']]*output));err=abs(risk-float(after[seed,name,world['name']]['expected_risk']));max_replayed_risk_error=max(max_replayed_risk_error,err);assert err<1e-13
                replay_checks.append(dict(seed=seed,policy=name,calls=calls,mask_matches=True,rings_risk_error=abs(rrisk-float(before[seed,name,'rings']['expected_risk']))))
    summary=dict(status='PASS',worlds_checked=len(report),half_prevalence_worlds_checked=sum(z['half'] for z in report),independent_density='Direct Rice-mixture density; 24-point Gauss quadrature on independently split segments',max_half_prevalence_error=max_prevalence,max_total_density_mass_error=max_mass_error,max_continuous_gain_error=max_gain_error,max_saved_probability_error=max_probability_error,max_parity_correction=max_parity,all_1920_repaired_masks_match_original=True,reexecuted_policy_masks=len(replay_checks),recomputed_iid_risk_rows=len(replay_checks)*12,max_recomputed_risk_error=max_replayed_risk_error,original_sources_and_policy_tables_unchanged=True,freeze_commit_replay_order_valid=True,diagnostic_repair_forward_examples=13631488,additional_independent_audit_forward_examples=forwards,sharpness_scope='Exact threshold-rearrangement characterization for the fixed piecewise-interpolated moment curves; numerical true-density integration. Underlying neural moments remain approximate. Intervals are model-specific, not one common posterior world.',tail_mass_bound=float(np.exp(-(3-1.2)**2/(2*.15**2))))
    (ROOT/'audit/continuous_repair_independent.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    with (ROOT/'audit/continuous_repair_worlds.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(report[0]));w.writeheader();w.writerows(report)
    (ROOT/'audit/continuous_repair_replays.json').write_text(json.dumps(replay_checks,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
