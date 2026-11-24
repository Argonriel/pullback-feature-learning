"""Correct threshold atoms in diagnostic worlds without changing any policy or mask."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;sys.path.insert(0,str(BASE))
import run as s
import policy as p
NODES,WEIGHTS=np.polynomial.legendre.leggauss(12)

class Integral:
    def __init__(self,r,curves):
        self.r=r;self.curves=curves;self.width=np.diff(r);self.x=(r[:-1,None]+r[1:,None])/2+self.width[:,None]*NODES/2
        self.rho=s.density(self.x);self.mass=(self.rho*WEIGHTS).sum(axis=1)*self.width/2;self.total=float(self.mass.sum())
    def values(self,key,x):return np.interp(x,self.r,self.curves[key],left=0,right=0)
    def full(self,key=None):
        values=1. if key is None else self.values(key,self.x.ravel()).reshape(self.x.shape)
        return (self.rho*values*WEIGHTS).sum(axis=1)*self.width/2
    def pieces(self,world):
        b=self.curves[world['field']+'_b'];t=world['threshold'];high=world['high'];atom=world['atom_probability'];lo=self.r[:-1];hi=self.r[1:];b0=b[:-1];b1=b[1:]
        seg=[]
        for j in range(len(lo)):
            if b0[j]==b1[j]==t:
                if atom>0:seg.append((lo[j],hi[j],atom))
                continue
            cuts=[lo[j],hi[j]]
            if (b0[j]-t)*(b1[j]-t)<0:cuts.insert(1,lo[j]+(t-b0[j])/(b1[j]-b0[j])*(hi[j]-lo[j]))
            for a,z in zip(cuts[:-1],cuts[1:]):
                mid=(a+z)/2;v=b0[j]+(b1[j]-b0[j])*(mid-lo[j])/(hi[j]-lo[j])
                if (v>t if high else v<t):seg.append((a,z,1.))
        return np.asarray(seg).reshape(-1,3)
    def world_integral(self,world,key=None):
        seg=self.pieces(world)
        if not len(seg):return 0.
        x=(seg[:,0,None]+seg[:,1,None])/2+(seg[:,1,None]-seg[:,0,None])*NODES/2;values=1. if key is None else self.values(key,x.ravel()).reshape(x.shape)
        return float(np.sum((seg[:,1]-seg[:,0])*seg[:,2]/2*np.sum(s.density(x)*values*WEIGHTS,axis=1)))
    def above_equal(self,field,t):
        b=self.curves[field+'_b'];lo=b[:-1];hi=b[1:];equal=(lo==hi)&(lo==t);above=(lo>t)&(hi>t);cross=(lo>t)!=(hi>t)
        mass=float(self.mass[above].sum());ids=np.flatnonzero(cross)
        if len(ids):
            cut=self.r[ids]+(t-lo[ids])/(hi[ids]-lo[ids])*self.width[ids]
            left=np.where(lo[ids]>t,self.r[ids],cut);right=np.where(lo[ids]>t,cut,self.r[ids+1]);x=(left[:,None]+right[:,None])/2+(right[:,None]-left[:,None])*NODES/2
            mass+=float(np.sum((right-left)/2*np.sum(s.density(x)*WEIGHTS,axis=1)))
        return mass,float(self.mass[equal].sum())
    def median(self,field):
        b=self.curves[field+'_b'];target=self.total/2
        candidates=np.unique(b[:-1][b[:-1]==b[1:]])
        for t in candidates:
            above,equal=self.above_equal(field,float(t))
            if equal>0 and above<=target<=above+equal:return float(t),float((target-above)/equal),above,equal
        lo=float(b.min());hi=float(b.max())
        for _ in range(70):
            mid=(lo+hi)/2;above,equal=self.above_equal(field,mid)
            if above>target:lo=mid
            else:hi=mid
        t=(lo+hi)/2;above,equal=self.above_equal(field,t)
        atom=float(np.clip((target-above)/equal,0,1)) if equal>0 else 0.
        return t,atom,above,equal

def probability(world,r,radial,curves):
    b=np.interp(r,radial,curves[world['field']+'_b'],left=0,right=0);t=world['threshold']
    out=(b>t).astype(float) if world['high'] else (b<t).astype(float)
    out[b==t]=world['atom_probability'];return out

def freeze():
    s.jo(ROOT/'freeze.json',dict(timestamp_utc=s.now(),source_sha256=s.sha(Path(__file__).read_bytes()),spec_sha256=s.sha((ROOT/'SPEC.md').read_bytes()),policy_sha256=s.sha((BASE/'policy.py').read_bytes()),original_outcomes_sha256=s.sha((BASE/'fresh'/'per_seed.csv').read_bytes()),original_policy_commit_sha256=s.sha((BASE/'fresh'/'policy_commit.json').read_bytes()),purpose='Evaluation-only atom and continuous-density repair; no policy, resolution, mask, data seed, or original-risk change.'))

def prepare():
    fr=json.loads((ROOT/'freeze.json').read_text());assert fr['source_sha256']==s.sha(Path(__file__).read_bytes());out=ROOT/'prepared';out.mkdir(exist_ok=False);worldrows=[];continuous=[];widths=[];manifest={};parity=[]
    for seed in range(1,17):
        rawpath=BASE/'fresh'/'diagnostics'/f'{seed}_moments.npz';curves=dict(np.load(rawpath));r=curves['r'];table=dict(np.load(BASE/'fresh'/'policies'/f'{seed}_policy.npz'));odd=np.array([k%2==1 for k in list(range(1,9))*2])
        for field in p.FIELDS:
            coefficients=p.coefficients_at(table,r,field);exact_even=np.all(coefficients[:,odd]==0,axis=1);before=curves[field+'_b'].copy();curves[field+'_b'][exact_even]=0.
            parity.append(dict(seed=seed,field=field,exact_even_nodes=int(exact_even.sum()),maximum_parity_correction=float(abs(before-curves[field+'_b']).max())))
        integ=Integral(r,curves);worlds=[]
        for field in p.FIELDS:
            threshold,atom,above,equal=integ.median(field)
            worlds.extend([dict(name=field+'_unrestricted_min',field=field,threshold=0.,high=True,atom_probability=0.,target_prevalence=-1.),dict(name=field+'_unrestricted_max',field=field,threshold=0.,high=False,atom_probability=0.,target_prevalence=-1.),dict(name=field+'_prevalence_half_min',field=field,threshold=threshold,high=True,atom_probability=atom,target_prevalence=.5),dict(name=field+'_prevalence_half_max',field=field,threshold=threshold,high=False,atom_probability=1-atom,target_prevalence=.5)])
        for world in worlds:
            prevalence=integ.world_integral(world);assert world['target_prevalence']<0 or abs(prevalence-.5)<2e-12
            worldrows.append(dict(seed=seed,world=world['name'],prevalence=prevalence,threshold=world['threshold'],atom_probability=world['atom_probability'],equal_threshold_mass=integ.above_equal(world['field'],world['threshold'])[1]))
        gainlookup={}
        for field in p.FIELDS:
            C=float(integ.full(field+'_C').sum())
            for world in worlds:
                gain=C-integ.world_integral(world,field+'_b');gainlookup[field,world['name']]=gain
                continuous.append(dict(seed=seed,field=field,world=world['name'],continuous_gain=gain,posterior_prevalence=integ.world_integral(world),C=C))
            for family in ('unrestricted','prevalence_half'):
                low=gainlookup[field,field+'_'+family+'_min'];high=gainlookup[field,field+'_'+family+'_max'];widths.append(dict(seed=seed,field=field,family=family,lower=low,upper=high,width=high-low))
        dest=out/f'{seed}_curves.npz';np.savez_compressed(dest,**curves);manifest[dest.name]=s.sha(dest.read_bytes());wp=out/f'{seed}_worlds.json';s.jo(wp,worlds);manifest[wp.name]=s.sha(wp.read_bytes())
    s.co(ROOT/'worlds.csv',worldrows);s.co(ROOT/'continuous_gains.csv',continuous);s.co(ROOT/'intervals.csv',widths);s.co(ROOT/'parity_checks.csv',parity)
    s.jo(ROOT/'diagnostic_commit.json',dict(timestamp_utc=s.now(),manifest=manifest,new_replay_started=False))

def replay():
    commit=json.loads((ROOT/'diagnostic_commit.json').read_text());assert all(s.sha((ROOT/'prepared'/k).read_bytes())==v for k,v in commit['manifest'].items());s.jo(ROOT/'replay_start.json',dict(timestamp_utc=s.now(),diagnostic_commit_sha256=s.sha((ROOT/'diagnostic_commit.json').read_bytes())))
    original=s.ci(BASE/'fresh'/'per_seed.csv');old={(int(r['seed']),r['policy'],r['world']):r for r in original};rows=[];checks=[]
    for seed in range(1,17):
        curves=dict(np.load(ROOT/'prepared'/f'{seed}_curves.npz'));worlds=json.loads((ROOT/'prepared'/f'{seed}_worlds.json').read_text());table=dict(np.load(BASE/'fresh'/'policies'/f'{seed}_policy.npz'));gt=dict(np.load(BASE/'fresh'/'policies'/f'{seed}_gate.npz'));gate=p.Gate(gt['edges'],gt['radius_scores'],gt['folded_scores']);x=s.sample(s.CONFIG['test_seed_base']+seed,s.CONFIG['test_n']);r=np.linalg.norm(x,axis=1);ties=np.random.default_rng(s.CONFIG['tie_seed_base']+seed).random(len(x));uniform=np.random.default_rng(s.CONFIG['group_seed_base']+seed).random(len(x)//4);probs={w['name']:probability(w,r,curves['r'],curves) for w in worlds};np.savez_compressed(ROOT/'prepared'/f'{seed}_test_posteriors.npz',**probs)
        for name in s.POLICIES:
            output,mask=p.deploy(x,lambda z:p.pred(table['theta'],z),table,gate,name,ties,uniform);maskhash=s.sha(np.packbits(mask).tobytes());assert maskhash==old[seed,name,'rings']['mask_sha256'];originalrisk=float(s.nll(output,s.post(r)).mean());err=abs(originalrisk-float(old[seed,name,'rings']['expected_risk']));assert err<1e-13
            checks.append(dict(seed=seed,policy=name,mask_sha256=maskhash,unchanged_mask=True,original_rings_risk_error=err))
            for world in worlds:
                risk=float(s.nll(output,probs[world['name']]).mean());prior=float(old[seed,name,world['name']]['expected_risk']);rows.append(dict(seed=seed,policy=name,world=world['name'],expected_risk=risk,original_diagnostic_risk=prior,diagnostic_risk_change=risk-prior,test_prevalence=float(probs[world['name']].mean()),mask_sha256=maskhash))
        print(f'corrected posterior replay {seed}/16',flush=True)
    s.co(ROOT/'per_seed.csv',rows);s.co(ROOT/'immutable_mask_checks.csv',checks);s.jo(ROOT/'replay_complete.json',dict(timestamp_utc=s.now(),rows=len(rows),checks=len(checks),all_masks_unchanged=True,max_original_rings_risk_error=max(r['original_rings_risk_error'] for r in checks),outcomes_sha256=s.sha((ROOT/'per_seed.csv').read_bytes())))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','replay']);args=ap.parse_args()
    with threadpool_limits(1):globals()[args.action]()
