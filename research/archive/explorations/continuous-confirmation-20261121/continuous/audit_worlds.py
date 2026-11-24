"""Independent numerical checks of continuous posterior worlds and real routing costs."""
from pathlib import Path
import sys,json,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run as s
import policy as p

def intervals(world,r,m):
    if world['kind']!='threshold' or world['constant_value']>=0:return None,0.
    b=m[world['field']+'_b'];threshold=world['threshold'];high=world['high'];result=[];flatmass=0.
    for j in range(len(r)-1):
        lo=r[j];hi=r[j+1];a=b[j];z=b[j+1]
        if a==z==threshold:flatmass+=.5*(s.density(np.array([lo]))[0]+s.density(np.array([hi]))[0])*(hi-lo);continue
        cuts=[lo,hi]
        if (a-threshold)*(z-threshold)<0:cuts.insert(1,lo+(threshold-a)/(z-a)*(hi-lo))
        for left,right in zip(cuts[:-1],cuts[1:]):
            mid=(left+right)/2;v=a+(z-a)*(mid-lo)/(hi-lo)
            if (v>threshold if high else v<threshold):result.append((left,right))
    return result,flatmass

def main():
    nodes,weights=np.polynomial.legendre.leggauss(12);audits=[];times=[]
    for seed in range(1,17):
        path=ROOT/'fresh'/'diagnostics'/f'{seed}_moments.npz';m=dict(np.load(path));r=m['r'];worlds=json.loads((ROOT/'fresh'/'diagnostics'/f'{seed}_worlds.json').read_text())
        for world in worlds:
            if world['kind']!='threshold':continue
            segments,flat=intervals(world,r,m)
            if segments is None:prevalence=world['constant_value']
            elif not segments:prevalence=0.
            else:
                seg=np.array(segments);xx=(seg[:,1,None]+seg[:,0,None])/2+(seg[:,1,None]-seg[:,0,None])/2*nodes;prevalence=float(np.sum((seg[:,1]-seg[:,0])/2*np.sum(s.density(xx)*weights,axis=1)))
            frozen=s.posterior_integral(r,m['density'],np.ones_like(r),world,m)
            audits.append(dict(seed=seed,world=world['name'],independent_continuous_prevalence=prevalence,frozen_piecewise_prevalence=frozen,prevalence_numerical_error=abs(prevalence-frozen),target_half='_prevalence_half_' in world['name'],half_error=abs(prevalence-.5) if '_prevalence_half_' in world['name'] else '',equal_threshold_flat_mass=flat,valid_half=abs(prevalence-.5)<1e-6 if '_prevalence_half_' in world['name'] else True))
        # All arrays are loaded before timings; no ZIP access inside deployment.
        table=dict(np.load(ROOT/'fresh'/'policies'/f'{seed}_policy.npz'));gt=dict(np.load(ROOT/'fresh'/'policies'/f'{seed}_gate.npz'));gate=p.Gate(gt['edges'],gt['radius_scores'],gt['folded_scores'])
        x=s.sample(846000+seed,4096);ties=np.random.default_rng(847000+seed).random(len(x));uniform=np.random.default_rng(848000+seed).random(len(x)//4);rng=np.random.default_rng(849000+seed)
        predict=lambda xx:p.pred(table['theta'],xx)
        for rep in range(5):
            for name in rng.permutation(s.POLICIES):
                start=time.perf_counter_ns()
                for j in range(16):p.deploy(x[j*256:(j+1)*256],predict,table,gate,str(name),ties[j*256:(j+1)*256],uniform[j*64:(j+1)*64])
                if rep:times.append(dict(seed=seed,policy=str(name),repeat=rep,ms_per_batch=(time.perf_counter_ns()-start)/16/1e6))
    s.co(ROOT/'posterior_world_audit.csv',audits);s.co(ROOT/'deployment_timing.csv',times)
    half=[r for r in audits if r['target_half']];summary=dict(half_world_count=len(half),invalid_half_worlds=sum(not r['valid_half'] for r in half),maximum_half_prevalence_error=max(r['half_error'] for r in half),maximum_independent_prevalence_integration_error=max(r['prevalence_numerical_error'] for r in audits),maximum_exact_threshold_plateau_mass=max(r['equal_threshold_flat_mass'] for r in audits),tail_mass_bound=float(np.exp(-(3-1.2)**2/(2*.15**2))))
    s.jo(ROOT/'posterior_world_audit.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
