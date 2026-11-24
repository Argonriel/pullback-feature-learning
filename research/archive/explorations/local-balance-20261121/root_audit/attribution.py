"""Prespecified attribution of local analysis and variance-sensitive confidence."""
from pathlib import Path
import csv, hashlib, json, datetime, sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'reference'))
import continuous_policy as prior

def main():
    freeze=HERE/'attribution_freeze.json'
    if '--freeze' in sys.argv:
        with freeze.open('x',encoding='utf8') as out:
            json.dump(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                design='On the SAME frozen antithetic observations: local empirical Bernstein (primary), global empirical Bernstein (bound attribution), local Hoeffding (concentration attribution). Controls are descriptive. No new forward calls.',
                fresh_models_opened=False),out,indent=2)
        return
    assert json.loads(freeze.read_text())['code_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows=[]
    for seed in range(1,17):
        f=dict(np.load(ROOT/'validation/policies'/f'{seed}_continuous.npz'))
        z=dict(np.load(HERE/'certificates'/f'{seed}_observations.npz'))
        n=len(z['W']);delta=.05/16;D=3+float(f['C_h'])
        angular,radial=prior.bound_at(f,z['r']);globalB=angular+radial
        E=f['anchor_residual']+f['angular_bound'];width=np.diff(f['radii'])
        kr=float(f['L'])*width*(f['gamma'][:-1]+f['gamma'][1:])
        t=np.clip(np.divide(E[1:]-E[:-1]+kr,2*kr,out=np.zeros_like(kr),where=kr>0),0,1)
        globalmax=float(np.max((E[:-1]+(E[1:]-E[:-1])*t+kr*t*(1-t))/8))
        local=dict(np.load(ROOT/'bounds/fresh'/f'{seed}_local.npz'))
        spans=dict(local=D/4+float(local['uniform_bound'])/2,global_bound=D/4+globalmax/2)
        for label,B,span in [('local',z['B'],spans['local']),('global_bound',globalB,spans['global_bound'])]:
            obs=z['W']-B/2
            eb=np.sqrt(2*np.var(obs,ddof=1)*np.log(2/delta)/n)+7*span*np.log(2/delta)/(3*(n-1))
            hd=span*np.sqrt(np.log(1/delta)/(2*n))
            rows.append(dict(seed=seed,bound=label,mean_centered_gain=float(z['W'].mean()),
                mean_penalty=float(B.mean()/2),mean_Z=float(obs.mean()),
                empirical_Bernstein_deduction=float(eb),empirical_Bernstein_lower=float(obs.mean()-eb),
                Hoeffding_deduction=float(hd),Hoeffding_lower=float(obs.mean()-hd),additional_forwards=0))
    with (HERE/'certificate_attribution.csv').open('x',newline='',encoding='utf8') as out:
        writer=csv.DictWriter(out,list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({label:{kind:sum(r['bound']==label and r[kind]>0 for r in rows) for kind in ['empirical_Bernstein_lower','Hoeffding_lower']} for label in ['local','global_bound']}))

if __name__=='__main__':main()
