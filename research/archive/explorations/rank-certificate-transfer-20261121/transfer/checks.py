"""Pre-confirmation checks of geometry, generic rank fitting, and exact CDF."""
from pathlib import Path
import sys,json,csv,hashlib,datetime
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import task,rank_policy as p

def split_quadrature_cdf(table,scores):
    nodes,weights=np.polynomial.legendre.leggauss(8);result=[];w=float(table['width']);edges=table['radii'];profile=table['gain_profile']
    for z in scores:
        value=0.
        for j in range(len(edges)-1):
            lo,hi=edges[j:j+2];slope=(profile[j+1]-profile[j])/(hi-lo);cuts=[lo,hi]
            if slope:
                for level in (z,z-w):
                    crossing=lo+(level-profile[j])/slope
                    if lo<crossing<hi:cuts.append(crossing)
            cuts=sorted(cuts)
            for a,b in zip(cuts[:-1],cuts[1:]):
                r=(a+b)/2+(b-a)*nodes/2;gain=profile[j]+slope*(r-lo);value+=float(weights@(task.radial_density(r)*np.clip((z-gain)/w,0,1))*(b-a)/2)
        result.append(value)
    return np.array(result)
def run():
    out=ROOT/'pilot_policy';out.mkdir(exist_ok=False);rows=[];x=task.sample_inputs(1035001,4096);ties=np.random.default_rng(1035002).random(len(x));keys=np.random.default_rng(1035003).random(len(x));partial=x.copy();partial[::3]=task.partner(partial[::3])
    for seed in range(1,5):
        for method,rate in (('pb',8),('gd',.3)):
            theta=np.load(ROOT/'pilot'/f'{seed}_{method}_{rate:g}.npz')['theta'];calls=[];original=task.predict
            def counted(t,xx):calls.append(len(xx));return original(t,xx)
            task.predict=counted
            try:table=p.calibrate(theta,1036000+seed+(100 if method=='gd' else 0))
            finally:task.predict=original
            assert sum(calls)==66048;np.savez_compressed(out/f'{seed}_{method}.npz',**table);h=task.predict(theta,x);hp=task.predict(theta,task.partner(x));assert np.array_equal(p.scores(x,table,'rank_even',keys),p.scores(task.partner(x),table,'rank_even',keys));assert np.array_equal(p.select(x,h,table,'rank_even',ties,keys),p.select(partial,h,table,'rank_even',ties,keys))
            for name in p.POLICIES:assert np.all(p.select(x,h,table,name,ties,keys).reshape(-1,256).sum(1)==64)
            probes=np.linspace(float(table['gain_profile'].min())-.1*float(table['width']),float(table['gain_profile'].max())+1.1*float(table['width']),37);cdf=p.population_cdf(table,probes);quad=split_quadrature_cdf(table,probes);cdf_error=float(abs(cdf-quad).max());assert cdf_error<1e-9
            fft_error=0.;partner_error=0.;phi=2*np.pi*(np.arange(p.NA)+.5)/p.NA
            for radial_index in (0,21,53,96,128):
                r=table['radii'][radial_index];xx=task.coordinates(r,phi);hh=table['h'][radial_index];other=np.roll(hh,p.NA//2);partner_error=max(partner_error,float(abs(other-task.predict(theta,task.partner(xx))).max()));A=np.logaddexp(0,hh)-np.logaddexp(0,(hh+other)/2);d=(hh-other)/2
                for k in p.FREQUENCIES:
                    for shift in (0,63,117,256,511):
                        phase=(shift+.25)/p.NA;angles=p.action_angle(xx,np.full(p.NA,k));score=table['gain_profile'][radial_index]+float(table['width'])*((k*angles/(2*np.pi)+phase)%1);q=p.rank_probability(p.score_cdf(table,score));index=(k-1)*p.NA+shift;fft_error=max(fft_error,abs(float(q@A/p.NA-table['action_objectives'][radial_index,index])),abs(float(q@d/p.NA-table['action_moments'][radial_index,index])))
            assert fft_error<1e-10 and partner_error<1e-10
            rows.append(dict(seed=seed,method=method,forward_examples=sum(calls),coordinate_draws=int(table['coordinate_examples']),cdf_analytic_vs_split_GL8=cdf_error,fft_direct_max_error=fft_error,partner_roll_max_error=partner_error,max_anchor_moment=float(abs(table['rank_balanced_moment']).max()),even_scores_bitwise_equal=True,even_partial_flip_masks_identical=True,all_quota_64=True,policy_sha256=hashlib.sha256((out/f'{seed}_{method}.npz').read_bytes()).hexdigest()));print(seed,method,rows[-1],flush=True)
    nodes,w=np.polynomial.legendre.leggauss(32);r=.4+(nodes+1)/2;rho=task.radial_density(r);posterior=(r-.4)**3/((r-.4)**3+(1.4-r)**3);mass=float(w@rho/2);prevalence=float(w@(rho*posterior)/2);bayes=0.
    for lo,hi in ((.4,.9),(.9,1.4)):
        r=(lo+hi)/2+(hi-lo)*nodes/2;u=r-.4;pr=u**3/(u**3+(1-u)**3);bayes+=float(w@(task.radial_density(r)*np.minimum(pr,1-pr))*(hi-lo)/2)
    assert abs(mass-1)<1e-12 and abs(prevalence-.5)<1e-12 and abs(bayes-.109375)<1e-12
    with (out/'checks.csv').open('x',newline='') as f:writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'SUMMARY.json').write_text(json.dumps(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=8,task_mass=mass,prevalence=prevalence,bayes_error=bayes,max_CDF_error=max(r['cdf_analytic_vs_split_GL8'] for r in rows),max_FFT_error=max(r['fft_direct_max_error'] for r in rows),all_checks_pass=True,new_confirmation_models_accessed=False),indent=2))

if __name__=='__main__':
    with threadpool_limits(1):run()
