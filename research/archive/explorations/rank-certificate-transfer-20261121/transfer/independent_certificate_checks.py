"""Independent uniform-subset numerical checks; no fitting or source mutation."""
from pathlib import Path
import sys,json,csv,hashlib,datetime,time,importlib.util
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'validation'));sys.path.insert(0,str(ROOT/'certificate'))
import adapters,known_cdf as kc,local_rank as local
spec=importlib.util.spec_from_file_location('root_certificate_reviewed',ROOT/'root_audit'/'certificate.py');rootcert=importlib.util.module_from_spec(spec);spec.loader.exec_module(rootcert)
MODELS=(1,8,16);COHORTS=('rings','transfer');NA=16384;OFFSET=.371

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def polar_jets(adapter,theta,r,phi):
    c=np.cos(phi);s=np.sin(phi);zero=np.zeros_like(r)
    if adapter.dimension==2:
        x=np.c_[r*c,r*s];xr=np.c_[c,s];xp=np.c_[-r*s,r*c];xpp=-x
    else:
        c2=np.cos(2*phi);s2=np.sin(2*phi);x=np.c_[c,s,r*c2,r*s2];xr=np.c_[zero,zero,c2,s2];xp=np.c_[-s,c,-2*r*s2,2*r*c2];xpp=np.c_[-c,-s,-4*r*c2,-4*r*s2]
    initial=[x,xr,np.zeros_like(x),xp,xpp];sign=-np.ones(adapter.dimension);sign[2:]=1
    def propagate(values):
        values=[v.copy() for v in values]
        for A,B,b in adapter.network.unpack(theta):
            z=[v@A.T for v in values];z[0]+=b;t=np.tanh(z[0]);d=1-t*t;dd=-2*t*d
            nonlinear=[t,d*z[1],dd*z[1]**2+d*z[2],d*z[3],dd*z[3]**2+d*z[4]]
            values=[v+.5*n@B.T for v,n in zip(values,nonlinear)]
        return np.stack([v[:,0] for v in values],axis=1)
    return .5*(propagate(initial)-propagate([v*sign for v in initial]))
def density(r,cohort):
    if cohort=='transfer':return adapters.task.radial_density(r)
    return .5*sum(r/.15**2*np.exp(-(r*r+mu*mu)/(2*.15**2))*np.i0(r*mu/.15**2) for mu in (.5,1.2))
def split_cdf(table,scores,cohort):
    if cohort=='transfer':return adapters.transfer_rank.population_cdf(table,scores)
    nodes,weights=np.polynomial.legendre.leggauss(32);answer=[];edges=table['radii'];a=table['gain_profile'];w=float(table['width'])
    for score in np.asarray(scores).ravel():
        value=0.
        for j in range(len(edges)-1):
            lo,hi=edges[j:j+2];slope=(a[j+1]-a[j])/(hi-lo);cuts=[lo,hi]
            if slope:
                for threshold in (score,score-w):
                    crossing=lo+(threshold-a[j])/slope
                    if lo<crossing<hi:cuts.append(crossing)
            cuts.sort()
            for left,right in zip(cuts[:-1],cuts[1:]):
                r=(left+right)/2+(right-left)*nodes/2;location=a[j]+slope*(r-lo);value+=float(weights@(density(r,cohort)*np.clip((score-location)/w,0,1))*(right-left)/2)
        answer.append(value)
    return np.array(answer).reshape(np.shape(scores))
def fast_cdf_setup(table,cohort):
    # Independent high-order population quadrature for dense angular diagnostics.
    nodes,weights=np.polynomial.legendre.leggauss(256);edges=table['radii'];r=((edges[:-1,None]+edges[1:,None])/2+np.diff(edges)[:,None]*nodes/2).ravel();mass=(np.diff(edges)[:,None]*weights/2).ravel()*density(r,cohort);a=np.interp(r,edges,table['gain_profile']);order=np.argsort(a);return a[order],mass[order]
def true_marginal(adapter,table,x,cohort):
    r=adapter.quotient(x);phi=np.arctan2(x[:,1],x[:,0]);edges=table['radii'];out=np.zeros(len(x));lo=np.clip(np.searchsorted(edges,r,side='right')-1,0,len(edges)-2);t=np.clip((r-edges[lo])/np.diff(edges)[lo],0,1);location=np.interp(r,edges,table['gain_profile']);outside=(r<edges[0])|(r>edges[-1]);queries=0
    for side,weight in ((0,1-t),(1,t)):
        anchor=lo+side;mix=table['rank_balanced_mix'][anchor]
        for component,mass in ((0,mix),(1,1-mix)):
            k=np.where(outside,2,table['rank_balanced_frequency'][anchor,component]);phase=np.where(outside,0.,table['rank_balanced_phase'][anchor,component]);score=location+float(table['width'])*((k*phi/(2*np.pi)+phase)%1);F=split_cdf(table,score,cohort);out+=weight*mass*adapter.rank.rank_probability(F);queries+=len(score)
    return out,queries
def orbit_moments(adapter,table,radii,cohort,qu,qv):
    phi=2*np.pi*(np.arange(NA)+OFFSET)/NA;j=np.arange(NA);cal_a,cal_mass=fast_cdf_setup(table,cohort);rows=[];calls=0
    for r in radii:
        x=adapter.coordinates(r,phi);h=adapter.predict(table['theta'],x);hp=np.roll(h,NA//2);d=(h-hp)/2;calls+=len(x);base={};location=np.interp(r,table['radii'],table['gain_profile'])
        for k in (1,2,3,4):
            frac=k*OFFSET-np.floor(k*OFFSET);score=location+float(table['width'])*(j+frac)/NA;F=adapter.rank.mixture_cdf(cal_a,cal_mass,score,float(table['width']));base[k]=np.interp(F,qu,qv)
        lo=int(np.clip(np.searchsorted(table['radii'],r,side='right')-1,0,len(table['radii'])-2));t=float(np.clip((r-table['radii'][lo])/(table['radii'][lo+1]-table['radii'][lo]),0,1));q=np.zeros(NA)
        for anchor,weight in ((lo,1-t),(lo+1,t)):
            mix=table['rank_balanced_mix'][anchor]
            for component,mass in ((0,mix),(1,1-mix)):
                k=int(table['rank_balanced_frequency'][anchor,component]);phase=float(table['rank_balanced_phase'][anchor,component]);index=(k*j+int(np.floor(k*OFFSET))+int(round(phase*NA)))%NA;q+=weight*mass*base[k][index]
        direct=adapter.predict(table['theta'],adapter.partner(x[::1024]));calls+=len(direct)
        rows.append(dict(r=float(r),b=float(np.mean(q*d)),b_subsample4096=float(np.mean((q*d)[::4])),partner_roll_error=float(abs(hp[::1024]-direct).max()),mean_rate=float(q.mean())))
    return rows,calls
def run():
    destination=HERE/'independent_checks';destination.mkdir(exist_ok=False);qt=kc.q_table();summaries=[];all_jets=[];all_moments=[]
    sources=['root_audit/certificate.py','root_audit/CERTIFICATE.md','certificate/local_rank.py','certificate/known_cdf.py','validation/adapters.py','transfer/task.py','transfer/rank_policy.py']
    commit=dict(timestamp_utc=now(),source_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'INDEPENDENT_CHECK_PROTOCOL.json'),reviewed_source_sha256={f:sha(ROOT/f) for f in sources},fresh_files_read_before_commit=False)
    (destination/'SOURCE_COMMIT.json').write_text(json.dumps(commit,indent=2))
    for cohort in COHORTS:
        adapter=adapters.get_adapter(cohort);task_name='rings' if cohort=='rings' else 'waveform'
        for seed in MODELS:
            start=time.perf_counter();table=dict(np.load(ROOT/'validation'/'policies'/f'{cohort}_{seed}_pb_rank.npz'));bound=dict(np.load(ROOT/'certificate'/'fresh'/task_name/f'{seed}_local.npz'));dist=dict(np.load(ROOT/'certificate'/'fresh'/task_name/f'{seed}_distribution.npz'));rng=np.random.default_rng((1012000 if cohort=='rings' else 1052000)+seed);lo,hi=table['radii'][[0,-1]];random_r=rng.uniform(lo,hi,32);radii=np.r_[random_r,table['radii'][[0,32,64,96,128]]];moments,calls=orbit_moments(adapter,table,radii,cohort,qt[0],qt[1]);bl,bh=local.pointwise_bound(bound,radii)
            for i,row in enumerate(moments):row.update(cohort=cohort,seed=seed,lower=float(bl[i]),upper=float(bh[i]),enclosed=bool(bl[i]-1e-10<=row['b']<=bh[i]+1e-10));all_moments.append(row)
            jet_violations=0;jet_passes=0
            for box in range(32):
                rr=rng.uniform(lo,hi);rw=(hi-lo)/2048;rl=max(lo,rr-rw/2);rh=min(hi,rr+rw/2);pl=rng.uniform(0,2*np.pi);ph=pl+np.pi/512;interval=local.difference_jets(table['theta'],np.array([rl]),np.array([rh]),np.array([pl]),np.array([ph]),task_name);rp=np.r_[rl,rh,rng.uniform(rl,rh,3)];pp=np.r_[pl,ph,rng.uniform(pl,ph,3)];point=polar_jets(adapter,table['theta'],rp,pp);jet_passes+=2*len(rp)
                for point_id in range(5):
                    for channel in range(5):
                        lower=float(interval[channel][0][0]);upper=float(interval[channel][1][0]);value=float(point[point_id,channel]);passed=lower-1e-10<=value<=upper+1e-10;jet_violations+=not passed;all_jets.append(dict(cohort=cohort,seed=seed,box=box,point=point_id,channel=channel,value=value,lower=lower,upper=upper,enclosed=passed))
            scores=np.linspace(float(table['gain_profile'].min())-.1*float(table['width']),float(table['gain_profile'].max())+1.1*float(table['width']),257);exact=split_cdf(table,scores,cohort);cl,ch=kc.bracket(dist,scores);cdf_violation=float(max(0,np.max(cl-exact),np.max(exact-ch)));aa,mm=fast_cdf_setup(table,cohort);fast=adapter.rank.mixture_cdf(aa,mm,scores,float(table['width']))
            x=adapter.sample_inputs((1012000 if cohort=='rings' else 1052000)+seed,256);x=np.concatenate([x,adapter.coordinates(np.array([hi+.1,hi+.5]),np.array([.317,1.719]))]);xp=adapter.partner(x);h=adapter.predict(table['theta'],x);hp=adapter.predict(table['theta'],xp);calls+=2*len(x);q,nq=true_marginal(adapter,table,x,cohort);qp,nqp=true_marginal(adapter,table,xp,cohort);ql,qh=rootcert.marginal_bounds(x,table,dist,qt,adapter);qpl,qph=rootcert.marginal_bounds(xp,table,dist,qt,adapter);wl,wh=rootcert.centered_bounds(h,hp,ql,qh,qpl,qph);m=(h+hp)/2;g=lambda z:np.logaddexp(z/2,-z/2);W=.5*(q*(g(h)-g(m))+qp*(g(hp)-g(m)));q_violation=float(max(0,np.max(ql-q),np.max(q-qh),np.max(qpl-qp),np.max(qp-qph)));W_violation=float(max(0,np.max(wl-W),np.max(W-wh)))
            path=destination/f'{cohort}_{seed}_cdf_W.npz';np.savez_compressed(path,score=scores,F_exact=exact,F_fast=fast,F_lower=cl,F_upper=ch,x=x,q=q,q_partner=qp,q_lower=ql,q_upper=qh,W=W,W_lower=wl,W_upper=wh)
            summary=dict(cohort=cohort,seed=seed,radial_probes=len(radii),angular_nodes=NA,orbit_enclosure_failures=sum(not row['enclosed'] for row in moments),max_orbit_subsample_difference=max(abs(row['b']-row['b_subsample4096']) for row in moments),min_orbit_margin=min(min(row['b']-row['lower'],row['upper']-row['b']) for row in moments),max_partner_roll_error=max(row['partner_roll_error'] for row in moments),jet_probes=800,jet_violations=jet_violations,cdf_violation=cdf_violation,cdf_fast_vs_split_max=float(abs(fast-exact).max()),q_interval_violation=q_violation,W_interval_violation=W_violation,outside_W_nonnegative=bool(np.all(W[-2:]>=-1e-12)),model_forward_examples=calls,analytic_jet_propagations=jet_passes,independent_exact_CDF_queries=257+nq+nqp,seconds=time.perf_counter()-start);summaries.append(summary);(destination/f'{cohort}_{seed}_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    for name,rows in [('summary.csv',summaries),('jet_probes.csv',all_jets),('moment_probes.csv',all_moments)]:
        with (destination/name).open('x',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    (destination/'COMPLETE.json').write_text(json.dumps(dict(timestamp_utc=now(),models=6,model_forward_examples=sum(r['model_forward_examples'] for r in summaries),analytic_jet_propagations=sum(r['analytic_jet_propagations'] for r in summaries),all_failures_preserved=True,source_sha256=sha(Path(__file__))),indent=2))

if __name__=='__main__':
    with threadpool_limits(1):run()
