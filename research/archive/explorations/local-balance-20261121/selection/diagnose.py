"""Frozen, label-free diagnostics of the actual global-rank inclusion marginal."""
from pathlib import Path
import sys,json,csv,hashlib,datetime,math,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import policy as p
OUT=ROOT/'diagnostics';NQ=262145

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def density(r):
    return .5*sum(r/.15**2*np.exp(-(r*r+mu*mu)/(2*.15**2))*np.i0(r*mu/.15**2) for mu in (.5,1.2))
def radial_nodes(edges,order):
    x,w=np.polynomial.legendre.leggauss(order);r=((edges[:-1,None]+edges[1:,None])/2+np.diff(edges)[:,None]*x/2).ravel();weights=(np.diff(edges)[:,None]*w/2).ravel()*density(r);return r,weights
def cdf_rule(table,order):
    r,w=radial_nodes(table['radii'],order);a=np.interp(r,table['radii'],table['gain_profile']);idx=np.argsort(a);return a[idx],w[idx]
def q_interpolator():
    u=np.linspace(0,1,NQ);q=p.rank_probability(u);lb=math.lgamma(256)-math.lgamma(192)-math.lgamma(64)
    bound2=191*np.exp(lb+190*np.log(190/253)+63*np.log(63/253))+63*np.exp(lb+191*np.log(191/253)+62*np.log(62/253))
    return u,q,float(bound2/(8*(NQ-1)**2))
def action_mix(table,r,name):
    edges=table['radii'];lo=int(np.clip(np.searchsorted(edges,r,side='right')-1,0,p.NR-2));t=float(np.clip((r-edges[lo])/(edges[lo+1]-edges[lo]),0,1));out=[]
    for anchor,w in ((lo,1-t),(lo+1,t)):
        mix=float(table[name+'_mix'][anchor])
        for component,v in ((0,mix),(1,1-mix)):
            if w*v>0:out.append((int(table[name+'_frequency'][anchor,component]),float(table[name+'_phase'][anchor,component]),w*v))
    return out
def evaluate(table,radial_order,na,offset,q_u,q_values,keep_h=False):
    r,weights=radial_nodes(table['radii'],radial_order);phi=2*np.pi*(np.arange(na)+offset)/na;unit=np.c_[np.cos(phi),np.sin(phi)];h=np.empty((len(r),na));calls=0
    for start in range(0,len(r),32):
        points=(r[start:start+32,None,None]*unit).reshape(-1,2);h[start:start+32]=p.core.representation(table['theta'],points)[:,0].reshape(-1,na);calls+=len(points)
    partner=np.roll(h,na//2,axis=1);d=(h-partner)/2;m=(h+partner)/2;A=np.logaddexp(0,h)-np.logaddexp(0,m);center=np.logaddexp(h/2,-h/2)-np.logaddexp(m/2,-m/2)
    moments=np.zeros((len(r),len(p.POLICIES),6));cal_a,cal_w=cdf_rule(table,32);j=np.arange(na);width=float(table['width']);profile=np.interp(r,table['radii'],table['gain_profile']);spot_error=0.
    for row in range(len(r)):
        base={}
        for k in p.FREQUENCIES:
            frac=k*offset-np.floor(k*offset);score=profile[row]+width*(j+frac)/na;F=p.mixture_cdf(cal_a,cal_w,score,width);base[k]=np.interp(F,q_u,q_values)
        for column,name in enumerate(p.POLICIES):
            q=np.zeros(na)
            for k,phase,weight in action_mix(table,r[row],name):
                phase_index=int(round(phase*na));assert abs(phase*na-phase_index)<1e-9
                index=(k*j+int(np.floor(k*offset))+phase_index)%na;q+=weight*base[k][index]
            moments[row,column]=[q@A[row]/na,q@d[row]/na,q@center[row]/na,q.mean(),q.min(),q.max()]
            if row in (0,len(r)//2,len(r)-1):
                ids=np.array([0,17,153,na-1]);x=r[row]*unit[ids];truth=p.marginal(x,table,name,true_cdf=True,cdf_order=32);spot_error=max(spot_error,float(abs(truth-q[ids]).max()))
    result=dict(radii=r,radial_weights=weights,moments=moments,columns=np.array(['C','b','centered','rate','q_min','q_max']),policies=np.array(p.POLICIES),radial_order=np.array(radial_order),angular_nodes=np.array(na),angular_offset=np.array(offset),model_forward_examples=np.array(calls),spot_exact_marginal_error=np.array(spot_error))
    if keep_h:result['h']=h
    for name in p.POLICIES:result[name+'_bound']=p.conditional_bound(table,r,name,alpha=.05/16)['total']
    return result
def summarise(value,seed,pass_name):
    w=value['radial_weights'];out=[]
    for col,name in enumerate(p.POLICIES):
        C,b,center,rate,qmin,qmax=value['moments'][:,col,:].T;bb=np.zeros_like(b) if name=='rank_even' else b
        out.append(dict(seed=seed,pass_name=pass_name,policy=name,C=float(w@C),centered_gain=float(w@center),b_mean=float(w@b),max_abs_b=float(abs(b).max()),numerical_width=float(w@abs(bb)),numerical_low=float(w@(C-np.maximum(bb,0))),numerical_high=float(w@(C-np.minimum(bb,0))),mean_rate=float(w@rate),min_rate=float(rate.min()),max_rate=float(rate.max()),min_q=float(qmin.min()),max_q=float(qmax.max()),formal_width_bound=float(w@value[name+'_bound']),radial_mass=float(w.sum()),spot_marginal_error=float(value['spot_exact_marginal_error'])))
    return out
def run():
    OUT.mkdir(exist_ok=False);freeze=json.loads((ROOT/'FREEZE.json').read_text());assert sha(ROOT/'policy.py')==freeze['source_sha256'];spec=json.loads((ROOT/'DIAGNOSTIC_FREEZE.json').read_text());qu,qv,qerr=q_interpolator();np.savez_compressed(OUT/'rank_probability.npz',u=qu,q=qv,error_bound=qerr)
    source_commit=dict(timestamp_utc=now(),source_sha256=sha(Path(__file__)),spec_sha256=sha(ROOT/'DIAGNOSTIC_FREEZE.json'),policy_source_sha256=sha(ROOT/'policy.py'),new_policy_files_accessed=False,new_test_outcomes_accessed=False,main_score_may_have_started=True,note='Fixed diagnostic specification preceded any table/outcome access by this investigator; source committed before table reads. Main outcome scoring had begun eight seconds before the diagnostic-spec timestamp, without revealing results to this investigator.')
    (OUT/'SOURCE_COMMIT.json').write_text(json.dumps(source_commit,indent=2));rows=[];checks=[]
    passes=[('main',4,4096,.173),('radial',2,4096,.173),('angular',4,2048,.173),('phase',2,4096,.619)]
    for seed in range(1,17):
        start=time.perf_counter();path=ROOT.parent/'validation'/'policies'/f'{seed}_new.npz';table=dict(np.load(path));modeldir=OUT/str(seed);modeldir.mkdir();calls=0
        for name,nr,na,phase in passes:
            value=evaluate(table,nr,na,phase,qu,qv,keep_h=name=='main');assert float(value['spot_exact_marginal_error'])<qerr+1e-11
            np.savez_compressed(modeldir/f'{name}.npz',**value);rows+=summarise(value,seed,name);calls+=int(value['model_forward_examples'])
        grid=np.linspace(float(table['gain_profile'].min()),float(table['gain_profile'].max())+float(table['width']),10001);a32,w32=cdf_rule(table,32);a64,w64=cdf_rule(table,64);F32=p.mixture_cdf(a32,w32,grid,float(table['width']));F64=p.mixture_cdf(a64,w64,grid,float(table['width']));emp=p.score_cdf(table,grid)
        np.savez_compressed(modeldir/'cdf_refinement.npz',score=grid,F32=F32,F64=F64,Fempirical=emp)
        check=dict(seed=seed,policy_sha256=sha(path),cdf32_64_max_difference=float(abs(F32-F64).max()),empirical_true64_max_difference=float(abs(emp-F64).max()),rank_q_interpolation_error_bound=qerr,model_forward_examples=calls,seconds=time.perf_counter()-start,DKW_family16_epsilon=float(np.sqrt(np.log(2/(.05/16))/(2*p.CDF_N))))
        checks.append(check);(modeldir/'provenance.json').write_text(json.dumps(check,indent=2));print(f'diagnostics {seed}/16 {check["seconds"]:.1f}s',flush=True)
    for filename,data in [('summary.csv',rows),('checks.csv',checks)]:
        with (OUT/filename).open('x',newline='') as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(timestamp_utc=now(),models=16,passes=4,rows=len(rows),forward_examples=sum(r['model_forward_examples'] for r in checks),source_sha256=sha(Path(__file__)),summary_sha256=sha(OUT/'summary.csv')),indent=2))

if __name__=='__main__':
    with threadpool_limits(1):run()
