"""Transfer adapter of the frozen global exact-quota rank family; only task geometry and radial law change."""
from pathlib import Path
import sys,time,math,json
import numpy as np
ROOT=Path(__file__).resolve().parent;STUDY=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
import task as core
NR=129;NA=512;RMIN=.4;RMAX=1.4;CDF_N=131072;FREQUENCIES=(1,2,3,4)
POLICIES=('rank_balanced','rank_even','rank_unbalanced')
J=np.arange(64);LOG_BINOM=np.array([math.lgamma(256)-math.lgamma(int(j)+1)-math.lgamma(256-int(j)) for j in J])
MODE=191/254
Q_LIPSCHITZ=float(np.exp(191*np.log(MODE)+63*np.log1p(-MODE)+math.lgamma(256)-math.lgamma(192)-math.lgamma(64)))

def rank_probability(u):
    """Exact binomial polynomial for inclusion in top64 of256 (stable log-sum)."""
    shape=np.shape(u);u=np.asarray(u,dtype=float).ravel();out=np.empty_like(u)
    for start in range(0,len(u),4096):
        x=np.clip(u[start:start+4096],1e-300,1-np.finfo(float).eps);lu=np.log(x);lv=np.log1p(-x)
        terms=LOG_BINOM[:,None]+(255-J)[:,None]*lu[None,:]+J[:,None]*lv[None,:];mx=terms.max(axis=0);out[start:start+len(x)]=np.exp(mx)*np.exp(terms-mx).sum(axis=0)
    out[u<=0]=0.;out[u>=1]=1.;return np.clip(out,0,1).reshape(shape)

def sample_radii(seed,n):
    return core.sample_radii(seed,n)

def orbit(theta,radii):
    phi=2*np.pi*(np.arange(NA)+.5)/NA;h=np.empty((len(radii),NA))
    for j in range(0,len(radii),32):
        x=core.coordinates(radii[j:j+32,None],phi[None,:]).reshape(-1,4);h[j:j+32]=core.predict(theta,x).reshape(-1,NA)
    return h

def prepare(theta,seed):
    start=time.perf_counter();r=np.linspace(RMIN,RMAX,NR);h=orbit(theta,r);other=np.roll(h,NA//2,axis=1);m=(h+other)/2;soft=np.logaddexp(0,h);gain=(.5*(soft+np.roll(soft,NA//2,axis=1))-np.logaddexp(0,m)).mean(axis=1);gain=np.maximum(gain,0)
    calr=sample_radii(seed,CDF_N);a=np.interp(calr,r,gain);sorted_a=np.sort(a)
    return dict(theta=np.asarray(theta),radii=r,h=h,gain_profile=gain,cdf_cal_radii=calr,cdf_a=sorted_a,width_scale=np.array(max(a.std(),1e-8)),cal_seed=np.array(seed),forward_examples=np.array(NR*NA),coordinate_examples=np.array(CDF_N),prepare_seconds=np.array(time.perf_counter()-start))

def mixture_cdf(a,weights,score,width):
    shape=np.shape(score);score=np.asarray(score).ravel();mass=np.r_[0.,np.cumsum(weights)];weighted=np.r_[0.,np.cumsum(weights*a)]
    lo=np.searchsorted(a,score-width,side='right');hi=np.searchsorted(a,score,side='right');value=mass[lo]+((hi-lo)*0.)
    value=mass[lo]+(score*(mass[hi]-mass[lo])-(weighted[hi]-weighted[lo]))/width
    return np.clip(value,0,1).reshape(shape)
def score_cdf(table,score):
    a=table['cdf_a'];return mixture_cdf(a,np.full(len(a),1/len(a)),score,float(table['width']))
def population_cdf(table,score,order=None):
    """Exact polynomial-integral formula for the known transfer input law.
    order is unused and accepted solely for the common evaluator API.
    """
    shape=np.shape(score);z=np.asarray(score,dtype=float).ravel();out=np.zeros_like(z);edges=table['radii'];a=table['gain_profile'];width=float(table['width'])
    for j in range(len(edges)-1):
        lo,hi=edges[j:j+2];slope=(a[j+1]-a[j])/(hi-lo);intercept=a[j]-slope*lo;mass=core.population_cdf_r(hi)-core.population_cdf_r(lo)
        if slope==0:
            out+=mass*np.clip((z-a[j])/width,0,1);continue
        rz=np.clip((z-intercept)/slope,lo,hi);rzw=np.clip((z-width-intercept)/slope,lo,hi)
        if slope>0:
            lower,upper=rzw,rz;full=core.population_cdf_r(lower)-core.population_cdf_r(lo)
        else:
            lower,upper=rz,rzw;full=core.population_cdf_r(hi)-core.population_cdf_r(upper)
        delta0=core.population_cdf_r(upper)-core.population_cdf_r(lower);delta1=core.population_first_moment_r(upper)-core.population_first_moment_r(lower)
        out+=full+((z-intercept)*delta0-slope*delta1)/width
    return np.clip(out,0,1).reshape(shape)

def mixture_at_zero(b,C):
    """Upper concave hull of scalar moment/objective actions at moment zero."""
    order=np.lexsort((-C,b));keep=[]
    for idx in order:
        if keep and b[idx]==b[keep[-1]]:continue
        keep.append(int(idx))
    hull=[]
    for idx in keep:
        while len(hull)>=2:
            i,j=hull[-2:]
            left=(C[j]-C[i])*(b[idx]-b[j]);right=(C[idx]-C[j])*(b[j]-b[i])
            if left<=right:hull.pop()
            else:break
        hull.append(idx)
    zero=[i for i in hull if b[i]==0]
    if zero:
        i=max(zero,key=lambda k:C[k]);return i,i,1.,C[i]
    for i,j in zip(hull[:-1],hull[1:]):
        if b[i]<0<b[j]:
            weight=b[j]/(b[j]-b[i]);return i,j,float(weight),float(weight*C[i]+(1-weight)*C[j])
    raise ArithmeticError('No feasible zero-moment hull segment')

def calibrate(theta,seed,tau=None,prepared=None):
    if tau is None:tau=1.
    if not np.isfinite(tau) or tau<=0:raise ValueError('tau must be positive and finite')
    base=prepare(theta,seed) if prepared is None else prepared;table={k:v for k,v in base.items()};width=float(tau)*float(base['width_scale']);table['width']=np.array(width);table['tau']=np.array(tau);start=time.perf_counter()
    h=base['h'];partner=np.roll(h,NA//2,axis=1);d=(h-partner)/2;A=np.logaddexp(0,h)-np.logaddexp(0,(h+partner)/2)
    actions_k=np.repeat(np.array(FREQUENCIES),NA);actions_phase=np.tile((np.arange(NA)+.25)/NA,len(FREQUENCIES));even=actions_k%2==0
    C=np.empty((NR,len(actions_k)));b=np.empty_like(C);j=np.arange(NA);rates=np.empty(NR)
    for row in range(NR):
        odd_w=rank_probability(score_cdf(table,base['gain_profile'][row]+width*(j+.75)/NA));even_w=rank_probability(score_cdf(table,base['gain_profile'][row]+width*(j+.25)/NA))
        rates[row]=odd_w.mean()
        for ki,k in enumerate(FREQUENCIES):
            index=(k*j+k//2)%NA;aa=np.bincount(index,weights=A[row],minlength=NA)/NA;bb=np.bincount(index,weights=d[row],minlength=NA)/NA;w=odd_w if k%2 else even_w
            C[row,ki*NA:(ki+1)*NA]=np.fft.ifft(np.fft.fft(w)*np.conj(np.fft.fft(aa))).real
            b[row,ki*NA:(ki+1)*NA]=0. if k%2==0 else np.fft.ifft(np.fft.fft(w)*np.conj(np.fft.fft(bb))).real
    for name in POLICIES:
        freq=np.empty((NR,2),int);phase=np.empty((NR,2));mix=np.ones(NR);obj=np.empty(NR);res=np.empty(NR)
        for row in range(NR):
            if name=='rank_balanced':i,z,lam,value=mixture_at_zero(b[row],C[row])
            else:
                choices=np.flatnonzero(even) if name=='rank_even' else np.arange(len(actions_k));i=int(choices[np.argmax(C[row,choices])]);z=i;lam=1.;value=C[row,i]
            freq[row]=actions_k[[i,z]];phase[row]=actions_phase[[i,z]];mix[row]=lam;obj[row]=value;res[row]=lam*b[row,i]+(1-lam)*b[row,z]
        table[name+'_frequency']=freq;table[name+'_phase']=phase;table[name+'_mix']=mix;table[name+'_objective']=obj;table[name+'_moment']=res
    table['rate_profile']=rates;table['action_objectives']=C;table['action_moments']=b;table['fit_seconds']=np.array(time.perf_counter()-start);table['cdf_dkw_epsilon']=np.array(np.sqrt(np.log(40)/(2*CDF_N)));table['rank_probability_lipschitz']=np.array(Q_LIPSCHITZ)
    assert np.max(abs(table['rank_balanced_moment']))<1e-10
    assert np.all(table['rank_even_frequency']%2==0)
    return table

def action_parameters(x,table,name,action_keys):
    r=core.quotient(x);edges=table['radii'];lo=np.clip(np.searchsorted(edges,r,side='right')-1,0,NR-2);t=np.clip((r-edges[lo])/(edges[lo+1]-edges[lo]),0,1);mix=table[name+'_mix'];p0=(1-t)*mix[lo];p1=(1-t)*(1-mix[lo]);p2=t*mix[lo+1]
    cut=np.stack([p0,p0+p1,p0+p1+p2],axis=1);which=np.sum(action_keys[:,None]>=cut,axis=1);anchor=lo+(which>=2);component=which%2;k=table[name+'_frequency'][anchor,component];phase=table[name+'_phase'][anchor,component]
    outside=(r>RMAX)|(r<RMIN);k=np.where(outside,2,k);phase=np.where(outside,0.,phase);return r,k,phase

def action_angle(x,k):
    first=np.asarray(x)[:,:2];phi=np.arctan2(first[:,1],first[:,0]);flip=(first[:,1]<0)|((first[:,1]==0)&(first[:,0]<0));canonical=np.where(flip[:,None],-first,first);folded=np.arctan2(canonical[:,1],canonical[:,0]);return np.where(k%2==0,folded,phi)

def scores(x,table,name,action_keys):
    r,k,phase=action_parameters(x,table,name,action_keys);phi=action_angle(x,k);u=(k*phi/(2*np.pi)+phase)%1.;a=np.interp(r,table['radii'],table['gain_profile']);return a+float(table['width'])*u

def select(x,first,table,name,ties,action_keys):
    if name not in POLICIES:raise ValueError(name)
    score=scores(x,table,name,np.asarray(action_keys));order=np.lexsort((np.asarray(ties).reshape(-1,256),-score.reshape(-1,256)),axis=1);mask=np.zeros_like(score.reshape(-1,256),dtype=bool);np.put_along_axis(mask,order[:,:64],True,axis=1);assert np.all(mask.sum(1)==64);return mask.ravel()

def marginal(x,table,name,true_cdf=False,cdf_order=16):
    """Offline integration over action keys of the actual rank inclusion marginal."""
    r=core.quotient(x);phi=np.arctan2(x[:,1],x[:,0]);edges=table['radii'];lo=np.clip(np.searchsorted(edges,r,side='right')-1,0,NR-2);t=np.clip((r-edges[lo])/(edges[lo+1]-edges[lo]),0,1);a=np.interp(r,edges,table['gain_profile']);answer=np.zeros(len(x))
    for side,anchorweight in ((0,1-t),(1,t)):
        anchor=lo+side;mix=table[name+'_mix'][anchor]
        for component,weight in ((0,mix),(1,1-mix)):
            k=table[name+'_frequency'][anchor,component];phase=table[name+'_phase'][anchor,component];outside=(r>RMAX)|(r<RMIN);k=np.where(outside,2,k);phase=np.where(outside,0.,phase);angle=action_angle(x,k);score=a+float(table['width'])*((k*angle/(2*np.pi)+phase)%1.)
            F=population_cdf(table,score,cdf_order) if true_cdf else score_cdf(table,score);answer+=anchorweight*weight*rank_probability(F)
    return answer


def network_constants(theta):
    """Global |h(x)-x_0|<=C_h and spatial Lipschitz L_h for the 4D residual net."""
    L=1.;C=0.
    for A,B,b in core.unpack(theta):
        L*=1+.5*np.linalg.norm(B,2)*np.linalg.norm(A,2)
        C+=.5*np.abs(B[0]).sum()
    return float(L),float(C)

def conditional_bound(table,r,name,alpha=.05):
    """Uniform-in-radius high-probability bound for actual top64 conditional moment.
    Empirical-CDF DKW + bounded-variation angular quadrature + radial variation.
    Even actions are handled structurally, with an exact zero bound.
    """
    r=np.asarray(r,dtype=float);edges=table['radii'];L,C=network_constants(table['theta']);D=np.full_like(edges,min(1+C,L))
    epsilon=np.sqrt(np.log(2/alpha)/(2*len(table['cdf_a'])));cdf_error=min(1.,Q_LIPSCHITZ*epsilon)
    low=rank_probability(score_cdf(table,table['gain_profile']));high=rank_probability(score_cdf(table,table['gain_profile']+float(table['width'])))
    freq=table[name+'_frequency'];mix=table[name+'_mix'];weights=np.stack([mix,1-mix],axis=1);odd=(freq%2==1);gamma=np.sum(weights*odd,axis=1)
    # Conservative BV/M rule, valid at jumps; no smooth midpoint assertion.
    angular=np.sum(weights*odd*(2*freq*(high-low)[:,None]*D[:,None]+2*np.pi*np.sqrt(1+4*edges[:,None]**2)*L*high[:,None]),axis=1)/NA
    cdf=gamma*cdf_error*D;residual=abs(table[name+'_moment']);E=angular+cdf+residual
    lo=np.clip(np.searchsorted(edges,r,side='right')-1,0,NR-2);delta=edges[lo+1]-edges[lo];t=np.clip((r-edges[lo])/delta,0,1);da=abs(table['gain_profile'][lo+1]-table['gain_profile'][lo])
    node=(1-t)*E[lo]+t*E[lo+1]
    radial=t*(1-t)*(L*delta*(gamma[lo]+gamma[lo+1])+Q_LIPSCHITZ*da*(gamma[lo]*D[lo]+gamma[lo+1]*D[lo+1])/float(table['width']))
    outside=(r>RMAX)|(r<RMIN);node=np.where(outside,0.,node);radial=np.where(outside,0.,radial)
    return dict(total=np.where(outside,0.,np.minimum(min(1+C,L),node+radial)),anchor=node,radial=radial,alpha=float(alpha),cdf_epsilon=float(epsilon),L_h=L,C_h=C)
