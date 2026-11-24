"""Frozen deterministic range bounds and same-law confidence calculations."""
import math
import numpy as np
BATCH=256;QUOTA=64;LOOKS=(512,2048);SHIFTS=(0.,.01,.1,.6)
EPS=(0.,.025,.05,.1,.2,.4,.8,1.)
MOMENTS=('A','W','B','R_known','R_unknown')
FAMILY=8*4*2*5*2*2+4
DELTA=.05/FAMILY
def lse(z):
    a=np.max(z,axis=-1);return a+np.log(np.exp(z-a[...,None]).sum(axis=-1))
def ce(z,p):return lse(z)-np.sum(p*z,axis=-1)
def gap(h,ht):return np.maximum(0.,.5*(np.logaddexp(0,h)+np.logaddexp(0,ht))-np.logaddexp(0,.5*(h+ht)))
def nll(h,p):return (1-p)*np.logaddexp(0,h)+p*np.logaddexp(0,-h)
def clip(grid):return float(min(.5,max(1e-6,np.quantile(grid['folded_jensen'],.95))))
def local(h,ht,u):
    d=(h-ht)/2;S=abs(d);K=.5*(np.logaddexp(0,h)-np.logaddexp(0,ht));t=K-d/2
    V=abs(u)*S/2;C=u*t;W=V-C
    assert np.min(W)>-1e-12 and np.max(abs(t)-S/2)<1e-12
    return gap(h,ht),S,W,V,t
def interval(v,item_range,side):
    n=len(v);z=math.log(2/DELTA);mean=float(v.mean())
    eb=math.sqrt(2*v.var(ddof=1)*z/n)+7*(item_range/4)*z/(3*(n-1))
    mcd=item_range*math.sqrt(math.log(1/DELTA)/(2*n*BATCH))
    value=max(0.,mean-eb,mean-mcd) if side=='lower' else min(item_range/4,mean+eb,mean+mcd)
    return dict(mean=mean,variance=float(v.var(ddof=1)),eb_radius=eb,mcd_radius=mcd,bound=value,
                item_range=item_range,batch_range=item_range/4,n_batches=n,delta=DELTA)
def parameter_interval(cosine):
    estimate=2*float(np.mean(cosine));radius=math.sqrt(8*math.log(2/DELTA)/len(cosine))
    return dict(estimate=estimate,radius=radius,lower=float(np.clip(estimate-radius,0,1)),
                upper=float(np.clip(estimate+radius,0,1)),n=len(cosine),delta=DELTA)
def sample(task,seed,n,a):
    rng=np.random.default_rng(seed);r=rng.uniform(.4,1.4,(n,2));accepted=[];count=0;proposals=0
    while count<n:
        size=max(1024,int((n-count)*(1+abs(a))*1.03)+8)
        phi=rng.uniform(0,2*np.pi,size);u=rng.random(size)
        keep=phi[u<=(1+a*np.cos(phi))/(1+abs(a))]
        accepted.append(keep);count+=len(keep);proposals+=size
    phi=np.concatenate(accepted)[:n]
    return task.coordinates(r[:,0],r[:,1],phi),dict(proposals=proposals,retained=n)
def posteriors(world,p0,d,u,eps):
    if world=='orbit_worst':
        # u is an orientation convention when a=0; u*d is even under T.
        mean=.5+.5*(1-eps)*np.sign(u*d);difference=eps*np.sign(d)
        return mean+difference/2,mean-difference/2
    direction=u if world=='cosine' else np.sign(d)
    amplitude=np.minimum(eps/2,np.minimum(p0,1-p0))
    return p0+amplitude*direction,p0-amplitude*direction
