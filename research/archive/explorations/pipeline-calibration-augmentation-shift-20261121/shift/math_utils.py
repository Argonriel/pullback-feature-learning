"""Loss identities and same-law finite-sample bounds, in nats."""
import math
import numpy as np
BATCH=256;QUOTA=64;LOOKS=(512,2048);SHIFTS=(0.,.01,.1,.6)
EPS=(0.,.025,.05,.1,.2,.4,.8,1.)
FAMILY=512;DELTA=.05/FAMILY
def lse(z):
    a=np.max(z,axis=-1);return a+np.log(np.exp(z-a[...,None]).sum(axis=-1))
def ce(z,p):return lse(z)-np.sum(p*z,axis=-1)
def multi_gap(z,zt):return .5*(lse(z)+lse(zt))-lse(.5*(z+zt))
def span(z):return np.max(z,axis=-1)-np.min(z,axis=-1)
def gap(h,ht):return np.maximum(0.,.5*(np.logaddexp(0,h)+np.logaddexp(0,ht))-np.logaddexp(0,.5*(h+ht)))
def nll(h,p):return (1-p)*np.logaddexp(0,h)+p*np.logaddexp(0,-h)
def clip(grid):return float(min(.5,max(1e-6,np.quantile(grid['folded_jensen'],.95))))
def bound(A,B,c,D):
    n=len(A);z=math.log(2/DELTA);s=math.sqrt(math.log(1/DELTA)/(2*n*BATCH))
    pa=math.sqrt(2*A.var(ddof=1)*z/n)+7*c/4*z/(3*(n-1))
    pb=math.sqrt(2*B.var(ddof=1)*z/n)+7*D/4*z/(3*(n-1))
    lae=float(A.mean()-pa);lam=float(A.mean()-c*s)
    ube=float(B.mean()+pb);ubm=float(B.mean()+D*s)
    return dict(n_batches=n,c=c,D=D,delta=DELTA,mean_A=float(A.mean()),mean_B=float(B.mean()),
        A_eb_lower=lae,A_mcd_lower=lam,B_eb_upper=ube,B_mcd_upper=ubm,
        A_lower=max(0.,lae,lam),B_upper=min(D/4,ube,ubm))
def sample(task,seed,n,a):
    """Exact rejection sampling for density (1+a cos(phi))/(2 pi)."""
    rng=np.random.default_rng(seed);r=rng.uniform(.4,1.4,(n,2));accepted=[];count=0;proposals=0
    while count<n:
        size=max(1024,int((n-count)*(1+abs(a))*1.03)+8)
        phi=rng.uniform(0,2*np.pi,size);u=rng.random(size)
        keep=phi[u<=(1+a*np.cos(phi))/(1+abs(a))]
        accepted.append(keep);count+=len(keep);proposals+=size
    phi=np.concatenate(accepted)[:n]
    return task.coordinates(r[:,0],r[:,1],phi),dict(proposals=proposals,retained=n)
