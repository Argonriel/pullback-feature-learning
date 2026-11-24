"""Local interval jets for the continuous conditional moment, in real arithmetic.

Floating-point arithmetic is padded and numerically checked, not directed-rounded.
"""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent.parent/'continuous-confirmation-20261121'
sys.path.insert(0,str(ROOT.parent/'source'))
from pullback_feature_learning import core

def add(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[1],-a[0]
def scale(a,s):return (a[0]*s,a[1]*s) if s>=0 else (a[1]*s,a[0]*s)
def mul(a,b):
    v=np.stack([a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]])
    return v.min(axis=0),v.max(axis=0)
def square(a):
    lo,hi=a;return np.where((lo<=0)&(hi>=0),0,np.minimum(lo*lo,hi*hi)),np.maximum(lo*lo,hi*hi)
def linear(a,w):
    pos=np.maximum(w,0);negative=np.minimum(w,0)
    return a[0]@pos.T+a[1]@negative.T,a[1]@pos.T+a[0]@negative.T
def pad(a):
    eps=2e-13*(1+np.maximum(abs(a[0]),abs(a[1])))
    return a[0]-eps,a[1]+eps
def trig(lo,hi,kind):
    if kind=='cos':lo=lo+np.pi/2;hi=hi+np.pi/2
    a=np.sin(lo);b=np.sin(hi);lower=np.minimum(a,b);upper=np.maximum(a,b)
    hasmax=np.ceil((lo-np.pi/2)/(2*np.pi))<=np.floor((hi-np.pi/2)/(2*np.pi))
    hasmin=np.ceil((lo+np.pi/2)/(2*np.pi))<=np.floor((hi+np.pi/2)/(2*np.pi))
    return np.where(hasmin,-1,lower),np.where(hasmax,1,upper)

def init_jets(rlo,rhi,plo,phi):
    c=trig(plo,phi,'cos');s=trig(plo,phi,'sin');unit=(np.column_stack([c[0],s[0]]),np.column_stack([c[1],s[1]]))
    r=(rlo[:,None],rhi[:,None]);value=mul(r,unit);zero=np.zeros_like(value[0]);rot=(np.column_stack([-s[1],c[0]]),np.column_stack([-s[0],c[1]]))
    return [value,unit,(zero,zero),mul(r,rot),neg(value)]

def tanh_ranges(z):
    lo=np.tanh(z[0]);hi=np.tanh(z[1]);t=(lo,hi)
    dmin=np.minimum(1-lo*lo,1-hi*hi);dmax=np.maximum(1-lo*lo,1-hi*hi);dmax=np.where((lo<=0)&(hi>=0),1,dmax)
    a=-2*lo*(1-lo*lo);b=-2*hi*(1-hi*hi);low=np.minimum(a,b);high=np.maximum(a,b);v=1/np.sqrt(3);ext=4/(3*np.sqrt(3))
    low=np.where((lo<=v)&(hi>=v),-ext,low);high=np.where((lo<=-v)&(hi>=-v),ext,high)
    return pad(t),pad((dmin,dmax)),pad((low,high))

def propagate(theta,jets):
    h=[pad(a) for a in jets]
    for A,B,bias in core.unpack(theta):
        z=[linear(a,A) for a in h];z[0]=(z[0][0]+bias,z[0][1]+bias)
        t,td,tdd=tanh_ranges(z[0]);nonlin=[t,mul(td,z[1]),add(mul(tdd,square(z[1])),mul(td,z[2])),mul(td,z[3]),add(mul(tdd,square(z[3])),mul(td,z[4]))]
        h=[pad(add(a,scale(linear(v,B),.5))) for a,v in zip(h,nonlin)]
    return [(a[:,0],b[:,0]) for a,b in h]

def trig_fields(c0,c1,derivative,plo,phi):
    # Only odd harmonics can contribute to E[u*d].
    k=np.tile(np.arange(1,9),2);odd=k%2==1;c0=c0*odd;c1=c1*odd;derivative=derivative*odd
    ci=(np.minimum(c0,c1),np.maximum(c0,c1));a=plo[:,None]*np.arange(1,9);z=phi[:,None]*np.arange(1,9)
    co=trig(a,z,'cos');si=trig(a,z,'sin')
    psi=(np.c_[co[0],si[0]],np.c_[co[1],si[1]])
    psip=(np.c_[-si[1]*np.arange(1,9),co[0]*np.arange(1,9)],np.c_[-si[0]*np.arange(1,9),co[1]*np.arange(1,9)])
    psipp=(-psi[1]*k*k,-psi[0]*k*k)
    def reduce(x):return x[0].sum(axis=1),x[1].sum(axis=1)
    return reduce(mul(ci,psi)),reduce(mul((derivative,derivative),psi)),reduce(mul(ci,psip)),reduce(mul(ci,psipp))

def interval_derivatives(theta,rlo,rhi,plo,phi,c0,c1,derivative):
    initial=init_jets(rlo,rhi,plo,phi);a=propagate(theta,initial);b=propagate(theta,[neg(v) for v in initial]);d=[scale(add(x,neg(y)),.5) for x,y in zip(a,b)]
    u,ur,up,upp=trig_fields(c0,c1,derivative,plo,phi)
    fpp=add(add(mul(upp,d[0]),scale(mul(up,d[3]),2)),mul(u,d[4]))
    frr=add(scale(mul(ur,d[1]),2),mul(u,d[2]))
    return fpp,frr

def exact_moments(theta,r,c,na):
    phi=(np.arange(na)+.5)*np.pi/na;unit=np.c_[np.cos(phi),np.sin(phi)];k=np.arange(1,9);psi=np.c_[np.cos(phi[:,None]*k),np.sin(phi[:,None]*k)];odd=np.tile(k%2==1,2)
    result=[]
    for j in range(0,len(r),16):
        x=(r[j:j+16,None,None]*unit).reshape(-1,2);a=core.representation(theta,x)[:,0].reshape(-1,na);b=core.representation(theta,-x)[:,0].reshape(-1,na);u=(c[j:j+16]*odd)@psi.T;result.extend((u*(a-b)/2).mean(axis=1)/8)
    return np.array(result)

def rice_cdf(r):
    """Positive Poisson-Gamma mixture truncated at K=160; exact-arithmetic tail bound."""
    r=np.asarray(r);x=r*r/(2*.15**2);poisson=np.exp(-x);cum=poisson.copy();out=np.zeros_like(r)
    for radius in [.5,1.2]:
        lam=radius*radius/(2*.15**2);weight=np.exp(-lam);acc=weight*(1-cum);pk=poisson.copy();ck=cum.copy()
        for k in range(1,161):
            pk=pk*x/k;ck=ck+pk;weight=weight*lam/k;acc=acc+weight*(1-ck)
        out+=.5*acc
    tail=max(np.exp(-lam+161*(1+np.log(lam/161))) for lam in [.5**2/(2*.15**2),1.2**2/(2*.15**2)])
    return np.clip(out,0,1),float(tail)

def construct(table,sub=4,na=256,chunk=4096):
    start=time.perf_counter();oldr=table['radii'];r=np.linspace(oldr[0],oldr[-1],(len(oldr)-1)*sub+1)
    c=np.column_stack([np.interp(r,oldr,table['balanced'][:,k]) for k in range(16)]);slope=np.diff(table['balanced'],axis=0)/np.diff(oldr)[:,None];parent=np.arange(len(r)-1)//sub
    midpoint=exact_moments(table['theta'],r,c,na);point_seconds=time.perf_counter()-start
    n=len(r)-1;fpp_lo=np.empty(n*na);fpp_hi=np.empty(n*na);frr_lo=np.empty(n*na);frr_hi=np.empty(n*na);h=np.pi/na
    start=time.perf_counter()
    for first in range(0,n*na,chunk):
        ids=np.arange(first,min(first+chunk,n*na));ri=ids//na;ai=ids%na
        pp,rr=interval_derivatives(table['theta'],r[ri],r[ri+1],ai*h,(ai+1)*h,c[ri],c[ri+1],slope[parent[ri]])
        fpp_lo[ids]=pp[0];fpp_hi[ids]=pp[1];frr_lo[ids]=rr[0];frr_hi[ids]=rr[1]
    angular=np.maximum(abs(fpp_lo),abs(fpp_hi)).reshape(n,na).mean(axis=1)*h*h/(24*8)
    brr_lo=frr_lo.reshape(n,na).mean(axis=1)/8;brr_hi=frr_hi.reshape(n,na).mean(axis=1)/8
    width=np.diff(r);radial=np.maximum(abs(brr_lo),abs(brr_hi))*width*width/8;endpoint=np.maximum(abs(midpoint[:-1]),abs(midpoint[1:]));bound=endpoint+angular+radial
    # A whole radial band with exactly even coefficients has b=0 analytically.
    odd=np.tile(np.arange(1,9)%2==1,2);even=np.all(c[:-1,odd]==0,axis=1)&np.all(c[1:,odd]==0,axis=1)
    bound[even]=0;angular[even]=0;radial[even]=0;endpoint[even]=0;brr_lo[even]=0;brr_hi[even]=0
    cdf,tail=rice_cdf(r);mass=np.diff(cdf);assert mass.min()>-1e-14;mass=np.maximum(mass,0)
    return dict(r=r,coefficient=c,moment=midpoint,angular=angular,radial=radial,endpoint=endpoint,bound=bound,brr_lo=brr_lo,brr_hi=brr_hi,mass=mass,exact_even_band=even,
        sub=np.array(sub),na=np.array(na),interval_jet_propagations=np.array(2*n*na),exact_point_forwards=np.array(2*len(r)*na),point_seconds=np.array(point_seconds),interval_seconds=np.array(time.perf_counter()-start),poisson_cdf_tail_bound=np.array(tail),uniform_bound=np.array(bound.max()),integrated_bound=np.array(mass@bound+tail*bound.max()),angular_integral=np.array(mass@angular+tail*angular.max()),radial_integral=np.array(mass@radial+tail*radial.max()),endpoint_integral=np.array(mass@endpoint+tail*endpoint.max()))

def pointwise_bound(table,r):
    r=np.asarray(r);grid=table['r'];idx=np.clip(np.searchsorted(grid,r,side='right')-1,0,len(grid)-2);width=grid[idx+1]-grid[idx];t=np.clip((r-grid[idx])/width,0,1);qlin=(1-t)*table['moment'][idx]+t*table['moment'][idx+1]
    lower=qlin-table['angular'][idx]-.5*table['brr_hi'][idx]*width*width*t*(1-t)
    upper=qlin+table['angular'][idx]-.5*table['brr_lo'][idx]*width*width*t*(1-t)
    even=table['exact_even_band'][idx];outside=(r<grid[0])|(r>grid[-1]);lower=np.where(even|outside,0,lower);upper=np.where(even|outside,0,upper)
    return lower,upper
