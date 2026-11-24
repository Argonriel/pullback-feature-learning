"""Bounded continuous fields and one-of-four exact-quota deployment."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;STUDY=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(STUDY/'source'));sys.path.insert(0,str(STUDY/'legacy'/'budgeted-rotation-final'))
from pullback_feature_learning import core
from gate import Gate,allocate
K=8;NR=129;NA=512;RMAX=3.;FIELDS=('balanced','unbalanced','even')

def basis(phi):
    return np.concatenate([np.cos(np.asarray(phi)[...,None]*np.arange(1,K+1)),np.sin(np.asarray(phi)[...,None]*np.arange(1,K+1))],axis=-1)
def pred(t,x):return core.representation(t,x)[:,0]
def orbit(t,r,na,phase=0.):
    angles=2*np.pi*np.arange(na)/na+phase;unit=np.column_stack([np.cos(angles),np.sin(angles)]);h=np.empty((len(r),na))
    for j in range(0,len(r),32):h[j:j+32]=pred(t,(r[j:j+32,None,None]*unit[None,:,:]).reshape(-1,2)).reshape(-1,na)
    return h,angles

def lp(a,b):
    # Atomic l1-ball program: one zero-moment atom or two opposite-moment atoms.
    signed_a=np.r_[a,-a];signed_b=np.r_[b,-b];n=len(a);best=0.;out=np.zeros(n)
    def candidate(i,w,j=None):
        c=np.zeros(n);c[i%n]+=w*(1 if i<n else -1)
        if j is not None:c[j%n]+=(1-w)*(1 if j<n else -1)
        return c
    for i in np.flatnonzero(abs(signed_b)<1e-14):
        if signed_a[i]>best:best=signed_a[i];out=candidate(i,1.)
    for i in np.flatnonzero(signed_b>1e-14):
        for j in np.flatnonzero(signed_b < -1e-14):
            w=-signed_b[j]/(signed_b[i]-signed_b[j]);value=w*signed_a[i]+(1-w)*signed_a[j]
            if value>best:best=value;out=candidate(i,w,j)
    return out

def constants(theta):
    L=1.;H=0.;C=0.
    for A,B,b in core.unpack(theta):
        an=np.linalg.norm(A,2);bn=np.linalg.norm(B,2);ell=1+.5*bn*an;hh=.5*bn*(4/(3*np.sqrt(3)))*an**2
        H=hh*L**2+ell*H;L=ell*L;C+=.5*np.abs(B[0]).sum()
    return float(L),float(H),float(C)

def fit(theta):
    radii=np.linspace(0,RMAX,NR);h,angles=orbit(theta,radii,NA);partner=np.roll(h,NA//2,axis=1);d=(h-partner)/2;A=np.logaddexp(0,h)-np.logaddexp(0,(h+partner)/2);psi=basis(angles)
    ac=A@psi/NA;bc=d@psi/NA;even=np.array([k%2==0 for k in list(range(1,K+1))*2]);bc[:,even]=0.
    coefficients={}
    for field in FIELDS:
        c=np.zeros((NR,2*K))
        for j in range(1,NR-1):
            if field=='balanced':c[j]=lp(ac[j],bc[j])
            else:
                eligible=np.flatnonzero(even) if field=='even' else np.arange(2*K);k=eligible[np.argmax(abs(ac[j,eligible]))];c[j,k]=np.sign(ac[j,k])
        coefficients[field]=c
        assert np.max(np.abs(c).sum(axis=1))<=1+1e-14
    L,H,C=constants(theta);k=np.tile(np.arange(1,K+1),2);c=coefficients['balanced'];gamma=abs(c).sum(1);kap=abs(c)@k;kap2=abs(c)@(k*k);u=c@psi.T
    residual=abs((u*d).mean(1));D=radii+C
    angular1=np.pi/(2*NA)*(kap*D+gamma*L*radii)
    angular2=(2*np.pi/NA)**2/24*(kap2*D+(2*kap+gamma)*radii*L+gamma*radii*radii*H)
    # Both are valid analytic quadrature bounds; retain each and their minimum.
    return dict(radii=radii,theta=theta,h=h,ac=ac,bc=bc,balanced=coefficients['balanced'],unbalanced=coefficients['unbalanced'],even=coefficients['even'],L=np.array(L),H=np.array(H),C_h=np.array(C),gamma=gamma,anchor_residual=residual,angular_bound1=angular1,angular_bound2=angular2,angular_bound=np.minimum(angular1,angular2))

def coefficients_at(table,r,field):
    return np.stack([np.interp(r,table['radii'],table[field][:,k],left=0,right=0) for k in range(2*K)],axis=-1)
def field_value(table,x,field):
    r=np.linalg.norm(x,axis=1);phi=np.arctan2(x[:,1],x[:,0]);return np.sum(coefficients_at(table,r,field)*basis(phi),axis=1)
def bound_at(table,r):
    r=np.asarray(r);idx=np.clip(np.searchsorted(table['radii'],r,side='right')-1,0,NR-2);lo=table['radii'][idx];width=table['radii'][idx+1]-lo;t=np.clip((r-lo)/width,0,1)
    E=table['anchor_residual']+table['angular_bound'];ang=((1-t)*E[idx]+t*E[idx+1])/8
    radial=float(table['L'])*t*(1-t)*width*(table['gamma'][idx]+table['gamma'][idx+1])/8
    outside=(r<0)|(r>RMAX);ang[outside]=0;radial[outside]=0
    return ang,radial

def selection(x,first,table,gate,name,ties,group_uniform):
    n=len(x);order=np.argsort(ties.reshape(-1,256),axis=1);offset=np.arange(n//256)[:,None]*256;groups=(order+offset).reshape(-1,4)
    mask=np.zeros(n,bool)
    if name.startswith('group_'):
        field=name[6:]
        if field=='confidence':chosen=np.argmin(abs(first[groups]),axis=1)
        else:
            u=np.zeros(n) if field=='uniform' else field_value(table,x,field);ug=u[groups];prob=.25+(ug-ug.mean(axis=1,keepdims=True))/6
            assert prob.min()>-1e-14 and prob.max()<.5+1e-14 and np.max(abs(prob.sum(axis=1)-1))<1e-13
            cum=np.cumsum(prob,axis=1);cum[:,-1]=1.;chosen=np.sum(group_uniform[:,None]>=cum,axis=1)
        mask[groups[np.arange(len(groups)),chosen]]=True
    elif name in ('confidence','radius','folded'):
        score=-abs(first) if name=='confidence' else gate.priority(x,name)
        mask=allocate(score.reshape(-1,256),ties.reshape(-1,256),64).ravel()
    elif name=='full':mask[:]=True
    elif name!='raw':raise ValueError(name)
    if name not in ('raw','full'):assert np.all(mask.reshape(-1,256).sum(axis=1)==64)
    return mask

def deploy(x,predict,table,gate,name,ties,group_uniform):
    h=predict(x);selected=selection(x,h,table,gate,name,ties,group_uniform);out=h.copy()
    if selected.any():out[selected]=.5*(h[selected]+predict(-x[selected]))
    return out,selected
