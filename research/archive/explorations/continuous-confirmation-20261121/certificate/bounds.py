"""Deterministic interval enclosures and tight rectangle minimization."""
import numpy as np

def sigmoid(x):return np.exp(-np.logaddexp(0.,-x))
def softplus(x):return np.logaddexp(0.,x)
def jensen(a,b):return .5*(softplus(a)+softplus(b))-softplus(.5*(a+b))
def objective(a,b,epsilon):return jensen(a,b)-epsilon*np.abs(a-b)/4

def independent_bound(al,au,bl,bu,epsilon):
    jmin=np.where(al>bu,jensen(al,bu),np.where(bl>au,jensen(au,bl),0.))
    return np.maximum(jmin,0)-epsilon*np.maximum(abs(al-bu),abs(au-bl))/4

def joint_bound(al,au,bl,bu,epsilon):
    """All rectangle corners, diagonal, and edge cubic stationary candidates.

    Mathematical minimization is exact. A conservative floating-point margin
    and explicit residual-triggered fallback protect numerical evaluation;
    this is not a formally verified directed-rounding implementation.
    """
    al,au,bl,bu=np.broadcast_arrays(al,au,bl,bu);shape=al.shape
    al,au,bl,bu=[np.asarray(x,float).ravel() for x in (al,au,bl,bu)]
    baseline=independent_bound(al,au,bl,bu,epsilon)
    if epsilon==0:return baseline.reshape(shape),dict(cubic_solves=0,fallback_cells=0,max_stationary_residual=0.)
    if not 0<epsilon<1:raise ValueError('This implementation requires 0 < epsilon < 1, or epsilon=0')
    n=len(al);value=np.minimum.reduce([objective(al,bl,epsilon),objective(al,bu,epsilon),objective(au,bl,epsilon),objective(au,bu,epsilon)])
    overlap=np.maximum(al,bl)<=np.minimum(au,bu);value=np.where(overlap,np.minimum(value,0),value)
    # Fix each of the four edges; a and b are interchangeable.
    fixed=np.concatenate([al,au,bl,bu]);lo=np.concatenate([bl,bl,al,al]);hi=np.concatenate([bu,bu,au,au]);c=np.exp(fixed/2)
    failed=np.zeros(4*n,bool);maxres=0.;solves=0
    for sign in (-1.,1.):
        # sign=-1: free endpoint <= fixed; sign=+1: free >= fixed.
        # eps*c*t^3 + (eps-2sign)t^2 + c(eps+2sign)t + eps=0.
        coeff=np.stack([epsilon*c,np.full(len(c),epsilon-2*sign),c*(epsilon+2*sign),np.full(len(c),epsilon)],axis=1)
        companion=np.zeros((len(c),3,3));companion[:,0,:]=-coeff[:,1:]/coeff[:,0,None];companion[:,1,0]=1;companion[:,2,1]=1
        roots=np.linalg.eigvals(companion);solves+=len(c)
        real=roots.real;realish=np.abs(roots.imag)<=1e-8*(1+np.abs(real));positive=real>0
        candidate=2*np.log(np.maximum(real,1e-300));valid=realish&positive&(candidate>=lo[:,None]-1e-9)&(candidate<=hi[:,None]+1e-9)&(sign*(candidate-fixed[:,None])>=-1e-9)
        residual=np.abs(.5*(sigmoid(candidate)-sigmoid(.5*(fixed[:,None]+candidate)))-sign*epsilon/4)
        if valid.any():maxres=max(maxres,float(residual[valid].max()))
        bad=valid&(residual>1e-7);failed|=bad.any(1)
        # Near-real nonreal roots are deliberately treated as numerical issues.
        near=(~realish)&positive&(np.abs(roots.imag)<1e-5*(1+abs(real)))
        failed|=near.any(1)
        candidate=np.maximum(lo[:,None],np.minimum(hi[:,None],candidate))
        vals=np.where(valid&~bad,objective(fixed[:,None],candidate,epsilon),np.inf).min(1).reshape(4,n).min(0)
        value=np.minimum(value,vals)
    failed=failed.reshape(4,n).any(0)
    pad=1e-10*(1+np.maximum.reduce([abs(al),abs(au),abs(bl),abs(bu)]))
    value=np.maximum(baseline,value-pad)
    value=np.where(failed,baseline,value)
    assert np.isfinite(value).all()
    return value.reshape(shape),dict(cubic_solves=solves,fallback_cells=int(failed.sum()),max_stationary_residual=maxres)

def polar_boxes(cells):
    r0,r1,a0,a1=cells.T;cl=np.cos(a1);cu=np.cos(a0)
    sl=np.minimum(np.sin(a0),np.sin(a1));su=np.maximum(np.sin(a0),np.sin(a1));su=np.where((a0<=np.pi/2)&(a1>=np.pi/2),1.,su)
    cx=np.stack([r0*cl,r0*cu,r1*cl,r1*cu]);sy=np.stack([r0*sl,r0*su,r1*sl,r1*su])
    return np.c_[cx.min(0),sy.min(0)]-1e-12,np.c_[cx.max(0),sy.max(0)]+1e-12

def interval_forward(theta,lo,hi,core):
    for a,b,bias in core.unpack(theta):
        ap=np.maximum(a,0);an=np.minimum(a,0);bp=np.maximum(b,0);bn=np.minimum(b,0)
        tl=np.tanh(lo@ap.T+hi@an.T+bias);tu=np.tanh(hi@ap.T+lo@an.T+bias)
        lo,hi=lo+.5*(tl@bp.T+tu@bn.T),hi+.5*(tu@bp.T+tl@bn.T)
    return lo[:,0],hi[:,0]

def enclose(theta,cells,core):
    lo,hi=polar_boxes(cells);al,au=interval_forward(theta,lo,hi,core);bl,bu=interval_forward(theta,-hi,-lo,core)
    return np.stack([al,au,bl,bu],axis=1)

def uniform_cells(nr,na):
    r=np.linspace(0,3,nr+1);a=np.linspace(0,np.pi,na+1)
    return np.c_[np.repeat(r[:-1],na),np.repeat(r[1:],na),np.tile(a[:-1],nr),np.tile(a[1:],nr)]

def canonical_coordinates(x):
    x=np.where(((x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0)))[:,None],-x,x)
    return np.linalg.norm(x,axis=1),np.arctan2(x[:,1],x[:,0])

def lookup_nodes(x,table):
    radius,angle=canonical_coordinates(x);nr=int(table['root_nr']);na=int(table['root_na'])
    ri=np.minimum((radius/3*nr).astype(int),nr-1);ai=np.minimum((angle/np.pi*na).astype(int),na-1);idx=ri*na+ai
    if 'left' in table:
        left=table['left'];right=table['right'];axis=table['axis'];threshold=table['threshold']
        active=left[idx]>=0
        while active.any():
            old=idx[active];v=np.where(axis[old]==0,radius[active],angle[active]);idx[active]=np.where(v<threshold[old],left[old],right[old]);active=left[idx]>=0
    return idx,radius<=3
