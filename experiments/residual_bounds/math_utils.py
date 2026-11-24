"""Odd density envelopes, exact joint endpoint penalty and finite-cell label CIs."""
import math
import numpy as np
BATCH=256;QUOTA=64;LOOKS=(512,2048);EPS=(0.,.025,.05,.1,.2,.4,.8,1.)
POLICIES=('symmetric_jensen','folded');MODES=('known','estimated');DEFECTS=(0.,.1)
LAWS=[dict(name='uniform',theta=[0.,0.],residual=0.,rho=0.,covered=True),
      dict(name='small_first',theta=[.01,0.],residual=0.,rho=0.,covered=True),
      dict(name='covered_third',theta=[0.,.10],residual=0.,rho=0.,covered=True),
      dict(name='covered_mixed',theta=[.04,.04],residual=0.,rho=0.,covered=True),
      dict(name='covered_residual',theta=[.01,.04],residual=.025,rho=.025,covered=True),
      dict(name='strong_third',theta=[0.,.60],residual=0.,rho=0.,covered=True),
      dict(name='violated_residual',theta=[0.,0.],residual=.60,rho=0.,covered=False)]
LABEL_LAWS=('covered_third','covered_residual')
# 224 base model/law/policy/look groups x18 moments x2 methods;
# four label-audit contexts x8 models x2 policies x2 looks x2 modes x2 methods.
MOMENT_EVENTS=8*7*2*2*18*2+4*8*2*2*2*2
FAMILY=MOMENT_EVENTS+7*2+4*8
DELTA=.05/FAMILY
def lse(z):
    a=z.max(-1);return a+np.log(np.exp(z-a[...,None]).sum(-1))
def nll(h,p):return (1-p)*np.logaddexp(0,h)+p*np.logaddexp(0,-h)
def clip(grid):return float(min(.5,max(1e-6,np.quantile(grid['folded_jensen'],.95))))
def features(x):
    u=x[:,0]/np.linalg.norm(x[:,:2],axis=1)
    # Odd polynomials avoid trigonometric roundoff breaking T-antisymmetry.
    return np.column_stack([u,4*u**3-3*u,16*u**5-20*u**3+5*u])
def imbalance(feature,law):return feature[:,:2]@np.asarray(law['theta'])+law['residual']*feature[:,2]
def sample(task,seed,n,law):
    rng=np.random.default_rng(seed);radii=rng.uniform(.4,1.4,(n,2));parts=[];count=proposals=0
    maximum=sum(abs(v) for v in law['theta'])+abs(law['residual']);assert maximum<=1
    while count<n:
        size=max(1024,int((n-count)*(1+maximum)*1.03)+8)
        phi=rng.uniform(0,2*np.pi,size);u=rng.random(size)
        b=law['theta'][0]*np.cos(phi)+law['theta'][1]*np.cos(3*phi)+law['residual']*np.cos(5*phi)
        keep=phi[u<=(1+b)/(1+maximum)];parts.append(keep);count+=len(keep);proposals+=size
    angle=np.concatenate(parts)[:n]
    return task.coordinates(radii[:,0],radii[:,1],angle),dict(proposals=proposals,retained=n)
def geometry(h,ht):
    d=(h-ht)/2;S=abs(d);J=np.maximum(0.,(np.logaddexp(0,h)+np.logaddexp(0,ht))/2-np.logaddexp(0,(h+ht)/2))
    t=(np.logaddexp(0,h)-np.logaddexp(0,ht))/2-d/2
    assert np.max(abs(t)-S/2)<1e-12
    return J,S,t
def coefficient_ci(feature):
    estimate=2*feature[:,:2].mean(0);radius=math.sqrt(8*math.log(2/DELTA)/len(feature))
    return dict(estimate=estimate.tolist(),radius=[radius,radius],n=len(feature),delta=DELTA)
def envelope(feature,center,radius,rho):
    center=np.asarray(center);radius=np.asarray(radius);b=feature[:,:2]@center
    uncertainty=abs(feature[:,:2])@radius+rho
    lower=np.clip(b-uncertainty,-1,1);upper=np.clip(b+uncertainty,-1,1)
    rbar=float(min(1.,np.sum(abs(center)+radius)+rho))
    return lower,upper,rbar
def penalty(S,t,lower,upper,eps):
    def at(r):return (abs(r)+(1-abs(r))*eps)*S/2-r*t
    return np.maximum(at(lower),at(upper))
def penalty_range(D,rbar,eps):return D*(eps/2+(1-eps/2)*rbar)
def interval(v,item_range,side):
    n=len(v);mean=float(v.mean());variance=float(v.var(ddof=1));z=math.log(2/DELTA)
    eb=math.sqrt(2*variance*z/n)+7*(item_range/4)*z/(3*(n-1))
    mc=item_range*math.sqrt(math.log(1/DELTA)/(2*n*BATCH))
    value=max(0.,mean-eb,mean-mc) if side=='lower' else min(item_range/4,mean+eb,mean+mc)
    return dict(mean=mean,variance=variance,eb_radius=eb,mcd_radius=mc,bound=value,item_range=item_range,
                batch_range=item_range/4,n_batches=n,delta=DELTA)
def cells(x):
    radii=np.column_stack([np.linalg.norm(x[:,:2],axis=1),np.linalg.norm(x[:,2:],axis=1)])
    radial=2*(radii[:,0]>=.9).astype(int)+(radii[:,1]>=.9)
    # Canonical half-circle also pairs the vertical axis; the radius is nonzero.
    side=((x[:,0]>0)|((x[:,0]==0)&(x[:,1]>=0))).astype(int)
    return 2*radial+side
def cell_posterior(cell,defect):
    radial=np.asarray(cell)//2;side=np.asarray(cell)%2
    base=np.where((radial==0)|(radial==3),.8,.2)
    return base+(2*side-1)*defect/2
def label_intervals(cell,y):
    rows=[];lo=[];hi=[]
    for k in range(8):
        count=int(np.sum(cell==k));ones=int(np.sum(y[cell==k]))
        if count:
            phat=ones/count;rad=math.sqrt(math.log(2/DELTA)/(2*count));lower=max(0.,phat-rad);upper=min(1.,phat+rad)
        else:phat=0.;rad=1.;lower=0.;upper=1.
        rows.append(dict(cell=k,count=count,ones=ones,estimate=phat,radius=rad,lower=lower,upper=upper,delta=DELTA))
        lo.append(lower);hi.append(upper)
    eps=max(max(hi[2*k+1]-lo[2*k],hi[2*k]-lo[2*k+1],0.) for k in range(4))
    return rows,float(min(1.,eps))
def posterior(world,p0,d,feature,r,eps):
    if world=='orbit_worst':
        orient=np.where(r==0,feature[:,0],r);mean=.5+.5*(1-eps)*np.sign(orient*d);delta=eps*np.sign(d)
        return mean+delta/2,mean-delta/2
    direction=feature[:,0] if world=='cosine' else np.sign(d);amp=np.minimum(eps/2,np.minimum(p0,1-p0))
    return p0+amp*direction,p0-amp*direction
