"""Four-dimensional two-harmonic waveform task and ordinary residual model."""
import time
import numpy as np
D=4;H=8;BLOCK=2*H*D+H;PARAMETER_COUNT=2*BLOCK
RMIN=.4;RMAX=1.4

def coordinates(r,phi):
    r,phi=np.broadcast_arrays(r,phi);return np.stack([np.cos(phi),np.sin(phi),r*np.cos(2*phi),r*np.sin(2*phi)],axis=-1)
def quotient(x):return np.linalg.norm(np.asarray(x)[...,2:],axis=-1)
def angle(x):return np.arctan2(np.asarray(x)[...,1],np.asarray(x)[...,0])
def posterior(x):
    u=np.clip(quotient(x)-RMIN,0,1);return u**3/(u**3+(1-u)**3)
def radial_density(r):
    u=np.asarray(r)-RMIN;value=15*u*(1-u)*(u**3+(1-u)**3);return np.where((u>=0)&(u<=1),value,0.)
def sample_radii(seed,n):
    rng=np.random.default_rng(seed);y=rng.integers(0,2,n);u=rng.beta(np.where(y,5,2),np.where(y,2,5));return RMIN+u
def sample(seed,n,balanced=False):
    rng=np.random.default_rng(seed)
    if balanced:
        if n%2:raise ValueError('Balanced training requires even sample size')
        y=np.r_[np.zeros(n//2,dtype=int),np.ones(n//2,dtype=int)];rng.shuffle(y)
    else:y=rng.integers(0,2,n)
    u=rng.beta(np.where(y,5,2),np.where(y,2,5));phi=rng.uniform(0,2*np.pi,n);x=coordinates(RMIN+u,phi);return x,2*y-1,posterior(x)
def sample_train(seed,n=256):
    x,y,p=sample(seed,n,balanced=True);return x,y
def sample_test(seed,n=65536):return sample(seed,n,balanced=False)
def transform(x,theta):
    x=np.asarray(x);out=np.empty_like(x)
    for pair,freq in ((0,1),(2,2)):
        c=np.cos(freq*theta);s=np.sin(freq*theta);out[...,pair]=c*x[...,pair]-s*x[...,pair+1];out[...,pair+1]=s*x[...,pair]+c*x[...,pair+1]
    return out
def partner(x):
    out=np.array(x,copy=True);out[...,:2]*=-1;return out
def initialize(seed):
    rng=np.random.default_rng(seed);parts=[]
    for _ in range(2):parts.extend([rng.normal(0,1/np.sqrt(D),(H,D)).ravel(),rng.normal(0,.1/np.sqrt(H),(D,H)).ravel(),np.zeros(H)])
    return np.concatenate(parts)
def unpack(theta):
    theta=np.asarray(theta)
    if theta.shape!=(PARAMETER_COUNT,):raise ValueError('Expected 144 parameters')
    return [(theta[o:o+H*D].reshape(H,D),theta[o+H*D:o+2*H*D].reshape(D,H),theta[o+2*H*D:o+BLOCK]) for o in (0,BLOCK)]
def representation(theta,x):
    z=np.asarray(x)
    for A,B,b in unpack(theta):z=z+.5*np.tanh(z@A.T+b)@B.T
    return z
def predict(theta,x):return representation(theta,x)[:,0]
def representation_jacobian(theta,x):
    z=np.asarray(x);jac=np.zeros((len(z),D,PARAMETER_COUNT));eye=np.eye(D)
    for block,(A,B,b) in enumerate(unpack(theta)):
        t=np.tanh(z@A.T+b);derivative=1-t*t;state=eye+.5*np.einsum('oi,ni,ij->noj',B,derivative,A);jac=np.einsum('noj,njp->nop',state,jac);offset=block*BLOCK
        jac[:,:,offset:offset+H*D]+=(.5*np.einsum('oi,ni,nj->noij',B,derivative,z)).reshape(len(z),D,H*D)
        jac[:,:,offset+H*D:offset+2*H*D]+=(.5*np.einsum('ok,nj->nokj',eye,t)).reshape(len(z),D,H*D)
        jac[:,:,offset+2*H*D:offset+BLOCK]+=.5*np.einsum('oi,ni->noi',B,derivative)
        z=z+.5*t@B.T
    return jac.reshape(D*len(z),PARAMETER_COUNT)
def loss(theta,x,y):return float(np.logaddexp(0,-y*predict(theta,x)).mean())
def parameter_gradient(theta,x,y):
    z=x;cache=[];blocks=unpack(theta)
    for A,B,b in blocks:t=np.tanh(z@A.T+b);cache.append((z,t));z=z+.5*t@B.T
    v=np.zeros_like(z);v[:,0]=-y*np.exp(-np.logaddexp(0,y*z[:,0]))/len(y);grads=[]
    for (A,B,b),(previous,t) in zip(reversed(blocks),reversed(cache)):
        db=.5*v.T@t;dt=.5*(v@B)*(1-t*t);da=dt.T@previous;bias=dt.sum(0);grads.append(np.r_[da.ravel(),db.ravel(),bias]);v=v+dt@A
    return np.concatenate(grads[::-1])
def direction(theta,x,y,method='pb',damping=.1):
    if method=='gd':return -parameter_gradient(theta,x,y)
    if method!='pb':raise ValueError(method)
    z=representation(theta,x);jac=representation_jacobian(theta,x);v=np.zeros_like(z);v[:,0]=-y*np.exp(-np.logaddexp(0,y*z[:,0]))/len(y);gram=jac.T@jac;gram.flat[::PARAMETER_COUNT+1]+=damping;return np.linalg.solve(gram,-jac.T@v.ravel())
def train(theta,x,y,steps=500,method='pb',rate=8.,damping=.1):
    theta=np.array(theta,copy=True);start=time.perf_counter();trace=[]
    for step in range(steps):
        theta+=rate*direction(theta,x,y,method,damping)
        if (step+1)%100==0:trace.append((step+1,loss(theta,x,y)))
        if not np.isfinite(theta).all():raise FloatingPointError('Non-finite training state')
    return theta,dict(seconds=time.perf_counter()-start,steps=steps,method=method,rate=rate,damping=damping,loss_trace=trace)
def network_constants(theta):
    L=1.;C=0.
    for A,B,b in unpack(theta):L*=1+.5*np.linalg.norm(B,2)*np.linalg.norm(A,2);C+=.5*np.abs(B[0]).sum()
    return float(L),float(C)


TRAIN_CONFIG={'gd':dict(steps=500,method='gd',rate=.3,damping=.1),'pb':dict(steps=500,method='pb',rate=8.,damping=.1)}
def sample_inputs(seed,n):
    rng=np.random.default_rng(seed);component=rng.integers(0,2,n);u=rng.beta(np.where(component,5,2),np.where(component,2,5));phi=rng.uniform(0,2*np.pi,n);return coordinates(RMIN+u,phi)
def population_cdf_r(r):
    u=np.clip(np.asarray(r)-RMIN,0,1);return 7.5*u**2-20*u**3+22.5*u**4-9*u**5
def population_first_moment_r(r):
    u=np.clip(np.asarray(r)-RMIN,0,1);return RMIN*population_cdf_r(r)+5*u**3-15*u**4+18*u**5-7.5*u**6
