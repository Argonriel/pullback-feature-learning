"""Two independent harmonic amplitudes: a two-dimensional orbit quotient.

Only model functions are reused from byte-identical core4d.py. Its legacy data
helpers are not part of this task's data-generating interface.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('two_amplitude_frozen_core', HERE/'core4d.py')
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
D=4; H=8; PARAMETER_COUNT=144; QUOTIENT_DIM=2
RMIN=.4; RMAX=1.4
SUPPORT_LO=np.array([RMIN,RMIN]); SUPPORT_HI=np.array([RMAX,RMAX])
for _name in ('initialize','unpack','representation','predict','representation_jacobian',
              'loss','parameter_gradient','direction','train','network_constants'):
    globals()[_name]=getattr(core,_name)

def coordinates(r1,r2,phi):
    r1,r2,phi=np.broadcast_arrays(r1,r2,phi)
    return np.stack([r1*np.cos(phi),r1*np.sin(phi),r2*np.cos(2*phi),r2*np.sin(2*phi)],axis=-1)

def quotient(x):
    x=np.asarray(x)
    return np.stack([np.linalg.norm(x[...,:2],axis=-1),np.linalg.norm(x[...,2:],axis=-1)],axis=-1)

def angle(x):
    x=np.asarray(x); return np.arctan2(x[...,1],x[...,0])

def partner(x):
    result=np.array(x,copy=True); result[...,:2]*=-1; return result

transform=core.transform

def posterior_quotient(r):
    r=np.asarray(r); z=2*(r-.9)
    return .5+.4*np.tanh(4*z[...,0])*np.tanh(4*z[...,1])

def posterior(x): return posterior_quotient(quotient(x))

def density_quotient(r):
    r=np.asarray(r); return np.all((r>=RMIN)&(r<=RMAX),axis=-1).astype(float)

def cdf_quotient(r): return np.prod(np.clip(np.asarray(r)-RMIN,0,1),axis=-1)

def rectangle_mass(lo,hi):
    return np.prod(np.maximum(0,np.clip(np.asarray(hi)-RMIN,0,1)-np.clip(np.asarray(lo)-RMIN,0,1)),axis=-1)

def sample_quotient(seed,n): return np.random.default_rng(seed).uniform(RMIN,RMAX,(n,2))

def _sample_rng(rng,n):
    r=rng.uniform(RMIN,RMAX,(n,2)); phi=rng.uniform(0,2*np.pi,n)
    return coordinates(r[:,0],r[:,1],phi)

def sample_inputs(seed,n): return _sample_rng(np.random.default_rng(seed),n)

def sample_test(seed,n=65536):
    rng=np.random.default_rng(seed); x=_sample_rng(rng,n); p=posterior(x)
    y=2*(rng.random(n)<p).astype(int)-1
    return x,y,p

def sample_train_with_metadata(seed,n=256):
    """Independent class-conditional rejection, fixed counts, final shuffle.

    This is stratified sampling: labels have fixed counts, not iid Bernoulli.
    Conditional on each class, accepted inputs have the correct population law.
    """
    if n%2 or n<=0: raise ValueError('n must be positive and even')
    rng=np.random.default_rng(seed); accepted={-1:[],1:[]}; counts={-1:0,1:0}; proposals=0; batches=0
    while min(counts.values())<n//2:
        x=_sample_rng(rng,4*n); y=2*(rng.random(4*n)<posterior(x)).astype(int)-1
        proposals+=len(x); batches+=1
        for label in (-1,1):
            take=x[y==label][:n//2-counts[label]]
            accepted[label].append(take); counts[label]+=len(take)
    x=np.concatenate([np.concatenate(accepted[label]) for label in (-1,1)])
    y=np.repeat([-1,1],n//2); order=rng.permutation(n)
    return x[order],y[order],dict(proposals=proposals,proposal_batches=batches,class_counts=[n//2,n//2])

def sample_train(seed,n=256):
    x,y,_=sample_train_with_metadata(seed,n); return x,y

def training_config():
    path=HERE/'TRAINING_CHOICE.json'
    if not path.exists(): raise RuntimeError('Pilot recipe has not been frozen')
    return json.loads(path.read_text())['config']

def expected_loss(theta,x,p=None):
    h=predict(theta,x); p=posterior(x) if p is None else p
    return float(np.mean(np.logaddexp(0,h)-p*h))
