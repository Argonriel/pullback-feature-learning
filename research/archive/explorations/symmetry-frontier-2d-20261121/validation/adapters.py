"""Shared data/model interfaces for the three fixed symmetry tasks."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(ROOT/'legacy'))
from pullback_feature_learning import core
from generator import rings

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
wave=module('frontier_waveform_task',ROOT/'reference/waveform_task.py')
two=module('frontier_two_invariants_task',ROOT/'task2d/task.py')

def ring_inputs(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);phi=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.c_[np.cos(phi),np.sin(phi)]+rng.normal(0,.15,(n,2))
def ring_coordinates(r,phi):
    r,phi=np.broadcast_arrays(r,phi);return np.stack([r*np.cos(phi),r*np.sin(phi)],axis=-1)
def ring_posterior(x):
    r=np.linalg.norm(x,axis=1);v=.15**2
    logit=(.5**2-1.2**2)/(2*v)+np.log(np.i0(1.2*r/v))-np.log(np.i0(.5*r/v))
    return np.exp(-np.logaddexp(0,-logit))
def ring_constants(theta):
    L=1.;C=0.
    for A,B,b in core.unpack(theta):L*=1+.5*np.linalg.norm(B,2)*np.linalg.norm(A,2);C+=.5*abs(B[0]).sum()
    return float(L),float(C)
def ring_transform(x,angle):
    c=np.cos(angle);s=np.sin(angle);return np.c_[c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]]

def get_adapter(cohort):
    if cohort=='rings':
        predict=lambda theta,x:core.representation(theta,x)[:,0]
        return SimpleNamespace(name=cohort,dimension=2,parameters=80,quotient_dim=1,network=core,
            training=dict(steps=500,method='pb',rate=8.,damping=.1),
            sample_train=rings,sample_inputs=ring_inputs,initialize=core.initialize_parameters,
            predict=predict,representation=core.representation,
            direction=lambda theta,x,y,method='pb',damping=.1:core.direction(theta,x,y,'pullback' if method=='pb' else method,damping=damping)[0],
            loss=lambda theta,x,y:float(np.logaddexp(0,-y*predict(theta,x)).mean()),
            partner=lambda x:-np.asarray(x),transform=ring_transform,
            quotient=lambda x:np.linalg.norm(x,axis=1),coordinates=ring_coordinates,
            angle=lambda x:np.arctan2(x[:,1],x[:,0]),posterior=ring_posterior,network_constants=ring_constants,
            support_lo=np.array([0.]),support_hi=np.array([3.]))
    if cohort in ('waveform','two_invariants'):
        task=wave if cohort=='waveform' else two
        cfg=wave.TRAIN_CONFIG['pb'] if cohort=='waveform' else two.training_config()
        return SimpleNamespace(name=cohort,dimension=4,parameters=144,
            quotient_dim=1 if cohort=='waveform' else 2,network=wave if cohort=='waveform' else two.core,
            training=cfg,sample_train=task.sample_train,sample_inputs=task.sample_inputs,
            initialize=task.initialize,predict=task.predict,representation=task.representation,
            direction=task.direction,loss=task.loss,partner=task.partner,transform=task.transform,
            quotient=task.quotient,coordinates=task.coordinates,angle=task.angle,
            posterior=task.posterior,network_constants=task.network_constants,
            support_lo=np.full(1 if cohort=='waveform' else 2,.4),
            support_hi=np.full(1 if cohort=='waveform' else 2,1.4))
    raise ValueError(cohort)

def nll(h,p):return (1-p)*np.logaddexp(0,h)+p*np.logaddexp(0,-h)
def worlds(adapter,x):
    z=np.asarray(adapter.quotient(x));z=z[:,None] if z.ndim==1 else z
    if adapter.name=='rings':position=np.clip((z[:,0]-.25)/1.25,0,1)
    else:position=np.clip(z[:,0]-.4,0,1)
    return dict(half=np.full(len(x),.5),decreasing=1-position,
                angular=.5+.4*np.cos(2*adapter.angle(x)))
