"""Cohort adapters; label posteriors are evaluation-only functions."""
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,str(ROOT/'legacy'/'budgeted-rotation-final'))
from pullback_feature_learning import core
from generator import rings

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

ring_rank=load_module('confirmation_ring_rank',ROOT/'reference'/'rank_policy.py')
task=load_module('task',ROOT/'transfer'/'task.py')
transfer_rank=load_module('confirmation_transfer_rank',ROOT/'transfer'/'rank_policy.py')

def ring_inputs(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);phi=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.column_stack((np.cos(phi),np.sin(phi)))+rng.normal(0,.15,(n,2))

def ring_coordinates(r,phi):
    r,phi=np.broadcast_arrays(r,phi)
    return np.stack((r*np.cos(phi),r*np.sin(phi)),axis=-1)

def ring_transform(x,angle):
    c=np.cos(angle);s=np.sin(angle)
    return np.column_stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]))

def ring_posterior(x):
    r=np.linalg.norm(x,axis=1)
    logit=(.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(1.2*r/.15**2))-np.log(np.i0(.5*r/.15**2))
    return np.exp(-np.logaddexp(0.,-logit))

def nll(h,p):
    return (1-p)*np.logaddexp(0.,h)+p*np.logaddexp(0.,-h)

def ring_network_constants(theta):
    L=1.;C=0.
    for A,B,b in core.unpack(theta):
        L*=1+.5*np.linalg.norm(B,2)*np.linalg.norm(A,2)
        C+=.5*np.abs(B[0]).sum()
    return float(L),float(C)

def get_adapter(cohort):
    if cohort=='rings':
        predict=lambda theta,x:core.representation(theta,x)[:,0]
        return SimpleNamespace(name=cohort,network=core,rank=ring_rank,dimension=2,parameters=80,
            methods=['pb'],training={'pb':dict(steps=500,rate=8.,damping=.1)},
            sample_train=lambda seed:rings(seed),sample_inputs=ring_inputs,
            initialize=core.initialize_parameters,predict=predict,
            network_constants=ring_network_constants,
            direction=lambda theta,x,y,method,damping:core.direction(theta,x,y,'pullback' if method=='pb' else method,damping=damping)[0],
            loss=lambda theta,x,y:float(np.logaddexp(0.,-y*predict(theta,x)).mean()),
            quotient=lambda x:np.linalg.norm(x,axis=1),angle=lambda x:np.arctan2(x[:,1],x[:,0]),
            coordinates=ring_coordinates,partner=lambda x:-np.asarray(x),transform=ring_transform,
            posterior=ring_posterior,world_position=lambda x:np.clip((np.linalg.norm(x,axis=1)-.25)/1.25,0,1))
    if cohort=='transfer':
        return SimpleNamespace(name=cohort,network=task,rank=transfer_rank,dimension=4,parameters=144,
            methods=['pb','gd'],training={m:{k:v for k,v in cfg.items() if k!='method'} for m,cfg in task.TRAIN_CONFIG.items()},
            sample_train=lambda seed:task.sample_train(seed,256),sample_inputs=task.sample_inputs,
            initialize=task.initialize,predict=task.predict,direction=task.direction,loss=task.loss,
            network_constants=task.network_constants,
            quotient=task.quotient,angle=task.angle,coordinates=task.coordinates,
            partner=task.partner,transform=task.transform,posterior=task.posterior,
            world_position=lambda x:np.clip(task.quotient(x)-task.RMIN,0,1))
    raise ValueError(cohort)

def worlds(adapter,x):
    position=adapter.world_position(x)
    return {'half':np.full(len(x),.5),'increasing':position,
            'decreasing':1-position,'threshold':(position>=.5).astype(float)}

def folded_bins(adapter,x):
    # First harmonic determines the canonical representative under the half-turn.
    pair=np.asarray(x)[:,:2]
    flip=(pair[:,1]<0)|((pair[:,1]==0)&(pair[:,0]<0))
    canonical=np.where(flip[:,None],-pair,pair)
    angle=np.arctan2(canonical[:,1],canonical[:,0])%np.pi
    return np.minimum((8*angle/np.pi).astype(int),7)

def allocate(score,ties):
    score=np.asarray(score).reshape(-1,256);ties=np.asarray(ties).reshape(-1,256)
    order=np.lexsort((ties,-score),axis=1);mask=np.zeros_like(score,dtype=bool)
    np.put_along_axis(mask,order[:,:64],True,axis=1)
    assert np.all(mask.sum(axis=1)==64)
    return mask.ravel()

def fit_folded(adapter,theta,x):
    h=adapter.predict(theta,x);other=adapter.predict(theta,adapter.partner(x))
    gain=.5*(np.logaddexp(0.,h)+np.logaddexp(0.,other))-np.logaddexp(0.,.5*(h+other))
    assert gain.min()>-1e-12
    r=adapter.quotient(x);edges=np.quantile(r,np.arange(1,16)/16)
    bins=np.searchsorted(edges,r);counts=np.bincount(bins,minlength=16)
    radial=np.bincount(bins,weights=gain,minlength=16)/counts
    cells=8*bins+folded_bins(adapter,x);cell_counts=np.bincount(cells,minlength=128)
    scores=np.divide(np.bincount(cells,weights=gain,minlength=128),cell_counts,
        out=np.repeat(radial,8),where=cell_counts>0)
    return dict(edges=edges,scores=scores,radius_scores=radial,counts=cell_counts,
        calibration_x=x,calibration_first=h,calibration_partner=other)

def fit_angles(adapter,theta,r,phase):
    angles=phase[:,None]+2*np.pi*np.arange(128)[None,:]/128
    x=adapter.coordinates(r[:,None],angles).reshape(-1,adapter.dimension)
    h=adapter.predict(theta,x).reshape(len(r),128);soft=np.logaddexp(0.,h)
    gains=np.stack([(.5*(soft+np.roll(soft,-j,axis=1))-np.logaddexp(0.,.5*(h+np.roll(h,-j,axis=1)))).mean(axis=1) for j in range(1,65)],axis=1)
    edges=np.quantile(r,np.arange(1,16)/16);bins=np.searchsorted(edges,r,side='right')
    counts=np.bincount(bins,minlength=16);global_curve=gains.mean(axis=0)
    curves=np.array([gains[bins==b].mean(axis=0) if counts[b]>=4 else global_curve for b in range(16)])
    return dict(edges=edges,indices=curves.argmax(axis=1)+1,curves=curves,counts=counts,
        global_curve=global_curve,calibration_h=h,calibration_radii=r,calibration_phases=phase)

def deploy(adapter,theta,x,name,tables,ties,action_keys,predict=None):
    predict=(lambda z:adapter.predict(theta,z)) if predict is None else predict
    first=predict(x)
    if name=='raw':return first,np.zeros(len(x),dtype=bool)
    if name in ('full','full_radial'):mask=np.ones(len(x),dtype=bool)
    elif name in ('confidence','confidence_radial'):mask=allocate(-np.abs(first),ties)
    elif name=='folded':
        tab=tables['folded'];b=np.searchsorted(tab['edges'],adapter.quotient(x))
        mask=allocate(tab['scores'][8*b+folded_bins(adapter,x)],ties)
    else:mask=adapter.rank.select(x,first,tables['rank'],name,ties,action_keys)
    if name in ('confidence_radial','full_radial'):
        tab=tables['angles'];b=np.searchsorted(tab['edges'],adapter.quotient(x[mask]),side='right')
        alpha=tab['indices'][b]*2*np.pi/128
        partner=adapter.transform(x[mask],alpha)
    else:partner=adapter.partner(x[mask])
    out=first.copy();out[mask]=.5*(first[mask]+predict(partner))
    quota=256 if name in ('full','full_radial') else 64
    assert np.all(mask.reshape(-1,256).sum(axis=1)==quota)
    return out,mask
