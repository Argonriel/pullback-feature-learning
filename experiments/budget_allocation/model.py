"""Unchanged 144-parameter model; true L2 objective and label-only calibration."""
from pathlib import Path
import importlib.util,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('calibration_training_frozen_task',HERE/'reference/task.py')
task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
WEIGHT_MASK=np.ones(144)
WEIGHT_MASK[64:72]=0.;WEIGHT_MASK[136:144]=0.
CHECKPOINTS=(100,250,500,1000)
BETA_MIN=float(np.exp(-3));BETA_MAX=float(np.exp(3))

def labeled_inputs(seed,n):
    # The posterior belongs to the generative mechanism, never to fit functions.
    x,y,_=task.sample_test(seed,n);return x,y

def regularized_objective(theta,x,y,lam):
    return task.loss(theta,x,y)+.5*lam*float(np.sum(WEIGHT_MASK*theta*theta))

def direction(theta,x,y,optimizer,lam,damping=.1):
    if lam==0:return task.direction(theta,x,y,method=optimizer,damping=damping)
    penalty=lam*WEIGHT_MASK*theta
    if optimizer=='gd':return -task.parameter_gradient(theta,x,y)-penalty
    if optimizer!='pb':raise ValueError(optimizer)
    z=task.representation(theta,x);jac=task.representation_jacobian(theta,x)
    v=np.zeros_like(z);v[:,0]=-y*np.exp(-np.logaddexp(0,y*z[:,0]))/len(y)
    gram=jac.T@jac;gram.flat[::145]+=damping
    return np.linalg.solve(gram,-jac.T@v.ravel()-penalty)

def trajectory(initial,x,y,xv,yv,optimizer,rate,lam,checkpoints=CHECKPOINTS,damping=.1):
    theta=initial.copy();states=[];rows=[];start=time.perf_counter()
    for step in range(1,max(checkpoints)+1):
        theta+=rate*direction(theta,x,y,optimizer,lam,damping)
        if not np.isfinite(theta).all():raise FloatingPointError(f'nonfinite parameters at {step}')
        if step in checkpoints:
            states.append(theta.copy());rows.append(dict(step=step,training_nll=task.loss(theta,x,y),
                regularized_training_objective=regularized_objective(theta,x,y,lam),validation_nll=task.loss(theta,xv,yv)))
    index=min(range(len(rows)),key=lambda i:(rows[i]['validation_nll'],rows[i]['step']))
    return np.stack(states),rows,index,time.perf_counter()-start

def temperature(logits,labels):
    """Minimize sampled-label NLL over bounded positive inverse temperature."""
    h=np.asarray(logits,dtype=float);y=np.asarray(labels,dtype=float)
    if h.shape!=y.shape or not np.isfinite(h).all() or not np.isin(y,[-1,1]).all():raise ValueError('invalid logits/labels')
    objective=lambda beta:float(np.mean(np.logaddexp(0,-y*beta*h)))
    derivative=lambda beta:float(np.mean(-y*h*np.exp(-np.logaddexp(0,y*beta*h))))
    lo,hi=BETA_MIN,BETA_MAX;dl,dh=derivative(lo),derivative(hi)
    if dl>=0:beta=lo;status='lower_bound';iterations=0
    elif dh<=0:beta=hi;status='upper_bound';iterations=0
    else:
        for iterations in range(1,81):
            mid=.5*(lo+hi)
            if derivative(mid)>0:hi=mid
            else:lo=mid
        beta=.5*(lo+hi);status='interior'
    return dict(beta=beta,temperature=1/beta,status=status,iterations=iterations,
       n_labels=len(y),raw_nll=objective(1.),calibrated_nll=objective(beta),derivative=derivative(beta),
       lower_derivative=dl,upper_derivative=dh,beta_min=BETA_MIN,beta_max=BETA_MAX)
