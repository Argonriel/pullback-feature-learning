"""Uniform posterior-defect certificates for fixed symmetric top-quota rules."""
import math
import numpy as np
BATCH=256
QUOTA=64
LOOKS=(512,2048)
EPSILONS=(0.,.025,.05,.1,.2,.4,.8,1.)
DELTA=.05/512

def gap(a,b):
    out=.5*(np.logaddexp(0.,a)+np.logaddexp(0.,b))-np.logaddexp(0.,.5*(a+b))
    if np.min(out,initial=0.) < -1e-12:raise ArithmeticError('Numerical negative Jensen gap')
    return np.maximum(out,0.)

def clip_from_grid(grid):
    return float(min(.5,max(1e-6,np.quantile(grid['folded_jensen'],.95))))

def moment_bounds(A,B,c,D):
    """A=mean selected clipped J, B=mean selected |d| per independent batch."""
    A=np.asarray(A);B=np.asarray(B);n=len(A)
    assert n==len(B) and n>1 and c>0 and D>=0
    z=math.log(2/DELTA);mcd_scale=math.sqrt(math.log(1/DELTA)/(2*n*BATCH))
    eb_a=math.sqrt(2*A.var(ddof=1)*z/n)+7*(c/4)*z/(3*(n-1))
    eb_b=math.sqrt(2*B.var(ddof=1)*z/n)+7*(D/4)*z/(3*(n-1))
    low_eb=float(A.mean()-eb_a);low_mcd=float(A.mean()-c*mcd_scale)
    up_eb=float(B.mean()+eb_b);up_mcd=float(B.mean()+D*mcd_scale)
    lower=max(0.,low_eb,low_mcd);upper=min(D/4,up_eb,up_mcd)
    critical=min(1.,2*lower/upper) if upper>0 else (1. if lower>0 else 0.)
    return dict(n_batches=n,delta=DELTA,c=c,D=D,mean_A=float(A.mean()),mean_B=float(B.mean()),
        A_eb_lower=low_eb,A_mcd_lower=low_mcd,B_eb_upper=up_eb,B_mcd_upper=up_mcd,
        A_lower=lower,B_upper=upper,epsilon_critical=critical,
        A_mcd_penalty=c*mcd_scale,B_mcd_penalty=D*mcd_scale,A_eb_penalty=eb_a,B_eb_penalty=eb_b)

def robust_line(result,epsilon):
    return result['A_lower']-.5*np.asarray(epsilon)*result['B_upper']

def perturb(p0,epsilon,q):
    """q is T-odd in [-1,1]; pair difference is 2*amplitude*q."""
    amplitude=np.minimum(.5*epsilon,np.minimum(p0,1-p0))
    return p0+amplitude*q,p0-amplitude*q,amplitude

def nll(logit,posterior):
    return (1-posterior)*np.logaddexp(0.,logit)+posterior*np.logaddexp(0.,-logit)
