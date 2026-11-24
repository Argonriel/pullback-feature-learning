"""Numerical audits of identities and concentration implementation, not proofs."""
from pathlib import Path
import json
import numpy as np
from bounds import jensen_gap, jensen_bound, empirical_bernstein_lower, rings_tail_bound

def main():
    rng=np.random.default_rng(1109001)
    midpoint=rng.uniform(-20,20,100000);d=rng.uniform(-20,20,100000)
    direct=jensen_gap(midpoint+d,midpoint-d)
    algebra=.5*np.log1p(np.sinh(d/2)**2/np.cosh(midpoint/2)**2)
    bound=np.logaddexp(d/2,-d/2)-np.log(2.)
    error=float(np.max(abs(direct-algebra)))
    assert error<2e-14 and np.all(direct<=bound+2e-14)
    x=rng.normal(size=(128,256));ties=rng.random(x.shape)
    score=x*x+.2*np.cos(x)
    order=np.lexsort((ties,-score),axis=1);mask=np.zeros_like(x,bool)
    np.put_along_axis(mask,order[:,:64],True,axis=1)
    first=3*x+.7*np.sin(2*x)+.4*x*x
    partner=-3*x-.7*np.sin(2*x)+.4*x*x
    p=.5+.4*np.tanh(x*x-1)
    averaged=.5*(first+partner)
    out=np.where(mask,averaged,first);out_t=np.where(mask,averaged,partner)
    loss=lambda h:np.logaddexp(0,h)-p*h
    antithetic=.5*((loss(first)-loss(out))+(loss(partner)-loss(out_t)))
    identity_error=float(np.max(abs(antithetic-mask*jensen_gap(first,partner))))
    assert identity_error<3e-14
    v=rng.uniform(-.125,.125,128);delta=.05/432
    result=empirical_bernstein_lower(v,.25,delta)
    pairwise=float(np.square(v[:,None]-v[None,:]).sum()/(2*len(v)*(len(v)-1)))
    variance_error=abs(pairwise-result['variance'])
    assert variance_error<1e-16
    reflected=empirical_bernstein_lower((v+.125)/.25,1.,delta)
    scaling_error=abs(result['lower']-(.25*reflected['lower']-.125))
    assert scaling_error<1e-15
    output=dict(seed=1109001,jensen_identity_max_error=error,
        antithetic_batch_identity_max_error=identity_error,
        unbiased_variance_pairwise_error=variance_error,affine_scaling_error=scaling_error,
        ring_tail_C20_R3=rings_tail_bound(20.,3.),all_passed=True,
        qualification='Numerical identities supplement, and do not replace, the mathematical proof.')
    with (Path(__file__).parent/'bounds_checks.json').open('x',encoding='utf8') as f:
        json.dump(output,f,indent=2)
    print(json.dumps(output))

if __name__=='__main__':main()
