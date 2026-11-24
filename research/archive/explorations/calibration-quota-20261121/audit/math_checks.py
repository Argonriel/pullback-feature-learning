"""Independent arithmetic checks; does not import investigator implementations."""
from itertools import product
from pathlib import Path
import json
import numpy as np

def sigmoid(x):return np.exp(-np.logaddexp(0.,-x))
def ce(a,p):return np.logaddexp(0.,a)-p*a

def known_prevalence_integral(b,w,prevalence,maximize):
    order=np.argsort(b)
    if maximize:order=order[::-1]
    p=np.zeros(len(b));left=prevalence
    for i in order:
        amount=min(left,w[i]);p[i]=amount/w[i];left-=amount
        if left<=1e-14:break
    assert abs(w@p-prevalence)<1e-13
    return float(w@(p*b)),p

def run():
    rng=np.random.default_rng(741001)
    a,b=rng.normal(size=(2,50000))*4
    p,q=rng.random((2,len(a)));m=(a+b)/2
    direct=.5*(ce(a,p)+ce(b,q)-ce(m,p)-ce(m,q))
    J=.5*(np.logaddexp(0.,a)+np.logaddexp(0.,b))-np.logaddexp(0.,m)
    identity=J-(p-q)*(a-b)/4
    max_identity=float(np.max(abs(direct-identity)))
    sharp=[]
    for epsilon in [0.,.1,.25,.5,.75,1.]:
        pp=.5+.5*epsilon*np.sign(a-b);qq=1-pp
        actual=.5*(ce(a,pp)+ce(b,qq)-ce(m,pp)-ce(m,qq))
        bound=J-epsilon*abs(a-b)/4
        sharp.append(dict(epsilon=epsilon,max_bound_equality_error=float(np.max(abs(actual-bound)))))
    assert max_identity<1e-13 and max(x['max_bound_equality_error'] for x in sharp)<1e-13
    four_point=[float((ce(0.,eta)-ce(1.,eta))/4) for eta in [1.,0.]]
    assert four_point[0]>0 and four_point[1]<0
    # Exact finite posterior feasible set, independently checked by enumeration.
    bb=np.array([-.8,-.15,.35,1.2]);ww=np.full(4,.25);C=.03;pi=.5
    allp=np.array(list(product([0.,.25,.5,.75,1.],repeat=4)))
    allp=allp[np.isclose(allp@ww,pi)]
    gains=C-(allp*bb)@ww
    largest,pmin=known_prevalence_integral(bb,ww,pi,True)
    smallest,pmax=known_prevalence_integral(bb,ww,pi,False)
    assert abs(gains.min()-(C-largest))<1e-14 and abs(gains.max()-(C-smallest))<1e-14
    bc=np.full(4,.3);constant_gains=C-(allp*bc)@ww
    assert np.ptp(constant_gains)<1e-14
    # A real selective two-view construction with conditional imbalance canceling
    # in a single radius bin. Every radius has two equiprobable orientations;
    # select only the first, h_first=-(r-.5), h_second=r-.5, mean logit=0.
    r=(np.arange(200000)+.5)/200000;u=r-.5
    h=-u;mm=np.zeros_like(h);br=-u/2
    cc=float(.5*np.mean(np.logaddexp(0.,h)-np.log(2.)))
    results=[]
    for name,posterior in [('increasing_step',(r>=.5).astype(float)),('decreasing_step',(r<.5).astype(float)),
                           ('increasing_smooth',sigmoid(8*u)),('decreasing_smooth',sigmoid(-8*u))]:
        gain=float(.5*np.mean(ce(h,posterior)-ce(mm,posterior)))
        formula=cc-float(np.mean(posterior*br))
        assert abs(gain-formula)<1e-14
        results.append(dict(posterior=name,class_probability=float(posterior.mean()),actual_gain=gain,
                            incorrectly_binned_identified_gain=cc))
    assert abs(br.mean())<1e-14 and results[0]['actual_gain']*results[1]['actual_gain']<0
    assert results[2]['actual_gain']*results[3]['actual_gain']<0
    # Minimum Jensen gap over a rectangle: independent dense candidate search.
    interval_errors=[]
    for _ in range(100):
        al,au=np.sort(rng.normal(size=2)*3);bl,bu=np.sort(rng.normal(size=2)*3)
        gap=lambda x,y:.5*(np.logaddexp(0.,x)+np.logaddexp(0.,y))-np.logaddexp(0.,(x+y)/2)
        if max(al,bl)<=min(au,bu):lower=0.
        elif au<bl:lower=float(gap(au,bl))
        else:lower=float(gap(al,bu))
        aa=rng.uniform(al,au,1000);bbb=rng.uniform(bl,bu,1000)
        assert np.min(gap(aa,bbb))>=lower-1e-14
        interval_errors.append(float(np.min(gap(aa,bbb))-lower))
    result=dict(status='checked',role='Independent mathematical and numerical diagnostics; not new trained-model replications',
        random_identity_pairs=len(a),max_antipodal_identity_error=max_identity,sharpness=sharp,
        four_point_gains=four_point,known_prevalence=dict(n_enumerated=len(allp),range=[float(gains.min()),float(gains.max())],constant_b_range_width=float(np.ptp(constant_gains))),
        bin_cancellation=dict(bin_mean_b=float(br.mean()),C=cc,posterior_worlds=results),interval_jensen_minimum_checks=len(interval_errors))
    return result

if __name__=='__main__':
    result=run();out=Path(__file__).resolve().parent/'mathematics.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
