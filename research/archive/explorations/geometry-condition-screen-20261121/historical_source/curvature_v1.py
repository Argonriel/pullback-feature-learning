"""Two-parameter residual-tanh slice: reduction and curvature sign tests.

This is an explicitly constrained slice, not the original all-80-parameter
training experiment. It preserves the two residual blocks and both output
coordinates. Numerical search is counterexample construction, not inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from pullback_feature_learning import core

C = .5*np.tanh(1.)
LAMBDA = .1


def embed(theta, bias=1.):
    """Only B1[1,0]=u and B2[0,0]=v are trainable original coordinates."""
    p=np.zeros(80)
    p[32]=1.                      # first active neuron is constant tanh(1)
    p[24]=theta[0]                # B1 row 2, neuron 1
    p[41]=1.                      # A2 neuron 1 reads second state coordinate
    p[72]=bias
    p[56]=theta[1]                # B2 row 1, neuron 1
    return p


def predict(theta, shift=0., bias=1.):
    u,v=theta
    s=shift+C*u+bias
    return .5*v*np.tanh(s)


def jac(theta, shift=0., bias=1.):
    u,v=theta
    s=shift+C*u+bias
    return np.array([[.5*v*C*(1-np.tanh(s)**2), .5*np.tanh(s)], [C,0.]])


def metric(theta,bias=1.):
    j=jac(theta,bias=bias)
    return j.T@j+LAMBDA*np.eye(2)


def flow_r(theta,p0,adaptive,bias=1.):
    g=jac(theta,bias=bias)[0]
    d=np.linalg.solve(metric(theta,bias),g) if adaptive else p0@g
    return d/(g@d)


def path(theta0,r_end,h,adaptive,bias=1.):
    p0=np.linalg.inv(metric(theta0,bias))
    r0=predict(theta0,bias=bias)
    n=int(np.ceil((r_end-r0)/h))
    hh=(r_end-r0)/n
    theta=np.array(theta0,dtype=float)
    snapshots=[(float(r0),theta.copy())]
    min_s=float(bias+C*theta[0])
    for i in range(n):
        k1=flow_r(theta,p0,adaptive,bias)
        k2=flow_r(theta+hh*k1/2,p0,adaptive,bias)
        k3=flow_r(theta+hh*k2/2,p0,adaptive,bias)
        k4=flow_r(theta+hh*k3,p0,adaptive,bias)
        theta+=hh*(k1+2*k2+2*k3+k4)/6
        min_s=min(min_s,float(bias+C*theta[0]))
        if (i+1)%max(1,n//100)==0 or i+1==n:
            snapshots.append((float(r0+hh*(i+1)),theta.copy()))
    return snapshots,min_s


def risk(theta,shifts,weights,bias=1.):
    logits=np.array([predict(theta,s,bias) for s in shifts])
    return float(weights@np.logaddexp(0.,-logits))


def search():
    """Small deterministic construction grid; no claim of statistical power."""
    candidates=[]
    for bias in (.25,.5,1.,2.):
        theta0=np.array([0.,.5])
        r0=predict(theta0,bias=bias)
        aa,_=path(theta0,3.,.004,True,bias)
        ff,_=path(theta0,3.,.004,False,bias)
        for dm in (-.1,-.2):
            for dp in (.25,1.,3.):
                if bias+dm<=0:
                    continue
                components=[]
                for (r,a),(_,f) in zip(aa[1:],ff[1:]):
                    comp=np.array([np.logaddexp(0.,-predict(a,s,bias))-
                                   np.logaddexp(0.,-predict(f,s,bias)) for s in (dm,dp)])
                    components.append((r,comp))
                for weight in (.1,.25,.5,.75,.9):
                    vals=[(r,float(np.array([weight,1-weight])@co)) for r,co in components]
                    early=min(vals,key=lambda x:abs(x[0]-(r0+.1)))
                    late=vals[-1]
                    if early[1]*late[1]<0 and min(abs(early[1]),abs(late[1]))>1e-7:
                        candidates.append(dict(bias=bias,theta0=theta0.tolist(),
                            shifts=[dm,dp],weights=[weight,1-weight],early=early,late=late))
    return candidates


def verify(config):
    theta0=np.array(config['theta0'])
    bias=config['bias']
    shifts=np.array(config['shifts'])
    weights=np.array(config['weights'])
    # Include the training atom with fixed positive mass; its loss contrast is
    # zero at matched r, so this does not manufacture the sign reversal.
    population_shifts=np.r_[0.,shifts]
    population_weights=np.r_[.1,.9*weights]
    checks=[]
    for theta in (theta0,theta0+np.array([.7,1.1])):
        for shift in (0.,*shifts):
            x=np.array([[0.,shift]])
            full=core.representation(embed(theta,bias),x)
            expect=np.array([[predict(theta,shift,bias),shift+C*theta[0]]])
            assert np.max(np.abs(full-expect))<1e-14
            fullj=core.representation_jacobian(embed(theta,bias),x)[:,[24,56]]
            assert np.max(np.abs(fullj-jac(theta,shift,bias)))<1e-14
        g=jac(theta,bias=bias)[0]
        d0=g/np.array([C*C+LAMBDA,LAMBDA])
        exact=d0/(1+g@d0)
        numerical=np.linalg.solve(metric(theta,bias),g)
        checks.append(float(np.max(np.abs(numerical-exact))))
        assert checks[-1]<1e-14
    rows=[]
    targets=[config['early'][0],3.]
    max_match=0.
    for h in (.002,.001,.0005):
        for target in targets:
            aa,ma=path(theta0,target,h,True,bias)
            ff,mf=path(theta0,target,h,False,bias)
            a,f=aa[-1][1],ff[-1][1]
            match=max(abs(predict(a,bias=bias)-target),abs(predict(f,bias=bias)-target))
            max_match=max(max_match,match)
            # All active tanh arguments remain positive, so tanh'' is negative.
            min_argument=min(ma,mf)+min(population_shifts)
            assert min_argument>0
            # Exact path invariant for adaptive PB, derived from constant D.
            kappa=LAMBDA*C*C/(C*C+LAMBDA)
            inv=(a[1]**2-theta0[1]**2)-(np.sinh(bias+C*a[0])**2-
                 np.sinh(bias+C*theta0[0])**2)/kappa
            rows.append(dict(step=h,target_training_logit=target,
                adaptive_parameters=a.tolist(),frozen_parameters=f.tolist(),
                matched_logit_error=match,min_active_tanh_argument=min_argument,
                adaptive_invariant_error=float(inv),
                population_risk_adaptive=risk(a,population_shifts,population_weights,bias),
                population_risk_frozen=risk(f,population_shifts,population_weights,bias),
                risk_difference=risk(a,population_shifts,population_weights,bias)-
                                risk(f,population_shifts,population_weights,bias)))
    for h in (.002,.001,.0005):
        rs=[r for r in rows if r['step']==h]
        assert rs[0]['risk_difference']*rs[1]['risk_difference']<0
    assert max_match<1e-8
    return dict(status='passed',role='constructed restricted-architecture counterexample; not a new mechanism',
        slice_trainable_original_indices=[24,56],damping=LAMBDA,c=C,
        config=config,population_shifts=population_shifts.tolist(),
        population_weights=population_weights.tolist(),population_labels=[1,1,1],
        reduction_max_error=max(checks),max_matched_logit_error=max_match,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    with threadpool_limits(1):
        candidates=search()
        if not candidates:
            result=dict(status='no_sign_reversal_in_bounded_search',candidates=[])
        else:
            result=verify(candidates[0])
            result['construction_search_candidates']=candidates
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))
