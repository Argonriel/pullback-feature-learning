"""Exact finite-batch PB screening; no training or novelty claim.

Linear logistic model f_theta(x)=theta*x, representation (f_theta(x),0).
We use an average geometry metric and fixed lambda; equivalently the summed
metric uses lambda_sum=batch_size*lambda and eta_sum=batch_size*eta.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

P=.01
K=200.
LAMBDA=.1
BATCHES=(1,2,8,32,128,512,1024)


def batch_stats(m):
    k=np.arange(m+1,dtype=float)
    logp=np.array([math.lgamma(m+1)-math.lgamma(int(i)+1)-math.lgamma(m-int(i)+1)
                   +i*math.log(P)+(m-i)*math.log1p(-P) for i in k])
    probs=np.exp(logp)
    # Re-normalize only for accumulated lgamma rounding; record discrepancy.
    mass_error=float(abs(probs.sum()-1))
    probs/=probs.sum()
    numerator=((m-k)-K*k)/m
    denominator=((m-k)+K*K*k)/m+LAMBDA
    same=.5*numerator/denominator
    independent_mean=.5*float(probs@numerator)*float(probs@(1/denominator))
    independent_second=.25*float(probs@(numerator*numerator))*float(probs@(1/denominator**2))
    same_mean=float(probs@same)
    same_second=float(probs@(same*same))
    grad_population=-.5*((1-P)-P*K)
    return dict(batch_size=m,pmf_mass_error=mass_error,
        same_batch_mean_update=same_mean,independent_metric_mean_update=independent_mean,
        same_batch_update_variance=same_second-same_mean**2,
        independent_metric_update_variance=independent_second-independent_mean**2,
        same_batch_population_loss_derivative=grad_population*same_mean,
        independent_metric_population_loss_derivative=grad_population*independent_mean,
        no_rare_probability=(1-P)**m,
        full_metric_direction=-grad_population/((1-P)+P*K*K+LAMBDA))


def verify():
    rows=[batch_stats(m) for m in BATCHES]
    assert rows[0]['same_batch_population_loss_derivative']>0
    assert all(r['independent_metric_population_loss_derivative']<0 for r in rows)
    assert rows[-1]['same_batch_population_loss_derivative']<0
    # Direct four-event independent-batch enumeration checks moment formulas.
    probabilities=np.array([1-P,P])
    x=np.array([1.,K])
    y=np.array([1.,-1.])
    u=np.array([[.5*y[i]*x[i]/(x[j]**2+LAMBDA) for j in range(2)] for i in range(2)])
    weights=np.outer(probabilities,probabilities)
    mean=float(np.sum(weights*u))
    variance=float(np.sum(weights*(u-mean)**2))
    assert abs(mean-rows[0]['independent_metric_mean_update'])<1e-12
    assert abs(variance-rows[0]['independent_metric_update_variance'])<1e-10
    # Sherman-Morrison singleton norm bound for logistic residual <=1.
    radii=np.logspace(-6,6,601)
    max_ratio=float(np.max(radii/(radii*radii+LAMBDA)))
    assert max_ratio<=1/(2*np.sqrt(LAMBDA))+1e-13
    # Damping sum/average invariance, not changing damping with batch size.
    normalization_error=0.
    for m in BATCHES:
        for k in (0,m//2,m):
            grad=-.5*((m-k)-K*k)/m
            average=-grad/(((m-k)+K*K*k)/m+LAMBDA)
            summed=m*(-grad/((m-k)+K*K*k+m*LAMBDA))
            normalization_error=max(normalization_error,abs(average-summed))
    assert normalization_error<1e-13
    # A diagnostic exhaustive, predeclared batch grid. This is not a selected
    # favorable scientific endpoint; every value is retained.
    transition_rows=[batch_stats(m) for m in range(1,1025)]
    first=next(r['batch_size'] for r in transition_rows if r['same_batch_mean_update']<0)
    result=dict(status='passed',role='Exact counterexample/reduction screen, not innovation',
        model='Linear logistic f(theta,x)=theta*x; two-coordinate representation (f,0)',
        data=[dict(probability=1-P,x=1.,y=1),dict(probability=P,x=K,y=-1)],
        theta=0.,average_metric_damping=LAMBDA,
        population_gradient=-.5*((1-P)-P*K),
        same_batch_singleton_equivalent_objective='E[log(1+exp(-Y*theta*X))/(X^2+lambda)]',
        direct_independent_mean_error=abs(mean-rows[0]['independent_metric_mean_update']),
        direct_independent_variance_error=abs(variance-rows[0]['independent_metric_update_variance']),
        normalization_max_error=normalization_error,
        singleton_global_direction_bound=1/(2*np.sqrt(LAMBDA)),
        selected_rows=rows,all_batch_sizes_1_through_1024=transition_rows,
        first_negative_same_batch_mean_in_grid=first,
        interpretation='The finite-batch direction reversal is ordinary self-normalized sample reweighting. Cross-fitting removes coupling but can increase variance; no learned geometry or feature learning is needed.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    value=verify()
    out.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('status','selected_rows','first_negative_same_batch_mean_in_grid')},indent=2))
