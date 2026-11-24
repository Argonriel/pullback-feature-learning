"""Verify inversion-parity reductions for the actual two-block residual map.

These are algebraic mechanism-screening checks, not a new training experiment
or a generalization predictor. The synthetic quadrature inputs are unrelated
to the frozen empirical cohorts. Output files must be new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from pullback_feature_learning import core

BIAS = np.r_[32:40, 72:80]
WEIGHT = np.setdiff1d(np.arange(80), BIAS)
RESIDUAL = np.r_[16:32, 56:72]


def paired_grid():
    # Full central pairs at two radii, with an even, monotone posterior.
    angle = (np.arange(32) + .17) * np.pi / 32
    unit = np.column_stack((np.cos(angle), np.sin(angle)))
    half = np.vstack((.5 * unit, 1.2 * unit))
    x = np.vstack((half, -half))
    p_half = np.r_[np.full(32, .1), np.full(32, .9)]
    return x, np.r_[p_half, p_half]


def risk(theta, x, p):
    f = core.representation(theta, x)[:, 0]
    return float(np.mean(np.logaddexp(0., f) - p * f))


def objects(theta, x, p, damping=.1):
    z = core.representation(theta, x)
    jac = core.representation_jacobian(theta, x)
    residue = (np.exp(-np.logaddexp(0., -z[:, 0])) - p) / len(x)
    q = jac[::2].T @ residue
    metric = jac.T @ jac + damping * np.eye(80)
    u = np.linalg.solve(metric, -q)
    return z, jac, q, metric, u


def verify():
    x, p = paired_grid()
    n = len(x) // 2
    base = core.initialize_parameters(190733)
    lam = .1
    rows = []
    worst = dict(representation_parity=0., weight_jacobian_parity=0.,
                 bias_jacobian_parity=0., metric_cross_block=0.,
                 weight_gradient_label_dependence=0., bias_gradient_label_reversal=0.,
                 restricted_risk_label_dependence=0., block_solve_error=0.)
    for epsilon in (1., .3, .1, .03, .01, .003, .001, .0003, .0001):
        theta = base.copy()
        theta[RESIDUAL] *= epsilon
        z, j, q, m, u = objects(theta, x, p, lam)
        jj = j.reshape(len(x), 2, 80)
        _, _, q_reversed, _, _ = objects(theta, x, 1-p, lam)
        weight_gradient_only = j[::2, :][:, WEIGHT].T @ (
            (np.exp(-np.logaddexp(0., -z[:,0]))-.5)/len(x))
        bias_gradient_only = -j[::2, :][:, BIAS].T @ ((p-.5)/len(x))
        u_bias = -np.linalg.solve(m[np.ix_(BIAS,BIAS)],q[BIAS])
        checks = dict(
            representation_parity=float(np.max(np.abs(z[:n]+z[n:]))),
            weight_jacobian_parity=float(np.max(np.abs(jj[:n,:,WEIGHT]+jj[n:,:,WEIGHT]))),
            bias_jacobian_parity=float(np.max(np.abs(jj[:n,:,BIAS]-jj[n:,:,BIAS]))),
            metric_cross_block=float(np.max(np.abs(m[np.ix_(WEIGHT,BIAS)]))),
            weight_gradient_label_dependence=float(np.max(np.abs(q[WEIGHT]-q_reversed[WEIGHT]))),
            bias_gradient_label_reversal=float(np.max(np.abs(q[BIAS]+q_reversed[BIAS]))),
            restricted_risk_label_dependence=abs(risk(theta,x,p)-risk(theta,x,1-p)),
            block_solve_error=float(np.max(np.abs(u[BIAS]-u_bias))))
        for key,val in checks.items():
            worst[key]=max(worst[key],val)
        assert np.max(np.abs(q[WEIGHT]-weight_gradient_only)) < 1e-13
        assert np.max(np.abs(q[BIAS]-bias_gradient_only)) < 1e-13
        a_bias = j[::2,:][:,BIAS]
        even_pb = a_bias @ u[BIAS]
        even_gd = -a_bias @ q[BIAS]
        rows.append(dict(epsilon=epsilon,bias_jacobian_norm=float(np.linalg.norm(j[:,BIAS])),
                         bias_gradient_norm=float(np.linalg.norm(q[BIAS])),
                         bias_pb_parameter_velocity_norm=float(np.linalg.norm(u[BIAS])),
                         even_pb_predictive_velocity_norm=float(np.linalg.norm(even_pb)),
                         even_gd_predictive_velocity_norm=float(np.linalg.norm(even_gd)),
                         pb_vs_gd_even_velocity_ratio=float(np.linalg.norm(even_pb)/np.linalg.norm(even_gd)),
                         even_pb_velocity_over_epsilon_squared=float(np.linalg.norm(even_pb)/epsilon**2)))
    assert max(worst.values()) < 2e-12, worst
    small = rows[-3:]
    for field, order in (('bias_jacobian_norm',1),('bias_gradient_norm',1),
                         ('bias_pb_parameter_velocity_norm',1),('even_pb_predictive_velocity_norm',2)):
        slope=float(np.polyfit(np.log([r['epsilon'] for r in small]),
                               np.log([r[field] for r in small]),1)[0])
        assert abs(slope-order)<.005, (field,slope)
    # At zero residual weights the entire tangent representation is odd, but
    # one GD step creates residual weights and the next can move biases.
    zero = base.copy()
    zero[RESIDUAL] = 0.
    z0,j0,q0,m0,u0 = objects(zero,x,p,lam)
    assert np.array_equal(j0[:,BIAS],np.zeros((2*len(x),len(BIAS))))
    assert np.linalg.norm(q0[RESIDUAL]) > 1e-3
    assert np.linalg.norm(q0[BIAS]) == 0.
    rng=np.random.default_rng(77441)
    tangent=(z0.ravel()+j0@rng.normal(size=80)).reshape(-1,2)
    assert np.max(np.abs(tangent[:n]+tangent[n:]))<1e-13
    delayed=[]
    for eta in (.2,.1,.05,.025):
        first=zero-eta*q0
        _,_,q1,_,_=objects(first,x,p,lam)
        second=first-eta*q1
        f2=core.representation(second,x)[:,0]
        even=(f2[:n]+f2[n:])/2
        delayed.append(dict(eta=eta,first_bias_gradient_norm=float(np.linalg.norm(q1[BIAS])),
                            second_bias_norm=float(np.linalg.norm(second[BIAS])),
                            second_even_prediction_norm=float(np.linalg.norm(even)),
                            even_over_eta_cubed=float(np.linalg.norm(even)/eta**3)))
    delayed_slope=float(np.polyfit(np.log([r['eta'] for r in delayed[-3:]]),
                                   np.log([r['second_even_prediction_norm'] for r in delayed[-3:]]),1)[0])
    assert abs(delayed_slope-3)<.08, delayed_slope
    # Exact deterministic special case: one hidden unit, one block, B_readout>0.
    # At zero bias the decreasing even feature sech^2(a.x) has negative
    # covariance with this monotone radial target; both GD and PB start in
    # that same signal-fitting direction.
    a=np.array([.8,-.4])
    t=x@a
    phi=1-np.tanh(t)**2
    covariance=float(np.mean((p-.5)*phi))
    assert covariance<0
    return dict(status='passed',role='algebraic_screen_only_no_training_sweep_or_novelty_claim',
                seed=190733,n_inputs=len(x),damping_sum_metric=lam,
                exact_parity_and_gradient_max_errors=worst,epsilon_scaling=rows,
                zero_residual=dict(initial_bias_jacobian_norm=float(np.linalg.norm(j0[:,BIAS])),
                    initial_residual_gradient_norm=float(np.linalg.norm(q0[RESIDUAL])),
                    two_step_checks=delayed,estimated_even_prediction_eta_power=delayed_slope),
                single_unit_monotone_radial_covariance=covariance,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                core_sha256=hashlib.sha256((REPO/'src/pullback_feature_learning/core.py').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(1):
        result=verify()
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],
        max_identity_error=max(result['exact_parity_and_gradient_max_errors'].values()),
        delayed_power=result['zero_residual']['estimated_even_prediction_eta_power'],
        asymptotic_pb_gd_ratio=result['epsilon_scaling'][-1]['pb_vs_gd_even_velocity_ratio'])))
