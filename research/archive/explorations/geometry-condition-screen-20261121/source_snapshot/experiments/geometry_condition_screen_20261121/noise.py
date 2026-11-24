"""Finite conditional-label audit of initial geometry feedback.

This is a local algebra/counterexample check, not a training experiment,
novel mechanism, prospective predictor, or claim about the rings family.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from pullback_feature_learning import core


def audit():
    theta = core.initialize_parameters(404027)
    x = np.array([[-.7,.2],[.4,-.6],[.8,.5],[-.2,-.8]])
    probabilities = np.array([.2,.7,.4,.8])
    damping = .1
    n = len(x)
    jac = core.representation_jacobian(theta,x)
    a = jac[::2]
    logits = core.representation(theta,x)[:,0]
    fitted = 1/(1+np.exp(-logits))
    metric = jac.T @ jac + damping*np.eye(80)
    preconditioner = np.linalg.solve(metric,np.eye(80))
    label_directions = preconditioner @ a.T/n
    mean_gradient = a.T @ (fitted-probabilities)/n
    mean_velocity = -preconditioner @ mean_gradient

    def derivative_metric(direction):
        dj = np.imag(core.representation_jacobian(theta+1e-25j*direction,x))/1e-25
        return dj.T@jac + jac.T@dj

    def quadratic(left,right):
        return -preconditioner @ derivative_metric(left) @ right

    def project(velocity, value):
        return value-velocity*((velocity@metric@value)/(velocity@metric@velocity))

    covariance_contraction = sum(probabilities[i]*(1-probabilities[i])*
        quadratic(label_directions[:,i],label_directions[:,i]) for i in range(n))
    expected_formula = quadratic(mean_velocity,mean_velocity)+covariance_contraction
    mean_raw=np.zeros(80)
    mean_projected=np.zeros(80)
    mean_progress=np.zeros(80)
    rows=[]
    total_weight=0.
    for bits in itertools.product((0.,1.),repeat=n):
        target=np.array(bits)
        weight=float(np.prod(np.where(target>0,probabilities,1-probabilities)))
        q=a.T@(fitted-target)/n
        v=-preconditioner@q
        b=quadratic(v,v)
        projected=project(v,b)
        descent=float(v@metric@v)
        progress=projected/descent**2
        mean_raw+=weight*b
        mean_projected+=weight*projected
        mean_progress+=weight*progress
        total_weight+=weight
        rows.append(dict(labels01=list(bits),probability=weight,
                         initial_loss_speed=descent,
                         raw_acceleration_norm=float(np.linalg.norm(b)),
                         projected_norm=float(np.linalg.norm(projected)),
                         progress_normalized_norm=float(np.linalg.norm(progress)),
                         loss_tangent_error=float(abs(q@projected))))
    raw_error=float(np.max(np.abs(mean_raw-expected_formula)))
    assert abs(total_weight-1)<1e-14
    assert raw_error<1e-12
    assert min(r['initial_loss_speed'] for r in rows)>1e-9
    projected_at_mean=project(mean_velocity,mean_raw)
    nonlinear_projection_error=float(np.linalg.norm(mean_projected-projected_at_mean))
    assert nonlinear_projection_error>1e-5

    # Independent central-difference derivative of the whole summed metric.
    derivative_checks=[]
    for name,v in [('mean',mean_velocity),('label0',label_directions[:,0])]:
        exact=derivative_metric(v)
        errors=[]
        for h in (.002,.001,.0005):
            plus=core.representation_jacobian(theta+h*v,x)
            minus=core.representation_jacobian(theta-h*v,x)
            central=(plus.T@plus-minus.T@minus)/(2*h)
            errors.append(float(np.linalg.norm(central-exact)))
        assert errors[-1]<1e-7
        derivative_checks.append(dict(direction=name,steps=[.002,.001,.0005],errors=errors))

    # A finite witness search establishes non-identifiability for an unrestricted
    # population law. It is not a selected successful scientific test endpoint.
    candidates=np.array([[.1,.9],[1.1,-.3],[-.8,.6]])
    ja=core.representation_jacobian(theta,candidates)[::2]
    projections=ja@mean_progress
    chosen=int(np.argmax(np.abs(projections)))
    star=candidates[chosen:chosen+1]
    astar=ja[chosen]
    fstar=float(core.representation(theta,star)[0,0])
    sstar=1/(1+np.exp(-fstar))
    k=float(astar@mean_progress)
    epsilon=min(sstar,1-sstar)/2
    common=float(mean_gradient@mean_progress)
    common_mass=min(.01,abs(k)*epsilon/(4*abs(common)+abs(k)*epsilon))
    law_rows=[]
    for label,posterior in [('lower',sstar-epsilon),('upper',sstar+epsilon)]:
        pop_gradient=common_mass*mean_gradient+(1-common_mass)*(sstar-posterior)*astar
        coefficient=float(.5*pop_gradient@mean_progress)
        law_rows.append(dict(law=label,posterior_at_unseen=posterior,
                            coefficient_of_squared_training_loss_decrease=coefficient,
                            sign='harm' if coefficient>0 else 'benefit'))
    assert law_rows[0]['coefficient_of_squared_training_loss_decrease']*law_rows[1]['coefficient_of_squared_training_loss_decrease']<0

    # Independent short flow integration in training-loss-decrease coordinates.
    # These local probes have two fixed, tiny progress values; they do not train
    # the old benchmark tasks or choose any generalization endpoint.
    support=np.vstack((x,star))
    input_weights=np.r_[np.repeat(common_mass/n,n),1-common_mass]
    laws=[np.r_[probabilities,r['posterior_at_unseen']] for r in law_rows]

    def population_risk(t,p):
        z=core.representation(t,support)[:,0]
        return float(input_weights@(np.logaddexp(0.,z)-p*z))

    def progress_field(t,target,adaptive):
        jt=core.representation_jacobian(t,x)
        z=core.representation(t,x)[:,0]
        qt=jt[::2].T@(1/(1+np.exp(-z))-target)/n
        if adaptive:
            ptq=np.linalg.solve(jt.T@jt+damping*np.eye(80),qt)
        else:
            ptq=preconditioner@qt
        return -ptq/float(qt@ptq)

    flow_checks=[]
    for progress in (1e-4,5e-5):
        contrasts=np.zeros(2)
        max_target_error=0.
        for row in rows:
            target=np.array(row['labels01'])
            original=float(np.mean(np.logaddexp(0.,logits)-target*logits))
            final=[]
            for adaptive in (False,True):
                t=theta.copy()
                h=progress/16
                for _ in range(16):
                    k1=progress_field(t,target,adaptive)
                    k2=progress_field(t+.5*h*k1,target,adaptive)
                    k3=progress_field(t+.5*h*k2,target,adaptive)
                    k4=progress_field(t+h*k3,target,adaptive)
                    t+=h*(k1+2*k2+2*k3+k4)/6
                z=core.representation(t,x)[:,0]
                loss=float(np.mean(np.logaddexp(0.,z)-target*z))
                max_target_error=max(max_target_error,abs(original-loss-progress))
                final.append(t)
            contrasts+=row['probability']*np.array([population_risk(final[1],p)-population_risk(final[0],p) for p in laws])
        assert max_target_error<1e-12
        assert contrasts[0]<0<contrasts[1]
        flow_checks.append(dict(training_loss_decrease=progress,rk4_steps=16,
                               maximum_target_error=max_target_error,
                               expected_risk_contrasts=contrasts.tolist(),
                               contrasts_divided_by_progress_squared=(contrasts/progress**2).tolist()))
    return dict(status='passed',scope='Actual full 80-parameter initial local calculation only; no training sweep.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_sha256=hashlib.sha256((REPO/'src/pullback_feature_learning/core.py').read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),initialization_seed=404027,
        x=x.tolist(),probabilities=probabilities.tolist(),damping=damping,
        conditional_label_patterns=rows,
        raw_noise_identity=dict(max_absolute_error=raw_error,
            mean_raw_norm=float(np.linalg.norm(mean_raw)),
            zero_noise_soft_target_norm=float(np.linalg.norm(quadratic(mean_velocity,mean_velocity))),
            noise_contraction_norm=float(np.linalg.norm(covariance_contraction))),
        projection_caveat=dict(mean_projected_norm=float(np.linalg.norm(mean_projected)),
            project_mean_raw_at_mean_velocity_norm=float(np.linalg.norm(projected_at_mean)),
            absolute_difference=nonlinear_projection_error,
            progress_normalized_mean_norm=float(np.linalg.norm(mean_progress))),
        derivative_checks=derivative_checks,
        independent_short_flow_checks=flow_checks,
        nonidentifiability_witness=dict(candidate_inputs=candidates.tolist(),selected_index=chosen,
            selected_x=star[0].tolist(),selected_logit=fstar,selected_prediction_probability=sstar,
            predictive_projection=k,common_training_support_total_mass=common_mass,
            unseen_mass=1-common_mass,common_support_posteriors=probabilities.tolist(),laws=law_rows,
            meaning='Same conditional law of every training-only label-replica statistic, opposite expected local equal-loss risk signs; unrestricted populations only.'),
        decision='Reject universal training-only replica-variability/noise-harm sign rule. Keep exact covariance identity as routine bookkeeping, not innovation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(1):
        result=audit()
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','raw_noise_identity','projection_caveat','nonidentifiability_witness','decision']},indent=2))
