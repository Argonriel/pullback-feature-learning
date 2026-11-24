"""An exact residual-network slice: commuting metrics do not imply one clock.

This is a deterministic counterexample to a universal condition, not a study
of the full 80-parameter flow or evidence of a new learning mechanism.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from pullback_feature_learning import core

X=np.eye(2)
INITIAL=np.array([.15,.35])
LAMBDA=.1
TARGETS=(.02,.06,.12,.18)


def embed(w):
    theta=np.zeros(80,dtype=np.result_type(w,float))
    theta[0],theta[3]=w
    theta[16:18]=2.
    return theta


def logits(w):
    return np.array([1.,0.])+np.tanh(w)


def losses(w):
    return np.logaddexp(0.,-logits(w))


def metric(w):
    return np.diag((1-np.tanh(w)**2)**2+LAMBDA)


def gradient(w):
    f=logits(w)
    return -.5*np.exp(-np.logaddexp(0.,f))*(1-np.tanh(w)**2)


def field(w,adaptive):
    q=gradient(w)
    p=np.linalg.solve(metric(w if adaptive else INITIAL),q)
    return -p/(q@p)


def rk4(w,ds,adaptive):
    k1=field(w,adaptive)
    k2=field(w+ds*k1/2,adaptive)
    k3=field(w+ds*k2/2,adaptive)
    k4=field(w+ds*k3,adaptive)
    return w+ds*(k1+2*k2+2*k3+k4)/6


def run(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    initial_loss=float(losses(INITIAL).mean())
    rows=[]
    checks=dict(max_slice_forward_error=0.,max_slice_jacobian_error=0.,
                max_loss_clock_error=0.,max_metric_commutator=0.)
    for ds in (2e-4,1e-4):
        states={True:INITIAL.copy(),False:INITIAL.copy()}
        s=0.
        for target in TARGETS:
            while s < target-1e-14:
                step=min(ds,target-s)
                for adaptive in states:
                    states[adaptive]=rk4(states[adaptive],step,adaptive)
                s+=step
            a,f=states[True],states[False]
            la,lf=losses(a),losses(f)
            for w in (a,f):
                z=core.representation(embed(w),X)[:,0]
                j=core.representation_jacobian(embed(w),X)[:,[0,3]]
                expected=np.zeros((4,2))
                expected[0,0],expected[2,1]=1-np.tanh(w)**2
                checks['max_slice_forward_error']=max(checks['max_slice_forward_error'],float(np.max(np.abs(z-logits(w)))))
                checks['max_slice_jacobian_error']=max(checks['max_slice_jacobian_error'],float(np.max(np.abs(j-expected))))
                checks['max_loss_clock_error']=max(checks['max_loss_clock_error'],abs(float(losses(w).mean())-(initial_loss-target)))
                m,m0=metric(w),metric(INITIAL)
                checks['max_metric_commutator']=max(checks['max_metric_commutator'],float(np.linalg.norm(m@m0-m0@m)))
            delta=la-lf
            qp=gradient(a)
            va=-np.linalg.solve(metric(a),qp)
            vf=-np.linalg.solve(metric(INITIAL),qp)
            cos=float(va@vf/(np.linalg.norm(va)*np.linalg.norm(vf)))
            rows.append(dict(ds=ds,loss_drop=target,adaptive=a.tolist(),frozen=f.tolist(),
                adaptive_train_loss=float(la.mean()),frozen_train_loss=float(lf.mean()),
                adaptive_minus_frozen_per_input_loss=delta.tolist(),
                risk_change_population_weight_08=float(np.array([.8,.2])@delta),
                risk_change_population_weight_02=float(np.array([.2,.8])@delta),
                predictor_rms_difference=float(np.sqrt(np.mean((logits(a)-logits(f))**2))),
                same_state_field_cosine=cos))
    refinements=[]
    for target in TARGETS:
        coarse,fine=[r for r in rows if r['loss_drop']==target]
        error=max(abs(coarse[k]-fine[k]) for k in ('risk_change_population_weight_08','risk_change_population_weight_02'))
        refinements.append(dict(loss_drop=target,max_risk_refinement_error=error))
        assert error<1e-10
        assert fine['risk_change_population_weight_08']*fine['risk_change_population_weight_02']<0
        assert fine['predictor_rms_difference']>1e-5
        assert fine['same_state_field_cosine']<1-1e-8
    assert checks['max_slice_forward_error']<1e-13
    assert checks['max_slice_jacobian_error']<1e-13
    assert checks['max_loss_clock_error']<1e-10
    assert checks['max_metric_commutator']==0
    result=dict(status='passed',role='counterexample_to_universal_condition_not_novelty',
        restricted_parameters=[0,3],fixed_parameters='All other 78 parameters fixed; B1[0,0:2]=2.',
        initial=INITIAL.tolist(),damping=LAMBDA,train_inputs=X.tolist(),train_labels=[1,1],
        initial_loss=initial_loss,training_clock='s=L(initial)-L(theta), integrated directly',
        population_laws='Same two inputs and deterministic +1 labels; weights (0.8,0.2) or (0.2,0.8).',
        sampling_note='One observation of each input has probability 0.32 under either iid law. No deliberate distribution shift is required.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,
        refinements=refinements,rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(checks=checks,primary=rows[-1],refinements=refinements),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    with threadpool_limits(1):
        run(args.output)
