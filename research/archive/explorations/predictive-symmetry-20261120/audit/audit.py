"""Independent numerical and provenance audit of the two frozen experiments.

This auditor never changes investigator code or results. Generic math checks are
independent synthetic diagnostics; investigator-result checks are appended as
frozen files become available.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
WORKSPACE = STUDY.parents[1]
SOURCE = STUDY / 'source'
if not SOURCE.exists():
    SOURCE = WORKSPACE / 'pullback-feature-learning' / 'src'
sys.path.insert(0, str(SOURCE))
from pullback_feature_learning import core


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def loss(theta, x, y):
    return np.logaddexp(0., -y * core.representation(theta, x)[:, 0]).mean()


def simple_root(theta, x, y, v, delta):
    """Independent dense initial scan followed by bisection on a fixed ray."""
    base = loss(theta, x, y)
    q = -core.parameter_gradient(theta, x, y) @ v
    high = 1.5 * delta / q
    low = 0.
    for _ in range(30):
        if base - loss(theta + high * v, x, y) >= delta:
            break
        low, high = high, high * 1.5
    else:
        return None
    for _ in range(70):
        middle = (low + high) / 2
        if base - loss(theta + middle * v, x, y) < delta:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def qvalues(theta, x, p, radii, angles, v=None):
    logits = core.representation(theta, x)[:, 0].reshape(radii, angles)
    eta = p.reshape(radii, angles)
    mean = logits.mean(1)
    risk = (np.logaddexp(0., logits) - eta * logits).mean()
    fit = (np.logaddexp(0., mean) - eta[:, 0] * mean).mean()
    result = np.array([risk, fit, risk - fit])
    if v is None:
        return result
    w = (core.representation_jacobian(theta, x)[::2] @ v).reshape(radii, angles)
    meanw = w.mean(1)
    dr = ((sigmoid(logits) - eta) * w).mean()
    df = ((sigmoid(mean) - eta[:, 0]) * meanw).mean()
    return result, np.array([dr, df, dr - df])


def generic_math_checks():
    rng = np.random.default_rng(944001)
    radii, angles = 12, 96
    r = np.linspace(.25, 1.4, radii)
    a = 2 * np.pi * np.arange(angles) / angles
    xx = np.stack([r[:, None]*np.cos(a), r[:, None]*np.sin(a)], axis=-1).reshape(-1, 2)
    pp = np.repeat(sigmoid(12*(r-.8)), angles)
    train_r = rng.uniform(.25, 1.4, 256)
    train_a = rng.uniform(0, 2*np.pi, 256)
    train_x = train_r[:, None] * np.c_[np.cos(train_a), np.sin(train_a)]
    train_y = 2*(rng.random(256) < sigmoid(12*(train_r-.8)))-1
    rows = []
    for seed in (944101, 944102, 944103):
        theta = core.initialize_parameters(seed)
        for method in ("gd", "pullback"):
            v, _ = core.direction(theta, train_x, train_y, method, .1)
            # Normalize ray parameter only; endpoints and normalized slopes are unchanged.
            v = v / np.linalg.norm(v)
            q0, deriv = qvalues(theta, xx, pp, radii, angles, v)
            h = 1.e-5
            fd = (qvalues(theta+h*v,xx,pp,radii,angles)-qvalues(theta-h*v,xx,pp,radii,angles))/(2*h)
            dl = core.parameter_gradient(theta,train_x,train_y) @ v
            for fraction in (.001, .02):
                delta = fraction * loss(theta, train_x, train_y)
                alpha = simple_root(theta,train_x,train_y,v,delta)
                if alpha is None:
                    rows.append(dict(seed=seed,method=method,fraction=fraction,status="root_not_bracketed",
                        directional_derivative_max_abs_error=float(np.max(abs(deriv-fd)))))
                    continue
                updated = theta + alpha*v
                q1, d1 = qvalues(updated,xx,pp,radii,angles,v)
                exact = -d1 / (core.parameter_gradient(updated,train_x,train_y)@v)
                dh = delta * .0001
                ap = simple_root(theta,train_x,train_y,v,delta+dh)
                am = simple_root(theta,train_x,train_y,v,delta-dh)
                clockfd=(qvalues(theta+ap*v,xx,pp,radii,angles)-qvalues(theta+am*v,xx,pp,radii,angles))/(2*dh)
                rows.append(dict(seed=seed,method=method,fraction=fraction,status="matched",alpha=float(alpha),
                    loss_match_error=float(abs(loss(theta,train_x,train_y)-loss(updated,train_x,train_y)-delta)),
                    directional_derivative_max_abs_error=float(np.max(abs(deriv-fd))),
                    actual_clock_derivative_max_abs_error=float(np.max(abs(exact-clockfd))),
                    leading_prediction_max_abs_error=float(np.max(abs(q1-q0-delta*deriv/(-dl))))))
    # Exact Fourier transfer for two-view rotation: no investigator code used.
    phi = 2*np.pi*np.arange(768)/768
    coefficients = {1: 1.3, 2: -.8, 3: .4, 4: .6, 7: -.2}
    f = lambda t: .37 + sum(amp*np.cos(k*t+.13*k) for k,amp in coefficients.items())
    fourier_errors=[]
    gains=[]
    for shift in (np.pi/6,np.pi/3,np.pi/2,np.pi):
        pmean = (f(phi)+f(phi+shift))/2
        expected = .37+sum(amp*np.cos(k*shift/2)*np.cos(k*phi+.13*k+k*shift/2) for k,amp in coefficients.items())
        fourier_errors.append(np.max(abs(pmean-expected)))
        eta=.73
        actual=(np.logaddexp(0,f(phi))-eta*f(phi)).mean()-(np.logaddexp(0,pmean)-eta*pmean).mean()
        jensen=(.5*(np.logaddexp(0,f(phi))+np.logaddexp(0,f(phi+shift)))-np.logaddexp(0,pmean)).mean()
        gains.append(abs(actual-jensen))
    # Same unlabeled law, opposite labeled gains for an orientation-sensitive gate.
    values=[]
    for eta in (0.,1.):
        values.append(float(.5*(np.logaddexp(0.,2.)-2*eta-np.log(2.))))
    return dict(kind="Synthetic independent mathematical checks, not experimental outcomes",
                actual_loss_matched_rays=rows,
                root_not_bracketed=sum(r["status"]!="matched" for r in rows),
                max_loss_match_error=max(r["loss_match_error"] for r in rows if r["status"]=="matched"),
                max_directional_derivative_absolute_error=max(r["directional_derivative_max_abs_error"] for r in rows),
                max_actual_clock_derivative_absolute_error=max(r["actual_clock_derivative_max_abs_error"] for r in rows if r["status"]=="matched"),
                max_two_view_fourier_transfer_error=float(max(fourier_errors)),
                max_invariant_unlabeled_gain_identity_error=float(max(gains)),
                noninvariant_gate_same_unlabeled_law_opposite_gains=values)


def main():
    from experiment_checks import audit_direction1, audit_direction2, audit_iid_supplement
    from followup_checks import audit_second_order, audit_bridge
    with threadpool_limits(1):
        result = dict(auditor="Independent numerical recomputation; source hashes identify checked revisions",
                      audit_source_sha256={p.name:digest(p) for p in (HERE/'audit.py',HERE/'experiment_checks.py',HERE/'followup_checks.py')},
                      mathematics=generic_math_checks(),direction1=audit_direction1(STUDY),direction2=audit_direction2(STUDY),
                      iid_supplement=audit_iid_supplement(STUDY),second_order_followup=audit_second_order(STUDY),
                      balanced_control=audit_second_order(STUDY,balanced=True),pb100_bridge=audit_bridge(STUDY),
                      pb500_bridge=audit_bridge(STUDY,late=True))
    (HERE/"audit.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
