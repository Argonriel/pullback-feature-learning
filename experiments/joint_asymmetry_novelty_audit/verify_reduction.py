"""Independent finite-dimensional checks for a novelty reduction, not training.

Enumerates vertices of the two-simplex TV polytope directly from halfspaces.
No formula-specific optimizer or external LP package is used.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np


def logsumexp(z):
    top = np.max(z, axis=-1, keepdims=True)
    return np.squeeze(top, axis=-1) + np.log(np.exp(z - top).sum(axis=-1))


def polytope_vertices(classes, epsilon):
    """Use TV(p,q)=max_A (p(A)-q(A)); eliminate last simplex coordinates."""
    dimension = 2 * (classes - 1)
    p_map = np.zeros((classes, dimension))
    q_map = np.zeros((classes, dimension))
    p_map[:-1, :classes - 1] = np.eye(classes - 1)
    q_map[:-1, classes - 1:] = np.eye(classes - 1)
    p_map[-1, :classes - 1] = -1
    q_map[-1, classes - 1:] = -1
    offset = np.zeros(classes)
    offset[-1] = 1
    matrix = [-row for row in p_map] + [-row for row in q_map]
    rhs = list(offset) + list(offset)
    for subset in itertools.product((0, 1), repeat=classes):
        if sum(subset) in (0, classes):
            continue
        matrix.append(np.asarray(subset) @ (p_map - q_map))
        rhs.append(epsilon)
    matrix, rhs = np.asarray(matrix), np.asarray(rhs)
    combinations = np.asarray(list(itertools.combinations(range(len(rhs)), dimension)))
    active = matrix[combinations]
    active_rhs = rhs[combinations]
    nonsingular = np.abs(np.linalg.det(active)) > 1e-10
    solutions = np.linalg.solve(active[nonsingular], active_rhs[nonsingular, :, None])[..., 0]
    feasible = np.all(solutions @ matrix.T <= rhs + 1e-10, axis=1)
    solutions = np.unique(np.round(solutions[feasible], 13), axis=0)
    if not len(solutions):
        raise AssertionError("No vertices found")
    p = solutions @ p_map.T + offset
    q = solutions @ q_map.T + offset
    return p, q


def support_checks(rng):
    rows = []
    for classes in (2, 3):
        for epsilon in (0., .01, .1, .4, .8, 1.):
            p, q = polytope_vertices(classes, epsilon)
            for replicate in range(25):
                r = (-1., 0., 1.)[replicate] if replicate < 3 else rng.uniform(-1, 1)
                z, zt = rng.normal(size=(2, classes)) * rng.uniform(.01, 5)
                d = (z - zt) / 2
                mid = (d.max() + d.min()) / 2
                span = np.ptp(d)
                lp_objectives = ((1+r) * p - (1-r) * q) @ d / 2
                radius = (abs(r) + (1-abs(r)) * epsilon) * span / 2
                lo, hi = r*mid-radius, r*mid+radius
                j = (logsumexp(z) + logsumexp(zt))/2 - logsumexp((z+zt)/2)
                k = (logsumexp(z) - logsumexp(zt))/2
                t = k-mid
                gain_lp = j + r*k - lp_objectives
                gain_lo, gain_hi = j+r*t-radius, j+r*t+radius
                rows.append(dict(classes=classes, epsilon=epsilon, r=float(r),
                                 z=z.tolist(), zt=zt.tolist(), vertices=len(p),
                                 support_min=float(lp_objectives.min()),
                                 support_max=float(lp_objectives.max()),
                                 formula_min=float(lo), formula_max=float(hi),
                                 gain_min=float(gain_lp.min()), gain_max=float(gain_lp.max()),
                                 formula_gain_min=float(gain_lo), formula_gain_max=float(gain_hi)))
    error = max(max(abs(row['support_min']-row['formula_min']),
                    abs(row['support_max']-row['formula_max']),
                    abs(row['gain_min']-row['formula_gain_min']),
                    abs(row['gain_max']-row['formula_gain_max'])) for row in rows)
    assert error < 1e-9
    return rows, error


def joint_identity_checks(rng):
    """Compare raw risks with a single joint-distribution signed-measure identity."""
    rows = []
    for case in range(250):
        orbits, classes = 4, 3
        joint = rng.dirichlet(np.ones(orbits * 2 * classes)).reshape(orbits, 2, classes)
        z = rng.normal(size=(orbits, 2, classes))*3
        # An orbit-even selector, independent of labels.
        mask = rng.uniform(size=(orbits, 1, 1))
        averaged = np.repeat(z.mean(axis=1, keepdims=True), 2, axis=1)
        raw_loss = logsumexp(z)[..., None] - z
        avg_loss = logsumexp(averaged)[..., None] - averaged
        gains = mask*(raw_loss-avg_loss)
        direct = float(np.sum(joint*gains))
        swapped_joint = joint[:, ::-1, :]
        symmetric_joint = (joint+swapped_joint)/2
        jensen_gain = float(np.sum(symmetric_joint*gains))
        signed_defect = float(np.sum((joint-swapped_joint)*gains)/2)
        marginal = joint.sum(axis=2)
        orbit_mass = marginal.sum(axis=1)
        r = (marginal[:, 0]-marginal[:, 1])/orbit_mass
        p, q = joint[:, 0]/marginal[:, 0, None], joint[:, 1]/marginal[:, 1, None]
        d = (z[:, 0]-z[:, 1])/2
        j = (logsumexp(z[:, 0])+logsumexp(z[:, 1]))/2-logsumexp(z.mean(axis=1))
        k = (logsumexp(z[:, 0])-logsumexp(z[:, 1]))/2
        decomposed = float(np.sum(orbit_mass*mask[:, 0, 0]*(j+r*k-
            np.sum(((1+r[:, None])*p-(1-r[:, None])*q)*d, axis=1)/2)))
        tv_joint = float(np.sum(abs(joint-swapped_joint))/2)
        epsilon = np.sum(abs(p-q), axis=1)/2
        tv_bound = float(np.sum(orbit_mass*(abs(r)+(1-abs(r))*epsilon)))
        assert tv_joint <= tv_bound + 1e-12
        rows.append(dict(case=case, direct_gain=direct, joint_identity=jensen_gain+signed_defect,
                         posterior_decomposition=decomposed, joint_tv=tv_joint, tv_upper=tv_bound))
    error = max(max(abs(row['direct_gain']-row['joint_identity']),
                    abs(row['direct_gain']-row['posterior_decomposition'])) for row in rows)
    assert error < 1e-11
    return rows, error


def shared_constraint_counterexample():
    """Two equal-mass orbits share both posteriors; opposite d cancels the defect."""
    epsilon = .4
    z = np.array([[1., 0.], [-1., 0.]])
    zt = -z
    p, q = polytope_vertices(2, epsilon)
    d = (z-zt)/2
    j = (logsumexp(z)+logsumexp(zt))/2-logsumexp((z+zt)/2)
    # r=0 in each orbit; the same p and q apply across the two orbits.
    gain_by_vertex = j.mean() - ((p-q) @ d.T / 2).mean(axis=1)
    rectangular_lower = float(np.mean(j-epsilon*np.ptp(d, axis=1)/2))
    constrained_min = float(gain_by_vertex.min())
    assert rectangular_lower < 0 < constrained_min
    assert abs(constrained_min-j.mean()) < 1e-12
    return dict(epsilon=epsilon, r=0, shared_posteriors=True,
                rectangular_lower=rectangular_lower, actual_feasible_min=constrained_min,
                actual_feasible_max=float(gain_by_vertex.max()), gap=constrained_min-rectangular_lower,
                interpretation='Pointwise sharpness does not imply sharpness under shared posterior constraints.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2026112107)
    support, support_error = support_checks(rng)
    joint, joint_error = joint_identity_checks(rng)
    counterexample = shared_constraint_counterexample()
    payload = dict(seed=2026112107, support_cases=support, joint_cases=joint,
                   shared_constraint_counterexample=counterexample)
    (args.output/'audit_data.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    summary = dict(status='PASS', python=platform.python_version(), numpy=np.__version__,
                   support_cases=len(support), support_max_error=support_error,
                   joint_cases=len(joint), joint_max_error=joint_error,
                   shared_constraint_counterexample=counterexample,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   meaning='Numerical verification of reductions; not a proof of novelty or a training experiment.')
    (args.output/'audit_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
