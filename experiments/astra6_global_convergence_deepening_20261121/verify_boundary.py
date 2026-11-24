"""One coupled-center PSD-boundary check; no training data or old result writes.

Requires Python 3 and NumPy only. Outputs a fresh research record directory.
The theorem is not inferred from finite-horizon numerical integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectral_function(x, function):
    values, vectors = np.linalg.eigh((x + x.T) / 2)
    return (vectors * function(values)) @ vectors.T


def make_problem():
    basis = []
    for offset in (0, 3):
        diagonal = np.zeros((5, 5))
        diagonal[offset, offset] = 1 / np.sqrt(2)
        diagonal[offset + 1, offset + 1] = -1 / np.sqrt(2)
        off = np.zeros((5, 5))
        off[offset, offset + 1] = off[offset + 1, offset] = 1 / np.sqrt(2)
        basis.extend([diagonal, off])
    central = np.diag([0.5, 0.5, 1, 0, 0]) / np.sqrt(1.5)
    basis.append(central)
    basis = np.array(basis)
    x0 = np.zeros((5, 5))
    x0[:2, :2] = [[1.4, 0.15], [0.15, 0.9]]
    x0[2, 2] = 0.8
    x0[3:, 3:] = [[1, 0.2], [0.2, 1.3]]
    target_traceless = np.array([[0.25, -0.35], [-0.35, -0.25]])
    damping = 2.0
    a = damping / 2
    constant = np.trace(x0[3:, 3:]) + a * np.linalg.slogdet(x0[3:, 3:])[1]
    radius = np.linalg.norm(target_traceless, ord=2)
    lower, upper = radius, max(radius + 1, constant + 1)
    for _ in range(120):
        middle = (lower + upper) / 2
        value = 2 * middle + a * np.log(middle * middle - radius * radius)
        if value < constant:
            lower = middle
        else:
            upper = middle
    center = (lower + upper) / 2
    endpoint = np.zeros((5, 5))
    endpoint[:2, :2] = [[0.8, 0.4], [0.4, 0.2]]
    endpoint[3:, 3:] = center * np.eye(2) + target_traceless
    target = np.einsum('aij,ij->a', basis, endpoint)
    triangular = np.array([[1, 0, 0, 0, 0], [.3, 1.1, 0, 0, 0],
                           [.25, -.15, .9, 0, 0], [-.2, .2, .1, 1.2, 0],
                           [.2, -.25, .15, .1, 1]])
    weights = [np.eye(5), triangular @ triangular.T]
    normals = np.array([np.diag([1, 1, -1, 0, 0]), np.diag([0, 0, 0, 1, 1])])
    return basis, x0, endpoint, target, weights, normals, damping


def invariants(x, normals, damping):
    return np.einsum('aij,ij->a', normals, x + damping / 2 * spectral_function(x, np.log))


def integrate(basis, x0, endpoint, target, weight, normals, damping, step, horizon):
    factor = spectral_function(x0, np.sqrt)
    init_invariants = invariants(x0, normals, damping)
    max_drift = 0.0
    max_loss_increase = 0.0
    minimum = np.linalg.eigvalsh(x0)[0]
    snapshots = []
    sample_times = [0, 100, 200, 400, 800, horizon]
    sample_steps = {int(round(t / step)): t for t in sample_times}
    previous_loss = float('inf')

    def rhs(factor):
        x = factor @ factor.T
        residual = np.einsum('aij,ij->a', basis, x) - target
        g = np.einsum('a,aij->ij', weight @ residual, basis)
        eigenvalues, eigenvectors = np.linalg.eigh(x)
        rotated = eigenvectors.T @ g @ eigenvectors
        h = eigenvectors @ (rotated / (eigenvalues[:, None] + eigenvalues[None, :] + damping)) @ eigenvectors.T
        return -h @ factor

    nsteps = int(round(horizon / step))
    for k in range(nsteps + 1):
        if k % max(1, int(round(1 / step))) == 0 or k in sample_steps:
            x = factor @ factor.T
            eigenvalues = np.linalg.eigvalsh(x)
            residual = np.einsum('aij,ij->a', basis, x) - target
            loss = float(residual @ weight @ residual / 2)
            max_loss_increase = max(max_loss_increase, loss - previous_loss)
            previous_loss = loss
            minimum = min(minimum, float(eigenvalues[0]))
            max_drift = max(max_drift, float(np.max(np.abs(invariants(x, normals, damping) - init_invariants))))
            if k in sample_steps:
                snapshots.append({
                    'time': sample_steps[k], 'loss': loss,
                    'measurement_residual': float(np.linalg.norm(residual)),
                    'endpoint_error': float(np.linalg.norm(x - endpoint)),
                    'regular_block_error': float(np.linalg.norm(x[3:, 3:] - endpoint[3:, 3:])),
                    'first_block_minimum_eigenvalue': float(np.linalg.eigvalsh(x[:2, :2])[0]),
                    'scalar_component': float(x[2, 2]),
                    'minimum_eigenvalue': float(eigenvalues[0]),
                })
        if k == nsteps:
            break
        k1 = rhs(factor)
        k2 = rhs(factor + step / 2 * k1)
        k3 = rhs(factor + step / 2 * k2)
        k4 = rhs(factor + step * k3)
        factor += step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return factor @ factor.T, {
        'step': step, 'horizon': horizon, 'steps': nsteps,
        'minimum_sampled_eigenvalue': minimum,
        'maximum_sampled_invariant_drift': max_drift,
        'maximum_sampled_loss_increase': max_loss_increase,
        'snapshots': snapshots,
    }


def inconsistent_weight_control():
    # J is diagonal, A is identity; target (-1,1) is outside the PSD cone.
    # Identity W has optimum (0,1); coupled W has optimum (0,1/2).
    target = np.array([-1.0, 1.0])
    weights = [np.eye(2), np.array([[1.0, .5], [.5, 1.0]])]
    optima = [np.array([0.0, 1.0]), np.array([0.0, .5])]
    return [{'weight': w.tolist(), 'target': target.tolist(), 'optimum': x.tolist(),
             'gradient': (w @ (x - target)).tolist(),
             'loss': float((x - target) @ w @ (x - target) / 2),
             'complementarity': float(x @ (w @ (x - target))),
             'weight_minimum_eigenvalue': float(np.linalg.eigvalsh(w)[0])}
            for w, x in zip(weights, optima)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=float, default=1600.0)
    parser.add_argument('--steps', type=float, nargs=2, default=[0.05, 0.025])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    basis, x0, endpoint, target, weights, normals, damping = make_problem()
    records, finals = [], []
    for wi, weight in enumerate(weights):
        pair = []
        for step in args.steps:
            final, record = integrate(basis, x0, endpoint, target, weight, normals, damping, step, args.horizon)
            record['weight_index'] = wi
            records.append(record)
            pair.append(final)
        finals.append(pair)
    refinements = [float(np.linalg.norm(pair[0] - pair[1])) for pair in finals]
    basis_gram = np.einsum('aij,bij->ab', basis, basis)
    projected_normals = np.einsum('aij,bij->ab', basis, normals)
    eigenvalues = np.linalg.eigvalsh(endpoint)
    controls = inconsistent_weight_control()
    checks = {
        'basis_orthonormal': bool(np.linalg.norm(basis_gram - np.eye(len(basis))) < 1e-12),
        'normal_orthogonality': bool(np.linalg.norm(projected_normals) < 1e-12),
        'endpoint_has_two_zero_eigenvalues': bool(np.max(np.abs(eigenvalues[:2])) < 1e-12 and eigenvalues[2] > 0),
        'step_refinement_below_1e_minus_7': bool(max(refinements) < 1e-7),
        'sampled_positivity': bool(all(r['minimum_sampled_eigenvalue'] > 0 for r in records)),
        'sampled_invariant_error_below_1e_minus_6': bool(max(r['maximum_sampled_invariant_drift'] for r in records) < 1e-6),
        'endpoint_errors_decrease_at_late_samples': bool(all(r['snapshots'][-1]['endpoint_error'] < r['snapshots'][-2]['endpoint_error'] < r['snapshots'][-3]['endpoint_error'] for r in records)),
        'inconsistent_controls_psd_kkt': bool(all(min(r['gradient']) >= 0 and abs(r['complementarity']) < 1e-14 for r in controls)),
    }
    result = {
        'scope': 'One new 5x5 coupled-center PSD-only boundary problem, two SPD weights, two RK4 steps. No reserved data; no old large-target run repeated.',
        'caveat': 'Finite-horizon differences are not endpoint differences. Sampled positivity/descent/invariants are numerical checks, not continuous-time certificates.',
        'source_sha256': sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
        'damping': damping, 'elapsed_seconds': time.time() - started,
        'endpoint_eigenvalues': eigenvalues.tolist(), 'regular_endpoint_center': float(np.trace(endpoint[3:, 3:]) / 2),
        'initial_target_block_commutator_norm': float(np.linalg.norm(x0[:2, :2] @ endpoint[:2, :2] - endpoint[:2, :2] @ x0[:2, :2])),
        'weight_eigenvalues': [np.linalg.eigvalsh(w).tolist() for w in weights],
        'refinement_errors': refinements,
        'finite_horizon_weight_gap': float(np.linalg.norm(finals[0][-1] - finals[1][-1])),
        'records': records, 'inconsistent_weight_control': controls, 'checks': checks,
        'passed': all(checks.values()),
    }
    np.savez(args.output / 'matrices.npz', basis=basis, x0=x0, endpoint=endpoint,
             target=target, weights=np.array(weights), normals=normals, finals=np.array(finals))
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'refinement_errors': refinements,
                      'final_endpoint_errors': [r['snapshots'][-1]['endpoint_error'] for r in records],
                      'maximum_invariant_drift': max(r['maximum_sampled_invariant_drift'] for r in records),
                      'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
