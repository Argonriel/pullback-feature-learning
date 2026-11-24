"""Minimal check of the classified, coupled-center finite-damping theorem.

This checks a new direct sum Sym(2) + Sym(3) + R with one measured
central direction. It is not a test of novelty or a replacement for proof.
Run with --output pointing to a fresh, empty directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def basis():
    items = []
    for offset, size in ((0, 2), (2, 3)):
        for k in range(1, size):
            a = np.zeros((6, 6))
            a[np.arange(offset, offset + k), np.arange(offset, offset + k)] = 1
            a[offset + k, offset + k] = -k
            items.append(a / np.sqrt(k * (k + 1)))
        for i in range(size):
            for j in range(i + 1, size):
                a = np.zeros((6, 6))
                a[offset + i, offset + j] = a[offset + j, offset + i] = 1 / np.sqrt(2)
                items.append(a)
    centers = np.array([np.diag([1, 1, 0, 0, 0, 0]) / np.sqrt(2),
                        np.diag([0, 0, 1, 1, 1, 0]) / np.sqrt(3),
                        np.diag([0, 0, 0, 0, 0, 1])])
    direction = np.array([1.0, -0.7, 0.4])
    direction /= np.linalg.norm(direction)
    items.append(np.einsum('i,ijk->jk', direction, centers))
    _, _, vt = np.linalg.svd(direction[None, :], full_matrices=True)
    normal = np.einsum('ai,ijk->ajk', vt[1:], centers)
    return np.array(items), normal


def coords(b, x):
    return np.einsum('aij,ij->a', b, x)


def logm(x):
    d, q = np.linalg.eigh(x)
    if d.min() <= 0:
        raise ValueError('Left the positive definite domain')
    return (q * np.log(d)) @ q.T


def flow(x, b, target, weight, damping):
    error = coords(b, x - np.eye(6)) - target
    g = np.einsum('a,aij->ij', weight @ error, b)
    d, q = np.linalg.eigh(x)
    sums = d[:, None] + d[None, :]
    return -q @ ((q.T @ g @ q) * sums / (sums + damping)) @ q.T


def integrate(b, normal, target, weight, damping, dt, horizon):
    x = np.eye(6)
    minimum_eigenvalue = 1.0
    invariant_max = 0.0
    initial_invariant = coords(normal, x)
    steps = round(horizon / dt)
    for k in range(steps):
        v1 = flow(x, b, target, weight, damping)
        v2 = flow(x + dt * v1 / 2, b, target, weight, damping)
        v3 = flow(x + dt * v2 / 2, b, target, weight, damping)
        v4 = flow(x + dt * v3, b, target, weight, damping)
        x = x + dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6
        x = (x + x.T) / 2
        if k % 40 == 0 or k == steps - 1:
            minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(x).min()))
            invariant = coords(normal, x + damping * logm(x) / 2)
            invariant_max = max(invariant_max, float(np.linalg.norm(invariant - initial_invariant)))
    return x, {'dt': dt, 'steps': steps, 'horizon': horizon,
               'measurement_residual': float(np.linalg.norm(coords(b, x - np.eye(6)) - target)),
               'minimum_sampled_eigenvalue': minimum_eigenvalue,
               'max_sampled_invariant_drift': invariant_max}


def predicted_endpoint(b, normal, target, damping):
    base = np.eye(6) + np.einsum('a,aij->ij', target, b)
    coefficient = np.zeros(len(normal))
    for iteration in range(30):
        x = base + np.einsum('a,aij->ij', coefficient, normal)
        residual = coords(normal, x + damping * logm(x) / 2 - np.eye(6))
        if np.linalg.norm(residual) < 2e-14:
            return x, iteration, float(np.linalg.norm(residual))
        # Normal matrices are central and commute with x, so Dlog_x[n]=x^-1 n.
        hessian = np.array([[np.trace(na @ (nb + damping * np.linalg.solve(x, nb) / 2))
                             for nb in normal] for na in normal])
        change = np.linalg.solve(hessian, residual)
        fraction = 1.0
        while fraction > 1e-10:
            candidate = base + np.einsum('a,aij->ij', coefficient - fraction * change, normal)
            if np.linalg.eigvalsh(candidate).min() > 0:
                candidate_residual = coords(normal, candidate + damping * logm(candidate) / 2 - np.eye(6))
                if np.linalg.norm(candidate_residual) < np.linalg.norm(residual):
                    break
            fraction /= 2
        coefficient -= fraction * change
    raise RuntimeError('Endpoint invariant solve did not converge')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    b, normal = basis()
    rng = np.random.default_rng(20261121)
    target = np.array([0.12, -0.08, 0.09, -0.11, 0.06, 0.10, -0.07, 0.14])
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    weights = [np.eye(8), (q * np.linspace(0.7, 2.1, 8)) @ q.T]
    damping, horizon = 1.7, 100.0
    predicted, newton_steps, prediction_residual = predicted_endpoint(b, normal, target, damping)
    records, endpoints = [], {}
    for wi, weight in enumerate(weights):
        for dt in (0.025, 0.0125):
            x, record = integrate(b, normal, target, weight, damping, dt, horizon)
            record.update(weight_index=wi, endpoint_prediction_error=float(np.linalg.norm(x - predicted)))
            endpoints[f'w{wi}_dt{dt}'] = x
            records.append(record)
    coarse_gap = np.linalg.norm(endpoints['w0_dt0.025'] - endpoints['w1_dt0.025'])
    fine_gap = np.linalg.norm(endpoints['w0_dt0.0125'] - endpoints['w1_dt0.0125'])
    refinements = [float(np.linalg.norm(endpoints[f'w{i}_dt0.025'] - endpoints[f'w{i}_dt0.0125'])) for i in range(2)]
    kkt_error = np.linalg.norm(predicted + damping * logm(predicted) / 2 - np.eye(6)
                               - np.einsum('a,aij->ij', coords(b, predicted + damping * logm(predicted) / 2 - np.eye(6)), b))
    result = {'scope': 'One new 6x6 coupled-center classified case; numerical consistency, not proof.',
              'damping': damping, 'target': target.tolist(), 'seed': 20261121,
              'basis_dimension': len(b), 'normal_inside_J_dimension': len(normal),
              'weight_spectra': [np.linalg.eigvalsh(w).tolist() for w in weights],
              'prediction_newton_steps': newton_steps, 'prediction_invariant_residual': prediction_residual,
              'ambient_KKT_residual': float(kkt_error), 'coarse_weight_gap': float(coarse_gap),
              'fine_weight_gap': float(fine_gap), 'step_refinement_errors': refinements,
              'records': records}
    result['passed'] = bool(fine_gap < 2e-10 and max(refinements) < 2e-9 and kkt_error < 2e-12
                            and all(r['measurement_residual'] < 1e-11 and r['endpoint_prediction_error'] < 2e-9
                                    and r['minimum_sampled_eigenvalue'] > 0.5 for r in records))
    np.savez(args.output / 'matrices.npz', basis=b, normal=normal, weights=np.array(weights),
             target=target, predicted=predicted, **endpoints)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit('Verification threshold failed; retain and inspect outputs.')


if __name__ == '__main__':
    main()
