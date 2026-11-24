"""Verify delayed static-weight response for a positive monotone metric mixture."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def r_apply(x, velocity, metric):
    eigenvalues, q = np.linalg.eigh(x)
    average = (eigenvalues[:, None] + eigenvalues[None, :]) / 2
    ratio = (eigenvalues[:, None] - eigenvalues[None, :]) / (2 * average)
    squared = ratio * ratio
    power = squared.copy()
    kernel = np.zeros_like(average)
    for j in range(1, 80):
        moment = 0.0 if metric == 'SLD' else 1 / 3
        kernel += (1 / (2 * j + 1) - moment) * power / average
        power *= squared
    return q @ ((q.T @ velocity @ q) * kernel) @ q.T


def null_projection(basis, matrix):
    return matrix - np.einsum('a,aij->ij', np.einsum('aij,ij->a', basis, matrix), basis)


def response(basis, epsilon, weights, metric, order):
    nodes, quadrature = np.polynomial.legendre.leggauss(order)
    answer = np.zeros_like(basis[0])
    # A fixed diagonal weight makes the zero-damping trajectory explicit.
    for left in range(0, 60, 3):
        for node, weight in zip(nodes, quadrature):
            t = left + 1.5 * (node + 1)
            decay = np.exp(-weights * t)
            x = np.eye(len(answer)) + epsilon * np.einsum('a,aij->ij', 1 - decay, basis)
            velocity = epsilon * np.einsum('a,aij->ij', weights * decay, basis)
            answer += 1.5 * weight * r_apply(x, velocity, metric)
    return null_projection(basis, answer)


def comm(a, b):
    return a @ b - b @ a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    f1 = np.diag([1.0, 0.0])
    f2 = np.array([[0., 1.], [1., 1.]]) / np.sqrt(3)
    star = np.zeros((2, 3, 3))
    star[0, 0, 1] = star[0, 1, 0] = 1 / np.sqrt(2)
    star[1, 0, 2] = star[1, 2, 0] = 1 / np.sqrt(2)
    cases = {'non_lie_triple': np.array([f1, f2]), 'lie_triple_star': star}
    records, arrays = [], {}
    for case, basis in cases.items():
        for metric in ('SLD', 'matched_mixture'):
            previous = None
            for epsilon in (0.12, 0.06, 0.03, 0.015):
                coarse = response(basis, epsilon, np.array([1., 2.]), metric, 24)
                fine = response(basis, epsilon, np.array([1., 2.]), metric, 48)
                norm = np.linalg.norm(fine)
                records.append({'case': case, 'metric': metric, 'epsilon': epsilon,
                                'contrast_norm': float(norm),
                                'quadrature_refinement': float(np.linalg.norm(fine - coarse)),
                                'observed_doubling_order': None if previous is None else float(np.log2(previous / norm))})
                arrays[f'{case}_{metric}_{epsilon}'] = fine
                previous = norm
    # Check the explicit mixed derivative coefficients, independently of fitting powers.
    derivative_checks = []
    tau = 1e-4
    for case, basis in cases.items():
        e, v = basis.sum(axis=0), basis[0] - basis[1]
        a2v = v.copy()
        for _ in range(4):
            a2v = comm(e, a2v)
        fifth = 77 / 28800 * null_projection(basis, a2v)
        sixth = -29 / 4800 * null_projection(basis, e @ a2v + a2v @ e)
        for epsilon in (0.04, 0.02, 0.01):
            plus = response(basis, epsilon, np.array([1 + tau, 1 - tau]), 'matched_mixture', 48)
            minus = response(basis, epsilon, np.array([1 - tau, 1 + tau]), 'matched_mixture', 48)
            actual = (plus - minus) / (2 * tau)
            predicted = epsilon**5 * fifth + epsilon**6 * sixth
            derivative_checks.append({'case': case, 'epsilon': epsilon,
                                      'actual_norm': float(np.linalg.norm(actual)),
                                      'predicted_norm': float(np.linalg.norm(predicted)),
                                      'absolute_error': float(np.linalg.norm(actual - predicted)),
                                      'relative_error': float(np.linalg.norm(actual - predicted) / np.linalg.norm(actual))})
    moment_checks = []
    for m in range(1, 5):
        nodes, weights = np.polynomial.legendre.leggauss(2 * m)
        positive = nodes > 0
        nodes, weights = nodes[positive], weights[positive]
        errors = [float(1 / (2 * j + 1) - np.dot(weights, nodes**(2 * j))) for j in range(2 * m + 1)]
        moment_checks.append({'atoms': m, 'nodes_squared': (nodes**2).tolist(), 'weights': weights.tolist(),
                              'moment_differences': errors, 'first_nonzero_metric_order': 4 * m,
                              'matched_moment_max_error': max(abs(x) for x in errors[:-1]),
                              'first_unmatched_moment': errors[-1]})
    final_orders = {(r['case'], r['metric']): r['observed_doubling_order'] for r in records if r['epsilon'] == .015}
    passed = (all(abs(final_orders[c, m] - p) < .15 for c, m, p in
                  [('non_lie_triple', 'SLD', 3), ('lie_triple_star', 'SLD', 4),
                   ('non_lie_triple', 'matched_mixture', 5), ('lie_triple_star', 'matched_mixture', 6)])
              and all(c['matched_moment_max_error'] < 1e-13 and c['first_unmatched_moment'] > 1e-8 for c in moment_checks)
              and max(r['quadrature_refinement'] / r['contrast_norm'] for r in records) < 1e-8)
    result = {'scope': 'First damping derivative only; target power and positive metric moment matching.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'mixture': '(2/3) normalized SLD + (1/3) normalized RLD',
              'records': records, 'mixed_derivative_checks': derivative_checks,
              'positive_quadrature_moment_checks': moment_checks, 'passed': bool(passed)}
    np.savez(args.output / 'response_matrices.npz', **arrays)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': passed, 'final_orders': {str(k): v for k, v in final_orders.items()},
                      'max_relative_quadrature_change': max(r['quadrature_refinement'] / r['contrast_norm'] for r in records),
                      'derivative_checks': derivative_checks, 'moments': moment_checks}, indent=2))
    if not passed:
        raise SystemExit('Inspect the preserved failed check.')


if __name__ == '__main__':
    main()
