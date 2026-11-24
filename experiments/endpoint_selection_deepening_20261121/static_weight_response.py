"""Exact static-weight path coefficients and bounded small-residual checks.

No damped flow or training is run. Fixed positive quadratic weights generate
matrix-exponential paths. All configured quadrature orders/epsilon values are
retained even when a numerical check fails.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


SCALE = 1.
EPSILONS = [.2, .1, .05, .025]
ORDERS = [32, 64]
KNOTS = [0., .25, .5, 1., 2., 4., 8., 16., 32., 64.]


def project_null(matrix, basis):
    return matrix - sum(float(np.sum(e * matrix)) * e for e in basis)


def comm(a, b):
    return a @ b - b @ a


def entropy_minus_lyapunov(x, tangent):
    """Stable application of R_X=.5 Dlog_X - S_X^-1."""
    eig, vec = np.linalg.eigh(x)
    total = eig[:, None] + eig[None, :]
    r = (eig[:, None] - eig[None, :]) / total
    # atanh(r)/r-1=sum_{k>=1} r^(2k)/(2k+1). All fixed cases have |r|<.5.
    assert np.max(np.abs(r)) < .5
    kernel = sum(r ** (2 * k) / (2 * k + 1) for k in range(1, 33)) / total
    local = vec.T @ tangent @ vec
    return vec @ (kernel * local) @ vec.T


def static_response(basis, target, hessian, epsilon, order):
    rates, vec = np.linalg.eigh(hessian)
    assert rates.min() > 0
    target_coords = np.array([np.sum(e * target) for e in basis])
    modes = vec.T @ target_coords
    nodes, weights = np.polynomial.legendre.leggauss(order)
    integral = np.zeros_like(target)
    minimum_eigenvalue = SCALE
    for left, right in zip(KNOTS[:-1], KNOTS[1:]):
        for node, qweight in zip(nodes, weights):
            t = (left + right) / 2 + (right - left) * node / 2
            decay = np.exp(-rates * t)
            z = np.einsum('i,ijk->jk', vec @ ((1 - decay) * modes), basis)
            velocity = np.einsum('i,ijk->jk', vec @ (rates * decay * modes), basis)
            x = SCALE * np.eye(len(target)) + epsilon * z
            minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(x)[0]))
            value = entropy_minus_lyapunov(x, epsilon * velocity)
            integral += (right - left) * qweight / 2 * value
    # Radial B=I path has R_X(dX)=0 exactly. Therefore this is its contrast.
    contrast = project_null(integral, basis)
    return dict(epsilon=epsilon, quadrature_order_per_interval=order,
                response_contrast=contrast.tolist(),
                minimum_spd_eigenvalue_at_quadrature_nodes=minimum_eigenvalue)


def cubic_coefficient(basis, target, hessian):
    rates, vec = np.linalg.eigh(hessian)
    coords = np.array([np.sum(e * target) for e in basis])
    modes = vec.T @ coords
    eigbasis = np.einsum('ji,jkl->ikl', vec, basis)
    k = np.zeros_like(target)
    for i, j, l in itertools.product(range(len(basis)), repeat=3):
        f = (1 - rates[j] / (rates[i] + rates[j])
             - rates[j] / (rates[l] + rates[j])
             + rates[j] / (rates[i] + rates[j] + rates[l]))
        k += modes[i] * modes[j] * modes[l] * f * (eigbasis[i] @ eigbasis[j] @ eigbasis[l])
    return -project_null(k - target @ target @ target / 3, basis) / (8 * SCALE**3)


def collect():
    e1 = np.array([[1., 0.], [0., 0.]])
    e2 = np.array([[0., 1.], [1., 1.]]) / np.sqrt(3)
    cubic_basis = np.array([e1, e2])
    cubic_hessian = np.array([[1., .25], [.25, 1.]])
    cubic_prediction = cubic_coefficient(cubic_basis, e1, cubic_hessian)
    mixed_prediction = 5 / (288 * SCALE**3) * project_null(comm(comm(e1, e2), e1), cubic_basis)
    mixed_records = []
    for delta in [.02, .01, .005]:
        plus = cubic_coefficient(cubic_basis, e1, np.array([[1., delta], [delta, 1.]]))
        minus = cubic_coefficient(cubic_basis, e1, np.array([[1., -delta], [-delta, 1.]]))
        actual = (plus - minus) / (2 * delta)
        mixed_records.append(dict(weight_perturbation=delta, central_derivative=actual.tolist(),
                                  max_error=float(np.max(np.abs(actual - mixed_prediction)))))
    u = np.zeros((3, 3)); u[0, 1] = u[1, 0] = 1.
    v = np.zeros((3, 3)); v[0, 2] = v[2, 0] = 1.
    quartic_basis = np.array([u, v]) / np.sqrt(2)
    quartic_prediction = np.array([[0., 0., 0.], [0., -7 / 240, -1 / 210], [0., -1 / 210, 7 / 240]]) / SCALE**4
    definitions = [
        ('cubic_full_spd_weights', cubic_basis, e1, cubic_hessian, 3, cubic_prediction),
        ('diagonal_weights_invisible_target', cubic_basis, e1, np.diag([1., 2.]), 3, np.zeros((2, 2))),
        ('quartic_lie_triple_plane', quartic_basis, u + v, np.diag([1., 2.]), 4, quartic_prediction),
    ]
    cases = []
    max_refinement = 0.
    for name, basis, target, hessian, power, prediction in definitions:
        records = [static_response(basis, target, hessian, eps, order)
                   for eps in EPSILONS for order in ORDERS]
        scaling = []
        for k, eps in enumerate(EPSILONS):
            low = np.array(records[2*k]['response_contrast'])
            high = np.array(records[2*k+1]['response_contrast'])
            refinement = float(np.max(np.abs(high - low)))
            max_refinement = max(max_refinement, refinement)
            scaled = high / eps**power
            scaling.append(dict(epsilon=eps, residual_power=power, scaled_contrast=scaled.tolist(),
                                scaled_coefficient_max_error=float(np.max(np.abs(scaled - prediction))),
                                quadrature_max_difference=refinement))
        cases.append(dict(name=name, basis=basis.tolist(), target_direction=target.tolist(),
                          positive_loss_hessian=hessian.tolist(), predicted_coefficient=prediction.tolist(),
                          records=records, scaling=scaling))
    triple = project_null(comm(comm(u, v), u), quartic_basis)
    conditions = {
        'quadrature_refinement': max_refinement < 1e-12,
        'mixed_cubic_formula_converges': mixed_records[-1]['max_error'] < mixed_records[0]['max_error'] / 8,
        'diagonal_target_contrast_zero': max(abs(x) for row in cases[1]['records'] for line in row['response_contrast'] for x in line) < 1e-13,
        'cubic_scaling_approaches_formula': cases[0]['scaling'][-1]['scaled_coefficient_max_error'] < cases[0]['scaling'][0]['scaled_coefficient_max_error'] / 4,
        'quartic_scaling_approaches_formula': cases[2]['scaling'][-1]['scaled_coefficient_max_error'] < cases[2]['scaling'][0]['scaled_coefficient_max_error'] / 20,
        'star_plane_projected_triple_zero': float(np.max(np.abs(triple))) < 1e-13,
    }
    return dict(status='passed' if all(conditions.values()) else 'failed', checks=conditions,
                role='Static positive-weight attainability and asymptotic coefficient checks; not a novelty or prediction-performance validation',
                isotropic_initial_scale=SCALE, quadrature_intervals=KNOTS,
                cubic_mixed_weight_prediction=mixed_prediction.tolist(), mixed_weight_records=mixed_records,
                cases=cases, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = collect()
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['status', 'checks', 'mixed_weight_records']}, indent=2))
    for case in result['cases']:
        print(json.dumps(dict(name=case['name'], scaling=case['scaling']), indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)
