"""Independent first-order endpoint review using exact undamped paths.

No damped ODE is integrated and no learning experiment is repeated. Integrate
the perturbation coefficient along a matrix-exponential quadratic-flow path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BASIS = np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]],
                  [[0., 1 / np.sqrt(2)], [1 / np.sqrt(2), 0.]]])
A = np.array([[1., 0., 0.], [.5, .5, 1 / np.sqrt(2)]])
TARGET = np.ones(2)
INIT = np.array([.1, .1, 0.])
NULL = np.array([0., -2., np.sqrt(2)])
STAR = INIT + A.T @ np.linalg.solve(A @ A.T, TARGET - A @ INIT)
KNOTS = [0., 1., 2., 4., 8., 16., 32., 64., 96.]


def matrix(z):
    return np.einsum('i,ijk->jk', z, BASIS)


def lyapunov(x):
    return np.array([[np.sum(e * (x @ f + f @ x)) for f in BASIS] for e in BASIS])


def coefficient(weight, order):
    b = A.T @ np.diag([1., weight]) @ A
    eig, vec = np.linalg.eigh(b)
    active = eig > 1e-12
    rates, v = eig[active], vec[:, active]
    coefficients = v.T @ (INIT - STAR)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    integral = np.zeros(3)
    min_eig = .1
    for left, right in zip(KNOTS[:-1], KNOTS[1:]):
        for node, quadrature_weight in zip(nodes, weights):
            t = (left + right) / 2 + (right - left) * node / 2
            residual_state = v @ (np.exp(-rates * t) * coefficients)
            z = STAR + residual_state
            x = matrix(z)
            min_eig = min(min_eig, float(np.linalg.eigvalsh(x)[0]))
            forcing = np.linalg.solve(lyapunov(x), b @ residual_state)
            integral += (right - left) * quadrature_weight / 2 * forcing
    null_response = NULL * (NULL @ integral) / (NULL @ NULL)
    # Null coordinate c maps to the literal matrix entry X12 by construction.
    c = float(null_response[2] / np.sqrt(2))
    tail_bound = (np.linalg.norm(b, 2) * np.linalg.norm(INIT - STAR)
                  * np.exp(-rates.min() * KNOTS[-1]) / (.2 * rates.min()))
    return dict(weight=weight, quadrature_order_per_interval=order,
                endpoint_x12_first_derivative=c, nullspace_response=null_response.tolist(),
                minimum_spd_eigenvalue_at_quadrature_nodes=min_eig,
                unintegrated_tail_norm_bound=float(tail_bound))


def one_form(a, b, s=.1):
    determinant = a * (s + b) - b * b
    denominator = 6 * (a + s + b) * determinant
    return np.array([-b * (s + 2 * b) / denominator,
                     (a * (s + 2 * b - a) + b * b) / denominator])


def verify():
    # The initial 16/32-node trial failed the refinement check; retain a
    # stricter 64/128-node comparison, rather than relax its tolerance.
    records = [coefficient(w, n) for w in [1., 9.] for n in [64, 128]]
    refinement = max(abs(records[i]['endpoint_x12_first_derivative']
                         - records[i + 1]['endpoint_x12_first_derivative']) for i in [0, 2])
    assert refinement < 1e-11
    alpha_error = 0.
    for a, b in [(.2, .02), (.6, .25), (1., .3)]:
        x = np.array([[a, b], [b, .1 + b]])
        tangent_a = np.array([1., 0., 0.])
        tangent_b = np.array([0., 1., np.sqrt(2)])
        direct = np.array([NULL @ np.linalg.solve(lyapunov(x), tangent) / 6
                           for tangent in [tangent_a, tangent_b]])
        alpha_error = max(alpha_error, float(np.max(np.abs(direct - one_form(a, b)))))
    assert alpha_error < 1e-13
    a, b, s, h = .6, .25, .1, 1e-5
    finite_curl = ((one_form(a + h, b)[1] - one_form(a - h, b)[1])
                   - (one_form(a, b + h)[0] - one_form(a, b - h)[0])) / (2 * h)
    exact_curl = s * (s + 3 * b - a) / (6 * (a + s + b)**2 * (a * (s + b) - b*b))
    assert abs(finite_curl - exact_curl) < 1e-8
    assert exact_curl > 0
    delta = records[3]['endpoint_x12_first_derivative'] - records[1]['endpoint_x12_first_derivative']
    assert delta > 0
    return dict(status='passed', role='Independent perturbation coefficient, not novelty validation',
                undamped_endpoint=matrix(STAR).tolist(), records=records,
                contrast_first_derivative_weight9_minus_weight1=delta,
                quadrature_refinement_max_difference=refinement,
                explicit_one_form_max_error=alpha_error,
                explicit_curl=exact_curl, finite_difference_curl=float(finite_curl),
                conditions='Tail bound uses the proved X(t)>=0.1I along these undamped paths; no uniform small-initialization claim.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = verify()
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))
