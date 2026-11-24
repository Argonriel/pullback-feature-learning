"""Independent polynomial-path check of the 13/576 mixed quartic factor.

Uses the frozen star-plane family and exact undamped paths. The earlier
static_weight_response.py and its outputs are not changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from static_weight_response import KNOTS, comm, project_null


def quartic_path_coefficient(delta, order):
    rates = np.array([1., 1. + delta])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u = np.zeros((3, 3)); u[0, 1] = u[1, 0] = 1.
    v = np.zeros((3, 3)); v[0, 2] = v[2, 0] = 1.
    basis = np.array([u, v]) / np.sqrt(2)
    integral = np.zeros((3, 3))
    for left, right in zip(KNOTS[:-1], KNOTS[1:]):
        for node, weight in zip(nodes, weights):
            t = (left + right) / 2 + (right - left) * node / 2
            decay = np.exp(-rates * t)
            z = (1 - decay[0]) * u + (1 - decay[1]) * v
            dz = rates[0] * decay[0] * u + rates[1] * decay[1] * v
            nested = comm(z, comm(z, dz))
            value = -(z @ nested + nested @ z) / 16
            integral += (right - left) * weight / 2 * value
    return project_null(integral, basis)


def run():
    u = np.zeros((3, 3)); u[0, 1] = u[1, 0] = 1.
    v = np.zeros((3, 3)); v[0, 2] = v[2, 0] = 1.
    e = u + v
    nested = comm(e, comm(e, v))
    prediction = 13 / 576 * (e @ nested + nested @ e)
    records = []
    for delta in [.02, .01, .005]:
        for order in [32, 64]:
            plus = quartic_path_coefficient(delta, order)
            minus = quartic_path_coefficient(-delta, order)
            derivative = (plus - minus) / (2 * delta)
            records.append(dict(delta=delta, quadrature_order=order,
                                plus_coefficient=plus.tolist(), minus_coefficient=minus.tolist(),
                                central_derivative=derivative.tolist(),
                                max_error=float(np.max(np.abs(derivative - prediction)))))
    refinement = max(float(np.max(np.abs(np.array(records[i]['central_derivative'])
                                        - np.array(records[i+1]['central_derivative'])))) for i in [0, 2, 4])
    checks = dict(quadrature_refinement=refinement < 1e-11,
                  central_difference_convergence=records[-1]['max_error'] < records[1]['max_error'] / 8,
                  nonzero_predicted_coefficient=float(np.linalg.norm(prediction)) > .01)
    return dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
                formula='(13/576) P_N {E,[E,[E,V]]}, s=1; E=U+V, D=diag(0,1) in orthonormal star basis',
                predicted_mixed_quartic_coefficient=prediction.tolist(), records=records,
                maximum_quadrature_refinement_difference=refinement,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependency_sha256=hashlib.sha256(Path(__file__).with_name('static_weight_response.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)
