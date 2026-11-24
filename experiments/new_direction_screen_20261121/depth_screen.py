"""Tiny algebra/implicit-bias audit; no performance experiment.

Usage: python depth_screen.py --output-dir <fresh external research directory>
Requires NumPy. All cases are fixed below before evaluation; no search or tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def diagonal_check(depth: int) -> dict:
    beta = np.array([0.13, 0.71, 1.4])
    energies = np.array([0.5, 1.7, 3.0])
    q = np.array([-0.4, 0.3, -0.7])
    damping = 0.23
    w = np.tile(beta ** (1.0 / depth), (depth, 1))
    jac = np.zeros((3, 3 * depth))
    for layer in range(depth):
        jac[:, 3 * layer : 3 * (layer + 1)] = np.diag(beta / w[layer])
    metric = jac.T @ np.diag(energies) @ jac + damping * np.eye(3 * depth)
    velocity = -np.linalg.solve(metric, jac.T @ q)
    actual = jac @ velocity
    predicted = -q / (energies + damping / depth * beta ** (2.0 / depth - 2))
    square_rates = 2 * w * velocity.reshape(depth, 3)
    return {
        "depth": depth,
        "predictor_velocity_max_error": float(np.max(np.abs(actual - predicted))),
        "squared_layer_balance_rate_error": float(np.max(np.ptp(square_rates, axis=0))),
    }


def prefix_check(depth: int) -> dict:
    weights = np.linspace(0.7, 1.2, depth)
    prefix = np.cumprod(weights)
    jac = np.zeros((depth, depth))
    for row in range(depth):
        jac[row, : row + 1] = prefix[row] / weights[: row + 1]
    q = -0.8
    energy = 1.3
    velocity = -np.linalg.solve(energy * jac.T @ jac, q * jac[-1])
    prefix_velocity = jac @ velocity
    expected = np.zeros(depth)
    expected[-1] = -q / energy
    return {
        "depth": depth,
        "prefix_velocity": prefix_velocity.tolist(),
        "coordinate_change_error": float(np.max(np.abs(prefix_velocity - expected))),
    }


def mirror_gradient(beta: np.ndarray, depth: int, damping: float) -> np.ndarray:
    energies = np.array([1.0, 4.0])
    if depth == 2:
        correction = damping / 2 * np.log(beta)
    else:
        correction = damping / (2 - depth) * beta ** ((2 - depth) / depth)
    return energies * beta + correction


def interpolation_check(epsilon: float, damping: float = 1e-4, depth: int = 3) -> dict:
    # Positive diagonal predictor, one fixed measurement beta1 + 2 beta2 = 1.
    # Its representation metric uses the SAME input, so c=(1,4).
    init = np.full(2, epsilon**depth)
    anchor = mirror_gradient(init, depth, damping)

    def stationarity(first: float) -> float:
        beta = np.array([first, (1 - first) / 2])
        delta = mirror_gradient(beta, depth, damping) - anchor
        return float(delta[0] - delta[1] / 2)

    low, high = 1e-30, 1 - 1e-14
    assert stationarity(low) < 0 < stationarity(high)
    for _ in range(180):
        mid = (low + high) / 2
        if stationarity(mid) < 0:
            low = mid
        else:
            high = mid
    first = (low + high) / 2
    beta = np.array([first, (1 - first) / 2])
    return {
        "layer_init_epsilon": epsilon,
        "damping": damping,
        "depth": depth,
        "anchor_scale_lambda_epsilon_power": damping * epsilon ** (2 - depth),
        "bregman_projection": beta.tolist(),
        "stationarity_error": abs(stationarity(first)),
        "measurement_error": abs(float(beta @ np.array([1.0, 2.0])) - 1),
    }


SYM_BASIS = np.array([
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 1.0]],
    [[0.0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 0.0]],
])


def lyapunov_gram(matrix: np.ndarray) -> np.ndarray:
    return np.array([
        [np.sum(ei * (matrix @ ej + ej @ matrix)) for ej in SYM_BASIS]
        for ei in SYM_BASIS
    ])


def matrix_check() -> dict:
    # Balanced unconstrained two-factor model X=W2 W1, at SPD X.
    matrix = np.diag([0.8, 1.7])
    factor = np.diag(np.sqrt(np.diag(matrix)))
    damping = 0.3
    gradient = np.array([[0.2, -0.7], [-0.7, 0.5]])
    full_basis = np.eye(4).reshape(4, 2, 2)
    jac = np.stack(
        [(factor @ e).ravel() for e in full_basis]
        + [(e @ factor).ravel() for e in full_basis], axis=1
    )
    v = -np.linalg.solve(jac.T @ jac + damping * np.eye(8), jac.T @ gradient.ravel())
    actual = (jac @ v).reshape(2, 2)
    k = lyapunov_gram(matrix)
    h = np.eye(3) + damping * np.linalg.inv(k)
    g = np.array([np.sum(e * gradient) for e in SYM_BASIS])
    predicted = np.einsum("i,ijk->jk", -np.linalg.solve(h, g), SYM_BASIS)

    # Necessary Hessian condition: partial_x H_zz == partial_z H_zx.
    # Its failure blocks an automatic Euclidean-coordinate mirror potential.
    def metric_at(x: np.ndarray) -> np.ndarray:
        return np.eye(3) + damping * np.linalg.inv(lyapunov_gram(x))

    defects = []
    for step in [1e-3, 1e-4, 1e-5]:
        dx = (metric_at(matrix + step * SYM_BASIS[0]) - metric_at(matrix - step * SYM_BASIS[0])) / (2 * step)
        dz = (metric_at(matrix + step * SYM_BASIS[2]) - metric_at(matrix - step * SYM_BASIS[2])) / (2 * step)
        defects.append({"finite_difference_step": step, "hessian_integrability_defect": float(dx[2, 2] - dz[2, 0])})
    a, b = np.diag(matrix)
    exact_defect = damping * (b - a) / (2 * a * (a + b) ** 2)
    return {
        "matrix": matrix.tolist(),
        "damping": damping,
        "full_parameter_vs_reduced_matrix_velocity_error": float(np.max(np.abs(actual - predicted))),
        "integrability_defects": defects,
        "analytic_integrability_defect": float(exact_defect),
        "interpretation": "Non-Hessian geometry is not a new implicit-bias or generalization theorem.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {
        "scope": "Frozen tiny reductions and one matrix integrability falsification; no training-performance claim.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagonal_checks": [diagonal_check(depth) for depth in [2, 3, 4]],
        "prefix_checks": [prefix_check(depth) for depth in [2, 3, 4]],
        "small_init_projection": [interpolation_check(epsilon) for epsilon in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]],
        "noncommuting_matrix_check": matrix_check(),
    }
    assert max(x["predictor_velocity_max_error"] for x in results["diagonal_checks"]) < 1e-12
    assert max(x["coordinate_change_error"] for x in results["prefix_checks"]) < 1e-12
    assert results["noncommuting_matrix_check"]["full_parameter_vs_reduced_matrix_velocity_error"] < 1e-12
    assert abs(results["noncommuting_matrix_check"]["integrability_defects"][-1]["hessian_integrability_defect"] - 0.027) < 1e-7
    (args.output_dir / "depth_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
