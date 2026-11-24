"""Numerically check elementary geometry-only contamination reductions.

This is a fixed-gradient linear algebra audit, not a training experiment or a
new attack algorithm. Geometry features are realizable as Jacobian rows for
the linear representation h_theta(x) = theta.T @ x. The supervised gradient
is held fixed throughout. See the separate POISONING_AUDIT.md research note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def audit() -> dict:
    metric = np.array([[2.0, 0.3], [0.3, 1.2]])
    gradient = np.array([1.0, -0.6])
    feature = np.array([1.5, 0.8])
    weight = 3.0
    clean_direction = np.linalg.solve(metric, gradient)
    inverse_feature = np.linalg.solve(metric, feature)
    poisoned_metric = metric + weight * np.outer(feature, feature)
    actual = np.linalg.solve(poisoned_metric, gradient)
    predicted = clean_direction - (
        weight * inverse_feature * (feature @ clean_direction)
        / (1.0 + weight * feature @ inverse_feature)
    )
    progress_drop = float(gradient @ (clean_direction - actual))
    drop_formula = float(
        weight * (feature @ clean_direction) ** 2
        / (1.0 + weight * feature @ inverse_feature)
    )
    assert np.allclose(actual, predicted, atol=1e-14, rtol=1e-14)
    assert np.isclose(progress_drop, drop_formula, atol=1e-14, rtol=1e-14)

    # A constant-size mixture has a removal term, unlike pure Gram addition.
    clean_second_moment = np.eye(2)
    damping = 0.1
    fraction = 0.02
    gradient_mixture = np.array([1.0, 1.0])
    attack_feature = np.array([30.0, 0.0])
    metric_mixture_clean = clean_second_moment + damping * np.eye(2)
    metric_mixture_attack = (
        (1.0 - fraction) * clean_second_moment
        + fraction * np.outer(attack_feature, attack_feature)
        + damping * np.eye(2)
    )
    direction_clean = np.linalg.solve(metric_mixture_clean, gradient_mixture)
    direction_attack = np.linalg.solve(metric_mixture_attack, gradient_mixture)
    delta = metric_mixture_attack - metric_mixture_clean
    resolvent_change = -np.linalg.solve(
        metric_mixture_attack, delta @ direction_clean
    )
    assert np.allclose(direction_attack - direction_clean, resolvent_change)
    assert gradient_mixture @ direction_attack > 0.0

    # Any target p with p.T(q-Mp)>0 is obtained by a PSD rank-one addition.
    target = np.array([0.15, -0.1])
    residual = gradient - metric @ target
    curvature = float(target @ residual)
    assert curvature > 0.0
    rank_one_addition = np.outer(residual, residual) / curvature
    target_actual = np.linalg.solve(metric + rank_one_addition, gradient)
    assert np.allclose(target_actual, target, atol=1e-14, rtol=1e-14)

    # A local metric perturbation alone cannot add a stationary point.
    quadratic_hessian = np.array([[2.0, 0.4], [0.4, 1.0]])
    quadratic_linear = np.array([0.3, -0.7])
    minimizer = np.linalg.solve(quadratic_hessian, quadratic_linear)
    stationary_residual = quadratic_hessian @ minimizer - quadratic_linear
    clean_stationarity = np.linalg.solve(metric, stationary_residual)
    poisoned_stationarity = np.linalg.solve(poisoned_metric, stationary_residual)
    assert np.allclose(clean_stationarity, 0.0, atol=1e-14)
    assert np.allclose(poisoned_stationarity, 0.0, atol=1e-14)

    return {
        "status": "reject_generic_novelty_standard_matrix_perturbation",
        "scope": "fixed-gradient linear representation; no model training",
        "rank_one_addition": {
            "formula_max_error": float(np.max(np.abs(actual - predicted))),
            "progress_drop": progress_drop,
            "progress_drop_formula_error": abs(progress_drop - drop_formula),
            "angle_degrees": angle_degrees(clean_direction, actual),
        },
        "normalized_replacement": {
            "poison_fraction": fraction,
            "poison_feature_norm": float(np.linalg.norm(attack_feature)),
            "clean_descent_direction": direction_clean.tolist(),
            "poisoned_descent_direction": direction_attack.tolist(),
            "angle_degrees": angle_degrees(direction_clean, direction_attack),
            "clean_progress": float(gradient_mixture @ direction_clean),
            "poisoned_progress": float(gradient_mixture @ direction_attack),
            "resolvent_max_error": float(np.max(np.abs(
                direction_attack - direction_clean - resolvent_change
            ))),
        },
        "rank_one_target_reachability": {
            "curvature_condition": curvature,
            "target": target.tolist(),
            "max_error": float(np.max(np.abs(target_actual - target))),
            "addition_trace": float(np.trace(rank_one_addition)),
        },
        "quadratic_stationarity": {
            "common_minimizer": minimizer.tolist(),
            "clean_update_norm": float(np.linalg.norm(clean_stationarity)),
            "poisoned_update_norm": float(np.linalg.norm(poisoned_stationarity)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    with arguments.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
