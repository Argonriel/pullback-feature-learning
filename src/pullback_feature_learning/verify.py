"""Export numerical evidence for the local mathematical claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .core import (direction, initialize_parameters, logistic_loss,
                   parameter_gradient, representation,
                   representation_gradient, representation_jacobian)
from .data import two_moons


def verify(config: dict) -> dict:
    x, y = two_moons(config["train_size"], config["noise"], config["train_seed"])
    theta = initialize_parameters(config["initialization_seed"])
    z = representation(theta, x)
    jac = representation_jacobian(theta, x)
    g = representation_gradient(z, y).ravel()
    grad = parameter_gradient(theta, x, y)
    h = 1e-6
    numerical = np.column_stack([
        ((representation(theta + h * v, x) - representation(theta - h * v, x))
         / (2 * h)).ravel() for v in np.eye(80)
    ])
    jac_error = float(np.linalg.norm(jac - numerical) / np.linalg.norm(jac))
    chain_error = float(np.linalg.norm(grad - jac.T @ g) / np.linalg.norm(grad))
    left, singular, _ = np.linalg.svd(jac, full_matrices=False)
    damping = config["methods"]["pullback"]["damping"]
    results = {}
    all_passed = jac_error < 1e-7 and chain_error < 1e-11
    for method in ("gd", "pullback"):
        u, _ = direction(theta, x, y, method, damping)
        velocity = jac @ u
        factor = singular**2
        if method == "pullback":
            factor = factor / (factor + damping)
        predicted = -left @ (factor * (left.T @ g))
        spectral_error = float(np.linalg.norm(velocity - predicted) / np.linalg.norm(velocity))
        linearization = []
        for parameter_step in (0.01, 0.005, 0.0025):
            eta = parameter_step / np.linalg.norm(u)
            actual = (representation(theta + eta * u, x) - z).ravel()
            absolute_error = float(np.linalg.norm(actual - eta * velocity))
            linearization.append({"parameter_step_norm": parameter_step,
                                  "eta": float(eta), "absolute_error": absolute_error,
                                  "relative_error": float(absolute_error / np.linalg.norm(eta * velocity))})
        ratios = [linearization[i]["absolute_error"] / linearization[i + 1]["absolute_error"]
                  for i in range(2)]
        eta_small = 1e-4 / max(1.0, np.linalg.norm(u))
        loss_change = logistic_loss(representation(theta + eta_small * u, x), y) - logistic_loss(z, y)
        residual = None
        if method == "pullback":
            residual = float(np.linalg.norm((jac.T @ jac + damping * np.eye(80)) @ u + jac.T @ g)
                             / np.linalg.norm(jac.T @ g))
        passed = (spectral_error < 1e-9 and grad @ u < 0 and loss_change < 0
                  and all(3.5 < ratio < 4.5 for ratio in ratios)
                  and (residual is None or residual < 1e-9))
        all_passed = all_passed and passed
        results[method] = {"passed": bool(passed), "spectral_relative_error": spectral_error,
                           "gradient_dot_direction": float(grad @ u),
                           "small_step_loss_change": float(loss_change),
                           "normal_equation_relative_residual": residual,
                           "linearization": linearization, "error_halving_ratios": ratios}
    return {"passed": bool(all_passed),
            "tolerances": {"jacobian_relative_error_max": 1e-7,
                           "chain_rule_relative_error_max": 1e-11,
                           "spectral_relative_error_max": 1e-9,
                           "normal_equation_relative_residual_max": 1e-9,
                           "error_halving_ratio_interval": [3.5, 4.5]},
            "jacobian_shape": list(jac.shape),
            "jacobian_finite_difference_relative_error": jac_error,
            "chain_rule_relative_error": chain_error, "methods": results,
            "config": config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/verification.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output.exists():
        parser.error("Output already exists; select a fresh path to preserve prior evidence.")
    with threadpool_limits(limits=config["threads"]):
        result = verify(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("config", "methods")}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
