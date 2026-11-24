"""Fixed 2-by-2 endpoint probe of damped full-representation pullback.

Usage: python depth_endpoint_probe.py --output-dir <fresh external directory>
This is a bounded discovery probe, not prospective generalization validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from depth_screen import SYM_BASIS, lyapunov_gram


def integrate(damping: float, weight: float, step: float) -> dict:
    # Constraints X11=1 and v^T X v=1 for v=(1,1)/sqrt(2).
    # X0=.1 I is SPD; both weights have exactly the same interpolation set.
    measurements = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 1 / np.sqrt(2)]])
    loss_weights = np.array([1.0, weight])
    target = np.ones(2)
    state = np.array([0.1, 0.1, 0.0])

    def rhs(z: np.ndarray) -> np.ndarray:
        matrix = np.einsum("i,ijk->jk", z, SYM_BASIS)
        gradient = measurements.T @ (loss_weights * (measurements @ z - target))
        if damping == 0:
            return -gradient
        gram = lyapunov_gram(matrix)
        return -np.linalg.solve(gram + damping * np.eye(3), gram @ gradient)

    min_eigenvalue = 0.1
    steps = 0
    for steps in range(1, int(150 / step) + 1):
        k1 = rhs(state)
        k2 = rhs(state + step * k1 / 2)
        k3 = rhs(state + step * k2 / 2)
        k4 = rhs(state + step * k3)
        state += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        matrix = np.einsum("i,ijk->jk", state, SYM_BASIS)
        min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(matrix)[0]))
        if np.max(np.abs(measurements @ state - target)) < 1e-11:
            break
    residual = float(np.max(np.abs(measurements @ state - target)))
    assert min_eigenvalue > 0.05
    assert residual < 1e-11
    return {
        "damping": damping,
        "second_loss_weight": weight,
        "rk4_step": step,
        "steps": steps,
        "time": steps * step,
        "matrix": matrix.tolist(),
        "unobserved_x12": float(matrix[0, 1]),
        "max_constraint_residual": residual,
        "minimum_path_eigenvalue": min_eigenvalue,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = [integrate(damping, weight, step)
               for damping in [0.0, 0.003, 0.03, 0.3]
               for weight in [1.0, 9.0]
               for step in [0.02, 0.01]]
    contrasts = []
    for damping in [0.0, 0.003, 0.03, 0.3]:
        values = {(r["second_loss_weight"], r["rk4_step"]): r["unobserved_x12"]
                  for r in records if r["damping"] == damping}
        delta = values[9.0, 0.01] - values[1.0, 0.01]
        contrasts.append({
            "damping": damping,
            "x12_weight9_minus_weight1": delta,
            "contrast_over_damping": delta / damping if damping else None,
            "max_endpoint_step_refinement_difference": max(abs(values[w, 0.02] - values[w, 0.01]) for w in [1.0, 9.0]),
        })
    result = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_hypothesis": "Positive damping can break loss-weight independence of the interpolating product, although both losses have identical constraints; lambda=0 is a negative control.",
        "metric_definition": "Full matrix representation Jacobian J, parameter metric J^T J + lambda I; not the observed-measurement Gauss-Newton metric.",
        "records": records,
        "contrasts": contrasts,
    }
    (args.output_dir / "depth_endpoint_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contrasts": contrasts, "max_residual": max(r["max_constraint_residual"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
