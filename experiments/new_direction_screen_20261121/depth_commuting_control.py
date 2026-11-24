"""Overlapping diagonal measurement control for the matrix endpoint probe.

Usage: python depth_commuting_control.py --output-dir <fresh external directory>
This follow-up tests the analytic mirror-flow prediction, with no parameter search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def integrate(damping: float, weight: float, step: float) -> dict:
    matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    weights = np.array([1.0, weight])
    beta = np.full(3, 0.1)

    def rhs(value: np.ndarray) -> np.ndarray:
        gradient = matrix.T @ (weights * (matrix @ value - 1))
        return -gradient * (2 * value) / (2 * value + damping)

    for index in range(1, int(150 / step) + 1):
        k1 = rhs(beta)
        k2 = rhs(beta + step * k1 / 2)
        k3 = rhs(beta + step * k2 / 2)
        k4 = rhs(beta + step * k3)
        beta += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        assert np.min(beta) > 0.05
        if np.max(np.abs(matrix @ beta - 1)) < 1e-11:
            break
    residual = float(np.max(np.abs(matrix @ beta - 1)))
    assert residual < 1e-11
    return {"damping": damping, "weight": weight, "step": step,
            "beta": beta.tolist(), "residual": residual, "steps": index}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = [integrate(damping, weight, step) for damping in [0.0, 0.003, 0.03, 0.3]
               for weight in [1.0, 9.0] for step in [0.005, 0.0025]]
    contrasts = []
    for damping in [0.0, 0.003, 0.03, 0.3]:
        v = {(r["weight"], r["step"]): np.array(r["beta"])
             for r in records if r["damping"] == damping}
        contrasts.append({
            "damping": damping,
            "max_endpoint_weight_difference": float(np.max(np.abs(v[9.0, 0.0025] - v[1.0, 0.0025]))),
            "max_refinement_difference": max(float(np.max(np.abs(v[w, 0.005] - v[w, 0.0025]))) for w in [1.0, 9.0]),
        })
    result = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "This diagonal model has a nontrivial interpolation nullspace but a separable mirror potential; loss weighting should not select another endpoint.",
              "records": records, "contrasts": contrasts}
    (args.output_dir / "depth_commuting_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
