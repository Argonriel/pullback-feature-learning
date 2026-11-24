"""Independent full-parameter replay of the tiny matrix endpoint witness.

Integrates both unconstrained 2-by-2 factors with the explicit 8-by-8
parameter metric. Does not import the reduced product-flow implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def integrate(damping: float, weight: float, step: float) -> dict:
    state = np.tile((np.sqrt(0.1) * np.eye(2)).ravel(), 2)
    identity = np.eye(2)
    measurement = np.full((2, 2), 0.5)

    def rhs(z: np.ndarray) -> np.ndarray:
        first, second = z[:4].reshape(2, 2), z[4:].reshape(2, 2)
        product = second @ first
        gradient = weight * (np.sum(product * measurement) - 1) * measurement
        gradient[0, 0] += product[0, 0] - 1
        jacobian = np.hstack((np.kron(second, identity), np.kron(identity, first.T)))
        if damping == 0:
            return -jacobian.T @ np.linalg.solve(jacobian @ jacobian.T, gradient.ravel())
        return -np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(8), jacobian.T @ gradient.ravel())

    max_balance_error = 0.0
    max_symmetry_error = 0.0
    min_product_eigenvalue = 0.1
    for count in range(1, int(150 / step) + 1):
        k1 = rhs(state)
        k2 = rhs(state + step * k1 / 2)
        k3 = rhs(state + step * k2 / 2)
        k4 = rhs(state + step * k3)
        state += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        first, second = state[:4].reshape(2, 2), state[4:].reshape(2, 2)
        product = second @ first
        max_balance_error = max(max_balance_error, float(np.max(np.abs(first @ first.T - second.T @ second))))
        max_symmetry_error = max(max_symmetry_error, float(np.max(np.abs(product - product.T))))
        min_product_eigenvalue = min(min_product_eigenvalue, float(np.linalg.eigvalsh((product + product.T) / 2)[0]))
        residual = max(abs(product[0, 0] - 1), abs(np.sum(product * measurement) - 1))
        if residual < 1e-12:
            break
    assert residual < 1e-12
    assert max_balance_error < 1e-10
    assert max_symmetry_error < 1e-10
    assert min_product_eigenvalue > 0.05
    return {
        "damping": damping,
        "second_loss_weight": weight,
        "rk4_step": step,
        "steps": count,
        "matrix": product.tolist(),
        "unobserved_x12": float(product[0, 1]),
        "max_constraint_residual": float(residual),
        "max_balance_error": max_balance_error,
        "max_symmetry_error": max_symmetry_error,
        "min_product_eigenvalue": min_product_eigenvalue,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=float, nargs=2, default=[0.01, 0.005])
    args = parser.parse_args()
    coarse, fine = args.steps
    assert coarse > fine > 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = [integrate(damping, weight, step)
               for damping in [0.0, 0.003, 0.3]
               for weight in [1.0, 9.0]
               for step in args.steps]
    contrasts = []
    for damping in [0.0, 0.003, 0.3]:
        values = {(row["second_loss_weight"], row["rk4_step"]): row["unobserved_x12"]
                  for row in records if row["damping"] == damping}
        contrasts.append({
            "damping": damping,
            "x12_weight9_minus_weight1": values[9.0, fine] - values[1.0, fine],
            "max_endpoint_refinement_difference": max(abs(values[w, coarse] - values[w, fine]) for w in [1.0, 9.0]),
        })
    output = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Independent original eight-parameter continuous-flow replay; lambda=0 uses the minimum-norm pullback velocity.",
        "records": records,
        "contrasts": contrasts,
    }
    (args.output_dir / "endpoint_parameter_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contrasts": contrasts, "max_residual": max(r["max_constraint_residual"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
