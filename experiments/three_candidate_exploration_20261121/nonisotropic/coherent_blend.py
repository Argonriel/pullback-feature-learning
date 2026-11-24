"""Minimal checks for finite-damping loss of diagonal-observation invariants.

Outputs go to an explicit research directory, not the repository. No SciPy.
Run: python coherent_blend.py --output PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def mobility(x: np.ndarray, u: np.ndarray, damping: float | None) -> np.ndarray:
    if damping is None:
        return x @ u + u @ x
    if damping == 0:
        return u.copy()
    d, v = np.linalg.eigh(x)
    sums = d[:, None] + d[None, :]
    return v @ ((sums / (sums + damping)) * (v.T @ u @ v)) @ v.T


def inverse_shifted_lyapunov(x, u, damping):
    d, v = np.linalg.eigh(x)
    return v @ ((v.T @ u @ v) / (d[:, None] + d[None, :] + damping)) @ v.T


def mobility_derivative(x, direction, u, damping):
    inner = inverse_shifted_lyapunov(x, u, damping)
    return damping * inverse_shifted_lyapunov(
        x, direction @ inner + inner @ direction, damping
    )


def horizontal(x, damping):
    n = len(x)
    basis = [np.diag(np.eye(n)[i]) for i in range(n)]
    fields = np.array([mobility(x, e, damping) for e in basis])
    observed = np.array([np.diag(f) for f in fields]).T
    coefficients = np.linalg.inv(observed)
    lifted = np.einsum("kij,kl->lij", fields, coefficients)
    return lifted, fields, observed, coefficients


def horizontal_derivative(x, damping, direction):
    h, _, _, coeff = horizontal(x, damping)
    basis = [np.diag(np.eye(len(x))[i]) for i in range(len(x))]
    df = np.array([mobility_derivative(x, direction, e, damping) for e in basis])
    da = np.array([np.diag(f) for f in df]).T
    return np.einsum("kij,kl->lij", df, coeff) - np.einsum(
        "kij,kl->lij", h, da @ coeff
    )


def bracket(x, damping, i, j):
    h, _, _, _ = horizontal(x, damping)
    return horizontal_derivative(x, damping, h[i])[j] - horizontal_derivative(
        x, damping, h[j]
    )[i]


def vec(x):
    return np.r_[np.diag(x), [np.sqrt(2) * x[i, j] for i in range(len(x)) for j in range(i + 1, len(x))]]


def curvature(a, b, c, damping):
    determinant = a * b - c * c
    q = (a + b) * determinant + damping * a * b
    return damping * c * (b - a) * determinant / (2 * q * q)


def simulate(x0, target, weight, damping, step, duration=40):
    def rhs(x):
        grad = np.diag(weight @ (np.diag(x) - target))
        return -mobility(x, grad, damping)

    x = x0.copy()
    mineig = float(np.linalg.eigvalsh(x)[0])
    steps = int(np.ceil(duration / step))
    step = duration / steps
    for _ in range(steps):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * step * k1)
        k3 = rhs(x + 0.5 * step * k2)
        k4 = rhs(x + step * k3)
        x += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x = (x + x.T) / 2
        mineig = min(mineig, float(np.linalg.eigvalsh(x)[0]))
    return {
        "endpoint": x.tolist(),
        "offdiagonal": float(x[0, 1]),
        "residual": float(np.linalg.norm(np.diag(x) - target)),
        "minimum_sampled_eigenvalue": mineig,
        "step": step,
        "duration": duration,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    x0 = np.array([[1.0, 0.3], [0.3, 2.0]])
    damping = 2.865
    h, _, observed, _ = horizontal(x0, damping)
    d, v = np.linalg.eigh(observed)
    sqrt_a = v @ np.diag(np.sqrt(d)) @ v.T
    inv_sqrt_a = v @ np.diag(1 / np.sqrt(d)) @ v.T
    weight0 = np.linalg.inv(observed)
    weight1 = inv_sqrt_a @ np.diag([1.0, 2.0]) @ inv_sqrt_a
    direction = sqrt_a @ np.ones(2)
    f = curvature(1, 2, 0.3, damping)
    predicted = -f * np.sqrt(np.linalg.det(observed)) / 6
    rows = []
    for amplitude in (0.1, 0.05, 0.025):
        target = np.diag(x0) + amplitude * direction
        pair = [simulate(x0, target, w, damping, 0.01) for w in (weight0, weight1)]
        gap = pair[1]["offdiagonal"] - pair[0]["offdiagonal"]
        rows.append({"amplitude": amplitude, "runs": pair, "gap": gap, "gap_over_amplitude_squared": gap / amplitude**2})
    refined = [simulate(x0, np.diag(x0) + 0.025 * direction, w, damping, 0.005) for w in (weight0, weight1)]
    controls = {}
    target = np.diag(x0) + 0.1 * direction
    for label, mode in (("product_euclidean", 0), ("pure_bures", None)):
        pair = [simulate(x0, target, w, mode, 0.01) for w in (weight0, weight1)]
        expected = 0.3 if mode == 0 else 0.3 * np.sqrt(np.prod(target) / 2)
        controls[label] = {"runs": pair, "predicted_offdiagonal": float(expected), "weight_gap": pair[1]["offdiagonal"] - pair[0]["offdiagonal"]}
    actual_bracket = bracket(x0, damping, 0, 1)
    finite_difference_rows = []
    for delta in (1e-3, 5e-4, 2.5e-4):
        fd = (horizontal(x0 + delta * h[0], damping)[0][1] - horizontal(x0 - delta * h[0], damping)[0][1]) / (2 * delta)
        fd -= (horizontal(x0 + delta * h[1], damping)[0][0] - horizontal(x0 - delta * h[1], damping)[0][0]) / (2 * delta)
        finite_difference_rows.append({"delta": delta, "error": float(np.linalg.norm(fd - actual_bracket))})
    rank_rows = []
    diag = np.array([1.0, 2.0, 3.0])
    coherence = np.array([[0.0, 0.2, -0.3], [0.2, 0.0, 0.4], [-0.3, 0.4, 0.0]])
    pairs = [(0, 1), (0, 2), (1, 2)]
    leading = np.diag([damping * coherence[i, j] * (diag[j] - diag[i]) / (2 * diag[i] * diag[j] * (diag[i] + diag[j] + damping)**2) for i, j in pairs])
    for epsilon in (0.02, 0.01, 0.005):
        x = np.diag(diag) + epsilon * coherence
        hh, _, _, _ = horizontal(x, damping)
        brackets = [bracket(x, damping, i, j) for i, j in pairs]
        vertical = np.array([[b[i, j] for b in brackets] for i, j in pairs])
        mat = np.array([vec(z) for z in list(hh) + brackets]).T
        rank_rows.append({"epsilon": epsilon, "rank": int(np.linalg.matrix_rank(mat, tol=1e-12)), "singular_values": np.linalg.svd(mat, compute_uv=False).tolist(), "vertical_brackets": vertical.tolist(), "scaled_leading_error": float(np.linalg.norm(vertical / epsilon - leading)), "determinant_scaled": float(np.linalg.det(vertical) / epsilon**3)})
    result = {
        "initial_state": x0.tolist(), "damping": damping,
        "observed_mobility": observed.tolist(), "target_direction": direction.tolist(),
        "weights": [weight0.tolist(), weight1.tolist()], "curvature": f,
        "analytic_bracket": actual_bracket.tolist(), "bracket_formula_error": float(abs(actual_bracket[0, 1] - f)),
        "finite_difference_bracket": finite_difference_rows,
        "predicted_quadratic_gap": predicted, "endpoint_rows": rows,
        "refined_runs": refined, "limit_controls": controls,
        "rank_checks": rank_rows, "rank_leading_matrix": leading.tolist(),
        "rank_leading_determinant": float(np.linalg.det(leading)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    assert result["bracket_formula_error"] < 1e-13
    assert all(r["rank"] == 6 for r in rank_rows)
    assert max(r["residual"] for row in rows for r in row["runs"]) < 1e-11
    assert all(row["gap"] < 0 for row in rows)
    assert abs(rows[-1]["gap_over_amplitude_squared"] / predicted - 1) < 0.03
    assert max(abs(a["offdiagonal"] - b["offdiagonal"]) for a, b in zip(rows[-1]["runs"], refined)) < 1e-10
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"predicted_gap_coefficient": predicted, "observed_coefficients": [r["gap_over_amplitude_squared"] for r in rows], "rank": [r["rank"] for r in rank_rows], "controls": {k: v["weight_gap"] for k, v in controls.items()}}, indent=2))


if __name__ == "__main__":
    main()
