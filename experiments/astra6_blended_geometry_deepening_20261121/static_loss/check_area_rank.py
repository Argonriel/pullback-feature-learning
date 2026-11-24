"""Bounded algebraic checks for the fixed-target signed-area rank theorem.

No training, datasets, or integrations. Output must be a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def sylvester_spd(c: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(c)
    transformed = vectors.T @ rhs @ vectors
    return vectors @ (transformed / (values[:, None] + values[None, :])) @ vectors.T


def area(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    gramian = sylvester_spd(c, np.outer(d, d))
    return (c @ gramian - gramian @ c) / 2


def derivative(c: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    gramian = sylvester_spd(c, np.outer(d, d))
    z = sylvester_spd(c, -(e @ gramian + gramian @ e))
    return (e @ gramian - gramian @ e + c @ z - z @ c) / 2


def symmetric_basis(n: int) -> list[np.ndarray]:
    result = []
    for i in range(n):
        for j in range(i, n):
            e = np.zeros((n, n))
            e[i, j] = e[j, i] = 1
            result.append(e)
    return result


def jacobian(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(len(c), 1)
    return np.column_stack([derivative(c, d, e)[upper] for e in symmetric_basis(len(c))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    n = 4
    c = np.diag([1.0, 2.0, 4.0, 7.0])
    rows = []
    for k in range(5):
        d = np.r_[np.ones(k), np.zeros(n - k)]
        matrix = jacobian(c, d)
        singular = np.linalg.svd(matrix, compute_uv=False)
        expected = k * (2 * n - k - 1) // 2
        actual = int(np.linalg.matrix_rank(matrix, tol=1e-11))
        rows.append({"krylov_dimension": k, "direction": d.tolist(), "rank": actual,
                     "expected_rank": expected, "singular_values": singular.tolist()})
        assert actual == expected
    d = np.ones(n)
    scalar = 2.0 * np.eye(n)
    scalar_matrix = jacobian(scalar, d)
    scalar_rank = int(np.linalg.matrix_rank(scalar_matrix, tol=1e-11))
    assert scalar_rank == n - 1
    e = np.array([[.3, -.2, .1, .4], [-.2, .5, -.1, .2],
                  [.1, -.1, -.4, .3], [.4, .2, .3, .1]])
    exact = derivative(c, d, e)
    fd = []
    for h in (1e-2, 5e-3, 2.5e-3):
        estimate = (area(c + h * e, d) - area(c - h * e, d)) / (2 * h)
        fd.append({"increment": h, "error_frobenius": float(np.linalg.norm(estimate - exact))})
    assert all(3.8 < fd[i]["error_frobenius"] / fd[i+1]["error_frobenius"] < 4.2 for i in (0, 1))
    gramian = sylvester_spd(c, np.outer(d, d))
    sqrt_c = np.diag(np.sqrt(np.diag(c)))
    s = sqrt_c @ gramian @ sqrt_c
    result = {
        "purpose": "Independent algebraic checks; proof is analytic, not inferred from rank samples.",
        "python": platform.python_version(), "numpy": np.__version__,
        "rates": np.diag(c).tolist(), "rank_rows": rows,
        "scalar_weight_rank": scalar_rank,
        "scalar_weight_expected_rank": n - 1,
        "finite_difference_direction": e.tolist(), "finite_differences": fd,
        "cyclic_gramian_eigenvalues": np.linalg.eigvalsh(gramian).tolist(),
        "cyclic_S_eigenvalues": np.linalg.eigvalsh(s).tolist(),
        "cyclic_lyapunov_residual": float(np.linalg.norm(c @ gramian + gramian @ c - np.outer(d, d))),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
