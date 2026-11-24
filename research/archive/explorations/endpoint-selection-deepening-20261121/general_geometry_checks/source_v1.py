"""Minimal checks of the projected inverse-Lyapunov closure criterion.

Usage: python general_geometry_check.py --output-dir <fresh external directory>
All matrices and checks are fixed; no fitting, training sweep, or search is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def anti(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b + b @ a


def orthonormal(matrices: list[np.ndarray]) -> np.ndarray:
    q, _ = np.linalg.qr(np.stack([m.ravel() for m in matrices], axis=1))
    return q.T.reshape(len(matrices), 3, 3)


def pn(matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return matrix - np.einsum("i,ijk->jk", np.einsum("ijk,jk->i", basis, matrix), basis)


def inverse_lyapunov(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    # Independent full nine-dimensional Kronecker solve; inputs/outputs symmetric.
    return np.linalg.solve(np.kron(x, np.eye(3)) + np.kron(np.eye(3), x), u.ravel()).reshape(3, 3)


def log_derivative(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    eig, q = np.linalg.eigh(x)
    loewner = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            if abs(eig[i] - eig[j]) < 1e-12:
                loewner[i, j] = 1 / ((eig[i] + eig[j]) / 2)
            else:
                loewner[i, j] = np.log(eig[i] / eig[j]) / (eig[i] - eig[j])
    return q @ (loewner * (q.T @ u @ q)) @ q.T


def curvature(x: np.ndarray, u: np.ndarray, v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    value = -inverse_lyapunov(x, anti(u, inverse_lyapunov(x, v)))
    value += inverse_lyapunov(x, anti(v, inverse_lyapunov(x, u)))
    return pn(value, basis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    eye = np.eye(3)
    u = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    v = np.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])
    w = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    star = orthonormal([u, v])
    triple_errors = [np.linalg.norm(pn(comm(comm(a, b), c), star))
                     for a in [u, v] for b in [u, v] for c in [u, v]]
    star_records = []
    for a in [0.2, 0.4]:
        x = eye + a * u
        actual = curvature(x, u, v, star)
        coefficient = a*a * (8 + a*a) / (2 * (1-a*a) * (4-a*a)**2)
        step = 1e-5
        fd = (inverse_lyapunov(x + step*u, v) - inverse_lyapunov(x - step*u, v)
              - inverse_lyapunov(x + step*v, u) + inverse_lyapunov(x - step*v, u)) / (2*step)
        defect = pn(inverse_lyapunov(x, v) - .5*log_derivative(x, v), star)
        defect_formula = -a/(4-a*a) - np.log1p(-a*a)/(4*a)
        star_records.append({
            "a": a,
            "exact_curvature_coefficient": coefficient,
            "full_matrix_curvature_error": float(np.linalg.norm(actual - coefficient*w)),
            "finite_difference_error": float(np.linalg.norm(actual - pn(fd, star))),
            "primitive_defect_norm": float(np.linalg.norm(defect)),
            "primitive_defect_formula_error": float(np.linalg.norm(defect - defect_formula*w)),
        })

    diag1 = np.diag([1., 0., 1.])
    diag2 = np.diag([0., 1., 1.])
    commuting = orthonormal([diag1, diag2])
    traceless = orthonormal([np.diag([1., -1., 0.]), np.diag([0., 1., -1.]), u, v, w])
    jordan = orthonormal([np.diag([1., 0., 0.]), np.diag([0., 1., 0.]), u])
    controls = []
    for name, basis in [("commuting_non_jordan", commuting), ("traceless_non_jordan", traceless), ("nonunital_jordan", jordan)]:
        z = np.einsum("i,ijk->jk", np.linspace(.07, .19, len(basis)), basis)
        tangent = np.einsum("i,ijk->jk", np.linspace(-.2, .3, len(basis)), basis)
        x = eye + z
        a_tangent = comm(z, comm(z, tangent))
        c_a_tangent = anti(z, a_tangent)
        r_tangent = .5*log_derivative(x, tangent) - inverse_lyapunov(x, tangent)
        controls.append({
            "name": name,
            "projected_A_norm": float(np.linalg.norm(pn(a_tangent, basis))),
            "projected_CA_norm": float(np.linalg.norm(pn(c_a_tangent, basis))),
            "projected_R_norm": float(np.linalg.norm(pn(r_tangent, basis))),
            "omega_norm": float(np.linalg.norm(pn(inverse_lyapunov(x, tangent), basis))),
            "jordan_closure_residual": max(float(np.linalg.norm(pn(anti(a, b), basis))) for a in basis for b in basis),
        })

    generic_u = np.diag([1., 0., 0.])
    generic_v = np.ones((3, 3)) / 3
    generic_basis = orthonormal([generic_u, generic_v])
    z = generic_u + .3*generic_v
    leading = pn(comm(comm(generic_u, generic_v), z), generic_basis) / 8
    leading_checks = []
    for epsilon in [.01, .005, .0025]:
        actual = curvature(eye + epsilon*z, generic_u, generic_v, generic_basis)
        leading_checks.append({"epsilon": epsilon,
                               "curvature_over_epsilon_error": float(np.linalg.norm(actual/epsilon-leading))})

    results = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "star_lie_triple_residual": float(max(triple_errors)),
        "star_extra_condition_failure_norm": float(np.linalg.norm(pn(anti(u, comm(u, comm(u, v))), star))),
        "star_checks": star_records,
        "controls": controls,
        "generic_leading_coefficient_norm": float(np.linalg.norm(leading)),
        "generic_leading_checks": leading_checks,
        "scope": "Numerical checks of independently derived identities; no proof by sampling and no finite-damping endpoint claim.",
    }
    assert results["star_lie_triple_residual"] < 1e-14
    assert results["star_extra_condition_failure_norm"] > 1
    assert max(r["full_matrix_curvature_error"] for r in star_records) < 1e-13
    assert max(r["finite_difference_error"] for r in star_records) < 1e-8
    assert max(r["projected_R_norm"] for r in controls) < 1e-13
    assert results["generic_leading_coefficient_norm"] > .01
    (args.output_dir / "general_geometry_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
