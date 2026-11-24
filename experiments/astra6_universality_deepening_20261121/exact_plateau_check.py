"""Exact arithmetic check for the guarded n=6 plateau counterexample.

This checks the finite algebraic certificate and strict rational spectral bound.
It does not purport to numerically verify universal quantifiers; those are proved
in REGULARITY_THEOREM.md. No third-party dependencies or random inputs are used.
"""

from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import platform


def mul(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n))
             for j in range(n)] for i in range(n)]


def sub(a, b):
    return [[x - y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(c, a):
    return [[c * x for x in row] for row in a]


def comm(a, b):
    return sub(mul(a, b), mul(b, a))


def inner(a, b):
    return sum(x * y for ar, br in zip(a, b) for x, y in zip(ar, br))


def block(top, tail):
    out = [[F(0) for _ in range(6)] for _ in range(6)]
    for i in range(2):
        for j in range(2):
            out[i][j] = F(top[i][j])
    for i, v in enumerate(tail, 2):
        out[i][i] = F(v)
    return out


def encoded(matrix):
    return [[str(x) for x in row] for row in matrix]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    k = F(4)
    delta = F(1, 2)
    z1 = block([[1, 0], [0, -1]], [k, -k, 0, 0])
    z2 = block([[1, 1], [1, 1]], [0, 0, k, -k])
    obstruction = comm(z1, comm(z1, z2))
    expected = block([[0, 4], [4, 0]], [0, 0, 0, 0])
    assert obstruction == expected
    assert inner(z1, z2) == 0
    p1 = inner(z1, obstruction) / inner(z1, z1)
    p2 = inner(z2, obstruction) / inner(z2, z2)
    residual = sub(sub(obstruction, scale(p1, z1)), scale(p2, z2))
    residual_norm_sq = inner(residual, residual)
    ratio_bound_sq = F(2) / (k - 1) ** 2
    ratio_bound_gap_sq = delta ** 2 - ratio_bound_sq
    # sqrt(2)<3/2 bounds the active-block radius by 3/(2K).
    active_min_lower_bound = 1 - 1/k - F(3, 2)/k
    assert residual_norm_sq == F(272, 9) > 0
    assert ratio_bound_sq == F(2, 9) < delta ** 2
    assert ratio_bound_gap_sq == F(1, 36)
    assert active_min_lower_bound == F(3, 8) > 0
    archive = Path(r"C:\Users\liwei\Documents\Research\pullback-feature-learning\explorations")
    dependencies = [
        archive / "three-candidate-exploration-20261121/universality/THEOREM.md",
        archive / "three-candidate-exploration-20261121/universality/REVIEW.md",
        archive / "endpoint-classification-audit-20261121/ALGEBRA_REDUCTION.md",
    ]
    results = {
        "status": "passed",
        "purpose": "Exact finite certificate for guarded global plateau counterexample",
        "python": platform.python_version(),
        "arithmetic": "fractions.Fraction; all reported certificates exact",
        "K": str(k),
        "delta": str(delta),
        "Z1": encoded(z1),
        "Z2": encoded(z2),
        "ad_Z1_squared_Z2": encoded(obstruction),
        "projection_coefficients": [str(p1), str(p2)],
        "normal_obstruction_norm_squared": str(residual_norm_sq),
        "active_ratio_upper_bound_squared": str(ratio_bound_sq),
        "plateau_delta_squared_minus_ratio_bound_squared": str(ratio_bound_gap_sq),
        "active_min_eigenvalue_strict_lower_bound": str(active_min_lower_bound),
        "errors": {"roundoff": "0; exact rational arithmetic", "statistical": "not applicable"},
        "limitations": [
            "Continuum spectral bound and global one-form exactness are proved in the theorem note, not inferred from samples.",
            "This counterexample is in the smooth positive kernel class, not the analytic monotone metric class.",
            "No finite-damping global endpoint claim is tested or implied."
        ],
        "source_sha256": digest(Path(__file__).resolve()),
        "theorem_sha256": digest(here / "REGULARITY_THEOREM.md"),
        "read_only_inputs_sha256": {str(path): digest(path) for path in dependencies},
    }
    target = here / "exact_plateau_results.json"
    target.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": results["status"], "output": str(target),
                      "normal_obstruction_norm_squared": str(residual_norm_sq),
                      "strict_ratio_margin_squared": str(ratio_bound_gap_sq)}))


if __name__ == "__main__":
    main()
