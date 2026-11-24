"""Exact arithmetic controls for the boundary theorem; no integration or data."""

from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import platform


def encoded(x):
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def main():
    here = Path(__file__).resolve().parent
    scalar = []
    for x, lam in [(Q(1), Q(2)), (Q(1, 3), Q(4)), (Q(7, 5), Q(2, 7))]:
        dt_dx = -(2 * x + lam) / (2 * x * x)
        vector_field = -2 * x * x / (2 * x + lam)
        check = dt_dx * vector_field
        assert check == 1
        scalar.append({"x": encoded(x), "lambda": encoded(lam), "identity_value": encoded(check)})
    kkt = []
    for r in [Q(0), Q(1, 2)]:
        z = [Q(0), 1 - r]
        residual = [Q(1), -r]
        grad = [residual[0] + r * residual[1], r * residual[0] + residual[1]]
        complementarity = sum(z[i] * grad[i] for i in range(2))
        assert min(grad) >= 0 and complementarity == 0 and min(1 - r, 1 + r) > 0
        kkt.append({"r": encoded(r), "endpoint": list(map(encoded, z)), "gradient": list(map(encoded, grad)), "complementarity": encoded(complementarity), "weight_eigenvalues": [encoded(1 - r), encoded(1 + r)]})
    result = {
        "status": "passed",
        "purpose": "Exact controls for strict-feasibility, singular-start, zero-damping, and inconsistent-weight claims; not theorem certification.",
        "scalar_implicit_solution_checks": scalar,
        "inconsistent_weight_kkt_checks": kkt,
        "arithmetic": "Python fractions.Fraction, exact rational arithmetic; no truncation or integration error",
        "python": platform.python_version(),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (here / "exact_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
