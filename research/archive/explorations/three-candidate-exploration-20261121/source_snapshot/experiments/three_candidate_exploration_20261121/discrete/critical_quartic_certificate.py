"""Exact rational quartic certificate after the prespecified factor-step check.

No trajectories are run. The degree-four invariant drift uses only the exact
linearized residual path and a quadratic conservation identity at lambda=2s.
"""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path


def add(*polynomials):
    ans = {}
    for p in polynomials:
        for key, value in p.items():
            ans[key] = ans.get(key, F(0)) + value
    return {k: v for k, v in ans.items() if v}


def scale(p, coefficient):
    return {k: coefficient*v for k, v in p.items() if coefficient*v}


def mul(p, q):
    ans = {}
    for (a, b), pv in p.items():
        for (c, d), qv in q.items():
            key = (a+c, b+d)
            ans[key] = ans.get(key, F(0))+pv*qv
    return {k: v for k, v in ans.items() if v}


def accumulated_quartic_drift(weights):
    eta = F(1, 2)
    q = [1-eta*b/2 for b in weights]
    variables = [{(1, 0): F(1)}, {(0, 1): F(1)}]
    v = [add({(0, 0): F(1)}, scale(z, -1)) for z in variables]
    u = [scale(z, -b/4) for z, b in zip(variables, weights)]
    vv = add(*(mul(z, z) for z in v))
    uu = add(*(mul(z, z) for z in u))
    vu = add(*(mul(a, b) for a, b in zip(v, u)))
    drift = add(scale(mul(vv, uu), eta**2/16),
                scale(mul(vu, vu), -eta**2/4),
                scale(mul(vu, uu), eta**3/4),
                scale(mul(uu, uu), -eta**4/8))
    assert (0, 0) not in drift
    total = sum((c/(1-q[0]**a*q[1]**b) for (a, b), c in drift.items()), F(0))
    return {"weights": [str(x) for x in weights], "linear_residual_factors": [str(x) for x in q],
            "drift_polynomial": {f"{a},{b}": str(c) for (a, b), c in sorted(drift.items())},
            "summed_quartic_invariant_drift": str(total)}, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows, totals = [], []
    for weights in [(F(1), F(1)), (F(1, 2), F(3, 2))]:
        row, total = accumulated_quartic_drift(weights)
        rows.append(row)
        totals.append(total)
    delta = (totals[1]-totals[0])/2
    result = {"lambda": 2, "s": 1, "eta": "1/2", "cases": rows,
              "quartic_scalar_endpoint_difference_exact": str(delta),
              "quartic_scalar_endpoint_difference_decimal": float(delta),
              "positive": delta > 0}
    out = Path(args.output)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
