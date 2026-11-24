"""Bounded deterministic checks for the nonanalytic universality extension.

Requires Python and NumPy only. This is not a theorem prover or training run.
All outputs go to the explicitly supplied new research output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


E = np.diag([1.0, -1.0])
H = np.array([[0.0, 1.0], [1.0, 0.0]])
V = np.eye(2) + H
N = H - np.eye(2)
BASIS = np.stack([E / math.sqrt(2.0), V / 2.0])
D = np.array([[0.0, math.sqrt(2.0)], [math.sqrt(2.0), 0.0]])


def discrepancy(v, kind):
    """Direct BKM-minus-metric difference a*(ell-m); no cancellation."""
    v = np.asarray(v)
    if kind == "bkm":
        return np.zeros_like(v)
    if kind == "flat":
        result = np.zeros_like(v)
        nz = v != 0
        result[nz] = -0.25 * np.exp(-1 / v[nz] ** 2)
        return result
    if kind == "plateau":
        result = np.zeros_like(v)
        nz = v * v > 0.25 ** 2
        result[nz] = -0.25 * np.exp(-1 / (v[nz] ** 2 - 0.25 ** 2))
        return result
    if kind == "sld":
        # Strictly convergent series at all declared test amplitudes.
        return sum(v ** (2 * k) / (2 * k + 1) for k in range(1, 160))
    raise ValueError(kind)


def quadrature(n, horizon):
    # Fixed intervals resolve early curvature and exponential tails separately.
    cuts = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float(horizon)]
    cuts = sorted(set(x for x in cuts if x <= horizon))
    x, w = leggauss(n)
    return (np.concatenate([(a+b)/2+(b-a)*x/2 for a, b in zip(cuts, cuts[1:])]),
            np.concatenate([(b-a)*w/2 for a, b in zip(cuts, cuts[1:])]))


def scalar_transform(epsilon, kind, order, horizon):
    t, w = quadrature(order, horizon)
    f = -np.expm1(-t)
    ep = np.exp(-t)
    h = t * ep
    hp = (1-t) * ep
    integrand = epsilon * (hp-h*ep/f) * discrepancy(epsilon*f, kind)
    return float(w @ integrand)


def matrix_line_response(epsilon, tau, kind, order, horizon):
    """Normal response minus the radial BKM endpoint term, exactly at lambda=0."""
    t, w = quadrature(order, horizon)
    beta, Q = np.linalg.eigh(np.eye(2) + tau * D)
    target_coordinates = np.array([math.sqrt(2.0) * epsilon, 0.0])
    amplitudes = Q.T @ target_coordinates
    zc = (-np.expm1(-t[:, None] * beta)) * amplitudes
    uc = np.exp(-t[:, None] * beta) * beta * amplitudes
    zc = zc @ Q.T
    uc = uc @ Q.T
    X = np.eye(2)[None, :, :] + np.einsum("ti,ijk->tjk", zc, BASIS)
    U = np.einsum("ti,ijk->tjk", uc, BASIS)
    eig, rot = np.linalg.eigh(X)
    assert eig.min() > 0
    a = (eig[:, :, None] + eig[:, None, :])/2
    v = (eig[:, :, None] - eig[:, None, :])/(2*a)
    transformed = np.swapaxes(rot, -1, -2) @ U @ rot
    RU = rot @ (discrepancy(v, kind) / a * transformed) @ np.swapaxes(rot, -1, -2)
    integral = np.einsum("t,tij->ij", w, RU)
    return float(np.sum(N * integral)/2), float(eig.min()), float(beta.min())


def exact_harmonic(n):
    return sum((Fraction(1, j) for j in range(1, n+1)), Fraction(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, epsilon in [("bkm", .5), ("sld", .2), ("flat", .6),
                          ("flat", .4), ("flat", .2), ("plateau", .2),
                          ("plateau", .6)]:
        coarse = scalar_transform(epsilon, kind, 32, 48)
        fine = scalar_transform(epsilon, kind, 64, 64)
        finite = []
        for tau in [.002, .001]:
            plus, eplus, bplus = matrix_line_response(epsilon, tau, kind, 64, 64)
            minus, eminus, bminus = matrix_line_response(epsilon, -tau, kind, 64, 64)
            fd = (plus-minus)/(2*tau)
            finite.append({"tau": tau, "central_weight_difference": fd,
                           "absolute_error": abs(fd-fine),
                           "relative_error": abs(fd-fine)/abs(fine) if fine else None,
                           "minimum_sampled_state_eigenvalue": min(eplus, eminus),
                           "minimum_weight_eigenvalue": min(bplus,bminus)})
        # For the scalar integral: |tail| <= eps sup|g| (T+3)e^-T.
        # Values here are monotone in |v| for every selected kernel.
        tail = epsilon * abs(float(discrepancy(np.array(epsilon), kind))) * 67 * math.exp(-64)
        records.append({"kernel": kind, "epsilon": epsilon,
                        "scalar_response": fine, "refinement_absolute": abs(coarse-fine),
                        "analytic_tail_upper_bound": tail,
                        "fixed_weight_matrix_checks": finite})

    exact = []
    for m in range(1, 5):
        h = Fraction(2 ** (4*m), (4*m+1) * math.comb(4*m, 2*m)**2)
        x, w = leggauss(2*m)
        positive = x > 0
        nodes, weights = x[positive]**2, w[positive]
        differences = [1/(2*j+1)-float(weights @ (nodes**j)) for j in range(2*m+1)]
        c = (exact_harmonic(4*m+1)-1)/(4*m)
        exact.append({"atoms": m, "exact_first_missed_moment": str(h),
                      "first_missed_moment_float": float(h),
                      "numeric_first_missed_moment": differences[-1],
                      "max_matched_error": max(abs(v) for v in differences[:-1]),
                      "canonical_first_response_coefficient": str(-c*h),
                      "sharp_non_lie_triple_order": 4*m+1,
                      "sharp_lie_triple_nonideal_order": 4*m+2,
                      "all_moment_error_lower_bound": str(h/(4**m))})

    # Exact algebra check for canonical S and D E=V, using rational matrices.
    AEV = E @ (E @ V - V @ E) - (E @ V - V @ E) @ E
    normal_coordinate = float(np.sum(N*AEV)/2)
    passed = (normal_coordinate == 4.0
              and all(r["fixed_weight_matrix_checks"][-1]["absolute_error"] <
                      max(1e-25, abs(r["scalar_response"])*2e-3) for r in records)
              and all(r["refinement_absolute"] < max(1e-28, abs(r["scalar_response"])*1e-8)
                      for r in records)
              and all(r["max_matched_error"] < 2e-14 for r in exact)
              and records[0]["scalar_response"] == 0
              and records[5]["scalar_response"] == 0
              and all(records[i]["scalar_response"] > 0 for i in [2,3,4,6]))
    out = {"scope": "Exact first damping response; independent matrix-path weight differences versus scalar transform. No finite-lambda trajectory claim.",
           "python": platform.python_version(), "numpy": np.__version__,
           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "canonical_matrices": {k: x.tolist() for k, x in [("E",E),("H",H),("V",V),("N",N),("orthonormal_weight_D",D)]},
           "normal_ad_E_squared_V_coordinate": normal_coordinate,
           "records": records, "classical_atom_ceiling_checks": exact,
           "passed": bool(passed)}
    (args.output / "results.json").write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": bool(passed), "records": len(records),
                      "largest_relative_weight_check_error": max((r["fixed_weight_matrix_checks"][-1]["relative_error"] or 0) for r in records),
                      "output": str(args.output / "results.json")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
