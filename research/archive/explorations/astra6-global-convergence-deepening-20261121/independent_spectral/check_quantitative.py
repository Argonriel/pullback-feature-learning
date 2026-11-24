"""Focused, deterministic 2-by-2 checks for the quantitative supplement.

No data, training, external dependencies beyond NumPy, or old experiments are used.
Run this file with the bundled Python interpreter; it writes only beside itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
I = np.eye(2)
E = np.diag([1.0, -1.0]) / math.sqrt(2.0)
F = np.array([[0.0, 1.0], [1.0, 0.0]]) / math.sqrt(2.0)
BASIS = np.array([E, F])


def sym(x):
    return 0.5 * (x + x.T)


def logm(x):
    values, vectors = np.linalg.eigh(sym(x))
    if min(values) <= 0:
        raise ValueError("non-SPD argument")
    return (vectors * np.log(values)) @ vectors.T


def divergence(y, x, damping):
    return float(0.5 * np.sum((y - x) ** 2) + damping / 2.0 *
                 np.trace(y @ (logm(y) - logm(x)) - y + x))


def coords(x):
    return np.einsum("aij,ij->a", BASIS, x)


def gradient(x, y, weight):
    r = coords(x - y)
    return np.einsum("a,aij->ij", weight @ r, BASIS)


def mobility(x, u, damping):
    values, vectors = np.linalg.eigh(sym(x))
    sums = values[:, None] + values[None, :]
    return sym(vectors @ ((sums / (sums + damping)) * (vectors.T @ u @ vectors)) @ vectors.T)


def rhs(x, y, weight, damping):
    return -mobility(x, gradient(x, y, weight), damping)


def loss(x, y, weight):
    r = coords(x - y)
    return float(0.5 * r @ weight @ r)


def bisect_decreasing(function, target, lower, upper, steps=180):
    for _ in range(steps):
        midpoint = (lower + upper) / 2.0
        if function(midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def endpoint_from_invariant(x0, y, damping):
    traceless = y - np.trace(y) / 2.0 * I
    radius = float(np.linalg.norm(traceless, 2))
    invariant = float(np.trace(x0) + damping / 2.0 * np.linalg.slogdet(x0)[1])
    lo, hi = radius, max(radius + 1.0, float(np.linalg.norm(x0, 2)) + 1.0)
    def value(center):
        return 2.0 * center + damping / 2.0 * math.log(center * center - radius * radius)
    while value(hi) < invariant:
        hi *= 2.0
    for _ in range(180):
        mid = (lo + hi) / 2.0
        if value(mid) < invariant:
            lo = mid
        else:
            hi = mid
    return traceless + (lo + hi) / 2.0 * I


def rk4(x, step, y, weight, damping):
    k1 = rhs(x, y, weight, damping)
    k2 = rhs(x + step / 2.0 * k1, y, weight, damping)
    k3 = rhs(x + step / 2.0 * k2, y, weight, damping)
    k4 = rhs(x + step * k3, y, weight, damping)
    return sym(x + step / 6.0 * (k1 + 2*k2 + 2*k3 + k4))


def integrate(x0, y, weight, damping, tolerance, horizon, bound_m):
    t, step, x = 0.0, 0.05, x0.copy()
    initial_d = divergence(y, x0, damping)
    initial_f = loss(x0, y, weight)
    beta = float(np.linalg.eigvalsh(weight)[0])
    exponent = 2 * beta * 2 * bound_m / (2 * bound_m + damping)
    invariant0 = float(np.trace(x0) + damping / 2 * np.linalg.slogdet(x0)[1])
    minima, maxima, divergences, loss_ratios, inv_errors = [], [], [], [], []
    accepted = rejected = 0
    max_accepted_error_ratio = 0.0
    min_step = float("inf")
    max_step = 0.0
    rejections = []
    def sample():
        values = np.linalg.eigvalsh(x)
        minima.append(float(min(values)))
        maxima.append(float(max(values)))
        divergences.append(divergence(y, x, damping))
        loss_ratios.append(loss(x, y, weight) * math.exp(exponent * t) / initial_f)
        inv_errors.append(abs(float(np.trace(x) + damping / 2 * np.linalg.slogdet(x)[1])-invariant0))
    sample()
    while t < horizon:
        step = min(step, horizon - t)
        whole = rk4(x, step, y, weight, damping)
        half = rk4(x, step / 2, y, weight, damping)
        twice = rk4(half, step / 2, y, weight, damping)
        difference = (twice - whole) / 15.0
        scale = tolerance * (1 + max(float(np.linalg.norm(x)), float(np.linalg.norm(twice))))
        error_ratio = float(np.linalg.norm(difference)) / scale
        trial = sym(twice + difference)
        positive = float(np.linalg.eigvalsh(trial)[0]) > 0
        if error_ratio <= 1 and positive:
            x, t = trial, t + step
            accepted += 1
            max_accepted_error_ratio = max(max_accepted_error_ratio, error_ratio)
            min_step, max_step = min(min_step, step), max(max_step, step)
            sample()
        else:
            rejected += 1
            rejections.append({"t": t, "step": step, "error_ratio": error_ratio, "positive": positive})
        factor = 2.0 if error_ratio == 0 else max(0.2, min(2.0, 0.9 * error_ratio ** (-0.2)))
        if not positive:
            factor = min(factor, 0.5)
        step = min(0.5, step * factor)
    reference = endpoint_from_invariant(x0, y, damping)
    return {
        "tolerance_absolute_and_relative": tolerance,
        "horizon": horizon,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "rejections": rejections,
        "min_accepted_step": min_step,
        "max_accepted_step": max_step,
        "max_accepted_error_ratio": max_accepted_error_ratio,
        "sampled_min_eigenvalue": min(minima),
        "sampled_max_eigenvalue": max(maxima),
        "max_reverse_bregman_increase_above_initial": max(divergences) - initial_d,
        "max_successive_reverse_bregman_increase": max(np.diff(divergences)),
        "max_loss_to_proved_exponential_bound_ratio": max(loss_ratios),
        "max_central_invariant_error": max(inv_errors),
        "endpoint": x.tolist(),
        "reference_endpoint": reference.tolist(),
        "endpoint_error_to_independent_invariant_solution": float(np.linalg.norm(x-reference)),
        "final_loss": loss(x, y, weight),
        "final_measurement_residual_norm": float(np.linalg.norm(coords(x-y))),
    }


def main():
    epsilon = math.exp(-10)
    cx = np.diag([epsilon, 1.0])
    cy = cx + 2*E + F + I
    cb = np.array([[1.0, -0.9], [-0.9, 1.0]])
    c = (1 + epsilon) / (3 + epsilon) * (1 + 10 / (1 - epsilon))
    exact_derivative = 0.8*c - 2.2
    velocity = rhs(cx, cy, cb, 2.0)
    differences = []
    for step in [1e-3, 1e-4, 1e-5, 1e-6]:
        estimate = (divergence(cy, cx + step*velocity, 2.0) -
                    divergence(cy, cx - step*velocity, 2.0))/(2*step)
        differences.append({"step": step, "estimate": estimate,
                            "absolute_error": abs(estimate-exact_derivative)})
    x0 = np.array([[1.8, 0.35], [0.35, 0.7]])
    y = np.array([[0.8, 0.25], [0.25, 1.5]])
    damping = 1.3
    weight = 0.8 * np.eye(2)
    eta = float(np.linalg.eigvalsh(y)[0])
    d0 = divergence(y, x0, damping)
    def h(value):
        return 0.5*(eta-value)**2 + damping/2*(eta*math.log(eta/value)-eta+value)
    lower = bisect_decreasing(h, d0, 0.0, eta)
    upper = float(np.linalg.norm(y, 2)) + math.sqrt(2*d0)
    elementary = eta * math.exp(-1 - 2*d0/(damping*eta))
    runs = [integrate(x0, y, weight, damping, tolerance, 40.0, lower)
            for tolerance in [2e-9, 2e-11]]
    results = {
        "purpose": "One noncommuting scalar-weight spectral/rate check at two tolerances, plus one exact failure of the arbitrary-weight Lyapunov shortcut.",
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counterexample": {
            "lambda": 2.0, "epsilon": epsilon, "X": cx.tolist(), "Y": cy.tolist(), "B": cb.tolist(),
            "measurement_basis": BASIS.tolist(),
            "target_min_eigenvalue": float(np.linalg.eigvalsh(cy)[0]),
            "weight_eigenvalues": np.linalg.eigvalsh(cb).tolist(),
            "C_offdiagonal_multiplier": c,
            "analytic_reverse_bregman_derivative": exact_derivative,
            "proved_strict_lower_bound_on_derivative": 11/15,
            "loss_derivative": -float(np.sum(gradient(cx, cy, cb)*mobility(cx, gradient(cx, cy, cb), 2.0))),
            "finite_difference_checks": differences,
        },
        "scalar_weight_validation": {
            "X0": x0.tolist(), "feasible_Y": y.tolist(), "B": weight.tolist(), "lambda": damping,
            "commutator_norm_X0_Y": float(np.linalg.norm(x0@y-y@x0)),
            "eta": eta, "D0": d0, "uniform_spectral_lower_root": lower,
            "root_equation_absolute_error": abs(h(lower)-d0),
            "elementary_spectral_lower_bound": elementary,
            "uniform_spectral_upper_bound": upper,
            "proved_loss_exponent_from_time_zero": 2*0.8*2*lower/(2*lower+damping),
            "runs": runs,
            "endpoint_refinement_difference": float(np.linalg.norm(np.array(runs[0]["endpoint"])-np.array(runs[1]["endpoint"]))),
        },
        "limits": "All extrema are sampled numerical checks. The continuous-time bound and all-parameter claim come from the proof. This is not an originality certificate or a training experiment.",
    }
    assert exact_derivative > 11/15
    assert min(np.linalg.eigvalsh(cy)) > 0
    assert differences[-2]["absolute_error"] < 1e-5
    for run in runs:
        assert run["sampled_min_eigenvalue"] >= lower
        assert run["sampled_max_eigenvalue"] <= upper
        assert run["max_loss_to_proved_exponential_bound_ratio"] <= 1 + 1e-8
        assert run["endpoint_error_to_independent_invariant_solution"] < 1e-7
    results["checks_passed"] = True
    (ROOT/"results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"checks_passed": True, "derivative": exact_derivative,
                      "spectral_lower_bound": lower, "loss_exponent": results["scalar_weight_validation"]["proved_loss_exponent_from_time_zero"],
                      "endpoint_errors": [r["endpoint_error_to_independent_invariant_solution"] for r in runs],
                      "refinement_difference": results["scalar_weight_validation"]["endpoint_refinement_difference"]}, indent=2))


if __name__ == "__main__":
    main()
