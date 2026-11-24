"""Verify an elementary curvature counterexample; this is not a new theorem.

The training representation is (u, a*u + b*v + c*u**2/2), its readout
is u, and an arbitrarily chosen unseen-input prediction is v.  The toy
is an alternative explanation/sanity check, not a residual-network result.
Only Python's standard library is required.  Existing output is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys


A = 0.7
B = 0.8
DAMPING = 0.1
TARGET_U = 1.0


def field(state: tuple[float, float], curvature: float,
          adaptive: bool) -> tuple[float, float]:
    """Damped representation-pullback flow for logistic label +1."""
    u, _ = state
    q = A + curvature * u if adaptive else A
    m00 = 1.0 + q * q + DAMPING
    m01 = B * q
    m11 = B * B + DAMPING
    determinant = m00 * m11 - m01 * m01
    loss_derivative = -1.0 / (1.0 + math.exp(u))
    return (-m11 * loss_derivative / determinant,
            m01 * loss_derivative / determinant)


def rk4(state: tuple[float, float], step: float, curvature: float,
        adaptive: bool) -> tuple[float, float]:
    def shifted(k: tuple[float, float], fraction: float):
        return tuple(state[j] + step * fraction * k[j] for j in (0, 1))

    k1 = field(state, curvature, adaptive)
    k2 = field(shifted(k1, 0.5), curvature, adaptive)
    k3 = field(shifted(k2, 0.5), curvature, adaptive)
    k4 = field(shifted(k3, 1.0), curvature, adaptive)
    return tuple(state[j] + step * (k1[j] + 2.0 * k2[j]
                                   + 2.0 * k3[j] + k4[j]) / 6.0
                 for j in (0, 1))


def integrate(curvature: float, adaptive: bool, step: float) -> dict:
    """Take actual RK4 steps, shortening the last step to attain u=1."""
    state = (0.0, 0.0)
    elapsed = 0.0
    steps = 0
    while steps < 100000:
        candidate = rk4(state, step, curvature, adaptive)
        actual_step = step
        if candidate[0] >= TARGET_U:
            low, high = 0.0, step
            for _ in range(60):
                middle = 0.5 * (low + high)
                candidate = rk4(state, middle, curvature, adaptive)
                if candidate[0] < TARGET_U:
                    low = middle
                else:
                    high = middle
            actual_step = 0.5 * (low + high)
            candidate = rk4(state, actual_step, curvature, adaptive)
            state = candidate
            elapsed += actual_step
            steps += 1
            break
        state = candidate
        elapsed += actual_step
        steps += 1
    else:
        raise RuntimeError("Failed to attain the declared training prediction")
    u, v = state
    exact_v = -B * (A * u + (curvature * u * u / 2.0
                             if adaptive else 0.0)) / (B * B + DAMPING)
    return {
        "curvature": curvature,
        "adaptive_metric": adaptive,
        "rk4_step": step,
        "accepted_steps": steps,
        "flow_time": elapsed,
        "training_prediction": u,
        "training_logistic_loss": math.log1p(math.exp(-u)),
        "unseen_prediction": v,
        "unseen_logistic_loss_label_positive": math.log1p(math.exp(-v)),
        "exact_unseen_prediction": exact_v,
        "exact_relation_absolute_error": abs(v - exact_v),
        "target_absolute_error": abs(u - TARGET_U),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    # J0 = [[1, 0], [a, b]] is independent of curvature c.
    gram00, gram01, gram11 = 1.0 + A * A, A * B, B * B
    trace = gram00 + gram11
    gap = math.sqrt((gram00 - gram11) ** 2 + 4.0 * gram01 ** 2)
    eigenvalues = [(trace - gap) / 2.0, (trace + gap) / 2.0]
    rows, contrasts = [], []
    for step in (0.005, 0.0025):
        fixed = integrate(0.0, False, step)
        rows.append(fixed)
        for curvature in (-1.0, 1.0):
            learned = integrate(curvature, True, step)
            rows.append(learned)
            predicted_delta = -B * curvature * TARGET_U ** 2 / (2.0 * (B * B + DAMPING))
            contrasts.append({
                "curvature": curvature,
                "rk4_step": step,
                "matched_training_prediction": TARGET_U,
                "unseen_prediction_difference": learned["unseen_prediction"] - fixed["unseen_prediction"],
                "exact_prediction_difference": predicted_delta,
                "unseen_logistic_risk_difference": learned["unseen_logistic_loss_label_positive"] - fixed["unseen_logistic_loss_label_positive"],
            })

    # Separate squared-loss endpoint example: labels y=s+epsilon,
    # epsilon equally likely +/-sigma.  This is an exact two-point
    # expectation, not an assertion about noisy logistic convergence.
    signal, noise_sd = 0.7, 0.4
    noise_rows = []
    for curvature in (-1.0, 1.0):
        coefficient = -B * curvature / (2.0 * (B * B + DAMPING))
        labels = [signal - noise_sd, signal + noise_sd]
        enumerated = sum(coefficient * y * y for y in labels) / 2.0
        analytic = coefficient * (signal * signal + noise_sd * noise_sd)
        noise_rows.append({
            "loss": "squared loss; exact converged endpoint u=y",
            "curvature": curvature,
            "signal": signal,
            "noise_standard_deviation": noise_sd,
            "noise_mean": 0.0,
            "enumerated_expected_prediction_difference": enumerated,
            "analytic_expected_prediction_difference": analytic,
            "noiseless_prediction_difference": coefficient * signal * signal,
            "noise_only_expected_difference": coefficient * noise_sd * noise_sd,
            "absolute_expectation_error": abs(enumerated - analytic),
        })

    max_relation_error = max(row["exact_relation_absolute_error"] for row in rows)
    max_target_error = max(row["target_absolute_error"] for row in rows)
    assert max_relation_error < 1e-10, max_relation_error
    assert max_target_error < 1e-12, max_target_error
    assert all(row["unseen_logistic_risk_difference"] * row["curvature"] > 0.0
               for row in contrasts)
    assert max(row["absolute_expectation_error"] for row in noise_rows) < 1e-14
    report = {
        "status": "verified elementary counterexample; no novelty claim",
        "scope": "Arbitrary train/test representation map; not an architecture theorem or a rings result",
        "parameters": {"a": A, "b": B, "damping": DAMPING, "target_u": TARGET_U},
        "initial_jacobian_shared_by_both_curvatures": [[1.0, 0.0], [A, B]],
        "initial_gram_eigenvalues_shared": eigenvalues,
        "initial_metric_eigenvalues_shared": [value + DAMPING for value in eigenvalues],
        "logistic_flow_rows": rows,
        "matched_progress_contrasts": contrasts,
        "separate_squared_loss_noise_expectation": noise_rows,
        "checks": {"max_exact_relation_error": max_relation_error,
                   "max_training_target_error": max_target_error,
                   "benefit_and_harm_at_identical_initial_spectra": True,
                   "zero_mean_noise_contributes_to_learned_geometry": True},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "checks": report["checks"]}))


if __name__ == "__main__":
    main()
