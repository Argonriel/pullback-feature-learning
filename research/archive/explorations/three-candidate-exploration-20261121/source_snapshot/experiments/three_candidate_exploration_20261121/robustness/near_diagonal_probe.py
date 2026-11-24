"""Small exact variational probe; no model training or benchmark selection.

Linearize two measurement matrices E11, E22 + theta*(E12+E21)
at theta=0 for full representation PB product flow on SPD(2).
"""
import json
from pathlib import Path
import numpy as np


def adaptive_rk4(rhs, state, horizon, rtol=2e-9, atol=2e-11):
    state = np.array(state, dtype=float)
    time, step, count = 0., .01, 0
    minimum = float(state[:2].min())
    def rk4(t, y, h):
        k1 = rhs(t, y)
        k2 = rhs(t+h/2, y+h*k1/2)
        k3 = rhs(t+h/2, y+h*k2/2)
        k4 = rhs(t+h, y+h*k3)
        return y+h*(k1+2*k2+2*k3+k4)/6
    while time < horizon:
        step = min(step, horizon-time)
        one = rk4(time, state, step)
        half = rk4(time, state, step/2)
        two = rk4(time+step/2, half, step/2)
        error = np.max(np.abs(two-one)/(atol+rtol*np.maximum(abs(state), abs(two))))/15
        if np.isfinite(error) and error <= 1:
            state = two+(two-one)/15
            time += step
            count += 1
            minimum = min(minimum, float(state[:2].min()))
        step *= min(2, max(.1, .85 * max(error, 1e-10)**(-.2))) if np.isfinite(error) else .1
        if step < 1e-12 or count > 200000:
            raise RuntimeError("Integrator stalled")
    return state, minimum, count


def probe(lam, weight, initial=0.1):
    weight = np.asarray(weight, dtype=float)
    rates = np.linalg.eigvalsh(weight)
    horizon = 50 * (1 + lam / (2 * initial)) / rates[0]

    def rhs(t, state):
        x, y = np.exp(np.clip(state[:2], -745, 700))
        q = weight @ (np.array([x, y]) - 1)
        logvel = -2 * q / (2 * np.array([x, y]) + lam)
        zvel = -(x+y)/(x+y+lam) * q[1]
        zvel += lam/(2*(x+y+lam)) * logvel.sum() * state[2]
        return np.array([*logvel, zvel])

    state, minimum, steps = adaptive_rk4(rhs, [np.log(initial), np.log(initial), 0], horizon)
    x, y = np.exp(state[:2])
    return {"lambda": lam, "weight": weight.tolist(), "success": True,
            "endpoint_error": float(np.linalg.norm([x-1, y-1])),
            "offdiag_theta_derivative": float(state[2]),
            "minimum_log_diagonal": minimum,
            "steps": steps}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weights = [[[1, 0], [0, 1]], [[1, -2], [-2, 5]], [[5, -2], [-2, 1]]]
    rows = [probe(lam, w) for lam in [0.3, .1, .03, .01, .003, .001] for w in weights]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
