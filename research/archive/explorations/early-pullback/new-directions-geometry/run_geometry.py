"""Train four direction/length arms and evaluate common-state branch updates."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core
from pullback_feature_learning.data import two_moons


ARMS = {"gd": ("gd", "gd"), "pb": ("pb", "pb"),
        "gd_at_pb_size": ("gd", "pb"), "pb_at_gd_size": ("pb", "gd")}
CHECKPOINTS = (0, 1, 10, 50, 100, 500)


def rms_norm(array, n):
    return np.linalg.norm(array.ravel()) / np.sqrt(n)


def accuracy(z, y):
    return np.mean(np.where(z[:, 0] >= 0, 1, -1) == y)


def candidates(theta, x, y, z):
    jac = core.representation_jacobian(theta, x)
    grad = core.representation_gradient(z, y).ravel()
    gd_step = -0.125 * core.parameter_gradient(theta, x, y)
    normal_matrix = jac.T @ jac
    normal_matrix.flat[::1 + len(theta)] += 0.1
    pb_step = 8.0 * np.linalg.solve(normal_matrix, -(jac.T @ grad))
    steps = {"gd": gd_step, "pb": pb_step}
    velocities = {name: jac @ step for name, step in steps.items()}
    lengths = {name: rms_norm(velocity, len(x)) for name, velocity in velocities.items()}
    return jac, grad, steps, velocities, lengths


def scaled_step(steps, lengths, direction, size):
    if direction == size:
        return steps[direction]
    if lengths[size] == 0:
        return np.zeros_like(steps[direction])
    if lengths[direction] == 0:
        raise ArithmeticError("Positive length with zero source direction.")
    return steps[direction] * (lengths[size] / lengths[direction])


def same_state_branches(seed, origin, step, theta, x, y, xv, yv, z, c):
    jac, grad, steps, velocities, lengths = c
    base_train = core.logistic_loss(z, y)
    base_val = core.logistic_loss(core.representation(theta, xv), yv)
    records = []
    for size in ("gd", "pb"):
        for direction in ("gd", "pb"):
            delta = scaled_step(steps, lengths, direction, size)
            vel = velocities[direction]
            cosine = np.dot(-grad, vel) / (np.linalg.norm(grad) * np.linalg.norm(vel))
            predicted = rms_norm(jac @ delta, len(x))
            np.testing.assert_allclose(predicted, lengths[size], atol=1e-12, rtol=1e-10)
            after = core.representation(theta + delta, x)
            val_after = core.representation(theta + delta, xv)
            records.append(dict(seed=seed, origin=origin, step_before=step, size=size,
                                direction=direction, target_rms=lengths[size], predicted_rms=predicted,
                                actual_rms=rms_norm(after - z, len(x)), alignment_cosine=cosine,
                                train_loss_change=core.logistic_loss(after, y) - base_train,
                                validation_loss_change=core.logistic_loss(val_after, yv) - base_val))
    return records


def follow(seed, arm, x, y, xv, yv, theta, branches):
    actual = predicted = max_match_error = max_finite_gap = 0.0
    checkpoints = []
    z = core.representation(theta, x)
    for step in range(501):
        if step in CHECKPOINTS:
            zv = core.representation(theta, xv)
            checkpoints.append(dict(seed=seed, arm=arm, step=step,
                                    train_loss=core.logistic_loss(z, y), train_accuracy=accuracy(z, y),
                                    validation_loss=core.logistic_loss(zv, yv), validation_accuracy=accuracy(zv, yv),
                                    cumulative_predicted_rms=predicted, cumulative_actual_rms=actual,
                                    max_matching_error=max_match_error, max_actual_predicted_gap=max_finite_gap))
        if step == 500:
            break
        c = candidates(theta, x, y, z)
        jac, _, steps, _, lengths = c
        if (step == 0 and arm == "gd") or (step in (99, 499) and arm in ("gd", "pb")):
            if step == 0:
                for name, method, rate in (("gd", "gd", 0.125), ("pb", "pullback", 8.0)):
                    ref, _ = core.direction(theta, x, y, method, damping=0.1)
                    np.testing.assert_allclose(steps[name], rate * ref, atol=2e-12, rtol=1e-10)
            origin = "shared" if step == 0 else arm
            branches.extend(same_state_branches(seed, origin, step, theta, x, y, xv, yv, z, c))
        direction, size = ARMS[arm]
        delta = scaled_step(steps, lengths, direction, size)
        length = rms_norm(jac @ delta, len(x))
        gap = abs(length - lengths[size])
        if not np.isfinite(length) or gap > 1e-12 + 1e-10 * lengths[size]:
            raise ArithmeticError(f"Feature step mismatch: {length}, {lengths[size]}")
        theta = theta + delta
        next_z = core.representation(theta, x)
        realized = rms_norm(next_z - z, len(x))
        if not np.isfinite(realized):
            raise ArithmeticError("Nonfinite feature movement.")
        actual += realized
        predicted += length
        max_match_error = max(max_match_error, gap)
        max_finite_gap = max(max_finite_gap, abs(realized - length))
        z = next_z
    return checkpoints


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    index = {(row["seed"], row["arm"], row["step"]): row for row in rows}
    assert len(index) == len(rows) == len(CHECKPOINTS) * 8 * len(ARMS)
    summary = {}
    for step in (100, 500):
        summary[step] = {}
        for arm in ARMS:
            group = [index[s, arm, step] for s in range(1, 9)]
            summary[step][arm] = {
                k: {"mean": float(np.mean([r[k] for r in group])),
                    "sample_sd": float(np.std([r[k] for r in group], ddof=1))}
                for k in ("train_loss", "validation_loss", "validation_accuracy",
                          "cumulative_predicted_rms", "cumulative_actual_rms")}
    pairs = []
    for step in (100, 500):
        for candidate, reference in (("pb", "gd_at_pb_size"), ("pb_at_gd_size", "gd"),
                                     ("pb", "gd")):
            for seed in range(1, 9):
                pairs.append(dict(seed=seed, step=step, candidate=candidate, reference=reference,
                                  **{k + "_difference": index[seed, candidate, step][k] -
                                     index[seed, reference, step][k] for k in
                                     ("validation_accuracy", "validation_loss")}))
    return summary, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    rows, branches = [], []
    with threadpool_limits(limits=1), np.errstate(invalid="raise", divide="raise", over="raise"):
        for seed in range(1, 9):
            x, y = two_moons(256, 0.15, 61000 + seed)
            xv, yv = two_moons(256, 0.15, 63000 + seed)
            initial = core.initialize_parameters(62000 + seed)
            for arm in ARMS:
                rows.extend(follow(seed, arm, x, y, xv, yv, initial.copy(), branches))
    summary, pairs = summarize(rows)
    assert len(branches) == 8 * 5 * 4
    write_csv(output / "checkpoints.csv", rows)
    write_csv(output / "branches.csv", branches)
    write_csv(output / "paired_differences.csv", pairs)
    results = dict(endpoints=summary, runs=8 * len(ARMS), steps_per_run=500,
                   max_matching_error=max(r["max_matching_error"] for r in rows),
                   checkout_core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                   protocol_sha256=hashlib.sha256(Path(__file__).with_name("protocol.md").read_bytes()).hexdigest(),
                   numpy=np.__version__, python=platform.python_version(), blas_threads=1)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(results, stream, allow_nan=False, indent=2)
    print(f"Completed {results['runs']} trajectories and {len(branches)} one-step diagnostics.")


if __name__ == "__main__":
    main()
