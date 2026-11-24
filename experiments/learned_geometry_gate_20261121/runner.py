"""Train full initial-tangent and nonlinear models with progress controls.

This runner never evaluates held-out inputs. All timing is diagnostic because
the common step controller and target search perform additional Jacobian work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pullback_feature_learning import core
from pullback_feature_learning.data import two_moons

ARMS = ("nonlinear_gd", "nonlinear_pb", "tangent_gd", "tangent_pb", "nonlinear_fixed_pb")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def make_data(task, seed, n=256):
    """Balanced class-conditional synthetic draws; no held-out state is read."""
    if n < 2 or n % 2:
        raise ValueError("An even positive sample count is required.")
    if task == "moons":
        return two_moons(n, 0.15, seed)
    if task != "rings":
        raise ValueError(f"Unknown task: {task}")
    rng = np.random.default_rng(seed)
    labels = np.repeat([-1., 1.], n // 2)
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.where(labels < 0, .5, 1.2)
    xy = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    xy += rng.normal(0, .15, xy.shape)
    order = rng.permutation(n)
    return xy[order], labels[order]


def predict(theta0, theta, x, linear=False):
    """Full representation map, including the unread coordinate on new inputs."""
    if not linear:
        return core.representation(theta, x)
    return (core.representation(theta0, x).ravel()
            + core.representation_jacobian(theta0, x) @ (theta - theta0)).reshape(-1, 2)


def direction(theta, x, y, z, arm, damping, j0):
    """Return direction, current predictive Jacobian, and parameter gradient."""
    linear = arm.startswith("tangent")
    jac = j0 if linear else core.representation_jacobian(theta, x)
    gradient = jac.T @ core.representation_gradient(z, y).ravel()
    if arm.endswith("gd"):
        return -gradient, jac, gradient
    metric_jac = j0 if (linear or arm == "nonlinear_fixed_pb") else jac
    metric = metric_jac.T @ metric_jac
    metric.flat[::core.PARAMETER_COUNT + 1] += damping
    return np.linalg.solve(metric, -gradient), jac, gradient


def train_one(config, task, seed, arm, damping, step_size, x, y, theta0):
    """Train once; return snapshots, sparse records, and one terminal summary."""
    linear = arm.startswith("tangent")
    theta = theta0.copy()
    j0 = core.representation_jacobian(theta0, x)
    z0 = core.representation(theta0, x)
    z = z0.copy()
    loss0 = core.logistic_loss(z, y)
    loss = loss0
    fractions = config["loss_fractions"]
    targets = [loss0 * float(q) for q in fractions]
    index = 0
    snapshots, trajectory = [], []
    ids = dict(task=task, seed=str(seed["id"]), cohort=seed.get("cohort", "unspecified"),
               arm=arm, damping=damping, step_size=step_size)
    diag = dict(cumulative_logit_path=0., cumulative_representation_path=0.,
                cumulative_second_coordinate_path=0., backtracks=0,
                target_searches=0, max_taylor_ratio=0., max_parameter_step=0.,
                scalar_clock=0., max_loss_increase=0.)
    start = time.perf_counter()

    def forward(t):
        if linear:
            return (z0.ravel() + j0 @ (t - theta0)).reshape(-1, 2)
        return core.representation(t, x)

    def record(step, fraction, status, target=None, save_state=True):
        row = dict(**ids, fraction=fraction, step=step, train_loss=loss,
                   initial_loss=loss0, target_error=0. if target is None else loss-target,
                   status=status, elapsed_training_seconds=time.perf_counter()-start, **diag)
        trajectory.append(row.copy())
        if save_state:
            snapshots.append((row, theta.copy()))
        return row

    record(0, 1., "initial")
    status = "max_steps"
    step = 0
    for step in range(1, int(config.get("max_steps", 6000)) + 1):
        update, jac, gradient = direction(theta, x, y, z, arm, damping, j0)
        slope = float(gradient @ update)
        velocity = jac[::2] @ update
        vrms = float(np.sqrt(np.mean(velocity ** 2)))
        dnorm = float(np.linalg.norm(update))
        if not np.isfinite(slope) or not np.isfinite(dnorm) or slope >= 0 or vrms < 1e-20:
            status = "stalled_or_nonfinite"
            step -= 1
            break
        eta = min(step_size / vrms, config.get("param_step_cap", .1) / dnorm)
        accepted = False
        for backtrack in range(61):
            trial_theta = theta + eta * update
            trial_z = forward(trial_theta)
            trial_loss = core.logistic_loss(trial_z, y)
            rem = float(np.sqrt(np.mean((trial_z[:, 0] - z[:, 0] - eta * velocity) ** 2)))
            if (np.isfinite(trial_loss) and np.all(np.isfinite(trial_theta))
                    and trial_loss <= loss + config.get("armijo", 1e-4) * eta * slope
                    and rem <= config.get("remainder_rtol", .05) * eta * vrms + 1e-12):
                accepted = True
                diag["backtracks"] += backtrack
                break
            eta *= .5
        if not accepted:
            status = "line_search_failed"
            step -= 1
            break
        crossing = trial_loss <= targets[index]
        if crossing:
            # Restrict this actual Euler update to the first target. No endpoint
            # interpolation is used; the resulting state seeds the next update.
            lower, upper = 0., eta
            for _ in range(80):
                mid = (lower + upper) / 2
                mid_z = forward(theta + mid * update)
                mid_loss = core.logistic_loss(mid_z, y)
                diag["target_searches"] += 1
                if mid_loss <= targets[index]:
                    upper = mid
                else:
                    lower = mid
                if mid_loss <= targets[index] and abs(mid_loss - targets[index]) <= config.get("target_atol", 1e-10):
                    break
            eta = upper
            trial_theta = theta + eta * update
            trial_z = forward(trial_theta)
            trial_loss = core.logistic_loss(trial_z, y)
            rem = float(np.sqrt(np.mean((trial_z[:, 0] - z[:, 0] - eta * velocity) ** 2)))
            # The smaller accepted step must independently retain the controls.
            if not (trial_loss <= loss + config.get("armijo", 1e-4) * eta * slope
                    and rem <= config.get("remainder_rtol", .05) * eta * vrms + 1e-12):
                raise ArithmeticError("A target-shortened step lost its acceptance conditions.")
            if abs(trial_loss - targets[index]) > config.get("target_atol", 1e-10):
                raise ArithmeticError("A target crossing failed its declared loss tolerance.")
        movement = trial_z - z
        diag["cumulative_logit_path"] += float(np.sqrt(np.mean(movement[:, 0] ** 2)))
        diag["cumulative_second_coordinate_path"] += float(np.sqrt(np.mean(movement[:, 1] ** 2)))
        diag["cumulative_representation_path"] += float(np.sqrt(np.mean(np.sum(movement ** 2, axis=1))))
        diag["max_taylor_ratio"] = max(diag["max_taylor_ratio"], rem / max(eta * vrms, 1e-300))
        diag["max_parameter_step"] = max(diag["max_parameter_step"], eta * dnorm)
        diag["scalar_clock"] += eta
        diag["max_loss_increase"] = max(diag["max_loss_increase"], trial_loss - loss)
        theta, z, loss = trial_theta, trial_z, trial_loss
        if crossing:
            record(step, fractions[index], "target", targets[index])
            index += 1
            if index == len(targets):
                status = "complete"
                break
        elif step % int(config.get("log_every", 100)) == 0:
            record(step, -1., "trajectory", save_state=False)
    if status != "complete":
        record(step, -1., status)
    summary = dict(**ids, status=status, completed_targets=index, step=step,
                   train_loss=loss, initial_loss=loss0,
                   elapsed_training_seconds=time.perf_counter()-start, **diag)
    return snapshots, trajectory, summary


def validate(config):
    if not config["tasks"] or any(t not in ("moons", "rings") for t in config["tasks"]):
        raise ValueError("Tasks must be moons and/or rings.")
    if any(a not in ARMS for a in config["arms"]):
        raise ValueError("Unknown arm.")
    q = config["loss_fractions"]
    if not q or not all(0 < v < 1 for v in q) or list(q) != sorted(set(q), reverse=True):
        raise ValueError("Loss fractions must be unique and strictly descending in (0,1).")
    if any(v <= 0 or not np.isfinite(v) for v in config["dampings"] + config["step_sizes"]):
        raise ValueError("Dampings and step sizes must be positive finite values.")
    ids = [str(seed["id"]) for seed in config["seeds"]]
    if len(ids) != len(set(ids)) or any("__" in s for s in ids):
        raise ValueError("Seed IDs must be unique and cannot contain __.")


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate(config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    source_paths = [Path(__file__), REPO / "src/pullback_feature_learning/core.py",
                    REPO / "src/pullback_feature_learning/data.py"]
    provenance = dict(started_utc=utcnow(), config_source=str(config_path.resolve()),
                      config_sha256=digest(config_path), python=sys.version,
                      platform=platform.platform(), numpy=np.__version__,
                      source_sha256={str(p.resolve()): digest(p) for p in source_paths},
                      timing_role="diagnostic_only_includes_shared_controller_and_jacobians")
    write_json(output / "provenance.json", provenance)
    bank, data, rows, trajectories, summaries, actual_seeds = {}, {}, [], [], [], []
    with threadpool_limits(limits=1):
        provenance["threadpools"] = threadpool_info()
        for task in config["tasks"]:
            for seed in config["seeds"]:
                offset = 10000 if task == "rings" else 0
                actual = dict(task=task, **seed)
                for field in ("train_seed", "init_seed", "test_seed"):
                    if field in actual:
                        actual[field] = int(actual[field]) + offset
                actual_seeds.append(actual)
                x, y = make_data(task, actual["train_seed"], config.get("n_train", 256))
                theta0 = core.initialize_parameters(actual["init_seed"])
                key = f"{task}__{seed['id']}"
                data.update({key + "__x": x, key + "__y": y, key + "__theta0": theta0})
                for arm in config["arms"]:
                    for damping in ([None] if arm.endswith("gd") else config["dampings"]):
                        for step_size in config["step_sizes"]:
                            snaps, trace, summary = train_one(config, task, seed, arm, damping,
                                                              step_size, x, y, theta0)
                            for row, theta in snaps:
                                state_key = f"s{len(bank):06d}"
                                bank[state_key] = theta
                                rows.append(dict(state_key=state_key, **row))
                            trajectories.extend(trace)
                            summaries.append(summary)
                            print(json.dumps({k: summary[k] for k in ("task", "seed", "arm", "damping",
                                              "step_size", "status", "step", "train_loss")}), flush=True)
                            # Retain completed prefixes if a later run is interrupted.
                            np.savez_compressed(output / "states.npz", **bank)
                            np.savez_compressed(output / "data.npz", **data)
                            write_csv(output / "snapshots.csv", rows)
                            write_json(output / "snapshots.json", rows)
                            write_json(output / "actual_seeds.json", actual_seeds)
                            write_csv(output / "trajectory.csv", trajectories)
                            write_csv(output / "summaries.csv", summaries)
    provenance["finished_utc"] = utcnow()
    provenance["run_count"] = len(summaries)
    provenance["completed_runs"] = sum(r["status"] == "complete" for r in summaries)
    provenance["state_count"] = len(bank)
    provenance["output_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.name != "provenance.json"}
    write_json(output / "provenance.json", provenance)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
