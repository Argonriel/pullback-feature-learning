"""Frozen-configuration paired evaluation at separate iteration/time endpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .core import direction, initialize_parameters, logistic_loss, representation
from .data import two_moons
from .pilot import fingerprint, spectral_diagnostic, write_csv, write_json


def validate_config(config):
    if config["stage"] != "formal_pending" or config["dtype"] != "float64":
        raise ValueError("Expected the frozen float64 formal configuration.")
    if config["termination_rule"] != "continue_until_both_budgets_reached":
        raise ValueError("Unexpected termination rule.")
    if config["time_endpoint_rule"] != "last_completed_update_at_or_before_budget":
        raise ValueError("Unexpected time-endpoint rule.")
    for key in ("iteration_endpoint", "threads", "log_every"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    budget = config["time_endpoint_seconds"]
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Time budget must be finite and positive.")
    if not isinstance(config["warmup_steps"], int) or config["warmup_steps"] < 0:
        raise ValueError("Warmup must be a nonnegative integer.")
    if set(config["methods"]) != {"gd", "pullback"}:
        raise ValueError("Both locked methods must be present.")
    for method, options in config["methods"].items():
        values = [options["learning_rate"]]
        if method == "pullback":
            values.append(options["damping"])
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Optimizer scales must be positive and finite.")
    seeds = config["seeds"]
    data_seeds = [seed[key] for seed in seeds for key in ("train_seed", "validation_seed", "test_seed")]
    if len(seeds) < 3 or len(set(data_seeds)) != len(data_seeds) or len({s["id"] for s in seeds}) != len(seeds):
        raise ValueError("At least three uniquely identified fresh paired splits are required.")


def train_trajectory(method, config, x, y, initial, clock=perf_counter):
    """Train using training data only; the injected clock measures updates only.

    Checkpoint copies and feature diagnostics are outside core-update timing.
    Exact time equality is eligible; one further update obtains a strict bracket.
    """
    target = config["iteration_endpoint"]
    budget = config["time_endpoint_seconds"]
    eta = config["methods"][method]["learning_rate"]
    damping = config["methods"][method].get("damping", 0.01)
    theta = initial.copy()
    step = 0
    core_seconds = diagnostic_seconds = warmup_seconds = displacement = 0.0
    saved = {}
    endpoints = {"iteration": None, "time": None, "time_crossing": None}
    reason = None
    phase = "warmup"
    start_wall = perf_counter()
    last_valid = (0, theta, 0.0, 0.0)

    def save(at_step, parameters, seconds, distance, label):
        if at_step not in saved:
            saved[at_step] = {"step": at_step, "training_seconds": float(seconds),
                              "cumulative_feature_displacement": float(distance),
                              "theta": parameters.tolist(), "labels": []}
        if label not in saved[at_step]["labels"]:
            saved[at_step]["labels"].append(label)
        return saved[at_step]

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            start = perf_counter()
            try:
                warm = initial.copy()
                for _ in range(config["warmup_steps"]):
                    u, _ = direction(warm, x, y, method, damping)
                    warm = warm + eta * u
                    if not np.isfinite(warm).all():
                        raise FloatingPointError("Nonfinite warmup parameters.")
            finally:
                warmup_seconds = perf_counter() - start
            phase = "training"
            start = perf_counter()
            previous_features = representation(theta, x)
            save(0, theta, 0.0, 0.0, "initial")
            diagnostic_seconds += perf_counter() - start
            while step < target or core_seconds <= budget:
                previous_theta, previous_time, previous_distance = theta, core_seconds, displacement
                start_update = clock()
                try:
                    u, _ = direction(theta, x, y, method, damping)
                    next_theta = theta + eta * u
                finally:
                    duration = clock() - start_update
                    if not np.isfinite(duration) or duration <= 0:
                        raise FloatingPointError("Core clock must advance by a finite positive duration.")
                    core_seconds += duration
                theta = next_theta
                step += 1
                start = perf_counter()
                try:
                    features = representation(theta, x)
                    if not np.isfinite(theta).all() or not np.isfinite(features).all():
                        raise FloatingPointError("Nonfinite training state.")
                    displacement += float(np.sqrt(np.mean(np.sum((features - previous_features)**2, axis=1))))
                    if not np.isfinite(displacement):
                        raise FloatingPointError("Nonfinite feature displacement.")
                    previous_features = features
                    if step % config["log_every"] == 0:
                        save(step, theta, core_seconds, displacement, "regular")
                    if step == target // 2:
                        save(step, theta, core_seconds, displacement, "iteration_middle")
                    if step == target:
                        endpoints["iteration"] = save(step, theta, core_seconds, displacement, "iteration")
                    if previous_time <= budget < core_seconds:
                        endpoints["time"] = save(step - 1, previous_theta, previous_time, previous_distance, "time")
                        endpoints["time_crossing"] = save(step, theta, core_seconds, displacement, "time_crossing")
                    last_valid = (step, theta, core_seconds, displacement)
                finally:
                    diagnostic_seconds += perf_counter() - start
            start = perf_counter()
            save(step, theta, core_seconds, displacement, "termination")
            diagnostic_seconds += perf_counter() - start
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
        reason = f"{phase}: {type(error).__name__}: {error}"
        start = perf_counter()
        save(*last_valid, "failure_prefix")
        diagnostic_seconds += perf_counter() - start
    complete = reason is None and step >= target and core_seconds > budget and all(endpoints.values())
    checkpoints = [saved[key] for key in sorted(saved)]
    return {"method": method, "status": "completed" if complete else "failed", "reason": reason,
            "total_steps": step, "training_seconds": core_seconds, "diagnostics_seconds": diagnostic_seconds,
            "warmup_seconds": warmup_seconds, "loop_wall_seconds": perf_counter() - start_wall,
            "initial_parameter_sha256": fingerprint(initial), "train_data_sha256": fingerprint(x, y),
            "history": [{k: v for k, v in point.items() if k != "theta"} for point in checkpoints],
            "checkpoints": checkpoints, "endpoints": endpoints}


def evaluate_checkpoint(point, splits):
    row = {k: v for k, v in point.items() if k not in ("theta", "labels")}
    row["labels"] = ";".join(point["labels"])
    theta = np.array(point["theta"], dtype=np.float64)
    for split, (x, y) in splits.items():
        z = representation(theta, x)
        loss = logistic_loss(z, y)
        if not np.isfinite(z).all() or not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite offline {split} evaluation.")
        row[f"{split}_loss"] = loss
        row[f"{split}_accuracy"] = float(np.mean(np.where(z[:, 0] >= 0, 1, -1) == y))
    return row


def summarize(endpoint_rows, trajectories, seed_ids):
    """Keep all per-seed values; no survivor-only aggregate after a failed run."""
    summary = {"methods": {}, "paired_differences": {}, "aggregate_complete": False}
    complete = len(trajectories) == 2 * len(seed_ids) and all(t["status"] == "completed" for t in trajectories)
    summary["aggregate_complete"] = complete
    if not complete:
        summary["reason"] = "At least one scheduled trajectory failed; inspect per-seed records."
        return summary
    metrics = ("train_loss", "validation_loss", "test_loss", "test_accuracy", "step", "training_seconds")
    keyed = {(r["seed_id"], r["method"], r["endpoint"]): r for r in endpoint_rows}
    for endpoint in ("iteration", "time"):
        summary["methods"][endpoint] = {}
        summary["paired_differences"][endpoint] = {}
        for method in ("gd", "pullback"):
            rows = [keyed[(seed, method, endpoint)] for seed in seed_ids]
            summary["methods"][endpoint][method] = {
                metric: {"mean": float(np.mean([r[metric] for r in rows])),
                         "sample_sd": float(np.std([r[metric] for r in rows], ddof=1))}
                for metric in metrics}
        for metric in ("test_loss", "test_accuracy"):
            differences = [keyed[(seed, "pullback", endpoint)][metric] - keyed[(seed, "gd", endpoint)][metric]
                           for seed in seed_ids]
            summary["paired_differences"][endpoint][metric] = {
                "definition": "pullback minus GD", "seed_order": seed_ids, "values": differences,
                "mean": float(np.mean(differences)), "sample_sd": float(np.std(differences, ddof=1))}
    return summary


def render_report(output, config, metrics, endpoint_rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gd": "#20639B", "pullback": "#D65F36"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for method in ("gd", "pullback"):
        for index, seed in enumerate(config["seeds"]):
            rows = sorted([r for r in metrics if r["method"] == method and r["seed_id"] == seed["id"]], key=lambda r: r["step"])
            for col, key, maximum in ((0, "step", config["iteration_endpoint"]), (1, "training_seconds", config["time_endpoint_seconds"])):
                within = [r for r in rows if r[key] <= maximum]
                for ax, metric in ((axes[0, col], "train_loss"), (axes[1, col], "test_accuracy")):
                    ax.plot([r[key] for r in within], [r[metric] for r in within],
                            color=colors[method], alpha=0.65, linestyle=("-", "--", ":")[index % 3],
                            label=f"{method}: {seed['id']}")
                    ax.grid(alpha=0.2)
    for col, xlabel in enumerate(("Iteration", "Core training time (seconds)")):
        axes[0, col].set(xlabel=xlabel, ylabel="Training logistic loss")
        axes[1, col].set(xlabel=xlabel, ylabel="Test accuracy", ylim=(0.0, 1.01))
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Frozen parameters: three fresh paired seeds; individual trajectories")
    fig.savefig(output / "learning_curves.svg")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    if summary["aggregate_complete"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
        for ax, endpoint in zip(axes, ("iteration", "time")):
            for index, seed in enumerate(config["seeds"]):
                values = [next(r["test_accuracy"] for r in endpoint_rows
                               if r["seed_id"] == seed["id"] and r["method"] == method and r["endpoint"] == endpoint)
                          for method in ("gd", "pullback")]
                ax.plot([0, 1], np.array(values) * 100, "o-", label=seed["id"], alpha=0.8)
            ax.set(xticks=[0, 1], xticklabels=["GD", "Pullback"], ylabel="Test accuracy (%)",
                   title="2,000 updates" if endpoint == "iteration" else "At or before 2.0 seconds")
            ax.grid(axis="y", alpha=0.2)
        axes[0].legend()
        fig.suptitle("Paired endpoints: each line is one fresh seed")
        fig.savefig(output / "paired_endpoints.svg")
        fig.savefig(output / "paired_endpoints.png", dpi=160)
        plt.close(fig)
    lines = ["# Formal paired evaluation", "",
             "Frozen parameters: GD learning rate 0.125; pullback learning rate 8 and damping 0.1.",
             "Three fresh paired seeds. Training finished before any test data were evaluated.",
             "The iteration and time endpoints are separate; neither is selected using test scores.", ""]
    if summary["aggregate_complete"]:
        lines += ["| Endpoint | Method | Test accuracy, mean +/- sample SD | Test loss, mean +/- sample SD | Mean updates | Mean core seconds |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for endpoint, methods in summary["methods"].items():
            for method, values in methods.items():
                accuracy, loss = values["test_accuracy"], values["test_loss"]
                lines.append(f"| {endpoint} | {method} | {accuracy['mean']:.2%} +/- {100*accuracy['sample_sd']:.2f} pp | "
                             f"{loss['mean']:.6f} +/- {loss['sample_sd']:.6f} | {values['step']['mean']:.1f} | {values['training_seconds']['mean']:.6f} |")
    else:
        lines += ["At least one trajectory failed. No survivor-only aggregate is reported."]
    lines += ["", "Sample SD describes variation across three seeds; it is not a confidence interval or a significance test.",
              "Time endpoints retain the last whole update at or before 2.0 seconds; crossing states are preserved.",
              "The time comparison uses iteration-tuned settings and is a cost diagnostic, not a separately time-tuned benchmark.",
              "Pullback damping was selected at the upper boundary of the fixed tuning grid.", "",
              "![Learning curves](learning_curves.svg)", "", "![Paired endpoints](paired_endpoints.svg)", "",
              "See endpoint_metrics.csv for per-seed endpoints, summary.json for paired differences, and trajectories.json for checkpoint weights and timing brackets.",
              "metrics.csv contains offline evaluations; spectra.csv and spectral_metadata.json preserve checkpoint diagnostics for the next analysis stage.",
              "Linear probes and the final feature-quality interpretation have not been performed.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(config, output, frozen_commit):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    wall_start = perf_counter()
    write_json(output / "config.json", config)
    schedule = [(seed["id"], method) for index, seed in enumerate(config["seeds"])
                for method in (("gd", "pullback") if index % 2 == 0 else ("pullback", "gd"))]
    write_json(output / "schedule.json", schedule)
    environment = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                   "frozen_config_commit": frozen_commit, "python": platform.python_version(),
                   "platform": platform.platform(), "processor": platform.processor(), "device": "cpu", "dtype": "float64",
                   "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "threadpoolctl")},
                   "test_generated": False, "test_evaluated": False, "test_used_for_training_or_selection": False}
    training = {}
    datasets = {}
    for seed in config["seeds"]:
        x, y = two_moons(config["train_size"], config["noise"], seed["train_seed"])
        theta = initialize_parameters(seed["initialization_seed"])
        training[seed["id"]] = x, y, theta
        datasets[seed["id"]] = {"train_x": x.tolist(), "train_y": y.tolist(), "initial_theta": theta.tolist()}
    write_json(output / "data.json", datasets)
    trajectories = []
    with threadpool_limits(limits=config["threads"]):
        environment["threadpools"] = [{k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()]
        write_json(output / "environment.json", environment)
        for index, (seed_id, method) in enumerate(schedule):
            trajectory = train_trajectory(method, config, *training[seed_id])
            trajectory["seed_id"] = seed_id
            trajectories.append(trajectory)
            write_json(output / "trajectories.json", trajectories)
            print(f"[{index+1}/{len(schedule)}] {seed_id}/{method}: {trajectory['status']}, "
                  f"{trajectory['total_steps']} updates, {trajectory['training_seconds']:.6f} core seconds", flush=True)
        environment["training_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if not all(t["status"] == "completed" for t in trajectories):
            write_json(output / "environment.json", environment)
            write_json(output / "summary.json", {"aggregate_complete": False,
                                                  "completed_trajectories": sum(t["status"] == "completed" for t in trajectories),
                                                  "scheduled_trajectories": len(trajectories),
                                                  "reason": "A training trajectory failed; offline test evaluation was not started."})
            write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - wall_start,
                                                "core_training_seconds": sum(t["training_seconds"] for t in trajectories),
                                                "online_diagnostics_seconds": sum(t["diagnostics_seconds"] for t in trajectories),
                                                "warmup_seconds": sum(t["warmup_seconds"] for t in trajectories)})
            (output / "README.md").write_text(
                "# Incomplete formal batch\n\nAll scheduled trajectories were retained. At least one training run failed. "
                "No survivor-only aggregate is reported. Validation and test arrays were not generated or evaluated. "
                "See trajectories.json for the saved prefixes and failure reasons.\n", encoding="utf-8")
            raise SystemExit("The formal batch contains failed trajectories; evidence was preserved before test generation.")
        evaluation_start = perf_counter()
        splits = {}
        for seed in config["seeds"]:
            seed_id = seed["id"]
            xv, yv = two_moons(config["validation_size"], config["noise"], seed["validation_seed"])
            xt, yt = two_moons(config["test_size"], config["noise"], seed["test_seed"])
            x, y, _ = training[seed_id]
            splits[seed_id] = {"train": (x, y), "validation": (xv, yv), "test": (xt, yt)}
            datasets[seed_id].update(validation_x=xv.tolist(), validation_y=yv.tolist(), test_x=xt.tolist(), test_y=yt.tolist())
        environment["test_generated"] = True
        environment["test_evaluation_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        environment["test_evaluation_complete"] = False
        write_json(output / "environment.json", environment)
        metrics, endpoint_rows, spectrum_rows, spectral_metadata = [], [], [], []
        for trajectory in trajectories:
            seed_id, method = trajectory["seed_id"], trajectory["method"]
            local_rows = {}
            for checkpoint in trajectory["checkpoints"]:
                row = {"seed_id": seed_id, "method": method, **evaluate_checkpoint(checkpoint, splits[seed_id])}
                metrics.append(row)
                local_rows[row["step"]] = row
                if any(label in checkpoint["labels"] for label in ("initial", "iteration_middle", "iteration", "time")):
                    x, y, _ = training[seed_id]
                    spectra, metadata = spectral_diagnostic(np.array(checkpoint["theta"]), x, y,
                                                           config["methods"]["pullback"]["damping"], checkpoint["step"])
                    spectrum_rows.extend({"seed_id": seed_id, "method": method, **entry} for entry in spectra)
                    spectral_metadata.append({"seed_id": seed_id, "method": method, "labels": checkpoint["labels"], **metadata})
            for endpoint in ("iteration", "time"):
                checkpoint = trajectory["endpoints"][endpoint]
                if checkpoint is not None:
                    endpoint_rows.append({"endpoint": endpoint, **local_rows[checkpoint["step"]]})
        environment["test_evaluated"] = True
        environment["test_evaluation_complete"] = True
        evaluation_seconds = perf_counter() - evaluation_start
    write_json(output / "environment.json", environment)
    write_json(output / "data.json", datasets)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "endpoint_metrics.csv", endpoint_rows)
    write_csv(output / "spectra.csv", spectrum_rows)
    write_json(output / "spectral_metadata.json", spectral_metadata)
    summary = summarize(endpoint_rows, trajectories, [seed["id"] for seed in config["seeds"]])
    write_json(output / "summary.json", summary)
    render_report(output, config, metrics, endpoint_rows, summary)
    write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - wall_start,
                                       "offline_evaluation_and_spectra_seconds": evaluation_seconds,
                                       "core_training_seconds": sum(t["training_seconds"] for t in trajectories),
                                       "online_diagnostics_seconds": sum(t["diagnostics_seconds"] for t in trajectories),
                                       "warmup_seconds": sum(t["warmup_seconds"] for t in trajectories)})
    print(json.dumps(summary["methods"], indent=2))
    if not summary["aggregate_complete"]:
        raise SystemExit("The formal batch contains failed trajectories; results were preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/formal.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/formal"))
    parser.add_argument("--frozen-config-commit", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    committed = subprocess.check_output(["git", "show", f"{args.frozen_config_commit}:configs/formal.json"], text=True)
    if json.loads(committed) != config:
        parser.error("The configuration differs from the frozen stage-4 settings.")
    run(config, args.output, args.frozen_config_commit)


if __name__ == "__main__":
    main()
