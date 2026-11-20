"""Bounded post-formal convergence diagnostic on existing development data."""

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

from .core import direction, initialize_parameters, representation
from .data import two_moons
from .formal import evaluate_checkpoint
from .pilot import fingerprint, write_csv, write_json


def validate_config(config):
    if config["stage"] != "long_training_diagnostic" or config["dtype"] != "float64":
        raise ValueError("Expected a float64 long-training diagnostic configuration.")
    for key in ("steps", "threads", "log_every", "train_size", "validation_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if type(config["warmup_steps"]) is not int or config["warmup_steps"] < 0:
        raise ValueError("Warmup steps must be nonnegative.")
    if not np.isfinite(config["noise"]) or config["noise"] < 0:
        raise ValueError("Noise must be finite and nonnegative.")
    points = config["checkpoint_steps"]
    if (any(type(s) is not int or not 0 <= s <= config["steps"] for s in points)
            or points != sorted(set(points)) or not {0, config["steps"]}.issubset(points)):
        raise ValueError("Checkpoint steps must be sorted, unique, in range, and include both endpoints.")
    if set(config["methods"]) != {"gd", "pullback"}:
        raise ValueError("Both methods are required.")
    for method, options in config["methods"].items():
        scales = [options["learning_rate"]]
        if method == "pullback":
            scales.append(options["damping"])
        if any(not np.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("Optimizer scales must be positive and finite.")
    seeds = config["development_seeds"]
    ids = [s["id"] for s in seeds]
    data_seeds = [s[k] for s in seeds for k in ("train_seed", "validation_seed")]
    if len(ids) < 3 or len(set(ids)) != len(ids) or len(set(data_seeds)) != len(data_seeds):
        raise ValueError("At least three unique paired development splits are required.")
    if "test_size" in config or any("test_seed" in s for s in seeds):
        raise ValueError("This diagnostic must not contain test data settings.")
    schedule = [(s["seed_id"], s["method"]) for s in config["schedule"]]
    expected = {(seed_id, method) for seed_id in ids for method in config["methods"]}
    if len(schedule) != len(expected) or set(schedule) != expected:
        raise ValueError("Schedule must contain every paired run exactly once.")
    rule = config["continuation_review"]
    if not (rule["start_step"] in points and rule["end_step"] == config["steps"]
            and rule["start_step"] < rule["end_step"]):
        raise ValueError("Continuation review must compare saved points ending at the fixed endpoint.")
    if not (0 < rule["minimum_relative_loss_decrease"] < 1
            and 1 <= rule["minimum_validation_seed_improvements"] <= len(seeds)
            and rule["automatic_extension"] is False):
        raise ValueError("Invalid bounded continuation-review rule.")


def train_trajectory(method, config, x, y, initial):
    """Run exactly the declared updates; warmup and diagnostics are timed separately."""
    eta = config["methods"][method]["learning_rate"]
    damping = config["methods"][method].get("damping", 0.01)
    theta = initial.copy()
    saved = {}
    step = 0
    core = state_core = diagnostics = warmup = displacement = 0.0
    reason = None
    phase = "warmup"
    started = perf_counter()
    special = set(config["checkpoint_steps"])

    def save():
        saved[step] = {"step": step, "theta": theta.tolist(), "training_seconds": state_core,
                       "cumulative_feature_displacement": displacement,
                       "labels": ["declared" if step in special else "regular"]}

    save()
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
                warmup = perf_counter() - start
            phase = "training"
            previous = representation(theta, x)
            for next_step in range(1, config["steps"] + 1):
                start = perf_counter()
                try:
                    u, _ = direction(theta, x, y, method, damping)
                    next_theta = theta + eta * u
                finally:
                    core += perf_counter() - start
                start = perf_counter()
                try:
                    features = representation(next_theta, x)
                    next_distance = displacement + float(np.sqrt(np.mean(np.sum((features - previous)**2, axis=1))))
                    if not (np.isfinite(next_theta).all() and np.isfinite(features).all()
                            and np.isfinite(next_distance)):
                        raise FloatingPointError("Nonfinite training state or feature displacement.")
                    theta, previous, displacement, step = next_theta, features, next_distance, next_step
                    state_core = core
                    if step in special or step % config["log_every"] == 0:
                        save()
                finally:
                    diagnostics += perf_counter() - start
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
        reason = f"{phase}: {type(error).__name__}: {error}"
    # A failed update never replaces the last valid parameters.
    if step not in saved:
        save()
    return {"method": method, "status": "completed" if reason is None and step == config["steps"] else "failed",
            "reason": reason, "total_steps": step, "training_seconds": core,
            "diagnostics_seconds": diagnostics, "warmup_seconds": warmup,
            "loop_wall_seconds": perf_counter() - started,
            "initial_parameter_sha256": fingerprint(initial), "train_data_sha256": fingerprint(x, y),
            "checkpoints": [saved[s] for s in sorted(saved)]}


def summarize(metrics, trajectories, config):
    """Describe all declared checkpoints without selecting a winning checkpoint."""
    ids = [s["id"] for s in config["development_seeds"]]
    expected = {(s, m) for s in ids for m in ("gd", "pullback")}
    keys = [(t["seed_id"], t["method"]) for t in trajectories]
    complete = (len(keys) == len(expected) and set(keys) == expected
                and all(t["status"] == "completed" and t["total_steps"] == config["steps"] for t in trajectories))
    result = {"aggregate_complete": complete, "endpoints": {}, "paired_differences": {}, "continuation_review": {}}
    if not complete:
        result["reason"] = "At least one scheduled run is missing or failed; no survivor-only aggregate."
        return result
    keyed = {(r["seed_id"], r["method"], r["step"]): r for r in metrics}
    if len(keyed) != len(metrics):
        raise ValueError("Duplicate checkpoint metric rows.")
    fields = ("train_loss", "validation_loss", "train_accuracy", "validation_accuracy",
              "training_seconds", "cumulative_feature_displacement")
    for step in config["checkpoint_steps"]:
        result["endpoints"][str(step)] = {}
        for method in ("gd", "pullback"):
            rows = [keyed[(seed_id, method, step)] for seed_id in ids]
            if not all(np.isfinite(r[k]) for r in rows for k in fields):
                raise ValueError("Nonfinite checkpoint metrics cannot be aggregated.")
            result["endpoints"][str(step)][method] = {
                k: {"mean": float(np.mean([r[k] for r in rows])),
                    "sample_sd": float(np.std([r[k] for r in rows], ddof=1))} for k in fields}
        result["paired_differences"][str(step)] = {}
        for metric in ("train_loss", "validation_loss", "validation_accuracy"):
            differences = [keyed[(seed_id, "pullback", step)][metric] - keyed[(seed_id, "gd", step)][metric]
                           for seed_id in ids]
            result["paired_differences"][str(step)][metric] = {
                "definition": "pullback minus GD", "seed_order": ids, "values": differences,
                "mean": float(np.mean(differences)), "sample_sd": float(np.std(differences, ddof=1))}
    rule = config["continuation_review"]
    for method in ("gd", "pullback"):
        start = result["endpoints"][str(rule["start_step"])][method]
        end = result["endpoints"][str(rule["end_step"])][method]
        reductions = {k: (start[k]["mean"] - end[k]["mean"]) / start[k]["mean"]
                      if start[k]["mean"] > 0 else 0.0 for k in ("train_loss", "validation_loss")}
        per_seed = {seed_id: keyed[(seed_id, method, rule["end_step"])]["validation_loss"]
                    < keyed[(seed_id, method, rule["start_step"])]["validation_loss"] for seed_id in ids}
        eligible = (all(v >= rule["minimum_relative_loss_decrease"] for v in reductions.values())
                    and sum(per_seed.values()) >= rule["minimum_validation_seed_improvements"])
        result["continuation_review"][method] = {
            "relative_mean_loss_decrease": reductions, "validation_improved_by_seed": per_seed,
            "validation_seed_improvements": sum(per_seed.values()), "criterion_met": bool(eligible)}
    result["further_training_supported_by_declared_criterion"] = any(
        row["criterion_met"] for row in result["continuation_review"].values())
    result["automatic_extension"] = False
    return result


def render_report(output, config, metrics, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for method, color in (("gd", "#20639B"), ("pullback", "#D65F36")):
        for index, seed in enumerate(config["development_seeds"]):
            rows = [r for r in metrics if r["method"] == method and r["seed_id"] == seed["id"] and r["step"] > 0]
            for col, field in enumerate(("step", "training_seconds")):
                for row, metric in enumerate(("train_loss", "validation_loss")):
                    axes[row, col].plot([r[field] for r in rows], [r[metric] for r in rows],
                                        color=color, linestyle=("-", "--", ":")[index % 3], alpha=.8,
                                        label=f"{method}: {seed['id']}")
    for row in range(2):
        for col in range(2):
            axes[row, col].set(xscale="log", yscale="log", xlabel="Updates" if col == 0 else "Core training time (s)",
                               ylabel="Training logistic loss" if row == 0 else "Development validation loss")
            axes[row, col].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Development convergence diagnostic: {config['steps']:,} fixed updates; no test evaluation")
    fig.savefig(output / "learning_curves.svg")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    lines = ["# Long-training development diagnostic", "",
             "Post-formal exploratory diagnostic with fixed settings and the original development seeds.",
             "These validation data were already used for tuning. This is not independent test evidence.",
             "No optimizer retuning, checkpoint selection, test evaluation, or automatic extension was performed.", "",
             "| Updates | Method | Train loss, mean | Validation loss, mean +/- SD | Validation accuracy, mean | Mean core seconds |",
             "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for step, methods in summary["endpoints"].items():
        for method, values in methods.items():
            val = values["validation_loss"]
            lines.append(f"| {int(step):,} | {method} | {values['train_loss']['mean']:.6g} | "
                         f"{val['mean']:.6g} +/- {val['sample_sd']:.6g} | "
                         f"{values['validation_accuracy']['mean']:.2%} | {values['training_seconds']['mean']:.4f} |")
    if not summary["aggregate_complete"]:
        lines += ["", summary["reason"]]
    lines += ["", "Sample SD describes three-seed variation, not a confidence interval.",
              "Settings were selected for 2,000 updates; this diagnoses their continuation, not optimally tuned long-horizon performance.",
              "Core time excludes warmup, feature diagnostics, checkpoint recording, plotting, and offline evaluation.",
              "The curves are observed checkpoints, with no interpolation used for endpoint metrics.", "",
              "## Bounded continuation review", "",
              "The declared rule is a practical diagnostic, not a statistical test or proof of convergence."]
    for method, review in summary["continuation_review"].items():
        changes = review["relative_mean_loss_decrease"]
        lines.append(f"- {method}: relative mean train-loss decrease = {changes['train_loss']:.2%}; "
                     f"validation-loss decrease = {changes['validation_loss']:.2%}; "
                     f"validation improved in {review['validation_seed_improvements']}/3 seeds; "
                     f"continuation criterion met = {review['criterion_met']}.")
    lines += ["", "![Learning curves](learning_curves.svg)", "",
              "See config.json for the declared budget and review rule, trajectories.json for weights, metrics.csv for all individual curves,",
              "summary.json for descriptive aggregates, and environment.json/timing.json for provenance and measured costs.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(config, output, protocol_commit):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    write_json(output / "config.json", config)
    write_json(output / "schedule.json", config["schedule"])
    environment = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_commit": protocol_commit,
                   "python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "dtype": "float64",
                   "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                   "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "threadpoolctl")},
                   "test_generated": False, "test_evaluated": False, "formal_data_generated": False}
    data, arrays, trajectories, metrics = {}, {}, [], []
    for seed in config["development_seeds"]:
        x, y = two_moons(config["train_size"], config["noise"], seed["train_seed"])
        initial = initialize_parameters(seed["initialization_seed"])
        arrays[seed["id"]] = x, y, initial
        data[seed["id"]] = {"train_x": x.tolist(), "train_y": y.tolist(), "initial_theta": initial.tolist()}
    write_json(output / "data.json", data)
    with threadpool_limits(limits=config["threads"]):
        environment["threadpools"] = [{k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()]
        write_json(output / "environment.json", environment)
        for i, item in enumerate(config["schedule"]):
            seed_id, method = item["seed_id"], item["method"]
            trajectory = train_trajectory(method, config, *arrays[seed_id])
            trajectory["seed_id"] = seed_id
            trajectories.append(trajectory)
            write_json(output / "trajectories.json", trajectories)
            print(f"[{i+1}/{len(config['schedule'])}] {seed_id}/{method}: {trajectory['status']}, "
                  f"{trajectory['total_steps']:,} updates, {trajectory['training_seconds']:.3f} core seconds", flush=True)
        environment["training_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        start = perf_counter()
        for seed in config["development_seeds"]:
            seed_id = seed["id"]
            xv, yv = two_moons(config["validation_size"], config["noise"], seed["validation_seed"])
            data[seed_id].update(validation_x=xv.tolist(), validation_y=yv.tolist())
            x, y, _ = arrays[seed_id]
            for trajectory in trajectories:
                if trajectory["seed_id"] == seed_id:
                    try:
                        for point in trajectory["checkpoints"]:
                            metrics.append({"seed_id": seed_id, "method": trajectory["method"],
                                            **evaluate_checkpoint(point, {"train": (x, y), "validation": (xv, yv)})})
                    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
                        trajectory["status"] = "failed"
                        detail = f"offline evaluation: {type(error).__name__}: {error}"
                        trajectory["reason"] = f"{trajectory['reason']}; {detail}" if trajectory["reason"] else detail
        evaluation_seconds = perf_counter() - start
    write_json(output / "environment.json", environment)
    write_json(output / "data.json", data)
    write_json(output / "trajectories.json", trajectories)
    if metrics:
        write_csv(output / "metrics.csv", metrics)
    summary = summarize(metrics, trajectories, config)
    write_json(output / "summary.json", summary)
    render_report(output, config, metrics, summary)
    write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - started,
                                        "core_training_seconds": sum(t["training_seconds"] for t in trajectories),
                                        "online_diagnostics_seconds": sum(t["diagnostics_seconds"] for t in trajectories),
                                        "warmup_seconds": sum(t["warmup_seconds"] for t in trajectories),
                                        "offline_evaluation_seconds": evaluation_seconds})
    print(json.dumps(summary["continuation_review"], indent=2))
    if not summary["aggregate_complete"]:
        raise SystemExit("One or more trajectories failed; partial evidence was preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/long_training.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/long-training"))
    parser.add_argument("--protocol-commit", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    committed = subprocess.check_output(["git", "show", f"{args.protocol_commit}:configs/long_training.json"], text=True)
    if json.loads(committed) != config:
        parser.error("Configuration differs from the recorded diagnostic protocol.")
    run(config, args.output, args.protocol_commit)


if __name__ == "__main__":
    main()
