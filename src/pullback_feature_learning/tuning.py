"""Predeclared equal-trial tuning using development validation data only."""

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
from .pilot import fingerprint, write_csv, write_json


def candidate_grid(config: dict) -> list[dict]:
    """Expand a deterministic grid with the same candidate count per method."""
    keys = ("gd_learning_rates", "pullback_learning_rates", "pullback_dampings")
    for key in keys:
        values = config[key]
        if (not values or any(not np.isfinite(v) or v <= 0 for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{key} must contain unique finite positive values.")
    if len(config[keys[0]]) != len(config[keys[1]]) * len(config[keys[2]]):
        raise ValueError("The candidate count must be equal across methods.")
    candidates = [{"id": f"gd_{i:02d}", "method": "gd", "learning_rate": float(eta), "damping": None}
                  for i, eta in enumerate(config[keys[0]])]
    pairs = [(eta, damping) for eta in config[keys[1]] for damping in config[keys[2]]]
    candidates.extend({"id": f"pullback_{i:02d}", "method": "pullback",
                       "learning_rate": float(eta), "damping": float(damping)}
                      for i, (eta, damping) in enumerate(pairs))
    return candidates


def aggregate_and_select(candidates: list[dict], results: list[dict], expected_seeds: list[str]) -> dict:
    """Rank complete configurations by terminal validation loss, accuracy, ID."""
    ids = {candidate["id"] for candidate in candidates}
    if len(ids) != len(candidates) or len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise ValueError("Candidate and expected-seed identifiers must be unique and nonempty.")
    keyed = {}
    for result in results:
        key = result["candidate_id"], result["seed_id"]
        if key in keyed or key[0] not in ids or key[1] not in expected_seeds:
            raise ValueError("Duplicate or unknown candidate/seed result.")
        keyed[key] = result
    rankings, selected = {}, {}
    for method in ("gd", "pullback"):
        rows = []
        for candidate in candidates:
            if candidate["method"] != method:
                continue
            available = [keyed[(candidate["id"], seed)] for seed in expected_seeds
                         if (candidate["id"], seed) in keyed]
            complete = [r for r in available if r["status"] == "completed"
                        and r.get("validation_loss") is not None
                        and r.get("validation_accuracy") is not None
                        and np.isfinite(r["validation_loss"]) and np.isfinite(r["validation_accuracy"])]
            eligible = len(complete) == len(expected_seeds)
            losses = [r["validation_loss"] for r in complete]
            accuracies = [r["validation_accuracy"] for r in complete]
            rows.append({**candidate, "eligible": eligible, "completed_seeds": len(complete),
                         "mean_validation_loss": float(np.mean(losses)) if eligible else None,
                         "std_validation_loss": float(np.std(losses, ddof=1)) if eligible and len(losses) > 1 else None,
                         "mean_validation_accuracy": float(np.mean(accuracies)) if eligible else None,
                         "training_seconds": float(sum(r["training_seconds"] for r in available)),
                         "diagnostics_seconds": float(sum(r["diagnostics_seconds"] for r in available))})
        rows.sort(key=lambda r: (not r["eligible"], r["mean_validation_loss"] if r["eligible"] else float("inf"),
                                 -r["mean_validation_accuracy"] if r["eligible"] else 0, r["id"]))
        if not rows or not rows[0]["eligible"]:
            raise ValueError(f"No complete eligible candidate for {method}.")
        rankings[method], selected[method] = rows, rows[0]
    return {"rankings": rankings, "selected": selected}


def validate_config(config):
    candidate_grid(config)
    if config.get("stage") != "tuning" or config["dtype"] != "float64":
        raise ValueError("Only float64 tuning is supported by this command.")
    if config["selection"] != "mean_terminal_validation_loss":
        raise ValueError("The selection objective must be terminal mean validation loss.")
    for field in ("steps", "threads", "log_every"):
        if not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer.")
    if not isinstance(config["warmup_steps"], int) or config["warmup_steps"] < 0:
        raise ValueError("Warmup steps must be nonnegative.")
    seeds = config["development_seeds"]
    data_seeds = [s[k] for s in seeds for k in ("train_seed", "validation_seed")]
    if not seeds or len({s["id"] for s in seeds}) != len(seeds) or len(set(data_seeds)) != len(data_seeds):
        raise ValueError("Development IDs and all data seeds must be unique.")
    formal_seeds = config["formal"]["seeds"]
    reserved = [s[k] for s in formal_seeds for k in ("train_seed", "validation_seed", "test_seed")]
    if len(reserved) != len(set(reserved)) or set(reserved) & set(data_seeds):
        raise ValueError("Formal data seeds must be unique and fresh.")


def trial(candidate, seed, config, arrays):
    x, y, xv, yv, initial = arrays
    theta = initial.copy()
    history = []
    core_seconds = diagnostic_seconds = warmup_seconds = 0.0
    completed_updates = 0
    phase = "warmup"
    failure = None
    terminal = None
    displacement = 0.0
    damping = candidate["damping"] if candidate["damping"] is not None else 0.01
    loop_start = perf_counter()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            warm = theta.copy()
            start = perf_counter()
            try:
                for _ in range(config["warmup_steps"]):
                    u, _ = direction(warm, x, y, candidate["method"], damping)
                    warm = warm + candidate["learning_rate"] * u
                    if not np.isfinite(warm).all():
                        raise FloatingPointError("Nonfinite warmup parameters.")
            finally:
                warmup_seconds = perf_counter() - start
            phase = "training"
            previous = representation(theta, x)
            for step in range(config["steps"] + 1):
                if step:
                    start = perf_counter()
                    try:
                        u, _ = direction(theta, x, y, candidate["method"], damping)
                        theta = theta + candidate["learning_rate"] * u
                    finally:
                        core_seconds += perf_counter() - start
                    completed_updates = step
                start = perf_counter()
                try:
                    z = representation(theta, x)
                    if not np.isfinite(theta).all() or not np.isfinite(z).all():
                        raise FloatingPointError("Nonfinite training state.")
                    if step:
                        displacement += float(np.sqrt(np.mean(np.sum((z - previous)**2, axis=1))))
                    previous = z
                    if step % config["log_every"] == 0 or step == config["steps"]:
                        zv = representation(theta, xv)
                        train_loss = logistic_loss(z, y)
                        val_loss = logistic_loss(zv, yv)
                        if not np.isfinite(zv).all() or not np.isfinite([train_loss, val_loss, displacement]).all():
                            raise FloatingPointError("Nonfinite evaluation metric.")
                        terminal = {"candidate_id": candidate["id"], "seed_id": seed["id"], "step": step,
                                    "train_loss": train_loss, "validation_loss": val_loss,
                                    "train_accuracy": float(np.mean(np.where(z[:, 0] >= 0, 1, -1) == y)),
                                    "validation_accuracy": float(np.mean(np.where(zv[:, 0] >= 0, 1, -1) == yv)),
                                    "training_seconds": core_seconds,
                                    "cumulative_feature_displacement": displacement}
                        history.append(terminal)
                finally:
                    diagnostic_seconds += perf_counter() - start
                if history and history[-1]["step"] == step:
                    history[-1]["diagnostics_seconds"] = diagnostic_seconds
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
        failure = f"{phase}: {type(error).__name__}: {error}"
    success = failure is None and completed_updates == config["steps"]
    result = {"candidate_id": candidate["id"], "seed_id": seed["id"], "method": candidate["method"],
              "status": "completed" if success else "failed", "failure_reason": failure,
              "completed_updates": completed_updates, "training_seconds": core_seconds,
              "diagnostics_seconds": diagnostic_seconds, "warmup_seconds": warmup_seconds,
              "trial_wall_seconds": perf_counter() - loop_start,
              "train_loss": terminal["train_loss"] if success else None,
              "validation_loss": terminal["validation_loss"] if success else None,
              "validation_accuracy": terminal["validation_accuracy"] if success else None,
              "initial_parameter_sha256": fingerprint(initial),
              "train_data_sha256": fingerprint(x, y), "validation_data_sha256": fingerprint(xv, yv)}
    final = {"candidate_id": candidate["id"], "seed_id": seed["id"], "completed_updates": completed_updates,
             "theta": theta.tolist() if np.isfinite(theta).all() else None}
    return result, history, final


def render_report(output, config, selection, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"gd": "#20639B", "pullback": "#D65F36"}
    for ax, method in zip(axes, ("gd", "pullback")):
        rows = selection["rankings"][method]
        eligible = [r for r in rows if r["eligible"]]
        ax.errorbar(range(len(eligible)), [r["mean_validation_loss"] for r in eligible],
                    yerr=[r["std_validation_loss"] or 0 for r in eligible], fmt="o", capsize=3,
                    color=colors[method])
        ax.set_xticks(range(len(eligible)), [r["id"] for r in eligible], rotation=70)
        ax.set(title=method, ylabel="Terminal validation loss (mean +/- seed SD)", xlabel="Candidates ranked by mean loss")
        ax.grid(alpha=0.2)
        if eligible and max(r["mean_validation_loss"] for r in eligible) > 10 * min(r["mean_validation_loss"] for r in eligible):
            ax.set_yscale("log")
    fig.suptitle("Development selection: 12 candidates per method, 3 paired seeds, 2,000 updates")
    fig.savefig(output / "validation_ranking.svg")
    fig.savefig(output / "validation_ranking.png", dpi=160)
    plt.close(fig)
    lines = ["# Validation-only tuning results", "",
             "This is development model selection, not independent final performance evaluation.",
             "The rules were committed before execution. Test and formal data were neither generated nor evaluated.", "",
             "| Method | Selected ID | Learning rate | Damping | Mean validation loss | Seed SD | Mean validation accuracy |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, row in selection["selected"].items():
        lines.append(f"| {method} | {row['id']} | {row['learning_rate']:g} | {row['damping'] or '-'} | "
                     f"{row['mean_validation_loss']:.6f} | {row['std_validation_loss']:.6f} | {row['mean_validation_accuracy']:.2%} |")
    lines += ["", "## Budget and recorded failures", "",
              "Each method was allocated 12 configurations x 3 development seeds x 2,000 updates.",
              "This equalizes candidate-run and update budgets, not elapsed computation. No failed trial is replaced.", ""]
    for method in ("gd", "pullback"):
        rows = [r for r in results if r["method"] == method]
        lines.append(f"- {method}: {sum(r['status']=='completed' for r in rows)}/{len(rows)} runs completed; "
                     f"{sum(r['training_seconds'] for r in rows):.3f} seconds core training; "
                     f"{sum(r['diagnostics_seconds'] for r in rows):.3f} seconds diagnostics.")
    lines += ["", "## Grid boundaries", ""]
    for method, row in selection["selected"].items():
        eta_grid = config["gd_learning_rates" if method == "gd" else "pullback_learning_rates"]
        boundary = row["learning_rate"] in (min(eta_grid), max(eta_grid))
        damping_boundary = method == "pullback" and row["damping"] in (min(config["pullback_dampings"]), max(config["pullback_dampings"]))
        lines.append(f"- {method}: learning-rate boundary = {boundary}; damping boundary = {damping_boundary}.")
    lines += ["", "The selected settings are the best eligible configurations within this fixed grid, not a global optimum.",
              "Boundary selections are reported without expanding the search after seeing the results.", "",
              "![Validation rankings](validation_ranking.svg)", "",
              "The frozen next-stage settings are in [formal.json](formal.json). No formal run has been executed.",
              "All per-seed outcomes, terminal checkpoints, and sampled learning curves are preserved in results.json, terminal_checkpoints.json, and histories.csv.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(config, output, preregistration_commit):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    candidates = candidate_grid(config)
    seeds = {s["id"]: s for s in config["development_seeds"]}
    schedule = [(c["id"], seed) for c in candidates for seed in seeds]
    np.random.default_rng(config["schedule_seed"]).shuffle(schedule)
    write_json(output / "config.json", config)
    write_json(output / "candidates.json", candidates)
    write_json(output / "schedule.json", schedule)
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    environment = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                   "preregistration_commit": preregistration_commit,
                   "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                   "python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
                   "device": "cpu", "dtype": "float64", "source_sha256": source,
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "threadpoolctl")},
                   "test_generated": False, "test_evaluated": False,
                   "formal_data_generated": False, "formal_experiments_executed": False}
    arrays = {}
    seed_data = {}
    for seed_id, seed in seeds.items():
        x, y = two_moons(config["train_size"], config["noise"], seed["train_seed"])
        xv, yv = two_moons(config["validation_size"], config["noise"], seed["validation_seed"])
        theta = initialize_parameters(seed["initialization_seed"])
        arrays[seed_id] = x, y, xv, yv, theta
        seed_data[seed_id] = {"train_x": x.tolist(), "train_y": y.tolist(), "validation_x": xv.tolist(),
                              "validation_y": yv.tolist(), "initial_theta": theta.tolist()}
    write_json(output / "development_data.json", seed_data)
    by_id = {c["id"]: c for c in candidates}
    results, histories, terminals = [], [], []
    with threadpool_limits(limits=config["threads"]):
        environment["threadpools"] = [{k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()]
        write_json(output / "environment.json", environment)
        with (output / "trial_log.jsonl").open("w", encoding="utf-8") as stream:
            for index, (candidate_id, seed_id) in enumerate(schedule):
                result, history, terminal = trial(by_id[candidate_id], seeds[seed_id], config, arrays[seed_id])
                results.append(result)
                histories.extend(history)
                terminals.append(terminal)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                print(f"[{index+1}/{len(schedule)}] {candidate_id}/{seed_id}: {result['status']}; "
                      f"validation loss={result['validation_loss']}", flush=True)
    write_json(output / "results.json", results)
    if histories:
        write_csv(output / "histories.csv", histories)
    else:
        (output / "histories.csv").write_text(
            "candidate_id,seed_id,step,train_loss,validation_loss,train_accuracy,validation_accuracy,"
            "training_seconds,cumulative_feature_displacement,diagnostics_seconds\n", encoding="utf-8")
    write_json(output / "terminal_checkpoints.json", terminals)
    selection = aggregate_and_select(candidates, results, list(seeds))
    write_json(output / "selection.json", selection)
    write_csv(output / "rankings.csv", selection["rankings"]["gd"] + selection["rankings"]["pullback"])
    formal = {**config["formal"], "selected_from": "mean_terminal_development_validation_loss",
              "selection_preregistration_commit": preregistration_commit,
              "dtype": config["dtype"], "threads": config["threads"], "noise": config["noise"],
              "train_size": config["train_size"], "validation_size": config["validation_size"],
              "log_every": config["log_every"], "warmup_steps": config["warmup_steps"],
              "methods": {method: {k: row[k] for k in ("learning_rate", "damping") if row[k] is not None}
                          for method, row in selection["selected"].items()}}
    write_json(output / "formal.json", formal)
    render_report(output, config, selection, results)
    write_json(output / "wall_time.json", {"total_wall_seconds": perf_counter() - started,
                                          "includes": "setup, all trials, diagnostics, serialization, and plotting"})
    print(json.dumps(selection["selected"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tuning.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/tuning"))
    parser.add_argument("--preregistration-commit", required=True,
                        help="Git commit containing the declared rules before this run.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Prevent a run from claiming a different preregistered grid.
    committed = subprocess.check_output(["git", "show", f"{args.preregistration_commit}:configs/tuning.json"], text=True)
    if json.loads(committed) != config:
        parser.error("The supplied configuration differs from the preregistered Git configuration.")
    run(config, args.output, args.preregistration_commit)


if __name__ == "__main__":
    main()
