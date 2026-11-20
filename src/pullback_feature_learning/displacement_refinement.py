"""Deterministic replay to refine archived displacement budgets to whole updates."""

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

from .core import direction, representation
from .formal import evaluate_checkpoint
from .pilot import write_csv, write_json


FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def replay_trajectory(method, config, x, y, initial, archived_points, targets):
    """Replay training only, verifying every archived state before any evaluation.

    Targets are fixed training-displacement ceilings. Their lower and upper
    endpoints are consecutive whole updates, without parameter interpolation.
    No replay timing substitutes for the original measured training times.
    """
    endpoint = config["iteration_endpoint"]
    expected = {p["step"]: p for p in archived_points if p["step"] <= endpoint}
    if (len(expected) != len([p for p in archived_points if p["step"] <= endpoint])
            or 0 not in expected or endpoint not in expected):
        raise ValueError("Unique archived initialization and terminal states are required.")
    maximum = expected[endpoint]["cumulative_feature_displacement"]
    if not targets or any(not np.isfinite(t) or not 0 <= t <= maximum for t in targets.values()):
        raise ValueError("Targets must be finite, nonnegative, and within the archived displacement range.")
    theta = initial.copy()
    displacement = 0.0
    steps = 0
    checks = []
    brackets = {name: {"target_displacement": target, "lower_checkpoint": None, "upper_checkpoint": None}
                for name, target in targets.items()}
    reason = None
    started = perf_counter()
    eta = config["methods"][method]["learning_rate"]
    damping = config["methods"][method].get("damping", 0.01)

    def point():
        return {"step": steps, "theta": theta.tolist(), "cumulative_feature_displacement": displacement, "labels": []}

    def record_and_verify():
        if steps in expected:
            original = expected[steps]
            parameter_error = float(np.max(np.abs(theta - np.asarray(original["theta"]))))
            distance_error = abs(displacement - original["cumulative_feature_displacement"])
            passed = np.array_equal(theta, np.asarray(original["theta"])) and displacement == original["cumulative_feature_displacement"]
            checks.append({"step": steps, "passed": bool(passed), "parameter_max_absolute_error": parameter_error,
                           "displacement_absolute_error": distance_error})
            if not passed:
                raise ValueError(f"Replay disagrees with archived step {steps}; exact parameter and displacement equality is required.")
        for record in brackets.values():
            if displacement <= record["target_displacement"]:
                record["lower_checkpoint"] = point()
            elif record["upper_checkpoint"] is None:
                record["upper_checkpoint"] = point()

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            previous = representation(theta, x)
            record_and_verify()
            for step in range(1, endpoint + 1):
                u, _ = direction(theta, x, y, method, damping)
                next_theta = theta + eta * u
                features = representation(next_theta, x)
                next_distance = displacement + float(np.sqrt(np.mean(np.sum((features - previous)**2, axis=1))))
                if not (np.isfinite(next_theta).all() and np.isfinite(features).all() and np.isfinite(next_distance)):
                    raise FloatingPointError("Nonfinite replay state.")
                theta, previous, displacement, steps = next_theta, features, next_distance, step
                record_and_verify()
            for record in brackets.values():
                lower, upper, target = record["lower_checkpoint"], record["upper_checkpoint"], record["target_displacement"]
                if lower is None or (upper is None and not (lower["step"] == endpoint and lower["cumulative_feature_displacement"] == target)):
                    raise ValueError("A requested budget lacks its whole-update bracket.")
                if upper is not None and (upper["step"] != lower["step"] + 1 or not lower["cumulative_feature_displacement"] <= target < upper["cumulative_feature_displacement"]):
                    raise ValueError("The displacement crossing is not bracketed by consecutive updates.")
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError, ValueError) as error:
        reason = f"{type(error).__name__}: {error}"
    return {"method": method, "status": "completed" if reason is None and steps == endpoint else "failed",
            "reason": reason, "total_steps": steps, "replay_wall_seconds": perf_counter() - started,
            "verification": checks, "verified_archived_states": sum(c["passed"] for c in checks),
            "expected_archived_states": len(expected), "final_state": point(), "brackets": brackets}


def evaluate_replays(replays, data):
    """No evaluation, including test feature reconstruction, before all replays pass."""
    if not replays or any(r["status"] != "completed" or r["verified_archived_states"] != r["expected_archived_states"] for r in replays):
        raise ValueError("Every replay and archived-state verification must pass before test evaluation.")
    rows = []
    for replay in replays:
        seed_id = replay["seed_id"]
        splits = {name: (np.array(data[seed_id][f"{name}_x"]), np.array(data[seed_id][f"{name}_y"]))
                  for name in ("train", "test")}
        for fraction, bracket in replay["brackets"].items():
            for side in ("lower", "upper"):
                state = bracket[f"{side}_checkpoint"]
                if state is not None:
                    metrics = evaluate_checkpoint(state, splits)
                    rows.append({"seed_id": seed_id, "method": replay["method"], "fraction": float(fraction), "side": side,
                                 "target_displacement": bracket["target_displacement"],
                                 "target_minus_displacement": bracket["target_displacement"] - state["cumulative_feature_displacement"],
                                 **metrics})
    return rows


def summarize(metrics, seed_ids):
    fields = ("train_loss", "train_accuracy", "test_loss", "test_accuracy", "step", "cumulative_feature_displacement", "target_minus_displacement")
    result = {}
    for fraction in FRACTIONS:
        selected = [r for r in metrics if r["fraction"] == fraction and r["side"] == "lower"]
        keyed = {(r["seed_id"], r["method"]): r for r in selected}
        expected = {(s, m) for s in seed_ids for m in ("gd", "pullback")}
        if len(selected) != len(expected) or set(keyed) != expected:
            raise ValueError("Each ceiling requires one retained state for every paired method and seed.")
        methods, paired = {}, {}
        for method in ("gd", "pullback"):
            rows = [keyed[(s, method)] for s in seed_ids]
            methods[method] = {field: {"mean": float(np.mean([r[field] for r in rows])),
                                       "sample_sd": float(np.std([r[field] for r in rows], ddof=1))} for field in fields}
            methods[method]["maximum_unused_budget"] = max(r["target_minus_displacement"] for r in rows)
        for field in ("train_loss", "test_loss", "test_accuracy"):
            values = [keyed[(s, "pullback")][field] - keyed[(s, "gd")][field] for s in seed_ids]
            paired[field] = {"definition": "pullback minus GD", "seed_order": seed_ids, "values": values,
                             "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
        result[str(fraction)] = {"methods": methods, "paired_differences": paired}
    return result


def render_figure(output, metrics, seed_ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for method, color in (("gd", "#20639B"), ("pullback", "#D65F36")):
        for index, seed_id in enumerate(seed_ids):
            rows = sorted([r for r in metrics if r["method"] == method and r["seed_id"] == seed_id and r["side"] == "lower"],
                          key=lambda r: r["fraction"])
            distance = [r["cumulative_feature_displacement"] for r in rows]
            for ax, field, scale in ((axes[0], "train_loss", 1), (axes[1], "test_accuracy", 100)):
                ax.plot(distance, [scale * r[field] for r in rows], color=color, linestyle=("-", "--", ":")[index % 3],
                        marker="o", markersize=4, alpha=.85, label=f"{method}: {seed_id}")
                ax.grid(alpha=.2)
                ax.set_xlabel("Realized cumulative RMS feature displacement")
    axes[0].set(ylabel="Training logistic loss", yscale="log")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Whole-update displacement ceilings: observed lower states; lines are visual guides only")
    fig.savefig(output / "displacement_refinement.svg")
    fig.savefig(output / "displacement_refinement.png", dpi=160)
    plt.close(fig)


def render_report(output, summary):
    lines = ["# Whole-update displacement refinement", "",
             "Deterministic replay of the six original training trajectories through step 2,000; no optimizer setting changed.",
             "Every shared archived parameter state and cumulative displacement matched exactly before any new test evaluation.",
             "The ceilings are the same 25%, 50%, 75%, and 100% of each pair's smaller original terminal displacement.",
             "Retained states are now the last whole update at or below each ceiling; each available crossing is the immediately next update.",
             "Realized distances still need not be exactly equal, but the residual gap is bounded by one observed update, not 100-update logging.", "",
             "| Ceiling fraction | Method | Mean updates | Mean realized displacement | Largest unused budget | Mean train loss | Mean test accuracy |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for fraction, records in summary.items():
        for method, values in records["methods"].items():
            lines.append(f"| {float(fraction):.0%} | {method} | {values['step']['mean']:.1f} | "
                         f"{values['cumulative_feature_displacement']['mean']:.6f} | {values['maximum_unused_budget']:.6f} | "
                         f"{values['train_loss']['mean']:.6f} | {values['test_accuracy']['mean']:.2%} |")
    lines += ["", "All lower and upper checkpoint metrics are in metrics.csv, and weights/verification checks are in replays.json.",
              "summary.json retains means, sample SDs, and paired pullback-minus-GD differences for the retained lower states.",
              "Ceilings and selection use training displacement only; no metric selects the checkpoints.",
              "Test data come from the previously evaluated formal split, so this is a secondary diagnostic, not new independent test evidence.",
              "Replay cost is recorded separately and never substitutes for the original training-time comparison.",
              "The earlier sparse-checkpoint diagnostic remains preserved; this refinement resolves its wide checkpoint gaps.", "",
              "![Whole-update displacement comparison](displacement_refinement.svg)", "",
              "Markers show retained observed states at their actual displacements. Connecting lines are visual guides, not interpolated metric estimates.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(source, output, protocol_commit):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    data = json.loads((source / "data.json").read_text(encoding="utf-8"))
    trajectories = json.loads((source / "trajectories.json").read_text(encoding="utf-8"))
    seed_ids = [s["id"] for s in config["seeds"]]
    expected = {(s, m) for s in seed_ids for m in ("gd", "pullback")}
    if (len(seed_ids) != 3 or len(trajectories) != 6 or config["iteration_endpoint"] != 2000
            or {(t["seed_id"], t["method"]) for t in trajectories} != expected
            or not all(t["status"] == "completed" for t in trajectories)):
        raise ValueError("The complete original six-trajectory formal archive is required.")
    base = {s: min(t["endpoints"]["iteration"]["cumulative_feature_displacement"] for t in trajectories if t["seed_id"] == s) for s in seed_ids}
    environment = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_commit": protocol_commit,
                   "python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "dtype": "float64",
                   "source_archive": source.as_posix(),
                   "input_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ("config.json", "data.json", "trajectories.json")},
                   "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "threadpoolctl")},
                   "test_evaluated": False, "test_used_for_selection": False, "optimizer_settings_changed": False,
                   "fractions": list(FRACTIONS), "base_displacement_by_seed": base}
    replays = []
    with threadpool_limits(limits=1):
        environment["threadpools"] = [{k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()]
        write_json(output / "environment.json", environment)
        for trajectory in trajectories:
            seed_id, method = trajectory["seed_id"], trajectory["method"]
            targets = {str(f): f * base[seed_id] for f in FRACTIONS}
            replay = replay_trajectory(method, config, np.array(data[seed_id]["train_x"]), np.array(data[seed_id]["train_y"]),
                                       np.array(data[seed_id]["initial_theta"]), trajectory["checkpoints"], targets)
            replay["seed_id"] = seed_id
            replays.append(replay)
            write_json(output / "replays.json", replays)
            print(f"{seed_id}/{method}: {replay['status']}; {replay['verified_archived_states']}/{replay['expected_archived_states']} archived states agree", flush=True)
            if replay["status"] != "completed":
                write_json(output / "summary.json", {"aggregate_complete": False, "reason": replay["reason"], "test_evaluated": False})
                write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - started})
                raise SystemExit("Replay failed verification; evidence preserved and no test evaluation performed.")
        environment["all_replays_verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "environment.json", environment)
        evaluation_start = perf_counter()
        environment["test_evaluation_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "environment.json", environment)
        try:
            metrics = evaluate_replays(replays, data)
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError, ValueError) as error:
            reason = f"offline evaluation: {type(error).__name__}: {error}"
            environment["evaluation_failure"] = reason
            write_json(output / "environment.json", environment)
            write_json(output / "summary.json", {"aggregate_complete": False, "reason": reason,
                                                  "test_evaluation_completed": False})
            write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - started,
                                                "replay_wall_seconds": sum(r["replay_wall_seconds"] for r in replays)})
            raise SystemExit("Offline evaluation failed; replay evidence and phase records were preserved.") from error
        evaluation_seconds = perf_counter() - evaluation_start
        environment["test_evaluated"] = True
        environment["test_evaluation_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary = summarize(metrics, seed_ids)
    write_json(output / "environment.json", environment)
    write_csv(output / "metrics.csv", metrics)
    write_json(output / "summary.json", {"aggregate_complete": True, "budgets": summary})
    render_figure(output, metrics, seed_ids)
    render_report(output, summary)
    write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - started,
                                       "replay_wall_seconds": sum(r["replay_wall_seconds"] for r in replays),
                                       "offline_evaluation_seconds": evaluation_seconds,
                                       "role": "Diagnostic replay cost; not a replacement for original training timings."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/formal-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/displacement-refinement"))
    parser.add_argument("--protocol-commit", required=True)
    args = parser.parse_args()
    for name in ("docs/displacement_refinement_protocol.md", "src/pullback_feature_learning/displacement_refinement.py"):
        committed = subprocess.check_output(["git", "show", f"{args.protocol_commit}:{name}"])
        if committed.replace(b"\r\n", b"\n") != Path(name).read_bytes().replace(b"\r\n", b"\n"):
            parser.error(f"{name} differs from the recorded refinement protocol.")
    run(args.source, args.output, args.protocol_commit)


if __name__ == "__main__":
    main()
