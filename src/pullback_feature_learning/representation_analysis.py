"""Archived-state feature geometry, local spectra, and displacement diagnostics."""

from __future__ import annotations

import argparse
import csv
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

from .core import representation, representation_gradient
from .pilot import write_csv, write_json


STATES = ("initial", "middle", "iteration", "time")
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SOURCE_FILES = ("config.json", "data.json", "trajectories.json", "metrics.csv", "spectra.csv", "spectral_metadata.json")


def select_displacement_bracket(rows, target):
    """Select an observed state without interpolating parameters or performance.

    The lower record is the last saved checkpoint at or below the target. The
    upper record is the next saved checkpoint, not necessarily the next update.
    Exact equality is retained, including a repeated-distance plateau's last state.
    """
    if not rows or not np.isfinite(target) or target < 0:
        raise ValueError("A nonempty saved trajectory and finite nonnegative target are required.")
    steps = [int(r["step"]) for r in rows]
    distances = np.array([float(r["cumulative_feature_displacement"]) for r in rows])
    if steps != sorted(set(steps)) or not np.isfinite(distances).all() or np.any(np.diff(distances) < 0):
        raise ValueError("Saved steps must increase strictly and finite cumulative distances must not decrease.")
    if target < distances[0] or target > distances[-1]:
        raise ValueError("The target must lie in the observed displacement range.")
    index = int(np.searchsorted(distances, target, side="right") - 1)
    lower = dict(rows[index])
    upper = dict(rows[index + 1]) if index + 1 < len(rows) else None
    return {"target_displacement": float(target), "lower_checkpoint": lower, "upper_checkpoint": upper,
            "realized_displacement": float(distances[index]), "unused_displacement": float(target - distances[index]),
            "exact_target": bool(distances[index] == target),
            "upper_displacement_gap": float(distances[index + 1] - target) if upper is not None else None,
            "bracket_steps": int(upper["step"] - lower["step"]) if upper is not None else None}


def summarize_spectrum(rows, metadata, gradient_norm, n, learning_rates):
    """Compute local motion diagnostics from archived singular values/alignment.

    Cosines compare -g with the first-order feature velocity at the same state.
    They do not measure the realized finite update or its generalization benefit.
    """
    singular = np.array([float(r["singular_value"]) for r in rows])
    alignment = np.array([float(r["gradient_alignment_fraction"]) for r in rows])
    filters = np.array([float(r["pullback_filter"]) for r in rows])
    active = singular > float(metadata["rank_tolerance"])
    if not (np.isfinite(singular).all() and np.isfinite(alignment).all()
            and np.isfinite(filters).all() and np.isfinite(gradient_norm) and gradient_norm >= 0):
        raise ValueError("Spectrum and gradient measurements must be finite.")
    if int(active.sum()) != metadata["numerical_rank"]:
        raise ValueError("Archived numerical-rank metadata disagrees with the singular values.")
    positive = singular[active]
    result = {"numerical_rank": int(active.sum()), "rank_tolerance": float(metadata["rank_tolerance"]),
              "largest_singular_value": float(singular.max()),
              "smallest_active_singular_value": float(positive.min()) if len(positive) else None,
              "active_condition_number": float(positive.max() / positive.min()) if len(positive) else None,
              "reachable_gradient_fraction": float(alignment[active].sum()),
              "unreachable_gradient_fraction": float(metadata["unreachable_gradient_fraction"]),
              "representation_gradient_norm": float(gradient_norm)}
    for method, weights in (("gd", singular**2), ("pullback", filters)):
        first = float(np.sum(alignment[active] * weights[active]))
        second = float(np.sum(alignment[active] * weights[active]**2))
        gain = float(np.sqrt(second))
        result[f"{method}_gradient_weighted_gain"] = first
        result[f"{method}_velocity_norm_over_gradient_norm"] = gain
        result[f"{method}_velocity_cosine_with_negative_gradient"] = first / gain if gain > 0 and gradient_norm > 0 else None
        result[f"{method}_predicted_step_rms_motion"] = float(learning_rates[method] * gradient_norm * gain / np.sqrt(n))
    return result


def selected_states(trajectory):
    by_step = {p["step"]: p for p in trajectory["checkpoints"]}
    return {"initial": by_step[0], "middle": by_step[1000],
            "iteration": trajectory["endpoints"]["iteration"], "time": trajectory["endpoints"]["time"]}


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["step"] = int(row["step"])
        for key in row:
            if key not in ("step", "seed_id", "method", "labels"):
                row[key] = float(row[key])
    return rows


def aggregate_displacement(records, seed_ids):
    fields = ("train_loss", "test_loss", "test_accuracy", "training_seconds", "cumulative_feature_displacement")
    result = {}
    for fraction in FRACTIONS:
        selected = {(r["seed_id"], r["method"]): r for r in records if r["fraction"] == fraction}
        expected = {(s, m) for s in seed_ids for m in ("gd", "pullback")}
        if len(selected) != len(expected) or set(selected) != expected:
            raise ValueError("Every displacement budget requires all paired seeds and methods.")
        methods = {}
        for method in ("gd", "pullback"):
            rows = [selected[(s, method)]["lower_checkpoint"] for s in seed_ids]
            methods[method] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                                   "sample_sd": float(np.std([r[k] for r in rows], ddof=1))} for k in fields}
        paired = {}
        for field in ("train_loss", "test_loss", "test_accuracy"):
            values = [selected[(s, "pullback")]["lower_checkpoint"][field]
                      - selected[(s, "gd")]["lower_checkpoint"][field] for s in seed_ids]
            paired[field] = {"definition": "pullback minus GD", "seed_order": seed_ids, "values": values,
                             "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
        result[str(fraction)] = {"methods": methods, "paired_differences": paired,
                                 "interpretation": "Common per-pair displacement ceilings; realized saved-state distances may differ."}
    return result


def render_figures(output, seeds, features, spectra, metrics, brackets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_colors = {"gd": "#20639B", "pullback": "#D65F36"}
    state_colors = {"initial": "#20639B", "middle": "#D65F36", "iteration": "#238B45", "time": "#7B3294"}
    for seed_id in seeds:
        local = [r for r in features if r["seed_id"] == seed_id]
        combined = np.concatenate([np.array(r["features"]) for r in local])
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        pad = .05 * np.maximum(hi - lo, 1)
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), constrained_layout=True, sharex=True, sharey=True)
        for row, method in enumerate(("gd", "pullback")):
            for col, state in enumerate(STATES):
                record = next(r for r in local if r["method"] == method and r["state"] == state)
                z, y = np.array(record["features"]), np.array(record["labels"])
                ax = axes[row, col]
                for label, color in ((-1, "#20639B"), (1, "#D65F36")):
                    ax.scatter(z[y == label, 0], z[y == label, 1], s=10, alpha=.65, c=color, label=f"class {label:+d}")
                ax.axvline(0, color="black", linewidth=.8, alpha=.5)
                ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]), ylim=(lo[1] - pad[1], hi[1] + pad[1]),
                       title=f"{method}: {state}\nstep {record['step']:,}")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=.15)
                if row == 1:
                    ax.set_xlabel("Representation coordinate 1")
                if col == 0:
                    ax.set_ylabel("Representation coordinate 2")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{seed_id}: training representations, common axes; vertical line is the fixed readout boundary")
        fig.savefig(output / f"feature_clouds_{seed_id}.svg")
        fig.savefig(output / f"feature_clouds_{seed_id}.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
        for col, method in enumerate(("gd", "pullback")):
            for state in STATES:
                record = next(r for r in local if r["method"] == method and r["state"] == state)
                rows = [r for r in spectra if r["seed_id"] == seed_id and r["method"] == method and int(r["step"]) == record["step"]]
                indices = [int(r["index"]) + 1 for r in rows]
                axes[0, col].plot(indices, [float(r["singular_value"]) for r in rows], color=state_colors[state], label=state)
                axes[1, col].plot(indices, [float(r["gradient_alignment_fraction"]) for r in rows], color=state_colors[state], alpha=.8)
            axes[0, col].set(title=method, yscale="log", ylabel="Jacobian singular value")
            axes[1, col].set(xlabel="Singular-value index (descending)", ylabel="Fraction of squared gradient projection")
            for ax in axes[:, col]:
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{seed_id}: local spectra and gradient alignment; axes shared in definition, not parameter state")
        fig.savefig(output / f"spectra_{seed_id}.svg")
        fig.savefig(output / f"spectra_{seed_id}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(seeds), figsize=(12, 3.8), constrained_layout=True, squeeze=False)
    for col, seed_id in enumerate(seeds):
        ax = axes[0, col]
        for method in ("gd", "pullback"):
            rows = sorted([r for r in metrics if r["seed_id"] == seed_id and r["method"] == method and r["step"] <= 2000], key=lambda r: r["step"])
            ax.plot([r["cumulative_feature_displacement"] for r in rows], [r["train_loss"] for r in rows], color=method_colors[method], label=method)
            selected = [r for r in brackets if r["seed_id"] == seed_id and r["method"] == method]
            ax.scatter([r["realized_displacement"] for r in selected], [r["lower_checkpoint"]["train_loss"] for r in selected], color=method_colors[method], s=24)
        targets = [r["target_displacement"] for r in brackets if r["seed_id"] == seed_id and r["method"] == "gd"]
        for target in targets:
            ax.axvline(target, color="gray", linewidth=.7, linestyle=":", alpha=.6)
        ax.set(title=seed_id, xlabel="Cumulative RMS feature displacement", ylabel="Training logistic loss", yscale="log")
        ax.grid(alpha=.15)
    axes[0, 0].legend()
    fig.suptitle("Saved states up to 2,000 updates; dots are retained states below common displacement ceilings")
    fig.savefig(output / "displacement_comparison.svg")
    fig.savefig(output / "displacement_comparison.png", dpi=160)
    plt.close(fig)


def render_report(output, spectral_rows, summary, seed_ids):
    lines = ["# Archived representation diagnostics", "",
             "This analysis uses fixed formal checkpoints and already reported classification metrics. No network or probe is trained here.",
             "Training-only point clouds use raw coordinates and common axis bounds within each seed across methods and states.", "",
             "## Local spectral summaries", "",
             "Entries are arithmetic means over three paired seeds. Per-state rows are in spectral_summary.csv.",
             "Both hypothetical update directions are evaluated locally at each archived state; these are not realized finite steps.", "",
             "| State | Trajectory | Mean unreachable gradient fraction | Mean GD velocity cosine | Mean pullback velocity cosine |",
             "| --- | --- | ---: | ---: | ---: |"]
    for state in STATES:
        for method in ("gd", "pullback"):
            rows = [r for r in spectral_rows if r["state"] == state and r["method"] == method]
            means = [np.mean([r[k] for r in rows]) for k in ("unreachable_gradient_fraction", "gd_velocity_cosine_with_negative_gradient", "pullback_velocity_cosine_with_negative_gradient")]
            lines.append(f"| {state} | {method} | {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} |")
    lines += ["", "A better local cosine means closer directional alignment with the training representation gradient, not better generalization.",
              "Full numerical rank does not mean all representation directions are reachable: J has 512 rows and only 80 columns.",
              "Individual singular-vector alignments can be unstable within nearly repeated singular-value subspaces.", "",
              "## Displacement-budget comparison", "",
              "For each pair, the common base budget is the smaller cumulative displacement at exactly 2,000 updates.",
              "Budgets are 25%, 50%, 75%, and 100% of that value. Retain the last saved state at or below each ceiling.",
              "Realized distances generally differ because logging is sparse. The next saved state and both gaps are retained in displacement_brackets.json.",
              "There is no parameter or metric interpolation, and this does not assert exactly equal realized motion.", "",
              "| Fraction of common per-pair budget | Method | Mean training loss | Mean test accuracy | Mean realized displacement |",
              "| ---: | --- | ---: | ---: | ---: |"]
    for fraction, record in summary.items():
        for method, values in record["methods"].items():
            lines.append(f"| {float(fraction):.0%} | {method} | {values['train_loss']['mean']:.6f} | "
                         f"{values['test_accuracy']['mean']:.2%} | {values['cumulative_feature_displacement']['mean']:.4f} |")
    lines += ["", "These are secondary descriptive comparisons. Ceiling selection uses training displacements only; test metrics are copied from the original formal archive.",
              "Three-seed means and sample SDs are descriptive, not significance evidence. Intermediate test scores do not select a model or replace formal endpoints.", "",
              "![Displacement comparison](displacement_comparison.svg)", ""]
    for seed_id in seed_ids:
        lines += [f"## {seed_id}", "", f"![Training feature clouds](feature_clouds_{seed_id}.svg)", "",
                  f"![Spectra and alignment](spectra_{seed_id}.svg)", ""]
    lines += ["Provenance is in environment.json. Full saved training feature coordinates are in feature_clouds.json; no test point clouds are produced.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(source, output, protocol_commit):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    data = json.loads((source / "data.json").read_text(encoding="utf-8"))
    trajectories = json.loads((source / "trajectories.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "spectral_metadata.json").read_text(encoding="utf-8"))
    metrics = read_metrics(source / "metrics.csv")
    with (source / "spectra.csv").open(newline="", encoding="utf-8") as stream:
        spectra = list(csv.DictReader(stream))
    seed_ids = [s["id"] for s in config["seeds"]]
    expected = {(s, m) for s in seed_ids for m in ("gd", "pullback")}
    if (len(seed_ids) != 3 or len(trajectories) != 6
            or {(t["seed_id"], t["method"]) for t in trajectories} != expected
            or not all(t["status"] == "completed" for t in trajectories) or config["iteration_endpoint"] != 2000):
        raise ValueError("This protocol requires the complete frozen six-trajectory 2,000-update formal archive.")
    environment = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_commit": protocol_commit,
                   "python": platform.python_version(), "platform": platform.platform(), "dtype": "float64", "device": "cpu",
                   "source_archive": source.as_posix(), "input_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
                   "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "threadpoolctl")},
                   "network_training_performed": False, "probe_training_performed": False,
                   "test_predictions_recomputed": False, "test_metrics_source": "Copied from the archived formal metrics.csv",
                   "states": list(STATES), "displacement_fractions": list(FRACTIONS)}
    features, spectral_rows, brackets = [], [], []
    learning_rates = {m: config["methods"][m]["learning_rate"] for m in ("gd", "pullback")}
    with threadpool_limits(limits=1):
        environment["threadpools"] = [{k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()]
        for trajectory in trajectories:
            seed_id, method = trajectory["seed_id"], trajectory["method"]
            x, y = np.array(data[seed_id]["train_x"]), np.array(data[seed_id]["train_y"])
            for state, point in selected_states(trajectory).items():
                theta = np.array(point["theta"], dtype=np.float64)
                z = representation(theta, x)
                features.append({"seed_id": seed_id, "method": method, "state": state, "step": point["step"],
                                 "features": z.tolist(), "labels": y.tolist()})
                local_spectrum = [r for r in spectra if r["seed_id"] == seed_id and r["method"] == method and int(r["step"]) == point["step"]]
                local_meta = next(r for r in metadata if r["seed_id"] == seed_id and r["method"] == method and r["step"] == point["step"])
                if len(local_spectrum) != 80:
                    raise ValueError("Every selected state must have its full archived spectrum.")
                row = summarize_spectrum(local_spectrum, local_meta, np.linalg.norm(representation_gradient(z, y)), len(x), learning_rates)
                spectral_rows.append({"seed_id": seed_id, "method": method, "state": state, "step": point["step"], **row})
        for seed_id in seed_ids:
            local = {m: sorted([r for r in metrics if r["seed_id"] == seed_id and r["method"] == m and r["step"] <= 2000], key=lambda r: r["step"])
                     for m in ("gd", "pullback")}
            if any(rows[-1]["step"] != 2000 for rows in local.values()):
                raise ValueError("The exact iteration endpoint is required for the displacement ceiling.")
            base = min(rows[-1]["cumulative_feature_displacement"] for rows in local.values())
            for fraction in FRACTIONS:
                for method in ("gd", "pullback"):
                    brackets.append({"seed_id": seed_id, "method": method, "fraction": fraction, "base_displacement": base,
                                     **select_displacement_bracket(local[method], fraction * base)})
    summary = aggregate_displacement(brackets, seed_ids)
    write_json(output / "environment.json", environment)
    write_json(output / "feature_clouds.json", features)
    write_csv(output / "spectral_summary.csv", spectral_rows)
    write_json(output / "displacement_brackets.json", brackets)
    write_json(output / "displacement_summary.json", summary)
    render_figures(output, seed_ids, features, spectra, metrics, brackets)
    render_report(output, spectral_rows, summary, seed_ids)
    write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - started})
    print(f"Archived analysis complete: {len(features)} feature states, {len(spectral_rows)} spectral summaries, {len(brackets)} displacement records.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/formal-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/representation-analysis"))
    parser.add_argument("--protocol-commit", required=True)
    args = parser.parse_args()
    for name in ("docs/representation_analysis_protocol.md", "src/pullback_feature_learning/representation_analysis.py"):
        committed = subprocess.check_output(["git", "show", f"{args.protocol_commit}:{name}"])
        current = Path(name).read_bytes()
        if committed.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n"):
            parser.error(f"{name} differs from the recorded analysis protocol.")
    run(args.source, args.output, args.protocol_commit)


if __name__ == "__main__":
    main()
