"""Run a short paired CPU pilot without inspecting the held-out test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .core import (direction, initialize_parameters, logistic_loss,
                   representation, representation_gradient, representation_jacobian)
from .data import two_moons


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def validate_config(config: dict):
    if config["stage"] != "pilot" or config["dtype"] != "float64":
        raise ValueError("This command supports only a float64 pilot.")
    if not isinstance(config["steps"], int) or config["steps"] < 1:
        raise ValueError("Steps must be a positive integer.")
    if not isinstance(config["threads"], int) or config["threads"] < 1:
        raise ValueError("Threads must be a positive integer.")
    if not isinstance(config["warmup_steps"], int) or config["warmup_steps"] < 0:
        raise ValueError("Warmup steps must be a nonnegative integer.")
    if set(config["methods"]) != {"gd", "pullback"}:
        raise ValueError("Both GD and pullback must be configured.")
    if len({config[k] for k in ("train_seed", "validation_seed", "test_seed")}) != 3:
        raise ValueError("Train, validation, and test seeds must differ.")
    for parameters in config["methods"].values():
        eta = parameters["learning_rate"]
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("Learning rates must be finite and positive.")
    damping = config["methods"]["pullback"]["damping"]
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("Damping must be finite and positive.")
    checkpoints = config["checkpoints"]
    if (0 not in checkpoints or config["steps"] not in checkpoints
            or any(not isinstance(c, int) or c < 0 or c > config["steps"] for c in checkpoints)):
        raise ValueError("Checkpoints must include the initial and final steps and lie within the run.")


def spectral_diagnostic(theta, x, y, damping, step):
    jac = representation_jacobian(theta, x)
    left, singular, _ = np.linalg.svd(jac, full_matrices=False)
    g = representation_gradient(representation(theta, x), y).ravel()
    coefficients = left.T @ g
    g_squared = float(g @ g)
    threshold = float(max(jac.shape) * np.finfo(np.float64).eps * singular[0])
    active = singular > threshold
    alignment = coefficients**2 / g_squared if g_squared else np.zeros_like(singular)
    rows = [{"step": step, "index": i, "singular_value": float(s),
             "gradient_alignment_fraction": float(alignment[i]),
             "numerically_active": bool(active[i]),
             "pullback_filter": float(s*s / (s*s + damping))}
            for i, s in enumerate(singular)]
    metadata = {"step": step, "rank_tolerance": threshold, "numerical_rank": int(active.sum()),
                "unreachable_gradient_fraction": float(max(0, 1 - alignment[active].sum())) if g_squared else 0.0}
    return rows, metadata


def train(method, config, x, y, xv, yv, initial):
    options = config["methods"][method]
    eta = options["learning_rate"]
    damping = config["methods"]["pullback"]["damping"]
    # Warm the computational path on disposable parameters, then reset exactly.
    warm = initial.copy()
    warm_start = perf_counter()
    for _ in range(config["warmup_steps"]):
        u, _ = direction(warm, x, y, method, damping)
        warm = warm + eta * u
    warmup_seconds = perf_counter() - warm_start
    theta = initial.copy()
    z_previous = representation(theta, x)
    cumulative_displacement = 0.0
    training_seconds = diagnostics_seconds = 0.0
    history, spectra, spectral_metadata, checkpoints, step_times = [], [], [], [], []
    start_wall = perf_counter()
    for step in range(config["steps"] + 1):
        if step:
            start_update = perf_counter()
            u, _ = direction(theta, x, y, method, damping)
            theta = theta + eta * u
            duration = perf_counter() - start_update
            training_seconds += duration
            step_times.append(duration)
        start_diagnostics = perf_counter()
        z = representation(theta, x)
        zv = representation(theta, xv)
        if not np.isfinite(theta).all() or not np.isfinite(z).all() or not np.isfinite(zv).all():
            raise FloatingPointError(f"Nonfinite pilot state for {method} at step {step}.")
        if step:
            cumulative_displacement += float(np.sqrt(np.mean(np.sum((z - z_previous)**2, axis=1))))
        z_previous = z
        row = {"step": step, "training_seconds": training_seconds,
               "train_loss": logistic_loss(z, y), "validation_loss": logistic_loss(zv, yv),
               "train_accuracy": float(np.mean(np.where(z[:, 0] >= 0, 1, -1) == y)),
               "validation_accuracy": float(np.mean(np.where(zv[:, 0] >= 0, 1, -1) == yv)),
               "cumulative_feature_displacement": cumulative_displacement}
        if step in config["checkpoints"]:
            checkpoints.append({"step": step, "theta": theta.tolist()})
            rows, metadata = spectral_diagnostic(theta, x, y, damping, step)
            spectra.extend(rows)
            spectral_metadata.append(metadata)
        diagnostics_seconds += perf_counter() - start_diagnostics
        row["diagnostics_seconds"] = diagnostics_seconds
        history.append(row)
    loop_wall_seconds = perf_counter() - start_wall
    summary = {"method": method, "steps": config["steps"], "options": options,
               "initial_parameter_sha256": fingerprint(initial),
               "training_seconds": training_seconds, "diagnostics_seconds": diagnostics_seconds,
               "loop_wall_seconds": loop_wall_seconds, "warmup_seconds": warmup_seconds,
               "mean_step_seconds": float(np.mean(step_times)),
               "median_step_seconds": float(np.median(step_times)),
               "p90_step_seconds": float(np.quantile(step_times, 0.9)),
               "loss_increase_count": sum(history[i]["train_loss"] > history[i-1]["train_loss"] + 1e-12
                                          for i in range(1, len(history))),
               "initial": history[0], "final": history[-1],
               "spectral_checkpoints": spectral_metadata}
    return history, spectra, checkpoints, summary


def plots(output, histories, checkpoints, x, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"gd": "#20639B", "pullback": "#D65F36"}
    labels = {"gd": "Gradient descent", "pullback": "Damped pullback"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for method, rows in histories.items():
        for ax, key, xlabel in zip(axes, ("step", "training_seconds", "cumulative_feature_displacement"),
                                   ("Iteration", "Training compute time (seconds)", "Cumulative feature displacement")):
            ax.plot([r[key] for r in rows], [r["train_loss"] for r in rows],
                    label=labels[method], color=colors[method])
            ax.set(xlabel=xlabel, ylabel="Training logistic loss")
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("CPU pilot: one paired seed, illustrative settings (not tuned)")
    fig.savefig(output / "learning_curves.svg")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    states = [("Initial features", checkpoints["gd"][0]["theta"]),
              ("GD: final features", checkpoints["gd"][-1]["theta"]),
              ("Pullback: final features", checkpoints["pullback"][-1]["theta"])]
    for ax, (title, theta) in zip(axes, states):
        z = representation(np.array(theta), x)
        for label, color in ((-1, "#20639B"), (1, "#D65F36")):
            ax.scatter(z[y == label, 0], z[y == label, 1], s=12, c=color, alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set(title=title, xlabel="Representation coordinate 1", ylabel="Representation coordinate 2")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Training point clouds; panel scales vary; no linear-probe result yet")
    fig.savefig(output / "feature_clouds.svg")
    fig.savefig(output / "feature_clouds.png", dpi=160)
    plt.close(fig)


def run(config, output):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    x, y = two_moons(config["train_size"], config["noise"], config["train_seed"])
    xv, yv = two_moons(config["validation_size"], config["noise"], config["validation_seed"])
    # The test split is specified, but neither generated nor evaluated in this pilot.
    initial = initialize_parameters(config["initialization_seed"])
    write_json(output / "data.json", {"train_x": x.tolist(), "train_y": y.tolist(),
                                     "validation_x": xv.tolist(), "validation_y": yv.tolist()})
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    environment = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                   "python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor(), "device": "cpu", "dtype": "float64",
                   "packages": {name: importlib.metadata.version(name)
                                for name in ("numpy", "matplotlib", "threadpoolctl")},
                   "source_sha256": source_hashes,
                   "train_data_sha256": fingerprint(x, y), "validation_data_sha256": fingerprint(xv, yv),
                   "initial_parameters_sha256": fingerprint(initial),
                   "test_evaluated": False, "test_generated": False}
    histories, checkpoints, summaries = {}, {}, {}
    with threadpool_limits(limits=config["threads"]):
        environment["threadpools"] = [{k: v for k, v in pool.items() if k != "filepath"}
                                      for pool in threadpool_info()]
        for method in ("gd", "pullback"):
            rows, spectra, states, summary = train(method, config, x, y, xv, yv, initial)
            histories[method], checkpoints[method], summaries[method] = rows, states, summary
            write_csv(output / f"{method}_history.csv", rows)
            write_csv(output / f"{method}_spectrum.csv", spectra)
            write_json(output / f"{method}_checkpoints.json", states)
    write_json(output / "environment.json", environment)
    write_json(output / "summary.json", summaries)
    start_plot = perf_counter()
    plots(output, histories, checkpoints, x, y)
    plotting_seconds = perf_counter() - start_plot
    lines = ["# CPU pilot results", "",
             "One paired seed, 200 steps by default, illustrative hyperparameters, no tuning.",
             "Test data were neither generated nor evaluated. These results establish feasibility, not optimizer superiority.", "",
             "| Method | Initial train loss | Final train loss | Final validation loss | Validation accuracy | Training seconds | Diagnostics seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, s in summaries.items():
        lines.append(f"| {method} | {s['initial']['train_loss']:.6f} | {s['final']['train_loss']:.6f} | "
                     f"{s['final']['validation_loss']:.6f} | {s['final']['validation_accuracy']:.2%} | "
                     f"{s['training_seconds']:.4f} | {s['diagnostics_seconds']:.4f} |")
    lines += ["", "Training compute includes forward/gradient evaluation, required Jacobian construction, Gram construction, linear solve, and parameter update.",
              "Evaluation, feature displacement, spectra, and checkpoint collection are recorded as diagnostics.",
              "Warmup, file writes, imports, and plotting are excluded from training compute; warmup and loop wall time are separately recorded in summary.json.",
              f"Plotting took {plotting_seconds:.3f} seconds; this is not included in either training compute or diagnostics.", "",
              "![Learning curves](learning_curves.svg)", "", "![Feature clouds](feature_clouds.svg)", "",
              "See config.json, environment.json, summary.json, per-step CSV files, and checkpoint JSON files for provenance.",
              "The held-out linear-probe evaluation and multi-seed tuned comparison remain future stages.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    for method, s in summaries.items():
        print(f"{method}: loss {s['initial']['train_loss']:.6f} -> {s['final']['train_loss']:.6f}; "
              f"training {s['training_seconds']:.4f}s; mean step {1000*s['mean_step_seconds']:.3f}ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/pilot"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(config, args.output)


if __name__ == "__main__":
    main()
