"""Frozen-feature logistic probes selected on validation data before test evaluation."""

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

from .core import logistic_loss, representation
from .pilot import fingerprint, write_csv, write_json


def objective_derivatives(beta, design, labels, alpha):
    """Mean logistic objective and exact derivatives; the last coefficient is unpenalized."""
    margins = labels * (design @ beta)
    probability_wrong = np.exp(-np.logaddexp(0.0, margins))
    probability_right = np.exp(-np.logaddexp(0.0, -margins))
    penalty = beta.copy()
    penalty[-1] = 0.0
    value = float(np.logaddexp(0.0, -margins).mean() + 0.5 * alpha * (penalty @ penalty))
    gradient = design.T @ (-labels * probability_wrong) / len(labels) + alpha * penalty
    weights = probability_wrong * probability_right
    hessian = design.T @ (weights[:, None] * design) / len(labels)
    hessian += alpha * np.diag([1.0] * (len(beta) - 1) + [0.0])
    return value, gradient, hessian


def fit_probe(train_features, train_labels, alpha, solverconfig):
    """Fit from zero using training-only standardization and deterministic damped Newton steps."""
    started = perf_counter()
    features = np.asarray(train_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.float64)
    if (features.ndim != 2 or features.shape[1] != 2 or len(features) == 0
            or labels.shape != (len(features),) or not np.isfinite(features).all()
            or set(np.unique(labels)) != {-1.0, 1.0} or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("A probe needs finite two-dimensional features, both binary classes, and positive alpha.")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        mean = features.mean(axis=0)
        scale = np.maximum(features.std(axis=0, ddof=0), solverconfig["scale_floor"])
        design = np.column_stack(((features - mean) / scale, np.ones(len(features))))
    if not np.isfinite(design).all() or not np.isfinite(scale).all():
        raise ValueError("Nonfinite standardized training features.")
    beta = np.zeros(3, dtype=np.float64)
    iterations = 0
    reason = None
    gradient_norm = objective = None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for iteration in range(solverconfig["max_iterations"] + 1):
                objective, gradient, hessian = objective_derivatives(beta, design, labels, alpha)
                gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
                if not np.isfinite(objective) or not np.isfinite(gradient_norm):
                    raise FloatingPointError("Nonfinite probe objective or gradient.")
                if gradient_norm <= solverconfig["gradient_tolerance"]:
                    break
                if iteration == solverconfig["max_iterations"]:
                    reason = "Maximum iterations reached before the gradient tolerance."
                    break
                delta = np.linalg.solve(hessian, -gradient)
                slope = float(gradient @ delta)
                if not np.isfinite(delta).all() or not np.isfinite(slope) or slope >= 0:
                    raise FloatingPointError("Newton direction is not a finite descent direction.")
                accepted = False
                for backtrack in range(solverconfig["max_backtracking"]):
                    step_size = 0.5 ** backtrack
                    proposal = beta + step_size * delta
                    proposed_objective, _, _ = objective_derivatives(proposal, design, labels, alpha)
                    if proposed_objective <= objective + solverconfig["armijo"] * step_size * slope:
                        beta = proposal
                        iterations += 1
                        accepted = True
                        break
                if not accepted:
                    reason = "Backtracking failed to find an Armijo step."
                    break
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
        reason = f"{type(error).__name__}: {error}"
    completed = reason is None and gradient_norm is not None and gradient_norm <= solverconfig["gradient_tolerance"]
    return {"status": "completed" if completed else "failed", "reason": reason,
            "alpha": float(alpha), "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": beta[:2].tolist(), "intercept": float(beta[2]),
            "coefficient_norm": float(np.linalg.norm(beta[:2])),
            "objective": objective if objective is not None and np.isfinite(objective) else None,
            "gradient_infinity_norm": gradient_norm if gradient_norm is not None and np.isfinite(gradient_norm) else None,
            "iterations": iterations, "fit_seconds": perf_counter() - started,
            "training_feature_sha256": fingerprint(features), "training_label_sha256": fingerprint(labels)}


def evaluate_probe(model, features, labels):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if (features.ndim != 2 or features.shape[1] != 2 or not len(features)
            or labels.shape != (len(features),) or not np.isfinite(features).all()
            or not set(np.unique(labels)).issubset({-1.0, 1.0})):
        raise ValueError("Probe evaluation requires finite two-dimensional features and matching binary labels.")
    standardized = (features - np.array(model["mean"])) / np.array(model["scale"])
    scores = standardized @ np.array(model["coef"]) + model["intercept"]
    loss = float(np.logaddexp(0.0, -labels * scores).mean())
    accuracy = float(np.mean(np.where(scores >= 0, 1, -1) == labels))
    if not np.isfinite(scores).all() or not np.isfinite(loss):
        raise FloatingPointError("Nonfinite probe evaluation.")
    return {"loss": loss, "accuracy": accuracy}


def select_candidate(candidates):
    """No failed candidate may be silently discarded; test metrics are never consulted."""
    if not candidates or any(c["model"]["status"] != "completed" for c in candidates):
        raise ValueError("A failed or missing candidate invalidates this selection.")
    if any(not np.isfinite(c["validation_loss"]) for c in candidates):
        raise ValueError("Selection requires finite validation loss for every candidate.")
    return min(candidates, key=lambda c: (c["validation_loss"], -c["model"]["alpha"]))


def failed_model(alpha, reason):
    """Serializable candidate evidence when feature construction or fit validation fails."""
    return {"status": "failed", "reason": reason, "alpha": float(alpha), "mean": None, "scale": None,
            "coef": None, "intercept": None, "coefficient_norm": None, "objective": None,
            "gradient_infinity_norm": None, "iterations": 0, "fit_seconds": 0.0,
            "training_feature_sha256": None, "training_label_sha256": None}


def validate_config(config):
    if config["stage"] != "frozen_feature_probes" or config["dtype"] != "float64":
        raise ValueError("Expected a float64 frozen-feature probe configuration.")
    if type(config["threads"]) is not int or config["threads"] < 1:
        raise ValueError("Thread count must be positive.")
    if config["selection"] != "minimum_validation_loss_tie_larger_alpha":
        raise ValueError("Unexpected probe selection rule.")
    ids = config["seed_ids"]
    if len(ids) < 3 or len(ids) != len(set(ids)) or config["methods"] != ["gd", "pullback"]:
        raise ValueError("At least three paired seeds and the two declared methods are required.")
    alphas = config["alphas"]
    if not alphas or len(set(alphas)) != len(alphas) or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Probe alphas must be unique, finite, and positive.")
    checkpoints = config["checkpoints"]
    if len({c["id"] for c in checkpoints}) != len(checkpoints):
        raise ValueError("Checkpoint identifiers must be unique.")
    if not any(c["id"] == "initial" and c.get("step") == 0 for c in checkpoints):
        raise ValueError("An initial checkpoint is required for the feature-learning comparison.")
    for checkpoint in checkpoints:
        if ("step" in checkpoint) == ("endpoint" in checkpoint):
            raise ValueError("Specify exactly one fixed step or stored endpoint per checkpoint.")
        if "step" in checkpoint and (type(checkpoint["step"]) is not int or checkpoint["step"] < 0):
            raise ValueError("Checkpoint steps must be nonnegative integers.")
        if "endpoint" in checkpoint and checkpoint["endpoint"] != "time":
            raise ValueError("Only the archived time endpoint is supported.")
    solver = config["solver"]
    for key in ("max_iterations", "max_backtracking"):
        if type(solver[key]) is not int or solver[key] < 1:
            raise ValueError("Solver iteration limits must be positive integers.")
    for key in ("gradient_tolerance", "scale_floor"):
        if not np.isfinite(solver[key]) or solver[key] <= 0:
            raise ValueError("Solver tolerances must be finite and positive.")
    if not np.isfinite(solver["armijo"]) or not 0 < solver["armijo"] < 0.5:
        raise ValueError("Armijo constant must lie strictly between zero and one half.")


def resolve_states(trajectories, config):
    keyed = {(t["seed_id"], t["method"]): t for t in trajectories}
    if len(keyed) != len(trajectories):
        raise ValueError("Duplicate archived trajectories.")
    states = []
    for seed_id in config["seed_ids"]:
        for method in config["methods"]:
            trajectory = keyed[(seed_id, method)]
            if trajectory["status"] != "completed":
                raise ValueError("Probe input must contain completed formal trajectories.")
            points = {p["step"]: p for p in trajectory["checkpoints"]}
            for checkpoint in config["checkpoints"]:
                point = (points[checkpoint["step"]] if "step" in checkpoint
                         else trajectory["endpoints"][checkpoint["endpoint"]])
                if point is None or not np.isfinite(point["theta"]).all():
                    raise ValueError("Missing or nonfinite archived checkpoint.")
                states.append({"seed_id": seed_id, "method": method,
                               "checkpoint": checkpoint["id"], **point})
    return states


def summarize(metrics, config):
    keyed = {(r["seed_id"], r["method"], r["checkpoint"]): r for r in metrics}
    expected = {(s, m, c["id"]) for s in config["seed_ids"] for m in config["methods"] for c in config["checkpoints"]}
    if len(metrics) != len(expected) or set(keyed) != expected:
        return {"aggregate_complete": False, "reason": "Incomplete or duplicate probe metric records."}
    fields = ("probe_train_loss", "probe_validation_loss", "probe_test_loss", "probe_test_accuracy",
              "fixed_test_loss", "fixed_test_accuracy", "probe_minus_fixed_test_loss", "probe_minus_fixed_test_accuracy")
    if any(not np.isfinite(r[field]) for r in metrics for field in fields):
        raise ValueError("Nonfinite probe metrics cannot be summarized.")
    result = {"aggregate_complete": True, "checkpoints": {}, "paired_differences": {}, "learned_minus_initial": {}}

    def describe(values):
        return {"values": values, "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}

    for checkpoint in config["checkpoints"]:
        cid = checkpoint["id"]
        result["checkpoints"][cid] = {}
        result["paired_differences"][cid] = {}
        result["learned_minus_initial"][cid] = {}
        for method in config["methods"]:
            rows = [keyed[(sid, method, cid)] for sid in config["seed_ids"]]
            result["checkpoints"][cid][method] = {field: describe([r[field] for r in rows]) for field in fields}
            result["learned_minus_initial"][cid][method] = {
                field: {"definition": "current frozen-feature probe minus initial-feature probe",
                        "seed_order": config["seed_ids"], **describe([
                            keyed[(sid, method, cid)][field] - keyed[(sid, method, "initial")][field]
                            for sid in config["seed_ids"]])}
                for field in ("probe_test_loss", "probe_test_accuracy")}
        for field in ("probe_test_loss", "probe_test_accuracy"):
            values = [keyed[(sid, "pullback", cid)][field] - keyed[(sid, "gd", cid)][field] for sid in config["seed_ids"]]
            result["paired_differences"][cid][field] = {
                "definition": "pullback minus GD", "seed_order": config["seed_ids"], **describe(values)}
    return result


def render_report(output, config, metrics, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoint_ids = [c["id"] for c in config["checkpoints"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, color, offset in (("gd", "#20639B", -.08), ("pullback", "#D65F36", .08)):
        for ax, metric, scale in ((axes[0], "probe_test_accuracy", 100), (axes[1], "probe_test_loss", 1)):
            entries = [summary["checkpoints"][cid][method][metric] for cid in checkpoint_ids]
            ax.errorbar(np.arange(len(entries)) + offset, [scale * v["mean"] for v in entries],
                        yerr=[scale * v["sample_sd"] for v in entries], marker="o", color=color,
                        capsize=3, label=method)
            ax.set_xticks(range(len(entries)), checkpoint_ids, rotation=30, ha="right")
            ax.grid(alpha=.2)
    axes[0].set(ylabel="Probe test accuracy (%)")
    axes[1].set(ylabel="Probe test logistic loss")
    axes[0].legend()
    fig.suptitle("Frozen-feature probes: mean +/- sample SD over three paired seeds")
    fig.savefig(output / "probe_comparison.svg")
    fig.savefig(output / "probe_comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Frozen-feature linear probes", "",
             "Regularized logistic probes use training-only standardization and validation-selected regularization.",
             "Every candidate was fitted and every selection saved before any probe test features were evaluated.",
             "The representation network and all archived checkpoints remain frozen. No checkpoint is selected by test performance.",
             "These are planned diagnostics on the previously evaluated formal splits, not a newly independent test study.", "",
             "| Checkpoint | Method | Fixed-readout test accuracy | Probe test accuracy, mean +/- SD | Probe test loss, mean +/- SD |",
             "| --- | --- | ---: | ---: | ---: |"]
    for cid in checkpoint_ids:
        for method in config["methods"]:
            row = summary["checkpoints"][cid][method]
            accuracy, loss = row["probe_test_accuracy"], row["probe_test_loss"]
            lines.append(f"| {cid} | {method} | {row['fixed_test_accuracy']['mean']:.2%} | "
                         f"{accuracy['mean']:.2%} +/- {100*accuracy['sample_sd']:.2f} pp | "
                         f"{loss['mean']:.6f} +/- {loss['sample_sd']:.6f} |")
    lines += ["", "The time checkpoint is each method's saved last whole update at or before two core training seconds.",
              "Sample SD describes seed variation; it is not a confidence interval or significance test.",
              "Probe fitting time is separate from the archived network-training time.", "",
              "![Probe comparison](probe_comparison.svg)", "",
              "candidate_fits.json contains every candidate model, solver diagnostic, and validation score.",
              "selections.json preserves all validation choices; metrics.csv and summary.json contain per-seed and paired results.",
              "Source artifact hashes and protocol revision are recorded in environment.json.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(config, output, protocol_commit):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    source = Path(config["source_directory"])
    write_json(output / "config.json", config)
    source_files = ("config.json", "data.json", "trajectories.json")
    environment = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_commit": protocol_commit,
                   "python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "dtype": "float64",
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "threadpoolctl")},
                   "input_sha256": {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in source_files},
                   "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                   "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                   "network_updated": False, "test_features_evaluated": False,
                   "test_evaluation_complete": False, "test_used_for_selection": False}
    data = json.loads((source / "data.json").read_text(encoding="utf-8"))
    trajectories = json.loads((source / "trajectories.json").read_text(encoding="utf-8"))
    states = resolve_states(trajectories, config)
    write_json(output / "states.json", states)
    all_candidates, selections, metrics, failures = [], [], [], []
    fit_seconds = selection_feature_seconds = test_seconds = 0.0
    with threadpool_limits(limits=config["threads"]):
        environment["threadpools"] = [{k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()]
        write_json(output / "environment.json", environment)
        for state in states:
            keys = {k: state[k] for k in ("seed_id", "method", "checkpoint", "step")}
            seed_data = data[state["seed_id"]]
            theta = np.array(state["theta"], dtype=np.float64)
            start = perf_counter()
            feature_error = None
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    train_features = representation(theta, np.array(seed_data["train_x"], dtype=np.float64))
                    validation_features = representation(theta, np.array(seed_data["validation_x"], dtype=np.float64))
                    train_labels = np.array(seed_data["train_y"], dtype=np.float64)
                    validation_labels = np.array(seed_data["validation_y"], dtype=np.float64)
                if not np.isfinite(train_features).all() or not np.isfinite(validation_features).all():
                    raise FloatingPointError("Nonfinite training or validation features.")
            except (KeyError, ValueError, FloatingPointError, OverflowError) as error:
                feature_error = f"selection feature construction: {type(error).__name__}: {error}"
            selection_feature_seconds += perf_counter() - start
            candidates = []
            for alpha in config["alphas"]:
                if feature_error is not None:
                    model = failed_model(alpha, feature_error)
                else:
                    try:
                        model = fit_probe(train_features, train_labels, alpha, config["solver"])
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
                        model = failed_model(alpha, f"fit validation: {type(error).__name__}: {error}")
                fit_seconds += model["fit_seconds"]
                candidate = {**keys, "model": model, "train_loss": None, "train_accuracy": None,
                             "validation_loss": None, "validation_accuracy": None}
                if model["status"] == "completed":
                    try:
                        for split, features, labels in (("train", train_features, train_labels),
                                                       ("validation", validation_features, validation_labels)):
                            score = evaluate_probe(model, features, labels)
                            candidate.update({f"{split}_{k}": v for k, v in score.items()})
                    except (ValueError, FloatingPointError, OverflowError) as error:
                        model["status"], model["reason"] = "failed", f"selection evaluation: {error}"
                candidates.append(candidate)
                all_candidates.append(candidate)
            try:
                selected = select_candidate(candidates)
                selections.append({**keys, "model": selected["model"],
                                   "train_loss": selected["train_loss"], "train_accuracy": selected["train_accuracy"],
                                   "validation_loss": selected["validation_loss"], "validation_accuracy": selected["validation_accuracy"]})
            except ValueError as error:
                failures.append({**keys, "reason": str(error)})
            write_json(output / "candidate_fits.json", all_candidates)
            write_json(output / "selections.json", selections)
        environment["selection_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "failures.json", failures)
        write_json(output / "environment.json", environment)
        if failures:
            write_json(output / "summary.json", {"aggregate_complete": False,
                                                  "reason": "At least one candidate failed; no probe test evaluation was performed.",
                                                  "failed_selections": len(failures)})
            write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - started,
                                                "fit_seconds": fit_seconds, "selection_feature_seconds": selection_feature_seconds,
                                                "test_evaluation_seconds": 0.0})
            (output / "README.md").write_text("# Incomplete probe selection\n\nAt least one candidate failed. All candidate diagnostics "
                                                "were preserved. Test representations were not evaluated.\n", encoding="utf-8")
            raise SystemExit("Probe candidate failure; all evidence preserved and test evaluation skipped.")
        environment["test_evaluation_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "environment.json", environment)
        start = perf_counter()
        for state, selected in zip(states, selections):
            try:
                seed_data = data[state["seed_id"]]
                theta = np.array(state["theta"], dtype=np.float64)
                environment["test_features_evaluated"] = True
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    features = representation(theta, np.array(seed_data["test_x"], dtype=np.float64))
                    labels = np.array(seed_data["test_y"], dtype=np.float64)
                    test_score = evaluate_probe(selected["model"], features, labels)
                    fixed_loss = logistic_loss(features, labels)
                    fixed_accuracy = float(np.mean(np.where(features[:, 0] >= 0, 1, -1) == labels))
                if not np.isfinite(fixed_loss):
                    raise FloatingPointError("Nonfinite fixed-readout comparison.")
            except (KeyError, ValueError, FloatingPointError, OverflowError) as error:
                failures.append({**{k: state[k] for k in ("seed_id", "method", "checkpoint", "step")},
                                 "reason": f"test evaluation: {type(error).__name__}: {error}"})
                continue
            metrics.append({"seed_id": state["seed_id"], "method": state["method"], "checkpoint": state["checkpoint"],
                            "step": state["step"], "network_training_seconds": state["training_seconds"],
                            "cumulative_feature_displacement": state["cumulative_feature_displacement"],
                            "selected_alpha": selected["model"]["alpha"],
                            "probe_train_loss": selected["train_loss"], "probe_train_accuracy": selected["train_accuracy"],
                            "probe_validation_loss": selected["validation_loss"], "probe_validation_accuracy": selected["validation_accuracy"],
                            "probe_test_loss": test_score["loss"], "probe_test_accuracy": test_score["accuracy"],
                            "fixed_test_loss": fixed_loss, "fixed_test_accuracy": fixed_accuracy,
                            "probe_minus_fixed_test_loss": test_score["loss"] - fixed_loss,
                            "probe_minus_fixed_test_accuracy": test_score["accuracy"] - fixed_accuracy})
        test_seconds = perf_counter() - start
        environment["test_evaluation_complete"] = not failures
        environment["test_evaluation_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "environment.json", environment)
    write_json(output / "failures.json", failures)
    if metrics:
        write_csv(output / "metrics.csv", metrics)
    summary = summarize(metrics, config)
    write_json(output / "summary.json", summary)
    if summary["aggregate_complete"]:
        render_report(output, config, metrics, summary)
    else:
        (output / "README.md").write_text("# Incomplete probe evaluation\n\nAll candidate fits and selections were retained. "
                                            "Numerical test-evaluation failures and any finite metric prefix were saved. "
                                            "No incomplete-seed aggregate is reported.\n", encoding="utf-8")
    write_json(output / "timing.json", {"total_wall_seconds": perf_counter() - started, "fit_seconds": fit_seconds,
                                        "selection_feature_seconds": selection_feature_seconds, "test_evaluation_seconds": test_seconds})
    print(f"Completed {len(all_candidates)} candidate fits and {len(selections)} validation-selected probes.", flush=True)
    if not summary["aggregate_complete"]:
        raise SystemExit("Incomplete probe evaluation; available evidence was preserved.")
    print(json.dumps({k: summary["paired_differences"][k] for k in summary["paired_differences"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/probes.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/probes"))
    parser.add_argument("--protocol-commit", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    repository = Path(__file__).resolve().parents[2]
    recorded = subprocess.check_output(["git", "show", f"{args.protocol_commit}:configs/probes.json"], cwd=repository, text=True)
    if json.loads(recorded) != config:
        parser.error("Probe configuration differs from the recorded protocol.")
    for relative in ("docs/probe_protocol.md", "src/pullback_feature_learning/probes.py"):
        recorded_bytes = subprocess.check_output(["git", "show", f"{args.protocol_commit}:{relative}"], cwd=repository)
        current_bytes = (repository / relative).read_bytes()
        if recorded_bytes.replace(b"\r\n", b"\n") != current_bytes.replace(b"\r\n", b"\n"):
            parser.error(f"{relative} differs from the recorded protocol revision.")
    run(config, args.output, args.protocol_commit)


if __name__ == "__main__":
    main()
