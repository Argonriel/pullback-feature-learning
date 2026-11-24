"""Independent read-only verification of frozen probes and archived geometry.

This auditor never imports project functions, fits a model, or updates a network.
It writes only the requested audit JSON after reading completed run artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits


START = perf_counter()
CHECKS = []
MAX_ERRORS = defaultdict(float)


def check(name, passed, details=None):
    CHECKS.append({"check": name, "passed": bool(passed), "details": details})


def close(name, actual, expected, bucket, atol=5e-12, rtol=5e-12):
    aa, ee = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    good = aa.shape == ee.shape and np.isfinite(aa).all() and np.isfinite(ee).all()
    error = float(np.max(np.abs(aa - ee))) if good and aa.size else (0.0 if good else math.inf)
    if np.isfinite(error):
        MAX_ERRORS[bucket] = max(MAX_ERRORS[bucket], error)
    check(name, good and np.allclose(aa, ee, atol=atol, rtol=rtol),
          {"maximum_absolute_error": error if np.isfinite(error) else None,
           "absolute_tolerance": atol, "relative_tolerance": rtol})


def compare_tree(name, actual, expected, bucket):
    errors, mismatches = [], []

    def visit(a, e, path):
        if isinstance(e, dict):
            if not isinstance(a, dict) or set(a) != set(e):
                mismatches.append(path + ": dictionary keys")
                return
            for k in e:
                visit(a[k], e[k], path + "/" + str(k))
        elif isinstance(e, list):
            if not isinstance(a, list) or len(a) != len(e):
                mismatches.append(path + ": list length")
                return
            for i, (av, ev) in enumerate(zip(a, e)):
                visit(av, ev, path + "/" + str(i))
        elif isinstance(e, (float, int)) and not isinstance(e, bool):
            if not isinstance(a, (float, int)) or not np.isfinite(a) or not np.isfinite(e):
                mismatches.append(path + ": invalid numeric value")
            else:
                errors.append(abs(a - e))
                if not math.isclose(a, e, abs_tol=5e-12, rel_tol=5e-12):
                    mismatches.append(path + ": numeric mismatch")
        elif a != e:
            mismatches.append(path + ": nonnumeric mismatch")

    visit(actual, expected, name)
    error = max(errors, default=0.0)
    MAX_ERRORS[bucket] = max(MAX_ERRORS[bucket], error)
    check(name, not mismatches, {"maximum_absolute_error": error, "mismatches": mismatches[:20]})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if key in ("seed_id", "method", "checkpoint", "state", "labels", "numerically_active"):
                continue
            row[key] = None if value == "" else float(value)
            if key in ("step", "index", "numerical_rank") and value != "":
                row[key] = int(value)
    return rows


def forward(theta, inputs):
    """Independent implementation of the two residual blocks; no project import."""
    params = np.asarray(theta, dtype=np.float64)
    h = np.asarray(inputs, dtype=np.float64)
    for offset in (0, 40):
        incoming = params[offset:offset + 16].reshape(8, 2)
        outgoing = params[offset + 16:offset + 32].reshape(2, 8)
        bias = params[offset + 32:offset + 40]
        activation = np.tanh(np.matmul(h, incoming.T) + bias)
        h = h + 0.5 * np.matmul(activation, outgoing.T)
    return h


def scores_for(model, features):
    return ((features - model["mean"]) / model["scale"]) @ np.asarray(model["coef"]) + model["intercept"]


def score_metrics(scores, labels):
    return {"loss": float(np.mean(np.logaddexp(0, -labels * scores))),
            "accuracy": float(np.mean(np.where(scores >= 0, 1, -1) == labels))}


def describe(values):
    return {"values": values, "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}


def stamp(text):
    return datetime.fromisoformat(text)


def verify_provenance(root, output, source, tag, protocol_paths):
    env = read_json(output / "environment.json")
    commit = env["protocol_commit"]
    commit_time = stamp(subprocess.check_output(["git", "show", "-s", "--format=%cI", commit], cwd=root, text=True).strip())
    check(tag + ": protocol commit precedes recorded run", commit_time <= stamp(env["recorded_at_utc"]),
          {"commit": commit, "commit_time": commit_time.isoformat(), "run_time": env["recorded_at_utc"]})
    for name, digest in env["input_sha256"].items():
        check(tag + ": input hash " + name, sha(source / name) == digest, {"sha256": digest})
    for name, digest in env["source_sha256"].items():
        path = root / "src/pullback_feature_learning" / name
        check(tag + ": runtime source hash " + name, sha(path) == digest, {"sha256": digest})
        committed = subprocess.check_output(["git", "show", f"{commit}:src/pullback_feature_learning/{name}"], cwd=root)
        check(tag + ": source matches protocol commit " + name,
              committed.replace(b"\r\n", b"\n") == path.read_bytes().replace(b"\r\n", b"\n"))
    for name in protocol_paths:
        committed = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        check(tag + ": committed protocol " + name,
              committed.replace(b"\r\n", b"\n") == (root / name).read_bytes().replace(b"\r\n", b"\n"))
    check(tag + ": CPU float64 and one BLAS thread",
          env["device"] == "cpu" and env["dtype"] == "float64" and bool(env["threadpools"])
          and all(p["num_threads"] == 1 for p in env["threadpools"]))
    return env


def verify_probes(root, output, source, data, trajectories):
    config = read_json(output / "config.json")
    env = verify_provenance(root, output, source, "probes", ["configs/probes.json", "docs/probe_protocol.md"])
    committed_config = json.loads(subprocess.check_output(["git", "show", f"{env['protocol_commit']}:configs/probes.json"], cwd=root, text=True))
    compare_tree("probes: frozen configuration", config, committed_config, "provenance")
    check("probes: canonical configuration fingerprint",
          hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == env["config_sha256"])
    check("probes: all selections finished before test evaluation",
          stamp(env["recorded_at_utc"]) <= stamp(env["selection_finished_at_utc"])
          <= stamp(env["test_evaluation_started_at_utc"]) <= stamp(env["test_evaluation_finished_at_utc"]),
          {k: v for k, v in env.items() if k.endswith("_at_utc")})
    check("probes: recorded no network update or test selection",
          env["network_updated"] is False and env["test_used_for_selection"] is False
          and env["test_features_evaluated"] is True and env["test_evaluation_complete"] is True)
    states = read_json(output / "states.json")
    candidates = read_json(output / "candidate_fits.json")
    selections = read_json(output / "selections.json")
    metrics = read_csv(output / "metrics.csv")
    summary = read_json(output / "summary.json")
    check("probes: no failed task", read_json(output / "failures.json") == [])
    expected_alphas = [1e-6, 1e-5, 1e-4, .001, .01, .1, 1.0, 10.0]
    check("probes: prescribed eight alpha candidates", config["alphas"] == expected_alphas)
    tasks = [(sid, method, cp["id"]) for sid in config["seed_ids"] for method in config["methods"] for cp in config["checkpoints"]]
    key = lambda r: (r["seed_id"], r["method"], r["checkpoint"])
    states_by_key = {key(r): r for r in states}
    selections_by_key = {key(r): r for r in selections}
    metrics_by_key = {key(r): r for r in metrics}
    grouped = defaultdict(list)
    for row in candidates:
        grouped[key(row)].append(row)
    for label, records, keyed in (("states", states, states_by_key), ("selections", selections, selections_by_key), ("metrics", metrics, metrics_by_key)):
        check("probes: exactly all 36 " + label, len(records) == len(tasks) == 36 and set(keyed) == set(tasks))
    check("probes: exactly 288 fits and all 36 task grids", len(candidates) == 288 and set(grouped) == set(tasks))
    check("probes: all saved metric numbers finite", all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v, (int, float))))
    trajectories_by_key = {(r["seed_id"], r["method"]): r for r in trajectories}
    predictions, expected_metrics, independent_gradients = {}, {}, []
    for task in tasks:
        sid, method, cid = task
        label = "/".join(task)
        state, selection = states_by_key[task], selections_by_key[task]
        trajectory = trajectories_by_key[(sid, method)]
        cp = next(cp for cp in config["checkpoints"] if cp["id"] == cid)
        point = (next(p for p in trajectory["checkpoints"] if p["step"] == cp["step"])
                 if "step" in cp else trajectory["endpoints"][cp["endpoint"]])
        compare_tree(label + ": exact frozen archived state", state, {"seed_id": sid, "method": method, "checkpoint": cid, **point}, "saved_states")
        split_features = {s: forward(point["theta"], data[sid][s + "_x"]) for s in ("train", "validation", "test")}
        split_labels = {s: np.asarray(data[sid][s + "_y"], dtype=float) for s in split_features}
        train = split_features["train"]
        mean = np.mean(train, axis=0)
        scale = np.maximum(np.std(train, axis=0, ddof=0), config["solver"]["scale_floor"])
        grid = grouped[task]
        check(label + ": full successful eight-alpha grid",
              len(grid) == 8 and sorted(r["model"]["alpha"] for r in grid) == expected_alphas
              and all(r["model"]["status"] == "completed" and r["model"]["reason"] is None for r in grid))
        reevaluated = {}
        for candidate in grid:
            model = candidate["model"]
            alpha = model["alpha"]
            name = label + f"/alpha={alpha:g}"
            close(name + ": training-only mean and scale", [model["mean"], model["scale"]], [mean, scale], "normalization", atol=1e-14, rtol=1e-14)
            check(name + ": training feature and label fingerprints",
                  model["training_feature_sha256"] == array_sha(train)
                  and model["training_label_sha256"] == array_sha(split_labels["train"]))
            coefficients = np.asarray(model["coef"])
            design = np.column_stack(((train - mean) / scale, np.ones(len(train))))
            beta = np.r_[coefficients, model["intercept"]]
            train_scores = design @ beta
            signed_scores = split_labels["train"] * train_scores
            wrong_probability = np.exp(-np.logaddexp(0, signed_scores))
            gradient = np.mean((-split_labels["train"] * wrong_probability)[:, None] * design, axis=0)
            gradient[:2] += alpha * coefficients
            gradient_norm = float(np.max(np.abs(gradient)))
            objective = float(np.mean(np.logaddexp(0, -signed_scores)) + alpha * np.sum(coefficients ** 2) / 2)
            independent_gradients.append(gradient_norm)
            close(name + ": independently evaluated objective, gradient and coefficient norm",
                  [model["objective"], model["gradient_infinity_norm"], model["coefficient_norm"]],
                  [objective, gradient_norm, np.linalg.norm(coefficients)], "objective_and_gradient")
            check(name + ": converged regularized objective",
                  gradient_norm <= config["solver"]["gradient_tolerance"] + 1e-14
                  and 0 <= model["iterations"] <= config["solver"]["max_iterations"]
                  and 0 <= model["fit_seconds"] and np.isfinite(model["fit_seconds"])
                  and objective <= math.log(2) + 1e-12,
                  {"independent_gradient_infinity_norm": gradient_norm, "iterations": model["iterations"]})
            computed = {}
            for split in ("train", "validation"):
                split_metric = score_metrics(scores_for(model, split_features[split]), split_labels[split])
                computed.update({split + "_" + k: v for k, v in split_metric.items()})
            compare_tree(name + ": train and validation predictions", {k: candidate[k] for k in computed}, computed, "candidate_prediction_metrics")
            reevaluated[alpha] = computed
        selected_alpha = min(expected_alphas, key=lambda a: (reevaluated[a]["validation_loss"], -a))
        selected_candidate = next(c for c in grid if c["model"]["alpha"] == selected_alpha)
        compare_tree(label + ": validation winner from entire candidate grid", selection, selected_candidate, "selection")
        model = selection["model"]
        test_scores = scores_for(model, split_features["test"])
        predictions[task] = test_scores
        probe_test = score_metrics(test_scores, split_labels["test"])
        fixed_test = score_metrics(split_features["test"][:, 0], split_labels["test"])
        expected_row = {"seed_id": sid, "method": method, "checkpoint": cid, "step": point["step"],
                        "network_training_seconds": point["training_seconds"],
                        "cumulative_feature_displacement": point["cumulative_feature_displacement"],
                        "selected_alpha": selected_alpha,
                        **{"probe_" + k: v for k, v in reevaluated[selected_alpha].items()},
                        **{"probe_test_" + k: v for k, v in probe_test.items()},
                        **{"fixed_test_" + k: v for k, v in fixed_test.items()},
                        **{"probe_minus_fixed_test_" + k: probe_test[k] - fixed_test[k] for k in probe_test}}
        compare_tree(label + ": all selected probe and fixed-readout metrics", metrics_by_key[task], expected_row, "selected_prediction_metrics")
        expected_metrics[task] = expected_row
    for sid in config["seed_ids"]:
        gd, pb = selections_by_key[(sid, "gd", "initial")], selections_by_key[(sid, "pullback", "initial")]
        fields = ["alpha", "coef", "intercept", "mean", "scale", "objective", "gradient_infinity_norm", "iterations", "training_feature_sha256", "training_label_sha256"]
        compare_tree(sid + ": paired initial probe equality", {k: gd["model"][k] for k in fields}, {k: pb["model"][k] for k in fields}, "initial_pair")
        close(sid + ": initial test prediction equality", predictions[(sid, "gd", "initial")], predictions[(sid, "pullback", "initial")], "initial_pair", atol=0, rtol=0)
    fields = ("probe_train_loss", "probe_validation_loss", "probe_test_loss", "probe_test_accuracy", "fixed_test_loss", "fixed_test_accuracy", "probe_minus_fixed_test_loss", "probe_minus_fixed_test_accuracy")
    expected_summary = {"aggregate_complete": True, "checkpoints": {}, "paired_differences": {}, "learned_minus_initial": {}}
    for cp in config["checkpoints"]:
        cid = cp["id"]
        expected_summary["checkpoints"][cid] = {}
        expected_summary["paired_differences"][cid] = {}
        expected_summary["learned_minus_initial"][cid] = {}
        for method in config["methods"]:
            rows = [expected_metrics[(s, method, cid)] for s in config["seed_ids"]]
            expected_summary["checkpoints"][cid][method] = {field: describe([r[field] for r in rows]) for field in fields}
            expected_summary["learned_minus_initial"][cid][method] = {
                field: {"definition": "current frozen-feature probe minus initial-feature probe", "seed_order": config["seed_ids"],
                        **describe([expected_metrics[(s, method, cid)][field] - expected_metrics[(s, method, "initial")][field] for s in config["seed_ids"]])}
                for field in ("probe_test_loss", "probe_test_accuracy")}
        for field in ("probe_test_loss", "probe_test_accuracy"):
            expected_summary["paired_differences"][cid][field] = {
                "definition": "pullback minus GD", "seed_order": config["seed_ids"],
                **describe([expected_metrics[(s, "pullback", cid)][field] - expected_metrics[(s, "gd", cid)][field] for s in config["seed_ids"]])}
    compare_tree("probes: all aggregate means, sample SDs and paired changes", summary, expected_summary, "probe_summary")
    timing = read_json(output / "timing.json")
    close("probes: total fit cost sums all candidate costs", timing["fit_seconds"], sum(r["model"]["fit_seconds"] for r in candidates), "timing")
    check("probes: finite nonnegative analysis costs", all(np.isfinite(v) and v >= 0 for v in timing.values()))
    return {"tasks": len(tasks), "candidate_fits": len(candidates), "selected_test_predictions": sum(len(v) for v in predictions.values()),
            "maximum_independent_gradient_infinity_norm": max(independent_gradients), "protocol_commit": env["protocol_commit"],
            "timing_seconds": timing}


def verify_representation(root, output, source, data, trajectories):
    env = verify_provenance(root, output, source, "representation", ["docs/representation_analysis_protocol.md"])
    config = read_json(source / "config.json")
    source_metrics = read_csv(source / "metrics.csv")
    spectra = read_csv(source / "spectra.csv")
    metadata = read_json(source / "spectral_metadata.json")
    features = read_json(output / "feature_clouds.json")
    spectral_summary = read_csv(output / "spectral_summary.csv")
    brackets = read_json(output / "displacement_brackets.json")
    displacement_summary = read_json(output / "displacement_summary.json")
    seed_ids = [s["id"] for s in config["seeds"]]
    fractions = (.25, .5, .75, 1.)
    check("representation: no training or new test predictions recorded",
          env["network_training_performed"] is False and env["probe_training_performed"] is False and env["test_predictions_recomputed"] is False)
    expected_state_keys = {(s, m, c) for s in seed_ids for m in ("gd", "pullback") for c in ("initial", "middle", "iteration", "time")}
    key = lambda r: (r["seed_id"], r["method"], r["state"])
    features_by_key = {key(r): r for r in features}
    spectral_by_key = {key(r): r for r in spectral_summary}
    for name, rows, keyed in (("training feature clouds", features, features_by_key), ("spectral summaries", spectral_summary, spectral_by_key)):
        check("representation: all 24 " + name, len(rows) == 24 and set(keyed) == expected_state_keys)
    for trajectory in trajectories:
        sid, method = trajectory["seed_id"], trajectory["method"]
        points = {p["step"]: p for p in trajectory["checkpoints"]}
        selected = {"initial": points[0], "middle": points[1000], "iteration": trajectory["endpoints"]["iteration"], "time": trajectory["endpoints"]["time"]}
        for state, point in selected.items():
            name = f"representation/{sid}/{method}/{state}"
            task = (sid, method, state)
            labels = np.asarray(data[sid]["train_y"], dtype=float)
            z = forward(point["theta"], data[sid]["train_x"])
            expected_features = {"seed_id": sid, "method": method, "state": state, "step": point["step"], "features": z.tolist(), "labels": labels.tolist()}
            compare_tree(name + ": training features from saved theta", features_by_key[task], expected_features, "representation_features")
            local = [r for r in spectra if r["seed_id"] == sid and r["method"] == method and r["step"] == point["step"]]
            meta = next(r for r in metadata if r["seed_id"] == sid and r["method"] == method and r["step"] == point["step"])
            singular = np.array([r["singular_value"] for r in local])
            alignment = np.array([r["gradient_alignment_fraction"] for r in local])
            active = singular > meta["rank_tolerance"]
            check(name + ": full finite ordered archived spectrum", len(local) == 80 and [r["index"] for r in local] == list(range(80))
                  and np.isfinite(singular).all() and np.all(singular >= 0) and np.all(np.diff(singular) <= 0)
                  and np.isfinite(alignment).all() and np.all(alignment >= 0))
            check(name + ": numerical rank and active flags", int(sum(active)) == meta["numerical_rank"]
                  and [r["numerically_active"] == "True" for r in local] == active.tolist())
            damping = config["methods"]["pullback"]["damping"]
            filters = singular ** 2 / (singular ** 2 + damping)
            close(name + ": damping filters", [r["pullback_filter"] for r in local], filters, "spectral_filters")
            close(name + ": reachable plus unreachable gradient fractions", float(alignment[active].sum()) + meta["unreachable_gradient_fraction"], 1.0, "spectral_fraction_partition")
            gradient = -labels * np.exp(-np.logaddexp(0, labels * z[:, 0])) / len(labels)
            gradient_norm = float(np.linalg.norm(gradient))
            positive = singular[active]
            expected = {"seed_id": sid, "method": method, "state": state, "step": point["step"],
                        "numerical_rank": int(sum(active)), "rank_tolerance": meta["rank_tolerance"],
                        "largest_singular_value": float(singular.max()), "smallest_active_singular_value": float(positive.min()),
                        "active_condition_number": float(positive.max() / positive.min()),
                        "reachable_gradient_fraction": float(alignment[active].sum()),
                        "unreachable_gradient_fraction": meta["unreachable_gradient_fraction"], "representation_gradient_norm": gradient_norm}
            projected_gradient = gradient_norm * np.sqrt(alignment[active])
            for direction, weights in (("gd", singular[active] ** 2), ("pullback", filters[active])):
                velocity = -weights * projected_gradient
                velocity_norm = float(np.linalg.norm(velocity))
                gain = velocity_norm / gradient_norm
                expected[direction + "_gradient_weighted_gain"] = float(np.dot(-projected_gradient, velocity) / gradient_norm ** 2)
                expected[direction + "_velocity_norm_over_gradient_norm"] = gain
                expected[direction + "_velocity_cosine_with_negative_gradient"] = float(np.dot(-projected_gradient, velocity) / (gradient_norm * velocity_norm))
                expected[direction + "_predicted_step_rms_motion"] = config["methods"][direction]["learning_rate"] * velocity_norm / np.sqrt(len(labels))
            compare_tree(name + ": independent local spectral summaries", spectral_by_key[task], expected, "spectral_summary")
    expected_brackets = []
    for sid in seed_ids:
        local = {m: sorted([r for r in source_metrics if r["seed_id"] == sid and r["method"] == m and r["step"] <= 2000], key=lambda r: r["step"]) for m in ("gd", "pullback")}
        check(sid + ": displacement inputs ordered through exact 2000 update endpoint", all(rows[-1]["step"] == 2000
              and len({r["step"] for r in rows}) == len(rows) and all(a["cumulative_feature_displacement"] <= b["cumulative_feature_displacement"] for a, b in zip(rows, rows[1:])) for rows in local.values()))
        base = min(rows[-1]["cumulative_feature_displacement"] for rows in local.values())
        for fraction in fractions:
            target = fraction * base
            for method in ("gd", "pullback"):
                rows = local[method]
                index = max(i for i, row in enumerate(rows) if row["cumulative_feature_displacement"] <= target)
                lower = rows[index]
                upper = rows[index + 1] if index + 1 < len(rows) else None
                record = {"seed_id": sid, "method": method, "fraction": fraction, "base_displacement": base,
                          "target_displacement": target, "lower_checkpoint": lower, "upper_checkpoint": upper,
                          "realized_displacement": lower["cumulative_feature_displacement"],
                          "unused_displacement": target - lower["cumulative_feature_displacement"],
                          "exact_target": target == lower["cumulative_feature_displacement"],
                          "upper_displacement_gap": upper["cumulative_feature_displacement"] - target if upper else None,
                          "bracket_steps": upper["step"] - lower["step"] if upper else None}
                expected_brackets.append(record)
    compare_tree("representation: all 24 displacement brackets selected only by training displacement", brackets, expected_brackets, "displacement_brackets")
    fields = ("train_loss", "test_loss", "test_accuracy", "training_seconds", "cumulative_feature_displacement")
    expected_summary = {}
    for fraction in fractions:
        keyed = {(r["seed_id"], r["method"]): r["lower_checkpoint"] for r in expected_brackets if r["fraction"] == fraction}
        methods = {}
        for method in ("gd", "pullback"):
            methods[method] = {}
            for field in fields:
                stat = describe([keyed[(s, method)][field] for s in seed_ids])
                methods[method][field] = {k: stat[k] for k in ("mean", "sample_sd")}
        paired = {field: {"definition": "pullback minus GD", "seed_order": seed_ids,
                          **describe([keyed[(s, "pullback")][field] - keyed[(s, "gd")][field] for s in seed_ids])}
                  for field in ("train_loss", "test_loss", "test_accuracy")}
        expected_summary[str(fraction)] = {"methods": methods, "paired_differences": paired,
                                          "interpretation": "Common per-pair displacement ceilings; realized saved-state distances may differ."}
    compare_tree("representation: displacement aggregate means, sample SDs and paired differences", displacement_summary, expected_summary, "displacement_summary")
    timing = read_json(output / "timing.json")
    check("representation: finite positive analysis cost", np.isfinite(timing["total_wall_seconds"]) and timing["total_wall_seconds"] > 0)
    return {"training_feature_states": len(features), "spectral_summaries": len(spectral_summary), "displacement_brackets": len(brackets),
            "largest_saved_bracket_update_interval": max(r["bracket_steps"] or 0 for r in brackets),
            "protocol_commit": env["protocol_commit"], "timing_seconds": timing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--probe-output", default="runs/probes-v1")
    parser.add_argument("--representation-output", default="runs/representation-analysis-v1")
    parser.add_argument("--output", default="reports/stage6_audit.json")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "reports/formal-v1"
    probe_output = root / args.probe_output
    representation_output = root / args.representation_output
    guarded = [p for directory in (source, probe_output, representation_output, root / "src/pullback_feature_learning") for p in directory.iterdir() if p.is_file()]
    before = {str(p): sha(p) for p in guarded}
    data = read_json(source / "data.json")
    trajectories = read_json(source / "trajectories.json")
    with threadpool_limits(limits=1):
        probes = verify_probes(root, probe_output, source, data, trajectories)
        representation = verify_representation(root, representation_output, source, data, trajectories)
    check("audit: all inspected inputs and outputs remained unchanged", before == {str(p): sha(p) for p in guarded})
    report = {"audit": "Independent frozen-feature probe and archived-representation audit", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "all_passed": all(c["passed"] for c in CHECKS), "checks_passed": sum(c["passed"] for c in CHECKS), "checks_total": len(CHECKS),
              "independence": "No project numerical functions imported; no fitting, network updates, or timing-controlled reruns. Predictions, gradients, aggregates and CSV-derived geometry recomputed independently.",
              "scope_limitations": ["The test split had been inspected in earlier formal fixed-readout experiments; these probes are representation diagnostics, not independent confirmation.",
                                    "Temporal ordering is verified from committed source, recorded timestamps and pipeline ordering tests, not a cryptographic execution trace.",
                                    "Archived singular values and alignments are treated as inputs; their derived summaries are independently recomputed without a new Jacobian SVD."],
              "probe_directory": args.probe_output, "representation_directory": args.representation_output,
              "probes": probes, "representation": representation, "maximum_absolute_errors": dict(MAX_ERRORS),
              "audit_script_sha256": sha(Path(__file__)), "audit_seconds": perf_counter() - START, "checks": CHECKS}
    destination = root / args.output
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_passed", "checks_passed", "checks_total", "maximum_absolute_errors", "audit_seconds")}, indent=2))
    print(json.dumps({"failures": [c for c in CHECKS if not c["passed"]]}, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
