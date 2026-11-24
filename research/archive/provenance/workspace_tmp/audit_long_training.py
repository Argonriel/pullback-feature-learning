"""Independent artifact audit of the bounded long-training diagnostic."""

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pullback_feature_learning.core import initialize_parameters, representation
from pullback_feature_learning.data import two_moons

REPO = Path(__file__).resolve().parents[1] / "pullback-feature-learning"
RUN = REPO / "runs" / "long-training-v1"
OUTPUT = REPO / "reports" / "long_training_audit.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(*arrays):
    value = hashlib.sha256()
    for array in arrays:
        value.update(np.ascontiguousarray(array).tobytes())
    return value.hexdigest()


def normalized(data):
    return data.replace(b"\r\n", b"\n")


checks = []
maximum_errors = {"metric_absolute_error": 0.0, "step_2000_parameter_absolute_error": 0.0,
                  "step_2000_metric_absolute_error": 0.0, "aggregate_absolute_error": 0.0}


def check(name, condition, detail=None):
    item = {"name": name, "passed": bool(condition)}
    if detail is not None:
        item["detail"] = detail
    checks.append(item)


def close(name, actual, expected, category="aggregate_absolute_error", atol=1e-12):
    actual, expected = float(actual), float(expected)
    error = abs(actual - expected)
    maximum_errors[category] = max(maximum_errors[category], error)
    check(name, np.isfinite(actual) and np.isfinite(expected) and error <= atol,
          {"actual": actual, "expected": expected, "absolute_error": error, "absolute_tolerance": atol})


def main():
    config = read(RUN / "config.json")
    env = read(RUN / "environment.json")
    trajectories = read(RUN / "trajectories.json")
    data = read(RUN / "data.json")
    summary = read(RUN / "summary.json")
    timing = read(RUN / "timing.json")
    schedule = read(RUN / "schedule.json")
    tuning_config = read(REPO / "configs/tuning.json")
    selected = read(REPO / "reports/tuning-v1/selection.json")["selected"]
    prior_states = read(REPO / "reports/tuning-v1/terminal_checkpoints.json")
    prior_results = read(REPO / "reports/tuning-v1/results.json")
    with (RUN / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    with (REPO / "reports/tuning-v1/histories.csv").open(newline="", encoding="utf-8") as stream:
        prior_history = list(csv.DictReader(stream))
    keyed = {(r["seed_id"], r["method"], int(r["step"])): r for r in metrics}
    check("metric_rows_unique", len(keyed) == len(metrics))
    check("fixed_100000_update_budget", config["steps"] == 100000)
    check("config_preserved", config == read(REPO / "configs/long_training.json"))
    check("schedule_preserved", schedule == config["schedule"])
    check("development_seeds_reused", config["development_seeds"] == tuning_config["development_seeds"])
    check("data_scale_unchanged", all(config[k] == tuning_config[k]
                                      for k in ("train_size", "validation_size", "noise", "dtype", "threads")))
    check("no_test_or_formal_generation", env["test_generated"] is False
          and env["test_evaluated"] is False and env["formal_data_generated"] is False)
    check("no_test_configuration", "test_size" not in config
          and all("test_seed" not in s for s in config["development_seeds"]))
    check("thread_environment", bool(env["threadpools"])
          and all(p["num_threads"] == 1 for p in env["threadpools"]))
    check("config_hash", env["config_sha256"] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest())
    protocol_commit = env["protocol_commit"]
    committed_config = subprocess.check_output(["git", "show", f"{protocol_commit}:configs/long_training.json"], cwd=REPO)
    check("protocol_commit_config", config == json.loads(committed_config))
    commit_date = subprocess.check_output(["git", "show", "-s", "--format=%cI", protocol_commit], cwd=REPO, text=True).strip()
    check("protocol_precedes_execution", datetime.fromisoformat(commit_date) <= datetime.fromisoformat(env["recorded_at_utc"]))
    check("training_timestamps_ordered", datetime.fromisoformat(env["recorded_at_utc"])
          < datetime.fromisoformat(env["training_finished_at_utc"]))
    for name, expected_hash in env["source_sha256"].items():
        path = REPO / "src/pullback_feature_learning" / name
        current = path.read_bytes()
        committed = subprocess.check_output(["git", "show", f"{protocol_commit}:src/pullback_feature_learning/{name}"], cwd=REPO)
        check(f"source_runtime_hash/{name}", hashlib.sha256(current).hexdigest() == expected_hash)
        check(f"source_commit_content/{name}", normalized(current) == normalized(committed))
    for method, options in config["methods"].items():
        check(f"selected_parameters/{method}", all(options[k] == selected[method][k] for k in options))
    expected_pairs = [(s["seed_id"], s["method"]) for s in schedule]
    check("trajectory_schedule", [(t["seed_id"], t["method"]) for t in trajectories] == expected_pairs)
    check("six_complete_trajectories", len(trajectories) == 6 and all(t["status"] == "completed"
          and t["reason"] is None and t["total_steps"] == 100000 for t in trajectories))
    expected_steps = sorted(set(range(0, config["steps"] + 1, config["log_every"])) | set(config["checkpoint_steps"]))
    check("all_checkpoint_metric_rows", len(metrics) == 6 * len(expected_steps))
    seed_ids = [s["id"] for s in config["development_seeds"]]
    check("data_keys_are_development_only", set(data) == set(seed_ids))
    datasets = {}
    for seed in config["development_seeds"]:
        seed_id = seed["id"]
        stored = data[seed_id]
        check(f"data_fields/{seed_id}", set(stored) == {"train_x", "train_y", "validation_x", "validation_y", "initial_theta"})
        x, y = two_moons(config["train_size"], config["noise"], seed["train_seed"])
        xv, yv = two_moons(config["validation_size"], config["noise"], seed["validation_seed"])
        initial = initialize_parameters(seed["initialization_seed"])
        for name, value in (("train_x", x), ("train_y", y), ("validation_x", xv), ("validation_y", yv), ("initial_theta", initial)):
            check(f"reconstructed_data/{seed_id}/{name}", np.array_equal(value, stored[name]))
        datasets[seed_id] = x, y, xv, yv, initial
    for trajectory in trajectories:
        sid, method = trajectory["seed_id"], trajectory["method"]
        prefix = f"{sid}/{method}"
        x, y, xv, yv, initial = datasets[sid]
        points = trajectory["checkpoints"]
        check(f"checkpoint_schedule/{prefix}", [p["step"] for p in points] == expected_steps)
        check(f"finite_parameters/{prefix}", all(np.isfinite(p["theta"]).all() and len(p["theta"]) == 80 for p in points))
        check(f"initial_parameters/{prefix}", np.array_equal(points[0]["theta"], initial))
        check(f"initial_hash/{prefix}", trajectory["initial_parameter_sha256"] == digest(initial))
        check(f"training_data_hash/{prefix}", trajectory["train_data_sha256"] == digest(x, y))
        times = [p["training_seconds"] for p in points]
        distances = [p["cumulative_feature_displacement"] for p in points]
        check(f"monotone_checkpoint_time/{prefix}", times[0] == 0 and all(a < b for a, b in zip(times, times[1:])))
        check(f"monotone_displacement/{prefix}", distances[0] == 0 and all(a <= b for a, b in zip(distances, distances[1:])))
        close(f"terminal_core_time/{prefix}", times[-1], trajectory["training_seconds"])
        check(f"positive_runtime/{prefix}", all(trajectory[k] > 0 for k in
              ("training_seconds", "diagnostics_seconds", "warmup_seconds", "loop_wall_seconds")))
        check(f"runtime_partition/{prefix}", trajectory["loop_wall_seconds"] >= sum(trajectory[k] for k in
              ("training_seconds", "diagnostics_seconds", "warmup_seconds")))
        candidate_id = selected[method]["id"]
        prior = next(t for t in prior_states if t["candidate_id"] == candidate_id and t["seed_id"] == sid)
        prior_result = next(t for t in prior_results if t["candidate_id"] == candidate_id and t["seed_id"] == sid)
        prior_row = next(t for t in prior_history if t["candidate_id"] == candidate_id and t["seed_id"] == sid and int(t["step"]) == 2000)
        checkpoint = next(p for p in points if p["step"] == 2000)
        parameter_error = float(np.max(np.abs(np.asarray(checkpoint["theta"]) - np.asarray(prior["theta"]))))
        maximum_errors["step_2000_parameter_absolute_error"] = max(maximum_errors["step_2000_parameter_absolute_error"], parameter_error)
        check(f"prior_2000_parameters/{prefix}", parameter_error <= 1e-12, {"absolute_error": parameter_error, "absolute_tolerance": 1e-12})
        close(f"prior_2000_displacement/{prefix}", checkpoint["cumulative_feature_displacement"],
              prior_row["cumulative_feature_displacement"], "step_2000_metric_absolute_error")
        check(f"prior_data_hashes/{prefix}", prior_result["train_data_sha256"] == trajectory["train_data_sha256"]
              and prior_result["initial_parameter_sha256"] == trajectory["initial_parameter_sha256"]
              and prior_result["validation_data_sha256"] == digest(xv, yv))
        for point in points:
            key = sid, method, point["step"]
            row = keyed[key]
            measured = []
            for split, xx, yy in (("train", x, y), ("validation", xv, yv)):
                z = representation(np.asarray(point["theta"], dtype=np.float64), xx)
                expected_loss = float(np.logaddexp(0.0, -yy * z[:, 0]).mean())
                expected_accuracy = float(np.mean(np.where(z[:, 0] >= 0, 1, -1) == yy))
                for field, expected in ((f"{split}_loss", expected_loss), (f"{split}_accuracy", expected_accuracy)):
                    error = abs(float(row[field]) - expected)
                    maximum_errors["metric_absolute_error"] = max(maximum_errors["metric_absolute_error"], error)
                    measured.append(np.isfinite(expected) and error <= 1e-12)
            measured.append(float(row["training_seconds"]) == point["training_seconds"])
            measured.append(float(row["cumulative_feature_displacement"]) == point["cumulative_feature_displacement"])
            check(f"saved_state_metrics/{prefix}/{point['step']}", all(measured))
        row2000 = keyed[(sid, method, 2000)]
        for field in ("train_loss", "validation_loss", "train_accuracy", "validation_accuracy"):
            close(f"prior_2000_metrics/{prefix}/{field}", row2000[field], prior_row[field], "step_2000_metric_absolute_error")
    check("summary_complete", summary["aggregate_complete"] is True)
    fields = ("train_loss", "validation_loss", "train_accuracy", "validation_accuracy", "training_seconds", "cumulative_feature_displacement")
    for step in config["checkpoint_steps"]:
        for method in config["methods"]:
            for field in fields:
                values = np.array([float(keyed[(sid, method, step)][field]) for sid in seed_ids])
                record = summary["endpoints"][str(step)][method][field]
                close(f"mean/{step}/{method}/{field}", record["mean"], values.mean())
                close(f"sample_sd/{step}/{method}/{field}", record["sample_sd"], values.std(ddof=1))
        for field in ("train_loss", "validation_loss", "validation_accuracy"):
            expected = np.array([float(keyed[(sid, "pullback", step)][field]) - float(keyed[(sid, "gd", step)][field]) for sid in seed_ids])
            record = summary["paired_differences"][str(step)][field]
            check(f"paired_values/{step}/{field}", record["definition"] == "pullback minus GD"
                  and record["seed_order"] == seed_ids and np.array_equal(record["values"], expected))
            close(f"paired_mean/{step}/{field}", record["mean"], expected.mean())
            close(f"paired_sample_sd/{step}/{field}", record["sample_sd"], expected.std(ddof=1))
    rule = config["continuation_review"]
    continuation = []
    for method in config["methods"]:
        record = summary["continuation_review"][method]
        decreases = {}
        for field in ("train_loss", "validation_loss"):
            a = np.mean([float(keyed[(sid, method, rule["start_step"])][field]) for sid in seed_ids])
            b = np.mean([float(keyed[(sid, method, rule["end_step"])][field]) for sid in seed_ids])
            decreases[field] = (a - b) / a if a > 0 else 0.0
            close(f"continuation_change/{method}/{field}", record["relative_mean_loss_decrease"][field], decreases[field])
        improved = {sid: float(keyed[(sid, method, rule["end_step"])]["validation_loss"])
                    < float(keyed[(sid, method, rule["start_step"])]["validation_loss"]) for sid in seed_ids}
        condition = all(x >= rule["minimum_relative_loss_decrease"] for x in decreases.values()) and sum(improved.values()) >= rule["minimum_validation_seed_improvements"]
        continuation.append(condition)
        check(f"continuation_seed_count/{method}", record["validation_improved_by_seed"] == improved and record["validation_seed_improvements"] == sum(improved.values()))
        check(f"continuation_decision/{method}", record["criterion_met"] == condition)
    check("combined_continuation_decision", summary["further_training_supported_by_declared_criterion"] == any(continuation))
    check("no_automatic_extension", summary["automatic_extension"] is False and rule["automatic_extension"] is False)
    for field, source in (("core_training_seconds", "training_seconds"), ("online_diagnostics_seconds", "diagnostics_seconds"), ("warmup_seconds", "warmup_seconds")):
        close(f"batch_time/{field}", timing[field], sum(t[source] for t in trajectories))
    check("batch_wall_time_bound", timing["total_wall_seconds"] >= sum(t["loop_wall_seconds"] for t in trajectories) + timing["offline_evaluation_seconds"])
    check("figures_present", (RUN / "learning_curves.svg").stat().st_size > 1000 and (RUN / "learning_curves.png").stat().st_size > 1000)
    report = {"passed": all(c["passed"] for c in checks), "checks_passed": sum(c["passed"] for c in checks),
              "checks_total": len(checks), "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_commit": protocol_commit, "audited_run": "runs/long-training-v1", "maximum_errors": maximum_errors,
              "scope": "Independent metric and summary recomputation, six exact 2,000-step prior-state comparisons, data reconstruction, provenance and bounded-budget checks; not a full 100,000-update replay.",
              "checks": checks}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    for item in checks:
        if not item["passed"]:
            print(json.dumps(item))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        sys.exit(main())
