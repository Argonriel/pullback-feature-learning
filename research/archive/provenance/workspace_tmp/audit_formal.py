"""Independent checks of the archived formal batch; no timed batch rerun."""
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning.core import direction, initialize_parameters
from pullback_feature_learning.data import two_moons

ROOT = Path.cwd()
SOURCE = ROOT / "runs/formal-v1"
start = perf_counter()


def read(name):
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def independent_forward(theta, inputs):
    state = np.asarray(inputs, dtype=np.float64)
    parameters = np.asarray(theta, dtype=np.float64)
    for offset in (0, 40):
        a = parameters[offset:offset + 16].reshape(8, 2)
        b = parameters[offset + 16:offset + 32].reshape(2, 8)
        bias = parameters[offset + 32:offset + 40]
        hidden = np.tanh(state @ a.T + bias)
        state = state + (hidden @ b.T) / 2.0
    return state


checks = []


def check(name, condition, details=None):
    checks.append({"name": name, "passed": bool(condition), **({"details": details} if details is not None else {})})


config = read("config.json")
environment = read("environment.json")
data = read("data.json")
trajectories = read("trajectories.json")
schedule = read("schedule.json")
summary = read("summary.json")
with (SOURCE / "metrics.csv").open(newline="", encoding="utf-8") as f:
    metrics = list(csv.DictReader(f))
with (SOURCE / "endpoint_metrics.csv").open(newline="", encoding="utf-8") as f:
    endpoints = list(csv.DictReader(f))
frozen = json.loads(subprocess.check_output(["git", "show", "8e24690f4f2af0e2c20a28701a2c0ea591cef4e7:configs/formal.json"], text=True))
check("Configuration equals frozen stage-4 commit", config == frozen)
check("Configuration file equals current repository specification", config == json.loads((ROOT / "configs/formal.json").read_text()))
check("Recorded source files match their SHA-256 fingerprints", all(hashlib.sha256((ROOT / "src/pullback_feature_learning" / name).read_bytes()).hexdigest() == value for name, value in environment["source_sha256"].items()), {"source_files": len(environment["source_sha256"])})
expected_schedule = [[seed["id"], method] for index, seed in enumerate(config["seeds"]) for method in (("gd", "pullback") if index % 2 == 0 else ("pullback", "gd"))]
actual_schedule = [[t["seed_id"], t["method"]] for t in trajectories]
check("All six scheduled paired runs are present in declared order", len(trajectories) == 6 and schedule == expected_schedule == actual_schedule)
check("Every trajectory completed successfully", all(t["status"] == "completed" and t["reason"] is None for t in trajectories))
check("Test evaluation starts after all training ends", datetime.fromisoformat(environment["training_finished_at_utc"]) <= datetime.fromisoformat(environment["test_evaluation_started_at_utc"]))
check("Offline test evaluation is complete and not used for selection", environment["test_generated"] and environment["test_evaluated"] and environment["test_evaluation_complete"] and not environment["test_used_for_training_or_selection"])
check("Float64 CPU with one BLAS thread recorded", environment["dtype"] == "float64" and environment["device"] == "cpu" and all(p["num_threads"] == 1 for p in environment["threadpools"]))

numeric_fields = [key for key in metrics[0] if key not in ("seed_id", "method", "labels")]
check("Every sampled metric is finite", all(np.isfinite(float(row[key])) for row in metrics for key in numeric_fields), {"metric_rows": len(metrics)})
check("Endpoint records cover exactly two endpoints for all six trajectories", len(endpoints) == 12 and len({(r["seed_id"], r["method"], r["endpoint"]) for r in endpoints}) == 12)
check("Every saved checkpoint has one offline metric row", len(metrics) == sum(len(t["checkpoints"]) for t in trajectories) and len({(r["seed_id"], r["method"], int(r["step"])) for r in metrics}) == len(metrics))

arrays = {}
for seed in config["seeds"]:
    seed_id = seed["id"]
    values = data[seed_id]
    split_arrays = {}
    for split, count in (("train", config["train_size"]), ("validation", config["validation_size"]), ("test", config["test_size"])):
        x, y = np.array(values[f"{split}_x"]), np.array(values[f"{split}_y"])
        expected_x, expected_y = two_moons(count, config["noise"], seed[f"{split}_seed"])
        check(f"{seed_id}: {split} data regenerate exactly from declared seed", np.array_equal(x, expected_x) and np.array_equal(y, expected_y))
        split_arrays[split] = x, y
    initial = np.array(values["initial_theta"])
    check(f"{seed_id}: initialization regenerates exactly", np.array_equal(initial, initialize_parameters(seed["initialization_seed"])))
    arrays[seed_id] = split_arrays, initial
    pair = [t for t in trajectories if t["seed_id"] == seed_id]
    check(f"{seed_id}: paired training data and initialization fingerprints agree", len(pair) == 2 and all(t["initial_parameter_sha256"] == digest(initial) and t["train_data_sha256"] == digest(*split_arrays["train"]) for t in pair))

maximum_metric_error = 0.0
maximum_replay_error = 0.0
maximum_replay_displacement_error = 0.0
maximum_crossing_error = 0.0
recomputed = {}

with threadpool_limits(limits=1):
    for trajectory in trajectories:
        seed_id, method = trajectory["seed_id"], trajectory["method"]
        prefix = f"{seed_id}/{method}"
        splits, initial = arrays[seed_id]
        points = trajectory["checkpoints"]
        by_step = {point["step"]: point for point in points}
        end = trajectory["endpoints"]
        check(f"{prefix}: checkpoint steps and core times are ordered", len(by_step) == len(points) and np.all(np.diff([p["step"] for p in points]) > 0) and np.all(np.diff([p["training_seconds"] for p in points]) > 0))
        check(f"{prefix}: displacement is finite and nondecreasing", all(np.isfinite(p["cumulative_feature_displacement"]) for p in points) and np.all(np.diff([p["cumulative_feature_displacement"] for p in points]) >= 0))
        check(f"{prefix}: initial snapshot equals paired initialization", np.array_equal(points[0]["theta"], initial) and points[0]["step"] == 0 and points[0]["training_seconds"] == 0)
        check(f"{prefix}: exact iteration endpoint is present", end["iteration"]["step"] == 2000 and end["iteration"] == by_step[2000] and "iteration" in by_step[2000]["labels"])
        check(f"{prefix}: strict adjacent time bracket is present", end["time"]["training_seconds"] <= 2.0 < end["time_crossing"]["training_seconds"] and end["time_crossing"]["step"] == end["time"]["step"] + 1 and end["time"] == by_step[end["time"]["step"]] and end["time_crossing"] == by_step[end["time_crossing"]["step"]])
        check(f"{prefix}: termination satisfies both budgets without replacing endpoints", trajectory["total_steps"] == points[-1]["step"] and trajectory["total_steps"] >= 2000 and trajectory["training_seconds"] > 2.0 and "termination" in points[-1]["labels"])
        check(f"{prefix}: initial, middle, and exact iteration snapshots preserved", {0, 1000, 2000}.issubset(by_step))
        eta = config["methods"][method]["learning_rate"]
        damping = config["methods"][method].get("damping", 0.01)
        x, y = splits["train"]
        time_theta = np.array(end["time"]["theta"])
        u, _ = direction(time_theta, x, y, method, damping)
        crossing_error = float(np.max(np.abs(time_theta + eta * u - np.array(end["time_crossing"]["theta"]))))
        maximum_crossing_error = max(maximum_crossing_error, crossing_error)
        check(f"{prefix}: crossing snapshot is exactly one optimizer update after time endpoint", crossing_error <= 1e-12, {"max_absolute_parameter_error": crossing_error})
        for endpoint in ("iteration", "time"):
            point = end[endpoint]
            row = next(r for r in endpoints if r["seed_id"] == seed_id and r["method"] == method and r["endpoint"] == endpoint)
            check(f"{prefix}/{endpoint}: metric row refers to the declared state", int(row["step"]) == point["step"] and float(row["training_seconds"]) == point["training_seconds"])
            calculated = {"step": float(point["step"]), "training_seconds": point["training_seconds"]}
            for split, (features, labels) in splits.items():
                representations = independent_forward(point["theta"], features)
                loss = float(np.mean(np.logaddexp(0.0, -labels * representations[:, 0])))
                accuracy = float(np.mean(np.where(representations[:, 0] >= 0, 1, -1) == labels))
                calculated[f"{split}_loss"] = loss
                calculated[f"{split}_accuracy"] = accuracy
                maximum_metric_error = max(maximum_metric_error, abs(loss - float(row[f"{split}_loss"])), abs(accuracy - float(row[f"{split}_accuracy"])))
            recomputed[(seed_id, method, endpoint)] = calculated
        theta = initial.copy()
        displacement = 0.0
        previous_features = independent_forward(theta, x)
        for _ in range(2000):
            u, _ = direction(theta, x, y, method, damping)
            theta = theta + eta * u
            features = independent_forward(theta, x)
            displacement += float(np.sqrt(np.mean(np.sum((features - previous_features)**2, axis=1))))
            previous_features = features
        replay_error = float(np.max(np.abs(theta - np.array(end["iteration"]["theta"]))))
        displacement_error = abs(displacement - end["iteration"]["cumulative_feature_displacement"])
        maximum_replay_error = max(maximum_replay_error, replay_error)
        maximum_replay_displacement_error = max(maximum_replay_displacement_error, displacement_error)
        check(f"{prefix}: deterministic 2000-update replay matches archived state", replay_error <= 1e-12, {"max_absolute_parameter_error": replay_error})
        check(f"{prefix}: replayed cumulative displacement matches", displacement_error <= 1e-10, {"absolute_error": displacement_error})

check("All 12 endpoint metrics match independent forward evaluation", maximum_metric_error <= 1e-12, {"maximum_absolute_error": maximum_metric_error, "endpoints": 12, "splits_per_endpoint": 3})
maximum_summary_error = 0.0
seed_ids = [seed["id"] for seed in config["seeds"]]
for endpoint in ("iteration", "time"):
    for method in ("gd", "pullback"):
        for metric, saved in summary["methods"][endpoint][method].items():
            numbers = [recomputed[(seed, method, endpoint)][metric] for seed in seed_ids]
            maximum_summary_error = max(maximum_summary_error, abs(float(np.mean(numbers)) - saved["mean"]), abs(float(np.std(numbers, ddof=1)) - saved["sample_sd"]))
    for metric in ("test_loss", "test_accuracy"):
        differences = [recomputed[(seed, "pullback", endpoint)][metric] - recomputed[(seed, "gd", endpoint)][metric] for seed in seed_ids]
        saved = summary["paired_differences"][endpoint][metric]
        maximum_summary_error = max(maximum_summary_error, abs(float(np.mean(differences)) - saved["mean"]), abs(float(np.std(differences, ddof=1)) - saved["sample_sd"]), float(np.max(np.abs(np.array(differences) - saved["values"]))))
        check(f"{endpoint}/{metric}: paired differences use the declared sign and seed order", saved["definition"] == "pullback minus GD" and saved["seed_order"] == seed_ids)
check("All endpoint aggregates and paired differences recompute correctly", maximum_summary_error <= 1e-12, {"maximum_absolute_error": maximum_summary_error, "sample_standard_deviation_ddof": 1})
check("Complete aggregate covers all three paired seeds", summary["aggregate_complete"] is True and len(seed_ids) == 3)

report = {"passed": all(c["passed"] for c in checks), "checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks),
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "input_artifacts": "runs/formal-v1 (canonical archive: reports/formal-v1)",
          "audit_seconds": perf_counter() - start,
          "maximum_endpoint_metric_absolute_error": maximum_metric_error,
          "maximum_2000_step_parameter_replay_error": maximum_replay_error,
          "maximum_replay_displacement_error": maximum_replay_displacement_error,
          "maximum_time_bracket_update_error": maximum_crossing_error,
          "maximum_summary_absolute_error": maximum_summary_error,
          "replayed_updates": 6 * 2000,
          "timed_formal_batch_rerun": False,
          "notes": ["The 2000-update endpoints were deterministically replayed with one BLAS thread.",
                    "Time endpoints were checked against archived adjacent states; no claim of identical clock-based endpoint counts on rerun is made.",
                    "Test metrics were independently recomputed from saved parameters and archived arrays, without parameter retuning."],
          "checks": checks}
(ROOT / "reports/formal_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
if not report["passed"]:
    raise SystemExit(1)
