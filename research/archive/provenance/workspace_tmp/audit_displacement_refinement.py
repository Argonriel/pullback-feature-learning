"""Independently audit whole-update displacement refinement without retuning.

Uses the original core update only for deterministic replay. Feature reconstruction,
path integration, budget selection, predictions and statistics are independent.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits


HELPER_PATH = Path(__file__).with_name("audit_stage6.py")
SPEC = importlib.util.spec_from_file_location("independent_stage6_helpers", HELPER_PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def read_metrics(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if key not in ("seed_id", "method", "side", "labels"):
                row[key] = int(value) if key == "step" else float(value)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", default="runs/displacement-refinement-v1")
    parser.add_argument("--output", default="reports/displacement_refinement_audit.json")
    args = parser.parse_args()
    started = perf_counter()
    root = args.root.resolve()
    source, output = root / "reports/formal-v1", root / args.run
    guarded = [p for directory in (source, output, root / "src/pullback_feature_learning") for p in directory.iterdir() if p.is_file()]
    original_hashes = {str(p): A.sha(p) for p in guarded}
    env = A.verify_provenance(root, output, source, "refinement", ["docs/displacement_refinement_protocol.md"])
    config = A.read_json(source / "config.json")
    data = A.read_json(source / "data.json")
    trajectories = A.read_json(source / "trajectories.json")
    replays = A.read_json(output / "replays.json")
    metrics = read_metrics(output / "metrics.csv")
    summary = A.read_json(output / "summary.json")
    timing = A.read_json(output / "timing.json")
    fractions = (.25, .5, .75, 1.)
    seed_ids = [s["id"] for s in config["seeds"]]
    paired_keys = {(s, m) for s in seed_ids for m in ("gd", "pullback")}
    source_keyed = {(r["seed_id"], r["method"]): r for r in trajectories}
    replays_keyed = {(r["seed_id"], r["method"]): r for r in replays}
    A.check("all six original source trajectories and complete replays",
            len(trajectories) == len(replays) == 6 and set(source_keyed) == set(replays_keyed) == paired_keys
            and all(r["status"] == "completed" and r["reason"] is None and r["total_steps"] == 2000 for r in replays))
    A.check("original optimizer settings and displacement-only selection retained",
            env["optimizer_settings_changed"] is False and env["test_used_for_selection"] is False
            and env["test_evaluated"] is True and env["fractions"] == list(fractions))
    A.check("all replay verification and bracket selection preceded test evaluation",
            A.stamp(env["recorded_at_utc"]) <= A.stamp(env["all_replays_verified_at_utc"])
            <= A.stamp(env["test_evaluation_started_at_utc"]) <= A.stamp(env["test_evaluation_finished_at_utc"]),
            {k: v for k, v in env.items() if k.endswith("_at_utc")})
    base = {s: min(source_keyed[(s, m)]["endpoints"]["iteration"]["cumulative_feature_displacement"] for m in ("gd", "pullback")) for s in seed_ids}
    A.compare_tree("all ceiling bases derive only from original training displacement", env["base_displacement_by_seed"], base, "target_definitions")
    A.check("source and replay scope stop at exactly 2000 updates", config["iteration_endpoint"] == 2000)
    sys.path.insert(0, str(root / "src"))
    from pullback_feature_learning.core import direction

    histories, distances, expected_brackets = {}, {}, {}
    archived_states_checked = 0
    replay_start = perf_counter()
    with threadpool_limits(limits=1):
        for pair in [(s, m) for s in seed_ids for m in ("gd", "pullback")]:
            sid, method = pair
            name = f"{sid}/{method}"
            original, replay = source_keyed[pair], replays_keyed[pair]
            x, y = np.asarray(data[sid]["train_x"], dtype=float), np.asarray(data[sid]["train_y"], dtype=float)
            theta = np.asarray(data[sid]["initial_theta"], dtype=float).copy()
            states = np.empty((2001, len(theta)), dtype=float)
            path = np.zeros(2001, dtype=float)
            states[0] = theta
            previous_features = A.forward(theta, x)
            rate = config["methods"][method]["learning_rate"]
            damping = config["methods"][method].get("damping", .01)
            for step in range(1, 2001):
                update, _ = direction(theta, x, y, method, damping)
                theta = theta + rate * update
                new_features = A.forward(theta, x)
                increment = float(np.sqrt(np.mean(np.sum((new_features - previous_features) ** 2, axis=1))))
                path[step] = path[step - 1] + increment
                states[step] = theta
                previous_features = new_features
            histories[pair], distances[pair] = states, path
            A.check(name + ": all 2001 replay states and path lengths finite", np.isfinite(states).all() and np.isfinite(path).all() and np.all(np.diff(path) >= 0))
            points = [p for p in original["checkpoints"] if p["step"] <= 2000]
            expected_verification = []
            for point in points:
                step = point["step"]
                parameter_error = float(np.max(np.abs(states[step] - np.asarray(point["theta"]))))
                displacement_error = abs(path[step] - point["cumulative_feature_displacement"])
                exact = np.array_equal(states[step], np.asarray(point["theta"])) and path[step] == point["cumulative_feature_displacement"]
                A.check(name + f": exact original checkpoint and cumulative path at step {step}", exact,
                        {"parameter_maximum_absolute_error": parameter_error, "path_absolute_error": displacement_error})
                A.MAX_ERRORS["archived_parameter_replay"] = max(A.MAX_ERRORS["archived_parameter_replay"], parameter_error)
                A.MAX_ERRORS["archived_cumulative_displacement"] = max(A.MAX_ERRORS["archived_cumulative_displacement"], displacement_error)
                expected_verification.append({"step": step, "passed": bool(exact), "parameter_max_absolute_error": parameter_error,
                                              "displacement_absolute_error": float(displacement_error)})
                archived_states_checked += 1
            A.compare_tree(name + ": replay verification evidence", replay["verification"], expected_verification, "verification_records")
            A.check(name + ": no omitted archived state", replay["verified_archived_states"] == replay["expected_archived_states"] == len(points))

            def checkpoint(step):
                return {"step": step, "theta": states[step].tolist(), "cumulative_feature_displacement": float(path[step]), "labels": []}

            A.compare_tree(name + ": final replay state", replay["final_state"], checkpoint(2000), "saved_parameters_and_path")
            expected = {}
            for fraction in fractions:
                target = fraction * base[sid]
                eligible = np.flatnonzero(path <= target)
                lower_step = int(eligible[-1])
                upper_step = lower_step + 1 if lower_step < 2000 else None
                lower = checkpoint(lower_step)
                upper = checkpoint(upper_step) if upper_step is not None else None
                expected[str(fraction)] = {"target_displacement": target, "lower_checkpoint": lower, "upper_checkpoint": upper}
                saved = replay["brackets"][str(fraction)]
                A.compare_tree(name + f": independently selected {fraction:g} whole-update bracket", saved, expected[str(fraction)], "selected_brackets")
                if upper is not None:
                    A.check(name + f": strict one-update crossing at fraction {fraction:g}", lower_step + 1 == upper_step and lower["cumulative_feature_displacement"] <= target < upper["cumulative_feature_displacement"])
                    lower_theta = np.asarray(saved["lower_checkpoint"]["theta"])
                    delta, _ = direction(lower_theta, x, y, method, damping)
                    next_theta = lower_theta + rate * delta
                    A.close(name + f": saved upper is one core update from lower at fraction {fraction:g}",
                            saved["upper_checkpoint"]["theta"], next_theta, "one_update_parameters", atol=0, rtol=0)
                    lower_features, upper_features = A.forward(lower_theta, x), A.forward(next_theta, x)
                    one_move = float(np.sqrt(np.mean(np.sum((upper_features - lower_features) ** 2, axis=1))))
                    A.close(name + f": one-update displacement increment at fraction {fraction:g}",
                            saved["upper_checkpoint"]["cumulative_feature_displacement"], saved["lower_checkpoint"]["cumulative_feature_displacement"] + one_move,
                            "one_update_displacement", atol=0, rtol=0)
                else:
                    A.check(name + f": absent upper only for exact terminal ceiling at fraction {fraction:g}", lower_step == 2000 and target == lower["cumulative_feature_displacement"])
            expected_brackets[pair] = expected
    independent_replay_seconds = perf_counter() - replay_start

    # All training-only independent replay and selection is finished before this
    # audit reconstructs any held-out features.
    row_key = lambda r: (r["seed_id"], r["method"], r["fraction"], r["side"])
    metrics_keyed = {row_key(r): r for r in metrics}
    expected_metrics = {}
    tested_examples = 0
    with threadpool_limits(limits=1):
        for pair, brackets in expected_brackets.items():
            sid, method = pair
            for fraction_text, record in brackets.items():
                fraction = float(fraction_text)
                for side in ("lower", "upper"):
                    state = record[side + "_checkpoint"]
                    if state is None:
                        continue
                    row = {"seed_id": sid, "method": method, "fraction": fraction, "side": side,
                           "target_displacement": record["target_displacement"],
                           "target_minus_displacement": record["target_displacement"] - state["cumulative_feature_displacement"],
                           "step": state["step"], "cumulative_feature_displacement": state["cumulative_feature_displacement"], "labels": ""}
                    for split in ("train", "test"):
                        z = A.forward(state["theta"], data[sid][split + "_x"])
                        y = np.asarray(data[sid][split + "_y"], dtype=float)
                        computed = A.score_metrics(z[:, 0], y)
                        row.update({split + "_" + k: v for k, v in computed.items()})
                        if split == "test":
                            tested_examples += len(y)
                    task = row_key(row)
                    expected_metrics[task] = row
                    A.compare_tree(f"{sid}/{method}/{fraction:g}/{side}: independently evaluated predictions", metrics_keyed[task], row, "prediction_metrics")
    A.check("exactly 24 lower and 21 upper states, with no invalid survivors", len(metrics) == len(metrics_keyed) == len(expected_metrics) == 45 and set(metrics_keyed) == set(expected_metrics)
            and sum(r["side"] == "lower" for r in metrics) == 24 and sum(r["side"] == "upper" for r in metrics) == 21)
    A.check("all saved metric numbers finite", all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v, (int, float))))
    fields = ("train_loss", "train_accuracy", "test_loss", "test_accuracy", "step", "cumulative_feature_displacement", "target_minus_displacement")
    expected_summary = {"aggregate_complete": True, "budgets": {}}
    for fraction in fractions:
        methods, paired = {}, {}
        for method in ("gd", "pullback"):
            rows = [expected_metrics[(sid, method, fraction, "lower")] for sid in seed_ids]
            methods[method] = {}
            for field in fields:
                stat = A.describe([r[field] for r in rows])
                methods[method][field] = {k: stat[k] for k in ("mean", "sample_sd")}
            methods[method]["maximum_unused_budget"] = max(r["target_minus_displacement"] for r in rows)
        for field in ("train_loss", "test_loss", "test_accuracy"):
            values = [expected_metrics[(sid, "pullback", fraction, "lower")][field] - expected_metrics[(sid, "gd", fraction, "lower")][field] for sid in seed_ids]
            paired[field] = {"definition": "pullback minus GD", "seed_order": seed_ids, **A.describe(values)}
        expected_summary["budgets"][str(fraction)] = {"methods": methods, "paired_differences": paired}
    A.compare_tree("all means, sample SDs, paired differences and unused-budget maxima", summary, expected_summary, "summary")
    A.close("recorded replay cost sums all six replays", timing["replay_wall_seconds"], sum(r["replay_wall_seconds"] for r in replays), "timing")
    A.check("replay costs finite, positive and explicitly diagnostic", all(np.isfinite(v) and v > 0 for k, v in timing.items() if k != "role") and timing["role"] == "Diagnostic replay cost; not a replacement for original training timings.")
    A.check("audit left all source artifacts and run files unchanged", original_hashes == {str(p): A.sha(p) for p in guarded})
    report = {"audit": "Independent whole-update displacement refinement audit", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "all_passed": all(c["passed"] for c in A.CHECKS), "checks_passed": sum(c["passed"] for c in A.CHECKS), "checks_total": len(A.CHECKS),
              "protocol_commit": env["protocol_commit"], "run_directory": args.run,
              "independence": "Six deterministic 2000-update replays use the frozen core direction. Feature forward passes, every cumulative path increment, target selection, held-out metrics and statistics are recomputed without analysis-runner functions.",
              "terminal_exception": "Exactly three 100% pullback ceilings coincide with step 2000 and legitimately have no upper checkpoint. All other 21 brackets span exactly one update.",
              "counts": {"paired_trajectories": 6, "independent_updates": 12000, "original_archived_states_verified": archived_states_checked,
                         "budgets": 24, "one_update_brackets": 21, "terminal_equalities": 3, "metric_rows": len(metrics), "test_predictions": tested_examples},
              "maximum_absolute_errors": dict(A.MAX_ERRORS), "independent_replay_seconds": independent_replay_seconds,
              "audit_seconds": perf_counter() - started, "recorded_run_timing_seconds": timing,
              "audit_script_sha256": A.sha(Path(__file__)), "audit_helper_sha256": A.sha(HELPER_PATH),
              "limitations": ["These are secondary diagnostics on previously inspected formal test data, not a new independent test.",
                              "Selected realized displacements differ by at most their observed crossing-update increments; exact equal movement is not asserted.",
                              "Execution chronology is supported by committed code, recorded phase timestamps and phase-ordering tests, not a cryptographic trace."],
              "checks": A.CHECKS}
    (root / args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_passed", "checks_passed", "checks_total", "counts", "maximum_absolute_errors", "independent_replay_seconds", "audit_seconds")}, indent=2))
    print(json.dumps({"failures": [c for c in A.CHECKS if not c["passed"]]}, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
