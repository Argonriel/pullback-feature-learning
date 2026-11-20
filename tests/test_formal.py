"""Formal endpoint and trajectory checks on tiny synthetic training arrays.

The injected clock makes update durations exact and deterministic. No formal
data split or held-out evaluation is used by these tests.
"""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from pullback_feature_learning import formal
from pullback_feature_learning.core import direction, initialize_parameters, parameter_gradient
from pullback_feature_learning.data import two_moons


class UpdateClock:
    """Supply paired start/end calls with a fixed positive update duration."""

    def __init__(self, duration):
        self.duration = duration
        self.calls = 0

    def __call__(self):
        update_index, end = divmod(self.calls, 2)
        self.calls += 1
        return (update_index + end) * self.duration


class FormalEndpointTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(733)
        self.x = rng.normal(size=(8, 2))
        self.y = np.where(self.x[:, 0] >= 0, 1.0, -1.0)
        self.initial = initialize_parameters(311)
        self.config = {
            "iteration_endpoint": 4,
            "time_endpoint_seconds": 1.0,
            "methods": {
                "gd": {"learning_rate": 0.125},
                "pullback": {"learning_rate": 8.0, "damping": 0.1},
            },
            "log_every": 100,
            "warmup_steps": 0,
        }

    def run_small(self, config=None, method="gd", duration=0.25):
        config = copy.deepcopy(config or self.config)
        clock = UpdateClock(duration)
        result = formal.train_trajectory(
            method, config, self.x, self.y, self.initial, clock=clock
        )
        return result, clock

    def replay(self, method, count):
        theta = self.initial.copy()
        options = self.config["methods"][method]
        for _ in range(count):
            if method == "gd":
                update = -parameter_gradient(theta, self.x, self.y)
            else:
                update, _ = direction(theta, self.x, self.y, method,
                                      options["damping"])
            theta = theta + options["learning_rate"] * update
        return theta

    def assert_checkpoint(self, checkpoint, method, step, elapsed):
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint["step"], step)
        self.assertEqual(checkpoint["training_seconds"], elapsed)
        assert_allclose(checkpoint["theta"], self.replay(method, step), rtol=0, atol=0)

    def test_exact_time_equality_is_eligible_and_next_update_brackets_it(self):
        result, clock = self.run_small()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_steps"], 5)
        self.assertEqual(clock.calls, 10)
        self.assert_checkpoint(result["endpoints"]["iteration"], "gd", 4, 1.0)
        self.assert_checkpoint(result["endpoints"]["time"], "gd", 4, 1.0)
        self.assert_checkpoint(result["endpoints"]["time_crossing"], "gd", 5, 1.25)

    def test_iteration_budget_can_require_continuing_after_time_crossing(self):
        config = copy.deepcopy(self.config)
        config["iteration_endpoint"] = 8
        result, clock = self.run_small(config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_steps"], 8)
        self.assertEqual(clock.calls, 16)
        self.assert_checkpoint(result["endpoints"]["iteration"], "gd", 8, 2.0)
        self.assert_checkpoint(result["endpoints"]["time"], "gd", 4, 1.0)
        self.assert_checkpoint(result["endpoints"]["time_crossing"], "gd", 5, 1.25)

    def test_time_budget_can_require_continuing_after_iteration_endpoint(self):
        config = copy.deepcopy(self.config)
        config["iteration_endpoint"] = 2
        for method in ("gd", "pullback"):
            with self.subTest(method=method):
                result, clock = self.run_small(config, method, duration=0.125)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["total_steps"], 9)
                self.assertEqual(clock.calls, 18)
                self.assert_checkpoint(result["endpoints"]["iteration"], method, 2, 0.25)
                self.assert_checkpoint(result["endpoints"]["time"], method, 8, 1.0)
                self.assert_checkpoint(result["endpoints"]["time_crossing"], method, 9, 1.125)

    def test_nonexact_time_budget_selects_previous_completed_update(self):
        config = copy.deepcopy(self.config)
        config["time_endpoint_seconds"] = 0.875
        result, _ = self.run_small(config)
        self.assertEqual(result["total_steps"], 4)
        self.assert_checkpoint(result["endpoints"]["time"], "gd", 3, 0.75)
        self.assert_checkpoint(result["endpoints"]["time_crossing"], "gd", 4, 1.0)
        self.assert_checkpoint(result["endpoints"]["iteration"], "gd", 4, 1.0)

    def test_first_update_crossing_keeps_initial_state_as_time_endpoint(self):
        config = copy.deepcopy(self.config)
        config["iteration_endpoint"] = 3
        result, clock = self.run_small(config, duration=2.0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_steps"], 3)
        self.assertEqual(clock.calls, 6)
        self.assert_checkpoint(result["endpoints"]["time"], "gd", 0, 0.0)
        self.assert_checkpoint(result["endpoints"]["time_crossing"], "gd", 1, 2.0)
        self.assert_checkpoint(result["endpoints"]["iteration"], "gd", 3, 6.0)

    def test_sparse_logging_still_preserves_endpoints_and_middle_checkpoint(self):
        config = copy.deepcopy(self.config)
        config["iteration_endpoint"] = 8
        result, _ = self.run_small(config)
        checkpoint_steps = {checkpoint["step"] for checkpoint in result["checkpoints"]}
        history_steps = {row["step"] for row in result["history"]}
        self.assertTrue({0, 4, 5, 8}.issubset(checkpoint_steps))
        self.assertTrue({0, 4, 5, 8}.issubset(history_steps))
        rows = sorted(result["history"], key=lambda row: row["step"])
        displacements = [row["cumulative_feature_displacement"] for row in rows]
        self.assertEqual(displacements[0], 0.0)
        self.assertTrue(np.isfinite(displacements).all())
        self.assertTrue(np.all(np.diff(displacements) >= 0))

    def test_warmup_resets_parameters_and_does_not_consume_measured_time(self):
        original = self.initial.copy()
        cold, cold_clock = self.run_small()
        config = copy.deepcopy(self.config)
        config["warmup_steps"] = 5
        warm, warm_clock = self.run_small(config)
        self.assertEqual(cold["total_steps"], warm["total_steps"])
        self.assertEqual(cold["training_seconds"], warm["training_seconds"])
        self.assertEqual(cold_clock.calls, warm_clock.calls)
        for endpoint in ("iteration", "time", "time_crossing"):
            assert_allclose(cold["endpoints"][endpoint]["theta"],
                            warm["endpoints"][endpoint]["theta"], rtol=0, atol=0)
        assert_allclose(self.initial, original, rtol=0, atol=0)

    def test_failed_update_preserves_prefix_without_claiming_completion(self):
        config = copy.deepcopy(self.config)
        config["time_endpoint_seconds"] = 10.0
        config["log_every"] = 1
        calls = 0

        def failing_direction(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise np.linalg.LinAlgError("Forced solve failure for the endpoint test.")
            return direction(*args, **kwargs)

        with patch.object(formal, "direction", side_effect=failing_direction):
            result, _ = self.run_small(config)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["total_steps"], 2)
        self.assertIn("Forced solve failure", result["reason"])
        self.assertIsNone(result["endpoints"]["iteration"])
        prefix = next(c for c in result["checkpoints"] if c["step"] == 2)
        self.assert_checkpoint(prefix, "gd", 2, 0.5)

    def test_warmup_failure_is_recorded_before_any_measured_update(self):
        config = copy.deepcopy(self.config)
        config["warmup_steps"] = 5
        clock = UpdateClock(0.25)
        with patch.object(formal, "direction", side_effect=FloatingPointError("Forced warmup failure.")):
            result = formal.train_trajectory(
                "gd", config, self.x, self.y, self.initial, clock=clock
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["total_steps"], 0)
        self.assertEqual(result["training_seconds"], 0.0)
        self.assertEqual(clock.calls, 0)
        self.assertIn("Forced warmup failure", result["reason"])
        self.assertIsNone(result["endpoints"]["iteration"])


class FormalSummaryTests(unittest.TestCase):
    def setUp(self):
        self.seeds = ["small1", "small2", "small3"]
        self.trajectories = [{"seed_id": seed, "method": method, "status": "completed"}
                             for seed in self.seeds for method in ("gd", "pullback")]
        self.rows = []
        accuracy = {"gd": [0.80, 0.86, 0.91], "pullback": [0.84, 0.85, 0.94]}
        loss = {"gd": [0.20, 0.15, 0.10], "pullback": [0.17, 0.18, 0.08]}
        for endpoint in ("iteration", "time"):
            for method in ("gd", "pullback"):
                for index, seed in enumerate(self.seeds):
                    increment = 0 if endpoint == "iteration" else (0.02 if method == "gd" else 0.01)
                    self.rows.append({"seed_id": seed, "method": method, "endpoint": endpoint,
                                      "train_loss": 0.04, "validation_loss": 0.12,
                                      "test_loss": loss[method][index],
                                      "test_accuracy": accuracy[method][index] + increment,
                                      "step": 10 if endpoint == "iteration" else 15,
                                      "training_seconds": 0.5 if endpoint == "iteration" else 1.99})

    def test_summary_uses_seed_sample_sd_and_paired_pullback_minus_gd(self):
        result = formal.summarize(self.rows, self.trajectories, self.seeds)
        self.assertTrue(result["aggregate_complete"])
        gd_accuracy = result["methods"]["iteration"]["gd"]["test_accuracy"]
        self.assertAlmostEqual(gd_accuracy["mean"], np.mean([0.80, 0.86, 0.91]))
        self.assertAlmostEqual(gd_accuracy["sample_sd"], np.std([0.80, 0.86, 0.91], ddof=1))
        differences = result["paired_differences"]["iteration"]["test_accuracy"]
        self.assertEqual(differences["seed_order"], self.seeds)
        assert_allclose(differences["values"], [0.04, -0.01, 0.03], rtol=0, atol=1e-15)
        self.assertAlmostEqual(differences["mean"], 0.02)
        self.assertAlmostEqual(differences["sample_sd"], np.std([0.04, -0.01, 0.03], ddof=1))
        self.assertAlmostEqual(result["paired_differences"]["time"]["test_accuracy"]["mean"], 0.01)

    def test_failed_trajectory_prevents_survivor_only_aggregate(self):
        trajectories = copy.deepcopy(self.trajectories)
        trajectories[-1]["status"] = "failed"
        rows = [row for row in self.rows
                if not (row["seed_id"] == "small3" and row["method"] == "pullback")]
        result = formal.summarize(rows, trajectories, self.seeds)
        self.assertFalse(result["aggregate_complete"])
        self.assertEqual(result["methods"], {})
        self.assertEqual(result["paired_differences"], {})
        self.assertIn("failed", result["reason"])


class FormalFailurePipelineTests(unittest.TestCase):
    def test_failed_batch_preserves_all_scheduled_runs_without_generating_test_data(self):
        config = {
            "stage": "formal_pending", "dtype": "float64", "threads": 1,
            "termination_rule": "continue_until_both_budgets_reached",
            "time_endpoint_rule": "last_completed_update_at_or_before_budget",
            "iteration_endpoint": 2, "time_endpoint_seconds": 0.01,
            "log_every": 1, "warmup_steps": 0, "noise": 0.1,
            "train_size": 4, "validation_size": 4, "test_size": 6,
            "methods": {"gd": {"learning_rate": 0.125},
                        "pullback": {"learning_rate": 8.0, "damping": 0.1}},
            "seeds": [
                {"id": "tiny1", "train_seed": 41, "validation_seed": 42,
                 "test_seed": 43, "initialization_seed": 71},
                {"id": "tiny2", "train_seed": 51, "validation_seed": 52,
                 "test_seed": 53, "initialization_seed": 72},
                {"id": "tiny3", "train_seed": 61, "validation_seed": 62,
                 "test_seed": 63, "initialization_seed": 73},
            ],
        }
        generated_seeds = []

        def recording_generator(n, noise, seed):
            generated_seeds.append(seed)
            return two_moons(n, noise, seed)

        def failed_trajectory(method, *args, **kwargs):
            return {"method": method, "status": "failed", "reason": "Forced pipeline failure.",
                    "total_steps": 0, "training_seconds": 0.0, "diagnostics_seconds": 0.0,
                    "warmup_seconds": 0.0, "loop_wall_seconds": 0.0,
                    "initial_parameter_sha256": "fixture", "train_data_sha256": "fixture",
                    "history": [], "checkpoints": [],
                    "endpoints": {"iteration": None, "time": None, "time_crossing": None}}

        with tempfile.TemporaryDirectory() as temp_directory:
            output = Path(temp_directory) / "failed-formal"
            with patch.object(formal, "two_moons", side_effect=recording_generator), \
                    patch.object(formal, "train_trajectory", side_effect=failed_trajectory), \
                    patch.object(formal, "render_report"), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    formal.run(config, output, "synthetic-test-commit")
            self.assertEqual(generated_seeds, [41, 51, 61])
            trajectories = json.loads((output / "trajectories.json").read_text(encoding="utf-8"))
            self.assertEqual(len(trajectories), 6)
            self.assertTrue(all(row["status"] == "failed" for row in trajectories))
            environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
            self.assertFalse(environment["test_generated"])
            self.assertFalse(environment["test_evaluated"])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["aggregate_complete"])
            self.assertEqual(summary.get("methods", {}), {})
            self.assertEqual(summary["scheduled_trajectories"], 6)
            self.assertEqual(summary["completed_trajectories"], 0)


if __name__ == "__main__":
    unittest.main()
