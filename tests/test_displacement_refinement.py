"""Whole-update crossings, exact replay agreement, and the pre-test gate."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from pullback_feature_learning.core import direction, initialize_parameters, representation
from pullback_feature_learning.displacement_refinement import evaluate_replays, replay_trajectory, run


def tiny_fixture(method="gd"):
    config = {"iteration_endpoint": 4, "methods": {"gd": {"learning_rate": 0.03},
                                                     "pullback": {"learning_rate": 2.0, "damping": 0.1}}}
    x = np.array([[-1.0, 0.2], [-0.5, 0.3], [0.5, -0.4], [1.0, -0.2]])
    y = np.array([-1, -1, 1, 1])
    initial = initialize_parameters(723)
    theta = initial.copy()
    previous = representation(theta, x)
    distance = 0.0
    points = [{"step": 0, "theta": theta.tolist(), "cumulative_feature_displacement": 0.0}]
    for step in range(1, 5):
        update, _ = direction(theta, x, y, method, config["methods"][method].get("damping", 0.01))
        theta = theta + config["methods"][method]["learning_rate"] * update
        features = representation(theta, x)
        distance += float(np.sqrt(np.mean(np.sum((features - previous)**2, axis=1))))
        points.append({"step": step, "theta": theta.tolist(), "cumulative_feature_displacement": distance})
        previous = features
    return config, x, y, initial, points


class WholeUpdateReplayTests(unittest.TestCase):
    def setUp(self):
        self.thread_limit = threadpool_limits(limits=1)
        self.thread_limit.__enter__()

    def tearDown(self):
        self.thread_limit.__exit__(None, None, None)

    def test_sparse_archive_replay_recovers_true_consecutive_crossing(self):
        for method in ("gd", "pullback"):
            with self.subTest(method=method):
                config, x, y, initial, points = tiny_fixture(method)
                target = (points[1]["cumulative_feature_displacement"] + points[2]["cumulative_feature_displacement"]) / 2
                result = replay_trajectory(method, config, x, y, initial, points[::2], {"between": target})
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["total_steps"], 4)
                self.assertEqual(result["verified_archived_states"], 3)
                lower = result["brackets"]["between"]["lower_checkpoint"]
                upper = result["brackets"]["between"]["upper_checkpoint"]
                self.assertEqual((lower["step"], upper["step"]), (1, 2))
                np.testing.assert_array_equal(lower["theta"], points[1]["theta"])
                np.testing.assert_array_equal(upper["theta"], points[2]["theta"])
                self.assertEqual(lower["cumulative_feature_displacement"], points[1]["cumulative_feature_displacement"])

    def test_zero_exact_and_terminal_targets_use_declared_boundary_convention(self):
        config, x, y, initial, points = tiny_fixture()
        targets = {"zero": 0.0, "exact": points[2]["cumulative_feature_displacement"],
                   "terminal": points[4]["cumulative_feature_displacement"]}
        result = replay_trajectory("gd", config, x, y, initial, points, targets)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["brackets"]["zero"]["lower_checkpoint"]["step"], 0)
        self.assertEqual(result["brackets"]["zero"]["upper_checkpoint"]["step"], 1)
        self.assertEqual(result["brackets"]["exact"]["lower_checkpoint"]["step"], 2)
        self.assertEqual(result["brackets"]["exact"]["upper_checkpoint"]["step"], 3)
        self.assertEqual(result["brackets"]["terminal"]["lower_checkpoint"]["step"], 4)
        self.assertIsNone(result["brackets"]["terminal"]["upper_checkpoint"])

    def test_parameter_disagreement_is_preserved_and_fails_replay(self):
        config, x, y, initial, points = tiny_fixture()
        original = copy.deepcopy(points)
        points[2]["theta"][0] += 1e-8
        result = replay_trajectory("gd", config, x, y, initial, points, {"zero": 0.0})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["total_steps"], 2)
        self.assertIn("step 2", result["reason"])
        self.assertFalse(result["verification"][-1]["passed"])
        np.testing.assert_array_equal(result["final_state"]["theta"], original[2]["theta"])

    def test_displacement_disagreement_fails_even_when_parameters_agree(self):
        config, x, y, initial, points = tiny_fixture()
        points[2]["cumulative_feature_displacement"] += 1e-12
        result = replay_trajectory("gd", config, x, y, initial, points, {"zero": 0.0})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["verification"][-1]["parameter_max_absolute_error"], 0.0)
        self.assertGreater(result["verification"][-1]["displacement_absolute_error"], 0.0)

    def test_failed_replay_cannot_reach_test_evaluation(self):
        config, x, y, initial, points = tiny_fixture()
        points[0]["theta"][0] += 1e-9
        result = replay_trajectory("gd", config, x, y, initial, points, {"zero": 0.0})
        with patch("pullback_feature_learning.displacement_refinement.evaluate_checkpoint") as evaluate:
            with self.assertRaises(ValueError):
                evaluate_replays([result], {})
            evaluate.assert_not_called()

    def test_incomplete_verification_cannot_reach_test_evaluation(self):
        record = {"status": "completed", "verified_archived_states": 1, "expected_archived_states": 2}
        with patch("pullback_feature_learning.displacement_refinement.evaluate_checkpoint") as evaluate:
            with self.assertRaises(ValueError):
                evaluate_replays([record], {})
            evaluate.assert_not_called()

    def test_invalid_or_out_of_range_targets_are_rejected(self):
        config, x, y, initial, points = tiny_fixture()
        for target in (-1.0, float("nan"), points[-1]["cumulative_feature_displacement"] + 1):
            with self.subTest(target=target), self.assertRaises(ValueError):
                replay_trajectory("gd", config, x, y, initial, points, {"bad": target})

    def test_offline_failure_preserves_phase_and_verified_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            source.mkdir()
            seeds = [f"tiny{i}" for i in range(3)]
            config = {"iteration_endpoint": 2000, "seeds": [{"id": s} for s in seeds]}
            data = {s: {"train_x": [[0.0, 0.0]], "train_y": [-1], "initial_theta": [0.0] * 80} for s in seeds}
            trajectories = [{"seed_id": s, "method": m, "status": "completed", "checkpoints": [],
                             "endpoints": {"iteration": {"cumulative_feature_displacement": 1.0}}}
                            for s in seeds for m in ("gd", "pullback")]
            for name, value in (("config", config), ("data", data), ("trajectories", trajectories)):
                (source / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            def fake_replay(method, *args):
                return {"method": method, "status": "completed", "verified_archived_states": 2,
                        "expected_archived_states": 2, "replay_wall_seconds": 0.01}
            with patch("pullback_feature_learning.displacement_refinement.replay_trajectory", side_effect=fake_replay), \
                    patch("pullback_feature_learning.displacement_refinement.evaluate_replays", side_effect=FloatingPointError("bad metric")), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run(source, output, "test-only")
            environment = json.loads((output / "environment.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            self.assertIn("test_evaluation_started_at_utc", environment)
            self.assertFalse(environment["test_evaluated"])
            self.assertIn("bad metric", environment["evaluation_failure"])
            self.assertFalse(summary["aggregate_complete"])
            self.assertEqual(len(json.loads((output / "replays.json").read_text())), 6)


if __name__ == "__main__":
    unittest.main()
