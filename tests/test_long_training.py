"""Bounded-training and continuation-policy checks using disposable tiny data."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from pullback_feature_learning import long_training
from pullback_feature_learning.core import direction, initialize_parameters, parameter_gradient


def small_config():
    seeds = [
        {"id": f"tiny{i}", "train_seed": 100 + 10 * i,
         "validation_seed": 101 + 10 * i, "initialization_seed": 102 + 10 * i}
        for i in range(1, 4)
    ]
    return {
        "stage": "long_training_diagnostic", "steps": 7, "threads": 1,
        "dtype": "float64", "noise": 0.15, "train_size": 8,
        "validation_size": 8, "development_seeds": seeds,
        "methods": {"gd": {"learning_rate": 0.125},
                    "pullback": {"learning_rate": 8.0, "damping": 0.1}},
        "log_every": 2, "checkpoint_steps": [0, 3, 7], "warmup_steps": 0,
        "schedule": [
            {"seed_id": seed["id"], "method": method}
            for i, seed in enumerate(seeds)
            for method in (("gd", "pullback") if i % 2 == 0 else ("pullback", "gd"))
        ],
        "continuation_review": {
            "start_step": 3, "end_step": 7,
            "minimum_relative_loss_decrease": 0.05,
            "minimum_validation_seed_improvements": 2,
            "automatic_extension": False,
        },
    }


class LongTrainingExecutionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(677)
        self.x = rng.normal(size=(8, 2))
        self.y = np.where(self.x[:, 0] >= 0, 1.0, -1.0)
        self.initial = initialize_parameters(733)
        self.config = small_config()

    def run_small(self, method="gd", config=None):
        return long_training.train_trajectory(
            method, config or self.config, self.x, self.y, self.initial
        )

    def replay_gd(self, steps):
        theta = self.initial.copy()
        for _ in range(steps):
            theta -= self.config["methods"]["gd"]["learning_rate"] * parameter_gradient(
                theta, self.x, self.y
            )
        return theta

    def test_stops_at_exact_limit_and_preserves_nonmultiple_checkpoints(self):
        result = self.run_small()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_steps"], 7)
        saved = {point["step"]: point for point in result["checkpoints"]}
        self.assertEqual(set(saved), {0, 2, 3, 4, 6, 7})
        for step, point in saved.items():
            assert_allclose(point["theta"], self.replay_gd(step), rtol=0, atol=0)
        self.assertEqual(saved[0]["cumulative_feature_displacement"], 0)
        distances = [saved[step]["cumulative_feature_displacement"] for step in sorted(saved)]
        self.assertTrue(np.all(np.diff(distances) >= 0))

    def test_warmup_is_discarded_and_input_arrays_remain_unchanged(self):
        original = [array.copy() for array in (self.x, self.y, self.initial)]
        warm_config = copy.deepcopy(self.config)
        warm_config["warmup_steps"] = 3
        for method in ("gd", "pullback"):
            with self.subTest(method=method):
                cold = self.run_small(method)
                warm = self.run_small(method, warm_config)
                self.assertEqual(cold["status"], "completed")
                self.assertEqual(warm["status"], "completed")
                self.assertEqual(cold["total_steps"], warm["total_steps"])
                for a, b in zip(cold["checkpoints"], warm["checkpoints"]):
                    self.assertEqual(a["step"], b["step"])
                    assert_allclose(a["theta"], b["theta"], rtol=0, atol=0)
                    self.assertEqual(a["cumulative_feature_displacement"],
                                     b["cumulative_feature_displacement"])
        for actual, expected in zip((self.x, self.y, self.initial), original):
            assert_allclose(actual, expected, rtol=0, atol=0)

    def test_failed_solve_preserves_last_successful_unscheduled_state(self):
        config = copy.deepcopy(self.config)
        config["log_every"] = 10
        calls = 0

        def fail_third(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise np.linalg.LinAlgError("Forced bounded-run solve failure.")
            return direction(*args, **kwargs)

        with patch.object(long_training, "direction", side_effect=fail_third):
            result = self.run_small(config=config)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["total_steps"], 2)
        self.assertIn("Forced bounded-run solve failure", result["reason"])
        prefix = result["checkpoints"][-1]
        self.assertEqual(prefix["step"], 2)
        assert_allclose(prefix["theta"], self.replay_gd(2), rtol=0, atol=0)

    def test_warmup_failure_does_not_claim_training_progress(self):
        config = copy.deepcopy(self.config)
        config["warmup_steps"] = 3
        with patch.object(long_training, "direction",
                          side_effect=FloatingPointError("Forced warmup failure.")):
            result = self.run_small(config=config)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["total_steps"], 0)
        self.assertEqual(result["training_seconds"], 0)
        self.assertEqual(result["checkpoints"][-1]["step"], 0)
        assert_allclose(result["checkpoints"][-1]["theta"], self.initial, rtol=0, atol=0)

    def test_nonfinite_proposed_update_does_not_replace_valid_parameters(self):
        with patch.object(long_training, "direction",
                          return_value=(np.full_like(self.initial, np.nan), {})):
            result = self.run_small()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["total_steps"], 0)
        self.assertEqual(result["checkpoints"][-1]["step"], 0)
        assert_allclose(result["checkpoints"][-1]["theta"], self.initial, rtol=0, atol=0)


class LongTrainingConfigTests(unittest.TestCase):
    def test_development_only_configuration_is_valid(self):
        long_training.validate_config(small_config())

    def test_test_data_settings_are_rejected(self):
        for setting in ("size", "seed"):
            with self.subTest(setting=setting):
                config = small_config()
                if setting == "size":
                    config["test_size"] = 8
                else:
                    config["development_seeds"][0]["test_seed"] = 999
                with self.assertRaises(ValueError):
                    long_training.validate_config(config)

    def test_duplicate_or_missing_scheduled_run_is_rejected(self):
        for case in ("duplicate", "missing"):
            with self.subTest(case=case):
                config = small_config()
                if case == "duplicate":
                    config["schedule"][-1] = config["schedule"][0].copy()
                else:
                    config["schedule"].pop()
                with self.assertRaises(ValueError):
                    long_training.validate_config(config)

    def test_automatic_extension_is_rejected(self):
        config = small_config()
        config["continuation_review"]["automatic_extension"] = True
        with self.assertRaises(ValueError):
            long_training.validate_config(config)


class LongTrainingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.seeds = [seed["id"] for seed in self.config["development_seeds"]]
        self.trajectories = [
            {"seed_id": seed, "method": method, "status": "completed", "total_steps": 7}
            for seed in self.seeds for method in ("gd", "pullback")
        ]
        self.rows = []
        for method in ("gd", "pullback"):
            for index, seed in enumerate(self.seeds):
                for step in (0, 3, 7):
                    factor = 0.8 if step == 7 else 1.0
                    self.rows.append({
                        "method": method, "seed_id": seed, "step": step,
                        "train_loss": factor * (0.1 + 0.01 * index),
                        "validation_loss": factor * (0.2 + 0.01 * index),
                        "train_accuracy": 0.9, "validation_accuracy": 0.85,
                        "training_seconds": step * 0.01,
                        "cumulative_feature_displacement": step * 0.02,
                    })

    def summarize(self, rows=None, trajectories=None):
        return long_training.summarize(
            self.rows if rows is None else rows,
            self.trajectories if trajectories is None else trajectories,
            self.config,
        )

    def eligible(self, result, method="pullback"):
        return result["continuation_review"][method]["criterion_met"]

    def test_endpoint_summary_uses_mean_and_sample_standard_deviation(self):
        result = self.summarize()
        self.assertTrue(result["aggregate_complete"])
        values = result["endpoints"]["7"]["gd"]["validation_loss"]
        expected = np.array([0.2, 0.21, 0.22]) * 0.8
        self.assertAlmostEqual(values["mean"], np.mean(expected))
        self.assertAlmostEqual(values["sample_sd"], np.std(expected, ddof=1))
        self.assertTrue(self.eligible(result))

    def test_missing_or_failed_run_prevents_survivor_only_aggregate(self):
        for case in ("missing", "failed"):
            with self.subTest(case=case):
                trajectories = copy.deepcopy(self.trajectories)
                if case == "missing":
                    trajectories.pop()
                else:
                    trajectories[-1]["status"] = "failed"
                result = self.summarize(trajectories=trajectories)
                self.assertFalse(result["aggregate_complete"])
                self.assertFalse(result.get("endpoints"))

    def test_training_improvement_alone_cannot_justify_extension(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["step"] == 7:
                row["validation_loss"] /= 0.8
        result = self.summarize(rows=rows)
        self.assertFalse(self.eligible(result))
        self.assertFalse(self.eligible(result, "gd"))

    def test_validation_improvement_alone_cannot_justify_extension(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["step"] == 7:
                row["train_loss"] /= 0.8
        result = self.summarize(rows=rows)
        self.assertFalse(self.eligible(result))

    def test_one_improving_seed_cannot_pass_despite_better_mean_validation(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["step"] == 7:
                row["validation_loss"] = {
                    "tiny1": 0.01, "tiny2": 0.22, "tiny3": 0.23
                }[row["seed_id"]]
        result = self.summarize(rows=rows)
        self.assertLess(result["endpoints"]["7"]["pullback"]["validation_loss"]["mean"],
                        result["endpoints"]["3"]["pullback"]["validation_loss"]["mean"] * 0.95)
        self.assertFalse(self.eligible(result))


if __name__ == "__main__":
    unittest.main()
