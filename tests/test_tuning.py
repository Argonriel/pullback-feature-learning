"""Selection-policy checks using fabricated validation results only.

These tests never generate data or train a model. They protect the declared
candidate budget, complete-seed requirement, and deterministic selection rule.
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

from pullback_feature_learning.tuning import aggregate_and_select, candidate_grid
from pullback_feature_learning import tuning
from pullback_feature_learning.core import initialize_parameters
from pullback_feature_learning.data import two_moons


class TuningPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "gd_learning_rates": [0.125, 0.5],
            "pullback_learning_rates": [8.0, 32.0],
            "pullback_dampings": [0.01],
            "steps": 2000,
            "development_seeds": [
                {"id": "dev1", "train_seed": 1701, "validation_seed": 1702,
                 "initialization_seed": 23},
                {"id": "dev2", "train_seed": 1801, "validation_seed": 1802,
                 "initialization_seed": 24},
                {"id": "dev3", "train_seed": 1901, "validation_seed": 1902,
                 "initialization_seed": 25},
            ],
            "selection": "mean_terminal_validation_loss",
            "threads": 1,
            "dtype": "float64",
            "noise": 0.15,
            "train_size": 256,
            "validation_size": 256,
            "log_every": 100,
            "warmup_steps": 5,
        }
        self.seeds = ["dev1", "dev2", "dev3"]
        self.candidates = [
            {"id": "gd_a", "method": "gd", "learning_rate": 0.125, "damping": None},
            {"id": "gd_b", "method": "gd", "learning_rate": 0.5, "damping": None},
            {"id": "pullback_a", "method": "pullback", "learning_rate": 8.0, "damping": 0.01},
            {"id": "pullback_b", "method": "pullback", "learning_rate": 32.0, "damping": 0.01},
        ]

    def result(self, candidate_id, seed_id, loss=0.3, accuracy=0.9, status="completed"):
        return {"candidate_id": candidate_id, "seed_id": seed_id,
                "status": status, "validation_loss": loss,
                "validation_accuracy": accuracy, "training_seconds": 0.01,
                "diagnostics_seconds": 0.002}

    def complete_results(self):
        return [self.result(candidate["id"], seed)
                for candidate in self.candidates for seed in self.seeds]

    def test_grid_is_deterministic_and_has_equal_candidate_budgets(self):
        original = copy.deepcopy(self.config)
        grid = candidate_grid(self.config)
        self.assertEqual(grid, candidate_grid(self.config))
        self.assertEqual(self.config, original)
        self.assertEqual(len(grid), 4)
        self.assertEqual(sum(row["method"] == "gd" for row in grid), 2)
        self.assertEqual(sum(row["method"] == "pullback" for row in grid), 2)
        self.assertEqual(len({row["id"] for row in grid}), len(grid))
        self.assertEqual(
            {(row["learning_rate"], row["damping"]) for row in grid
             if row["method"] == "pullback"},
            {(8.0, 0.01), (32.0, 0.01)},
        )

    def test_unequal_candidate_counts_are_rejected(self):
        config = copy.deepcopy(self.config)
        config["gd_learning_rates"].append(1.0)
        with self.assertRaises(ValueError):
            candidate_grid(config)

    def test_learning_rates_and_damping_must_be_finite_and_positive(self):
        for key in ("gd_learning_rates", "pullback_learning_rates", "pullback_dampings"):
            for invalid in (0.0, -1.0, np.nan, np.inf):
                with self.subTest(key=key, invalid=invalid):
                    config = copy.deepcopy(self.config)
                    config[key][0] = invalid
                    with self.assertRaises(ValueError):
                        candidate_grid(config)

    def test_selection_uses_mean_terminal_loss_not_best_seed(self):
        results = self.complete_results()
        for row in results:
            if row["candidate_id"] == "gd_a":
                row["validation_loss"] = {"dev1": 0.01, "dev2": 0.5, "dev3": 0.5}[row["seed_id"]]
            elif row["candidate_id"] == "gd_b":
                row["validation_loss"] = 0.2
        output = aggregate_and_select(self.candidates, results, self.seeds)
        self.assertEqual(output["selected"]["gd"]["id"], "gd_b")
        self.assertAlmostEqual(output["selected"]["gd"]["mean_validation_loss"], 0.2)
        candidate_a = next(row for row in output["rankings"]["gd"] if row["id"] == "gd_a")
        self.assertAlmostEqual(candidate_a["mean_validation_loss"], 1.01 / 3)
        self.assertTrue(output["selected"]["gd"]["eligible"])
        self.assertEqual(output["selected"]["gd"]["completed_seeds"], 3)

    def test_missing_seed_cannot_win_on_good_partial_results(self):
        results = [row for row in self.complete_results()
                   if not (row["candidate_id"] == "gd_a" and row["seed_id"] == "dev3")]
        for row in results:
            if row["candidate_id"] == "gd_a":
                row["validation_loss"] = 0.001
        output = aggregate_and_select(self.candidates, results, self.seeds)
        self.assertEqual(output["selected"]["gd"]["id"], "gd_b")
        incomplete = next(row for row in output["rankings"]["gd"] if row["id"] == "gd_a")
        self.assertFalse(incomplete["eligible"])

    def test_failed_seed_disqualifies_candidate_without_replacement(self):
        results = self.complete_results()
        for row in results:
            if row["candidate_id"] == "gd_a":
                row["validation_loss"] = 0.001
                if row["seed_id"] == "dev3":
                    row.update(status="failed", validation_loss=None, validation_accuracy=None)
        output = aggregate_and_select(self.candidates, results, self.seeds)
        self.assertEqual(output["selected"]["gd"]["id"], "gd_b")
        self.assertEqual(len(output["rankings"]["gd"]), 2)

    def test_nonfinite_completed_metrics_cannot_win(self):
        for key in ("validation_loss", "validation_accuracy"):
            for invalid in (np.nan, np.inf):
                with self.subTest(key=key, invalid=invalid):
                    results = self.complete_results()
                    for row in results:
                        if row["candidate_id"] == "gd_a":
                            row["validation_loss"] = 0.001
                            if row["seed_id"] == "dev3":
                                row[key] = invalid
                    output = aggregate_and_select(self.candidates, results, self.seeds)
                    self.assertEqual(output["selected"]["gd"]["id"], "gd_b")

    def test_no_complete_candidate_for_a_method_is_an_error(self):
        results = [row for row in self.complete_results()
                   if not (row["candidate_id"].startswith("gd_") and row["seed_id"] == "dev3")]
        with self.assertRaises(ValueError):
            aggregate_and_select(self.candidates, results, self.seeds)

    def test_duplicate_candidate_seed_results_are_rejected(self):
        results = self.complete_results()
        results.append(copy.deepcopy(results[0]))
        with self.assertRaises(ValueError):
            aggregate_and_select(self.candidates, results, self.seeds)

    def test_ties_use_validation_accuracy_then_candidate_id(self):
        results = self.complete_results()
        for row in results:
            if row["candidate_id"] == "gd_b":
                row["validation_accuracy"] = 0.95
        output = aggregate_and_select(self.candidates, results, self.seeds)
        self.assertEqual(output["selected"]["gd"]["id"], "gd_b")
        self.assertEqual(output["selected"]["pullback"]["id"], "pullback_a")

    def test_result_order_does_not_change_selection(self):
        results = self.complete_results()
        baseline = aggregate_and_select(self.candidates, results, self.seeds)
        permuted = aggregate_and_select(list(reversed(self.candidates)), list(reversed(results)), self.seeds)
        self.assertEqual(baseline["selected"], permuted["selected"])
        self.assertEqual(baseline["rankings"], permuted["rankings"])


class TuningExecutionTests(unittest.TestCase):
    """Exercise tiny, disposable runs; the real tuning grid is never used."""

    def setUp(self):
        self.config = {
            "stage": "tuning", "steps": 3, "threads": 1, "dtype": "float64",
            "noise": 0.15, "train_size": 8, "validation_size": 8,
            "development_seeds": [
                {"id": "dev1", "train_seed": 41, "validation_seed": 42,
                 "initialization_seed": 43},
                {"id": "dev2", "train_seed": 51, "validation_seed": 52,
                 "initialization_seed": 53},
            ],
            "gd_learning_rates": [0.125], "pullback_learning_rates": [2.0],
            "pullback_dampings": [0.01],
            "selection": "mean_terminal_validation_loss",
            "schedule_seed": 56, "log_every": 1, "warmup_steps": 2,
            "formal": {
                "stage": "formal_pending", "test_size": 12,
                "seeds": [{"id": "formal1", "train_seed": 61,
                           "validation_seed": 62, "test_seed": 63,
                           "initialization_seed": 64}],
            },
        }
        self.seed = self.config["development_seeds"][0]
        self.candidate = candidate_grid(self.config)[0]
        x, y = two_moons(8, 0.15, 41)
        xv, yv = two_moons(8, 0.15, 42)
        self.arrays = (x, y, xv, yv, initialize_parameters(43))

    def test_warmup_does_not_change_training_trajectory(self):
        original = tuple(array.copy() for array in self.arrays)
        without_warmup = {**self.config, "warmup_steps": 0}
        result_a, history_a, final_a = tuning.trial(
            self.candidate, self.seed, without_warmup, self.arrays)
        result_b, history_b, final_b = tuning.trial(
            self.candidate, self.seed, self.config, self.arrays)
        self.assertEqual(result_a["status"], "completed")
        self.assertEqual(result_b["status"], "completed")
        np.testing.assert_array_equal(final_a["theta"], final_b["theta"])
        for row_a, row_b in zip(history_a, history_b):
            for key in ("train_loss", "validation_loss", "validation_accuracy",
                        "cumulative_feature_displacement"):
                self.assertEqual(row_a[key], row_b[key])
        for actual, expected in zip(self.arrays, original):
            np.testing.assert_array_equal(actual, expected)

    def test_warmup_failure_is_recorded_without_training_updates(self):
        with patch.object(tuning, "direction", side_effect=FloatingPointError("fixture failure")):
            result, history, final = tuning.trial(
                self.candidate, self.seed, self.config, self.arrays)
        self.assertEqual(result["status"], "failed")
        self.assertIn("warmup", result["failure_reason"])
        self.assertEqual(result["completed_updates"], 0)
        self.assertEqual(result["training_seconds"], 0)
        self.assertGreaterEqual(result["warmup_seconds"], 0)
        self.assertIsNone(result["validation_loss"])
        self.assertEqual(history, [])
        np.testing.assert_array_equal(final["theta"], self.arrays[-1])

    def test_training_failure_preserves_prefix_and_disqualifies_terminal_score(self):
        original_direction = tuning.direction
        calls = 0

        def fail_second_training_update(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == self.config["warmup_steps"] + 2:
                raise np.linalg.LinAlgError("fixture solve failure")
            return original_direction(*args, **kwargs)

        with patch.object(tuning, "direction", side_effect=fail_second_training_update):
            result, history, final = tuning.trial(
                self.candidate, self.seed, self.config, self.arrays)
        self.assertEqual(result["status"], "failed")
        self.assertIn("training", result["failure_reason"])
        self.assertEqual(result["completed_updates"], 1)
        self.assertEqual([row["step"] for row in history], [0, 1])
        self.assertIsNone(result["validation_loss"])
        self.assertEqual(final["completed_updates"], 1)
        self.assertTrue(np.isfinite(final["theta"]).all())

    def test_run_uses_only_development_data_and_pairs_initial_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with (patch.object(tuning, "two_moons", wraps=two_moons) as generator,
                  patch.object(tuning, "render_report"), redirect_stdout(io.StringIO())):
                tuning.run(self.config, output, "unit-test-fixture")
            self.assertEqual([call.args[2] for call in generator.call_args_list], [41, 42, 51, 52])
            environment = json.loads((output / "environment.json").read_text())
            self.assertFalse(environment["test_generated"])
            self.assertFalse(environment["test_evaluated"])
            self.assertFalse(environment["formal_data_generated"])
            results = json.loads((output / "results.json").read_text())
            self.assertEqual(len(results), 4)
            self.assertTrue(all(row["status"] == "completed" for row in results))
            for seed_id in ("dev1", "dev2"):
                paired = [row for row in results if row["seed_id"] == seed_id]
                for key in ("initial_parameter_sha256", "train_data_sha256", "validation_data_sha256"):
                    self.assertEqual(len({row[key] for row in paired}), 1)
            self.assertTrue((output / "selection.json").is_file())
            self.assertTrue((output / "formal.json").is_file())
            with self.assertRaises(FileExistsError):
                tuning.run(self.config, output, "unit-test-fixture")

    def test_all_warmup_failures_leave_evidence_and_no_selected_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with (patch.object(tuning, "direction", side_effect=FloatingPointError("fixture failure")),
                  patch.object(tuning, "render_report"), redirect_stdout(io.StringIO())):
                with self.assertRaises(ValueError):
                    tuning.run(self.config, output, "unit-test-fixture")
            results = json.loads((output / "results.json").read_text())
            self.assertEqual(len(results), 4)
            self.assertTrue(all(row["status"] == "failed" for row in results))
            self.assertEqual(len((output / "trial_log.jsonl").read_text().splitlines()), 4)
            self.assertTrue((output / "terminal_checkpoints.json").is_file())
            self.assertFalse((output / "selection.json").exists())
            self.assertFalse((output / "formal.json").exists())


if __name__ == "__main__":
    unittest.main()
