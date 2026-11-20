"""Independent convex-probe correctness and split-isolation checks on disposable data."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pullback_feature_learning import probes
from pullback_feature_learning.core import initialize_parameters


SOLVER = {"max_iterations": 100, "gradient_tolerance": 1e-9, "armijo": 1e-4,
          "max_backtracking": 40, "scale_floor": 1e-12}


def tiny_config(source):
    return {"stage": "frozen_feature_probes", "source_directory": str(source), "dtype": "float64", "threads": 1,
            "seed_ids": ["tiny1", "tiny2", "tiny3"], "methods": ["gd", "pullback"],
            "checkpoints": [{"id": "initial", "step": 0}, {"id": "later", "step": 2}],
            "alphas": [0.01, 1.0], "solver": SOLVER.copy(),
            "selection": "minimum_validation_loss_tie_larger_alpha"}


def make_archive(source):
    source.mkdir()
    data, trajectories = {}, []
    for index in range(1, 4):
        sid = f"tiny{index}"
        rng = np.random.default_rng(index)
        x = rng.normal(size=(20, 2))
        y = np.where(x[:, 0] + .3 * x[:, 1] > 0, 1, -1)
        xv = rng.normal(size=(20, 2))
        yv = np.where(xv[:, 0] + .3 * xv[:, 1] > 0, 1, -1)
        # Distinctive test inputs allow a spy to detect premature representation evaluation.
        xt = np.array([[999., 999.], [-999., -999.]])
        theta = initialize_parameters(90 + index)
        data[sid] = {"train_x": x.tolist(), "train_y": y.tolist(), "validation_x": xv.tolist(),
                     "validation_y": yv.tolist(), "test_x": xt.tolist(), "test_y": [1, -1]}
        for method in ("gd", "pullback"):
            points = [{"step": step, "theta": theta.tolist(), "training_seconds": step * .01,
                       "cumulative_feature_displacement": 0.0, "labels": ["tiny"]} for step in (0, 2)]
            trajectories.append({"seed_id": sid, "method": method, "status": "completed", "checkpoints": points,
                                 "endpoints": {"time": points[-1]}})
    for filename, value in (("data.json", data), ("trajectories.json", trajectories), ("config.json", {"tiny": True})):
        (source / filename).write_text(json.dumps(value), encoding="utf-8")


class ProbeNumericalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.features = rng.normal(size=(40, 2)) * [2, .3] + [5, -4]
        self.labels = np.where(self.features[:, 0] - 5 + self.features[:, 1] + 4 > 0, 1, -1)

    def test_objective_gradient_and_hessian_match_independent_finite_differences(self):
        x = np.column_stack((self.features, np.ones(len(self.features))))
        beta = np.array([.17, -.13, .35])
        alpha = .23
        value, gradient, hessian = probes.objective_derivatives(beta, x, self.labels, alpha)
        step = 1e-5
        finite_gradient = np.zeros(3)
        finite_hessian = np.zeros((3, 3))
        for i in range(3):
            e = np.eye(3)[i] * step
            a, ga, _ = probes.objective_derivatives(beta + e, x, self.labels, alpha)
            b, gb, _ = probes.objective_derivatives(beta - e, x, self.labels, alpha)
            finite_gradient[i] = (a - b) / (2 * step)
            finite_hessian[:, i] = (ga - gb) / (2 * step)
        assert_allclose(gradient, finite_gradient, rtol=1e-8, atol=1e-8)
        assert_allclose(hessian, finite_hessian, rtol=1e-8, atol=1e-8)
        self.assertGreater(value, 0)
        self.assertTrue(np.all(np.linalg.eigvalsh(hessian) > 0))

    def test_intercept_has_no_regularization_penalty(self):
        design = np.column_stack((self.features, np.ones(len(self.features))))
        beta = np.array([.2, -.3, 4.])
        a = probes.objective_derivatives(beta, design, self.labels, 0)
        b = probes.objective_derivatives(beta, design, self.labels, 2)
        self.assertAlmostEqual(b[0] - a[0], .2**2 + .3**2)
        assert_allclose(b[1] - a[1], [.4, -.6, 0])
        assert_allclose(b[2] - a[2], np.diag([2, 2, 0]))

    def test_fitted_probe_reaches_stationary_convex_solution(self):
        model = probes.fit_probe(self.features, self.labels, .1, SOLVER)
        self.assertEqual(model["status"], "completed")
        x = np.column_stack(((self.features - model["mean"]) / model["scale"], np.ones(len(self.features))))
        beta = np.r_[model["coef"], model["intercept"]]
        objective, gradient, hessian = probes.objective_derivatives(beta, x, self.labels, .1)
        self.assertLessEqual(np.max(np.abs(gradient)), SOLVER["gradient_tolerance"])
        self.assertTrue(np.all(np.linalg.eigvalsh(hessian) > 0))
        self.assertAlmostEqual(model["objective"], objective)
        for displacement in np.eye(3) * .01:
            for sign in (-1, 1):
                self.assertGreater(probes.objective_derivatives(beta + sign * displacement, x, self.labels, .1)[0], objective)

    def test_normalization_uses_training_data_and_evaluation_does_not_mutate_model(self):
        model = probes.fit_probe(self.features, self.labels, .1, SOLVER)
        assert_allclose(model["mean"], self.features.mean(axis=0), atol=0, rtol=0)
        assert_allclose(model["scale"], self.features.std(axis=0, ddof=0), atol=0, rtol=0)
        before = copy.deepcopy(model)
        probes.evaluate_probe(model, self.features + 100, self.labels)
        self.assertEqual(model, before)

    def test_constant_features_have_finite_scale_and_known_intercept_optimum(self):
        features = np.ones((20, 2)) * [1, -3]
        labels = np.array([1] * 15 + [-1] * 5)
        model = probes.fit_probe(features, labels, .2, SOLVER)
        self.assertEqual(model["status"], "completed")
        assert_allclose(model["coef"], 0, atol=1e-12)
        self.assertAlmostEqual(model["intercept"], np.log(3), places=8)
        assert_allclose(model["scale"], [1e-12, 1e-12])

    def test_repeated_fits_start_from_zero_and_return_identical_parameters(self):
        a = probes.fit_probe(self.features, self.labels, 1e-6, SOLVER)
        probes.fit_probe(self.features, self.labels, 10, SOLVER)
        b = probes.fit_probe(self.features, self.labels, 1e-6, SOLVER)
        for field in ("coef", "intercept", "mean", "scale", "objective", "iterations"):
            self.assertEqual(a[field], b[field])

    def test_iteration_limit_is_failure_not_an_eligible_candidate(self):
        model = probes.fit_probe(self.features, self.labels, 1e-6, {**SOLVER, "max_iterations": 1})
        self.assertEqual(model["status"], "failed")
        self.assertGreater(model["gradient_infinity_norm"], SOLVER["gradient_tolerance"])

    def test_single_class_training_data_are_rejected(self):
        with self.assertRaises(ValueError):
            probes.fit_probe(self.features, np.ones(len(self.features)), .1, SOLVER)


class ProbeSelectionTests(unittest.TestCase):
    def test_selection_uses_validation_loss_and_exact_tie_prefers_larger_alpha(self):
        candidates = [{"model": {"status": "completed", "alpha": alpha}, "validation_loss": loss, "test_loss": test}
                      for alpha, loss, test in ((.1, .3, 0.0), (1, .2, .9), (10, .2, 1.0))]
        self.assertEqual(probes.select_candidate(candidates)["model"]["alpha"], 10)
        candidates[-1]["validation_loss"] += 1e-14
        self.assertEqual(probes.select_candidate(candidates)["model"]["alpha"], 1)

    def test_failed_or_nonfinite_candidate_invalidates_the_entire_selection(self):
        candidates = [{"model": {"status": "completed", "alpha": .1}, "validation_loss": .1},
                      {"model": {"status": "failed", "alpha": 1}, "validation_loss": 100}]
        with self.assertRaises(ValueError):
            probes.select_candidate(candidates)
        candidates[1]["model"]["status"] = "completed"
        candidates[1]["validation_loss"] = float("nan")
        with self.assertRaises(ValueError):
            probes.select_candidate(candidates)


class ProbePipelineTests(unittest.TestCase):
    def test_all_selections_are_saved_before_first_test_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "source", directory / "output"
            make_archive(source)
            config = tiny_config(source)
            real_representation = probes.representation
            test_calls = []

            def spy(theta, inputs):
                if np.max(np.abs(inputs)) > 900:
                    selected = json.loads((output / "selections.json").read_text())
                    candidates = json.loads((output / "candidate_fits.json").read_text())
                    self.assertEqual(len(selected), 12)
                    self.assertEqual(len(candidates), 24)
                    test_calls.append(True)
                return real_representation(theta, inputs)

            with patch.object(probes, "representation", side_effect=spy), patch.object(probes, "render_report"), redirect_stdout(io.StringIO()):
                probes.run(config, output, "disposable-test-revision")
            self.assertEqual(len(test_calls), 12)
            selected = json.loads((output / "selections.json").read_text())
            for sid in config["seed_ids"]:
                initial = [row for row in selected if row["seed_id"] == sid and row["checkpoint"] == "initial"]
                for field in ("coef", "intercept", "alpha", "mean", "scale"):
                    self.assertEqual(initial[0]["model"][field], initial[1]["model"][field])
            environment = json.loads((output / "environment.json").read_text())
            self.assertLessEqual(environment["selection_finished_at_utc"], environment["test_evaluation_started_at_utc"])
            self.assertFalse(environment["network_updated"])
            self.assertTrue(environment["test_features_evaluated"])

    def test_any_candidate_failure_preserves_evidence_and_skips_all_test_features(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "source", directory / "output"
            make_archive(source)
            config = tiny_config(source)
            real_representation, real_fit = probes.representation, probes.fit_probe

            def guarded_representation(theta, inputs):
                self.assertLess(np.max(np.abs(inputs)), 900)
                return real_representation(theta, inputs)

            def failed_fit(features, labels, alpha, solver):
                model = real_fit(features, labels, alpha, solver)
                if alpha == .01:
                    model.update(status="failed", reason="Injected numerical failure")
                return model

            with patch.object(probes, "representation", side_effect=guarded_representation), patch.object(probes, "fit_probe", side_effect=failed_fit):
                with self.assertRaises(SystemExit):
                    probes.run(config, output, "disposable-test-revision")
            candidates = json.loads((output / "candidate_fits.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(candidates), 24)
            self.assertFalse(summary["aggregate_complete"])
            self.assertEqual(summary["failed_selections"], 12)
            self.assertFalse((output / "metrics.csv").exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                probes.run(tiny_config("unused"), output, "unused")

    def test_bad_training_features_are_recorded_and_block_all_test_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "source", directory / "output"
            make_archive(source)
            config = tiny_config(source)
            with patch.object(probes, "representation", return_value=np.full((20, 2), np.nan)):
                with self.assertRaises(SystemExit):
                    probes.run(config, output, "disposable-test-revision")
            candidates = json.loads((output / "candidate_fits.json").read_text())
            self.assertEqual(len(candidates), 24)
            self.assertTrue(all(c["model"]["status"] == "failed" for c in candidates))
            environment = json.loads((output / "environment.json").read_text())
            self.assertFalse(environment["test_features_evaluated"])

    def test_numerical_test_failure_preserves_other_metrics_without_full_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "source", directory / "output"
            make_archive(source)
            config = tiny_config(source)
            real_representation = probes.representation
            failed = []

            def one_failure(theta, inputs):
                if np.max(np.abs(inputs)) > 900 and not failed:
                    failed.append(True)
                    raise FloatingPointError("Injected test feature failure")
                return real_representation(theta, inputs)

            with patch.object(probes, "representation", side_effect=one_failure), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    probes.run(config, output, "disposable-test-revision")
            environment = json.loads((output / "environment.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            failures = json.loads((output / "failures.json").read_text())
            self.assertTrue(environment["test_features_evaluated"])
            self.assertFalse(environment["test_evaluation_complete"])
            self.assertFalse(summary["aggregate_complete"])
            self.assertEqual(len(failures), 1)
            self.assertEqual(len((output / "metrics.csv").read_text().splitlines()), 12)


class ProbeSummaryTests(unittest.TestCase):
    def metrics(self):
        rows = []
        for i, sid in enumerate(("tiny1", "tiny2", "tiny3")):
            for method in ("gd", "pullback"):
                for cid in ("initial", "later"):
                    value = .3 + .01 * i - (.1 if cid == "later" else 0) + (.02 if method == "pullback" else 0)
                    rows.append({"seed_id": sid, "method": method, "checkpoint": cid,
                                 **{k: value for k in ("probe_train_loss", "probe_validation_loss", "probe_test_loss",
                                                      "probe_test_accuracy", "fixed_test_loss", "fixed_test_accuracy",
                                                      "probe_minus_fixed_test_loss", "probe_minus_fixed_test_accuracy")}})
        return rows

    def test_summary_preserves_pairing_sample_sd_and_learned_initial_differences(self):
        result = probes.summarize(self.metrics(), tiny_config("unused"))
        self.assertTrue(result["aggregate_complete"])
        gd = result["checkpoints"]["initial"]["gd"]["probe_test_loss"]
        self.assertAlmostEqual(gd["mean"], .31)
        self.assertAlmostEqual(gd["sample_sd"], .01)
        paired = result["paired_differences"]["later"]["probe_test_loss"]
        assert_allclose(paired["values"], [.02, .02, .02], atol=1e-15)
        learned = result["learned_minus_initial"]["later"]["pullback"]["probe_test_loss"]
        assert_allclose(learned["values"], [-.1, -.1, -.1], atol=1e-15)

    def test_missing_or_duplicate_metrics_do_not_form_survivor_aggregate(self):
        rows = self.metrics()
        self.assertFalse(probes.summarize(rows[:-1], tiny_config("unused"))["aggregate_complete"])
        self.assertFalse(probes.summarize(rows + rows[:1], tiny_config("unused"))["aggregate_complete"])


if __name__ == "__main__":
    unittest.main()
