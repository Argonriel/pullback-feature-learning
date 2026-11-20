"""Saved-state displacement boundaries and local spectral diagnostics."""

import unittest

import numpy as np

from pullback_feature_learning.representation_analysis import select_displacement_bracket, summarize_spectrum


def points(distances=(0.0, 1.0, 3.0, 6.0)):
    return [{"step": i * 100, "cumulative_feature_displacement": d, "train_loss": 1 / (i + 1)}
            for i, d in enumerate(distances)]


class DisplacementBracketTests(unittest.TestCase):
    def test_zero_budget_keeps_initial_state_and_next_saved_checkpoint(self):
        result = select_displacement_bracket(points(), 0.0)
        self.assertEqual(result["lower_checkpoint"]["step"], 0)
        self.assertEqual(result["upper_checkpoint"]["step"], 100)
        self.assertTrue(result["exact_target"])

    def test_nonexact_target_preserves_both_gaps_without_interpolation(self):
        result = select_displacement_bracket(points(), 2.5)
        self.assertEqual(result["lower_checkpoint"]["step"], 100)
        self.assertEqual(result["upper_checkpoint"]["step"], 200)
        self.assertEqual(result["realized_displacement"], 1.0)
        self.assertEqual(result["unused_displacement"], 1.5)
        self.assertEqual(result["upper_displacement_gap"], 0.5)
        self.assertEqual(result["lower_checkpoint"]["train_loss"], 0.5)
        self.assertEqual(result["bracket_steps"], 100)
        self.assertFalse(result["exact_target"])

    def test_exact_target_uses_equal_checkpoint(self):
        result = select_displacement_bracket(points(), 3.0)
        self.assertEqual(result["lower_checkpoint"]["step"], 200)
        self.assertEqual(result["upper_checkpoint"]["step"], 300)
        self.assertEqual(result["unused_displacement"], 0.0)

    def test_terminal_equality_has_no_manufactured_upper_state(self):
        result = select_displacement_bracket(points(), 6.0)
        self.assertEqual(result["lower_checkpoint"]["step"], 300)
        self.assertIsNone(result["upper_checkpoint"])
        self.assertIsNone(result["upper_displacement_gap"])
        self.assertIsNone(result["bracket_steps"])

    def test_plateau_retains_last_equal_saved_state(self):
        result = select_displacement_bracket(points((0.0, 1.0, 1.0, 2.0)), 1.0)
        self.assertEqual(result["lower_checkpoint"]["step"], 200)
        self.assertEqual(result["upper_checkpoint"]["step"], 300)

    def test_outside_observed_range_and_nonfinite_targets_are_rejected(self):
        for target in (-1.0, 6.1, float("nan"), float("inf")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                select_displacement_bracket(points(), target)

    def test_invalid_saved_trajectories_are_rejected(self):
        malformed = [[], points((0.0, 2.0, 1.0)), points((0.0, float("nan"), 3.0)), list(reversed(points()))]
        duplicate = points()
        duplicate[1]["step"] = 0
        malformed.append(duplicate)
        for rows in malformed:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                select_displacement_bracket(rows, 1.0)


class LocalSpectrumTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"singular_value": 2.0, "gradient_alignment_fraction": 0.25, "pullback_filter": 4 / 4.1},
                     {"singular_value": 1.0, "gradient_alignment_fraction": 0.5, "pullback_filter": 1 / 1.1}]
        self.metadata = {"rank_tolerance": 1e-12, "numerical_rank": 2, "unreachable_gradient_fraction": 0.25}
        self.rates = {"gd": 0.125, "pullback": 8.0}

    def test_local_diagnostics_match_explicit_orthogonal_feature_velocity(self):
        result = summarize_spectrum(self.rows, self.metadata, 2.0, 4, self.rates)
        g = 2.0 * np.sqrt([0.25, 0.5, 0.25])
        for method, q in (("gd", [4.0, 1.0, 0.0]), ("pullback", [4 / 4.1, 1 / 1.1, 0.0])):
            velocity = -np.array(q) * g
            expected_cosine = np.dot(-g, velocity) / (np.linalg.norm(g) * np.linalg.norm(velocity))
            self.assertAlmostEqual(result[f"{method}_velocity_cosine_with_negative_gradient"], expected_cosine)
            self.assertAlmostEqual(result[f"{method}_predicted_step_rms_motion"], self.rates[method] * np.linalg.norm(velocity) / 2)
        self.assertEqual(result["active_condition_number"], 2.0)
        self.assertEqual(result["reachable_gradient_fraction"], 0.75)

    def test_zero_gradient_has_undefined_cosine_and_zero_motion(self):
        rows = [{**r, "gradient_alignment_fraction": 0.0} for r in self.rows]
        result = summarize_spectrum(rows, {**self.metadata, "unreachable_gradient_fraction": 0.0}, 0.0, 4, self.rates)
        for method in self.rates:
            self.assertIsNone(result[f"{method}_velocity_cosine_with_negative_gradient"])
            self.assertEqual(result[f"{method}_predicted_step_rms_motion"], 0.0)

    def test_numerical_rank_disagreement_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_spectrum(self.rows, {**self.metadata, "numerical_rank": 1}, 2.0, 4, self.rates)


if __name__ == "__main__":
    unittest.main()
