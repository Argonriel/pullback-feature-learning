"""Independent numerical checks for the representation-space update.

Run with ``python -m unittest discover -s tests -v`` after installing the package.
These checks use small deterministic problems, not the pilot or test split.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from pullback_feature_learning.core import (
    direction,
    initialize_parameters,
    logistic_loss,
    parameter_gradient,
    representation,
    representation_gradient,
    representation_jacobian,
)


class NumericalCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.theta = initialize_parameters(23)
        rng = np.random.default_rng(127)
        self.x = rng.normal(size=(9, 2))
        self.y = np.array([-1, 1, 1, -1, 1, -1, -1, 1, 1], dtype=float)

    def test_shapes_and_reproducible_initialization(self):
        self.assertEqual(self.theta.shape, (80,))
        self.assertEqual(representation(self.theta, self.x).shape, (9, 2))
        self.assertEqual(representation_jacobian(self.theta, self.x).shape, (18, 80))
        assert_allclose(self.theta, initialize_parameters(23), rtol=0, atol=0)
        self.assertFalse(np.array_equal(self.theta, initialize_parameters(24)))

    def test_full_jacobian_against_central_differences(self):
        """Check every parameter and both coordinates independently of backprop."""
        h = 1e-6
        numerical = np.empty((18, 80))
        for column in range(80):
            perturbation = np.zeros(80)
            perturbation[column] = h
            numerical[:, column] = (
                representation(self.theta + perturbation, self.x)
                - representation(self.theta - perturbation, self.x)
            ).reshape(-1) / (2 * h)
        assert_allclose(
            representation_jacobian(self.theta, self.x), numerical,
            rtol=3e-6, atol=2e-9,
        )

    def test_chain_rule_matches_independent_reverse_mode(self):
        z = representation(self.theta, self.x)
        jacobian = representation_jacobian(self.theta, self.x)
        expected = jacobian.T @ representation_gradient(z, self.y).reshape(-1)
        assert_allclose(
            parameter_gradient(self.theta, self.x, self.y), expected,
            rtol=2e-12, atol=2e-13,
        )

    def test_parameter_gradient_against_loss_directional_differences(self):
        gradient = parameter_gradient(self.theta, self.x, self.y)
        rng = np.random.default_rng(310)
        for _ in range(5):
            vector = rng.normal(size=80)
            vector /= np.linalg.norm(vector)
            h = 1e-5
            plus = logistic_loss(representation(self.theta + h * vector, self.x), self.y)
            minus = logistic_loss(representation(self.theta - h * vector, self.x), self.y)
            assert_allclose((plus - minus) / (2 * h), gradient @ vector,
                            rtol=3e-6, atol=2e-10)

    def test_stable_logistic_loss_and_representation_gradient(self):
        z = np.array([[1000.0, 4.0], [-1000.0, -8.0], [0.3, 12.0]])
        y = np.array([-1.0, 1.0, -1.0])
        expected_loss = np.logaddexp(0.0, -y * z[:, 0]).mean()
        self.assertTrue(np.isfinite(logistic_loss(z, y)))
        assert_allclose(logistic_loss(z, y), expected_loss, rtol=1e-14)
        gradient = representation_gradient(z, y)
        self.assertTrue(np.isfinite(gradient).all())
        assert_allclose(gradient[:, 1], 0.0, atol=0, rtol=0)
        h = 1e-4
        for row in range(3):
            perturbation = np.zeros_like(z)
            perturbation[row, 0] = h
            numerical = (logistic_loss(z + perturbation, y)
                         - logistic_loss(z - perturbation, y)) / (2 * h)
            assert_allclose(numerical, gradient[row, 0], rtol=2e-7, atol=2e-9)

    def test_pullback_satisfies_damped_normal_equation(self):
        damping = 0.03
        jacobian = representation_jacobian(self.theta, self.x)
        g = representation_gradient(representation(self.theta, self.x), self.y).reshape(-1)
        update, _ = direction(self.theta, self.x, self.y,
                              method="pullback", damping=damping)
        system = jacobian.T @ jacobian + damping * np.eye(80)
        residual = system @ update + jacobian.T @ g
        self.assertLess(np.linalg.norm(residual) / np.linalg.norm(jacobian.T @ g), 1e-11)

    def test_svd_predicts_both_representation_velocities(self):
        jacobian = representation_jacobian(self.theta, self.x)
        g = representation_gradient(representation(self.theta, self.x), self.y).reshape(-1)
        left, singular_values, _ = np.linalg.svd(jacobian, full_matrices=False)
        damping = 0.03
        for method in ("gd", "pullback"):
            with self.subTest(method=method):
                update, _ = direction(self.theta, self.x, self.y,
                                      method=method, damping=damping)
                factors = singular_values**2
                if method == "pullback":
                    factors = factors / (factors + damping)
                expected = -left @ (factors * (left.T @ g))
                assert_allclose(jacobian @ update, expected, rtol=2e-10, atol=2e-12)

    def test_small_updates_descend_for_both_methods(self):
        gradient = parameter_gradient(self.theta, self.x, self.y)
        initial_loss = logistic_loss(representation(self.theta, self.x), self.y)
        for method in ("gd", "pullback"):
            with self.subTest(method=method):
                update, _ = direction(self.theta, self.x, self.y,
                                      method=method, damping=0.03)
                self.assertLess(gradient @ update, -1e-12)
                step_size = 1e-4 / max(1.0, np.linalg.norm(update))
                new_loss = logistic_loss(
                    representation(self.theta + step_size * update, self.x), self.y
                )
                self.assertLess(new_loss, initial_loss)

    def test_first_order_feature_remainder_is_quadratic(self):
        vector = np.random.default_rng(81).normal(size=80)
        vector /= np.linalg.norm(vector)
        initial = representation(self.theta, self.x).reshape(-1)
        velocity = representation_jacobian(self.theta, self.x) @ vector
        errors = []
        for step_size in (0.02, 0.01, 0.005):
            actual = representation(self.theta + step_size * vector, self.x).reshape(-1)
            errors.append(np.linalg.norm(actual - initial - step_size * velocity))
        self.assertGreater(min(errors), 1e-12)
        for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
            self.assertGreater(ratio, 3.7)
            self.assertLess(ratio, 4.3)

    def test_pullback_metric_includes_unread_representation_coordinate(self):
        """A zero second-coordinate loss gradient does not remove its metric cost."""
        jacobian = representation_jacobian(self.theta, self.x)
        g = representation_gradient(representation(self.theta, self.x), self.y).reshape(-1)
        damping = 0.03
        actual, _ = direction(self.theta, self.x, self.y,
                              method="pullback", damping=damping)
        full = np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(80),
                               -jacobian.T @ g)
        readout_jacobian = jacobian[::2]
        readout_only = np.linalg.solve(
            readout_jacobian.T @ readout_jacobian + damping * np.eye(80),
            -readout_jacobian.T @ g[::2],
        )
        assert_allclose(actual, full, rtol=2e-11, atol=2e-12)
        self.assertGreater(np.linalg.norm(full - readout_only),
                           1e-5 * np.linalg.norm(full))


if __name__ == "__main__":
    unittest.main()
