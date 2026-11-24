"""Independent algebraic and numerical checks for the initial-tangent gate."""
import json
import time

import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

import runner
from pullback_feature_learning import core


def verify():
    rng = np.random.default_rng(603991)
    theta0 = core.initialize_parameters(603992)
    x = rng.normal(size=(24, 2))
    y = np.where(rng.normal(size=len(x)) > 0, 1., -1.)
    j0 = core.representation_jacobian(theta0, x)
    z0 = core.representation(theta0, x)
    delta = rng.normal(size=80)
    zlinear = runner.predict(theta0, theta0 + delta, x, True)
    assert_allclose(zlinear.ravel() - z0.ravel(), j0 @ delta, atol=1e-14, rtol=2e-13)
    assert np.linalg.norm(zlinear[:, 1] - z0[:, 1]) > .01
    assert np.linalg.norm(zlinear - core.representation(theta0 + delta, x)) > .01

    lam = .1
    for suffix in ("gd", "pb"):
        left = runner.direction(theta0, x, y, z0, "tangent_" + suffix, lam, j0)[0]
        right = core.direction(theta0, x, y, "pullback" if suffix == "pb" else "gd", lam)[0]
        assert_allclose(left, right, atol=1e-13, rtol=1e-11)
    g = core.representation_gradient(z0, y).ravel()
    full = np.linalg.solve(j0.T @ j0 + lam * np.eye(80), -j0.T @ g)
    a = j0[::2]
    readout = np.linalg.solve(a.T @ a + lam * np.eye(80), -a.T @ g[::2])
    assert np.linalg.norm(full - readout) > 1e-4 * np.linalg.norm(full)

    # The whole nonlinear logistic trajectory equals fixed-kernel recurrence,
    # not merely its first-order approximation at the initial state.
    metric = j0.T @ j0 + lam * np.eye(80)
    kernel = a @ np.linalg.solve(metric, a.T)
    theta = theta0.copy()
    logits = z0[:, 0].copy()
    worst = 0.
    for _ in range(71):
        z = runner.predict(theta0, theta, x, True)
        d = runner.direction(theta, x, y, z, "tangent_pb", lam, j0)[0]
        theta += 2.7 * d
        gy = -y * np.exp(-np.logaddexp(0., y * logits)) / len(y)
        logits -= 2.7 * (kernel @ gy)
        err = float(np.max(np.abs(runner.predict(theta0, theta, x, True)[:, 0] - logits)))
        worst = max(worst, err)
    assert worst < 1e-11

    theta = theta0 + .08 * delta
    zn = core.representation(theta, x)
    fixed, jn, gradient = runner.direction(theta, x, y, zn, "nonlinear_fixed_pb", lam, j0)
    expected_gradient = core.parameter_gradient(theta, x, y)
    assert_allclose(gradient, expected_gradient, rtol=1e-11, atol=1e-13)
    assert_allclose(metric @ fixed, -expected_gradient, rtol=1e-11, atol=1e-13)
    current = runner.direction(theta, x, y, zn, "nonlinear_pb", lam, j0)[0]
    assert np.linalg.norm(fixed - current) > 1e-5

    # Duplicating the geometry data doubles the SUM metric. Duplicating labels
    # leaves the MEAN gradient unchanged. Doubling lambda and eta restores it.
    xx, yy = np.tile(x, (2, 1)), np.tile(y, 2)
    double = core.direction(theta0, xx, yy, "pullback", 2 * lam)[0]
    assert_allclose(2 * double, full, rtol=1e-11, atol=1e-13)

    cfg = dict(loss_fractions=[.95, .9], max_steps=800, param_step_cap=.1,
               remainder_rtol=.05, armijo=1e-4, target_atol=1e-10)
    smoke = []
    start = time.perf_counter()
    for task in ("moons", "rings"):
        xx, yy = runner.make_data(task, 603993, 256)
        for arm in runner.ARMS:
            snaps, trace, summary = runner.train_one(cfg, task, {"id": "smoke"}, arm, lam,
                                                     .01, xx, yy, theta0)
            assert summary["status"] == "complete", summary
            assert summary["max_parameter_step"] <= .1 + 1e-12
            for row, state in snaps:
                check = runner.predict(theta0, state, xx, arm.startswith("tangent"))
                assert_allclose(core.logistic_loss(check, yy), row["train_loss"], atol=2e-14, rtol=0)
                if row["status"] == "target":
                    assert abs(row["target_error"]) <= 1e-10
            smoke.append({k: summary[k] for k in ("task", "arm", "step", "elapsed_training_seconds")})
    return dict(status="passed", checks=["full_two_coordinate_affine_map", "initial_direction_equality",
              "unread_coordinate_metric_included", "71_step_fixed_kernel_trajectory", "nonlinear_fixed_metric_gradient",
              "sum_metric_duplicate_scaling", "all_five_arms_progress_controller_both_tasks"],
                fixed_kernel_max_error=worst, smoke=smoke, smoke_seconds=time.perf_counter()-start)


if __name__ == "__main__":
    with threadpool_limits(1):
        print(json.dumps(verify(), indent=2, allow_nan=False))
