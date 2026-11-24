"""Compare periodic metric refresh and optimizer switching on paired development data."""

import argparse
import csv
import json
import platform
import random
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from pullback_feature_learning import core
from pullback_feature_learning.data import two_moons


METHODS = ("gd", "pullback", "pb100_then_gd", "refresh10", "refresh50")
STEPS = (100, 500, 2000)
BUDGET_NS = 100_000_000
DAMPING = 0.1


def build_metric(theta, x):
    jac = core.representation_jacobian(theta, x)
    _, singular, right_t = np.linalg.svd(jac, full_matrices=False)
    return right_t.T, 1 / (singular**2 + DAMPING)


def apply_metric(metric, gradient):
    vectors, scales = metric
    return vectors @ (scales * (vectors.T @ gradient))


def step(method, completed, theta, x, y, metric):
    if method.startswith("refresh"):
        interval = int(method[7:])
        refresh = completed % interval == 0
        if refresh:
            metric = build_metric(theta, x)
        gradient = core.parameter_gradient(theta, x, y)
        return theta - 8 * apply_metric(metric, gradient), metric, int(refresh)
    use_pullback = method == "pullback" or (method == "pb100_then_gd" and completed < 100)
    name, rate = ("pullback", 8.0) if use_pullback else ("gd", 0.125)
    change, _ = core.direction(theta, x, y, name, DAMPING)
    return theta + rate * change, None, int(use_pullback)


def validation_row(pair, method, endpoint, theta, xv, yv, count, elapsed, refreshes, next_ns=None):
    z = core.representation(theta, xv)
    return dict(pair=pair, method=method, endpoint=endpoint, updates=count,
                core_seconds=elapsed * 1e-9,
                next_core_seconds=None if next_ns is None else next_ns * 1e-9,
                metric_builds=refreshes, val_loss=core.logistic_loss(z, yv),
                val_accuracy=float(np.mean(np.where(z[:, 0] >= 0, 1, -1) == yv)))


def train(pair, method):
    x, y = two_moons(256, 0.15, 61000 + pair)
    xv, yv = two_moons(256, 0.15, 63000 + pair)
    theta = core.initialize_parameters(62000 + pair)
    rows = []
    count = elapsed = builds = 0
    metric = None
    while count < max(STEPS) or elapsed <= BUDGET_NS:
        start = perf_counter_ns()
        try:
            proposed, metric, refreshed = step(method, count, theta, x, y, metric)
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            return rows, dict(pair=pair, method=method, failed_on_update=count + 1,
                              reason=str(error))
        next_elapsed = elapsed + (perf_counter_ns() - start)
        if elapsed <= BUDGET_NS < next_elapsed:
            rows.append(validation_row(pair, method, "time_100ms", theta, xv, yv, count,
                                       elapsed, builds, next_elapsed))
        if not np.isfinite(proposed).all():
            return rows, dict(pair=pair, method=method, failed_on_update=count + 1)
        theta = proposed
        count += 1
        elapsed = next_elapsed
        builds += refreshed
        if count in STEPS:
            rows.append(validation_row(pair, method, f"step_{count}", theta, xv, yv, count,
                                       elapsed, builds))
    return rows, None


def numerical_check():
    x, y = two_moons(256, 0.15, 61800)
    theta = core.initialize_parameters(62800)
    reference, _ = core.direction(theta, x, y, "pullback", DAMPING)
    candidate = -apply_metric(build_metric(theta, x), core.parameter_gradient(theta, x, y))
    difference = float(np.linalg.norm(reference - candidate) / np.linalg.norm(reference))
    if difference > 1e-9:
        raise ArithmeticError(f"Fresh metric did not match reference pullback: {difference}")
    for method in METHODS:
        current, metric = theta.copy(), None
        for index in range(6):
            current, metric, _ = step(method, index, current, x, y, metric)
    return difference


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    schedule = [(i, m) for i in range(1, 9) for m in METHODS]
    random.Random(20261120).shuffle(schedule)
    rows, failures = [], []
    with threadpool_limits(limits=1):
        agreement = numerical_check()
        info = threadpool_info()
        for pair, method in schedule:
            observed, failure = train(pair, method)
            rows.extend(observed)
            if failure:
                failures.append(failure)
    with (output / "paired_endpoints.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                    thread_pools=info, schedule=schedule, failures=failures,
                    fresh_metric_relative_error=agreement)
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{len(rows)} endpoints from {len(schedule)} runs, {len(failures)} failures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
