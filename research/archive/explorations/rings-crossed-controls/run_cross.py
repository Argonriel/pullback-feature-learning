"""Collect matched fixed-step and elapsed-time endpoints from each trajectory."""

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from time import perf_counter_ns, get_clock_info

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from pullback_feature_learning import core
import generator


HERE = Path(__file__).resolve().parent
TIME_NS = 100_000_000
STEP_LIMIT = 500
CELLS = (("gd", "original"), ("gd", "selected"), ("hybrid", "original"), ("hybrid", "selected"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def balanced_schedule():
    rng = random.Random(22772026)
    schedule = []
    for repeat in range(1, 4):
        pairs = list(range(1, 17))
        rng.shuffle(pairs)
        letters = list(CELLS)
        rng.shuffle(letters)
        for block, pair in enumerate(pairs):
            # A four-treatment Williams design balances position and previous treatment.
            for shift in (0, 1, 3, 2):
                method, policy = letters[(block + shift) % 4]
                schedule.append((pair, repeat, method, policy))
    return schedule


def run_trajectory(pair, repeat, method, policy, rate, order):
    xt, yt = generator.rings(91000 + pair)
    xv, yv = generator.rings(93000 + pair)
    theta = core.initialize_parameters(92000 + pair)
    count = elapsed = 0
    snapshots = {}
    while len(snapshots) < 2:
        pulling = method == "hybrid" and count < 100
        optimizer, eta = ("pullback", 8.0) if pulling else ("gd", rate)
        begin = perf_counter_ns()
        direction, _ = core.direction(theta, xt, yt, method=optimizer, damping=0.1)
        candidate = theta + eta * direction
        duration = perf_counter_ns() - begin
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"Nonfinite parameter on update {count+1}")
        if elapsed <= TIME_NS < elapsed + duration:
            snapshots["time100ms"] = (theta, count, elapsed, elapsed + duration)
        elapsed += duration
        count += 1
        theta = candidate
        if count == STEP_LIMIT:
            snapshots["step500"] = (theta, count, elapsed, None)
    rows = []
    for endpoint, (state, updates, lower_ns, crossing_ns) in snapshots.items():
        z = core.representation(state, xv)
        loss = core.logistic_loss(z, yv)
        if not np.isfinite(z).all() or not np.isfinite(loss):
            raise FloatingPointError("Nonfinite validation result")
        rows.append(dict(pair=pair, repeat=repeat, method=method, policy=policy, gd_rate=rate,
                         endpoint=endpoint, updates=updates, core_ns=lower_ns, cross_ns=crossing_ns,
                         pullback_updates=min(updates, 100) if method == "hybrid" else 0,
                         validation_loss=loss, validation_accuracy=np.mean(np.where(z[:, 0] >= 0, 1, -1) == yv),
                         train_seed=91000 + pair, init_seed=92000 + pair, validation_seed=93000 + pair,
                         state_sha256=sha(state.tobytes()), run_order=order))
    return rows


def main(out):
    selected = json.loads((HERE / "prior_selection.json").read_text())["rates"]
    assert selected == {"gd": 0.5, "warmup": 2.0}
    out.mkdir(parents=True, exist_ok=False)
    schedule = balanced_schedule()
    rows, errors = [], []
    with threadpool_limits(limits=1), np.errstate(over="raise", invalid="raise", divide="raise"):
        xt, yt = generator.rings(90801)
        for method, eta in (("pullback", 8.0), ("gd", .125)):
            theta = core.initialize_parameters(90802)
            for _ in range(24):
                direction, _ = core.direction(theta, xt, yt, method, .1)
                theta += eta * direction
        info = threadpool_info()
        assert all(x["num_threads"] == 1 for x in info if x["user_api"] == "blas")
        write_json(out / "design_record.json", dict(
            seed_blocks=16, repeats=3, budget_ns=TIME_NS, fixed_updates=STEP_LIMIT, schedule=schedule,
            python=sys.version, numpy=np.__version__, platform=platform.platform(), processor=platform.processor(),
            libraries=info, clock=vars(get_clock_info("perf_counter")),
            sha256={p.name: sha(p.read_bytes()) for p in (Path(__file__), HERE/"PLAN.md",
                                                          HERE/"prior_selection.json", HERE/"generator.py", Path(core.__file__))}))
        for index, (pair, repeat, method, policy) in enumerate(schedule, 1):
            rate = .125 if policy == "original" else selected["gd" if method == "gd" else "warmup"]
            try:
                rows.extend(run_trajectory(pair, repeat, method, policy, rate, index))
            except (ArithmeticError, np.linalg.LinAlgError) as exc:
                errors.append(dict(order=index, pair=pair, repeat=repeat, method=method,
                                   policy=policy, error=repr(exc)))
    write_json(out / "failures.json", errors)
    if rows:
        with (out / "rows.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(schedule)} trajectories, {len(rows)} endpoints, {len(errors)} failed.")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
