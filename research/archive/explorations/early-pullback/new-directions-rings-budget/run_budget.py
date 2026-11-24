"""Select rates on tuning pairs before running fresh budgeted evaluation pairs."""
import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import generator
from generator import rings
from pullback_feature_learning import core


BUDGET = 100_000_000
METHODS = ("gd", "warmup")
RATES = (0.03125, 0.125, 0.5, 2.0, 8.0)


def save_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, record):
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draws(pair, phase):
    base = 81000 if phase == "tuning" else 84000
    return rings(base + pair), core.initialize_parameters(base + 1000 + pair), rings(base + 2000 + pair)


def run(seed, phase, order, repeat, method, rate):
    (x, y), theta, (xv, yv) = draws(seed, phase)
    record = dict(phase=phase, order=order, seed=seed, repeat=repeat, method=method, gd_rate=rate,
                  status="failed", failure="", updates=0, pullback_updates=0,
                  lower_ns=0, crossing_ns="", validation_loss="", validation_accuracy="", state_sha256="")
    elapsed, steps = 0, 0
    try:
        while True:
            pb = (method == "warmup" and steps < 100)
            name, eta = ("pullback", 8.0) if pb else ("gd", rate)
            begin = perf_counter_ns()
            delta, _ = core.direction(theta, x, y, name, damping=0.1)
            after = theta + eta * delta
            ns = perf_counter_ns() - begin
            if elapsed + ns > BUDGET:
                record["crossing_ns"] = elapsed + ns
                break
            if not np.isfinite(after).all():
                raise FloatingPointError("Nonfinite accepted parameters.")
            steps += 1
            elapsed += ns
            theta = after
        zv = core.representation(theta, xv)
        loss = core.logistic_loss(zv, yv)
        if not np.isfinite(zv).all() or not np.isfinite(loss):
            raise FloatingPointError("Nonfinite held-out features or objective.")
        record.update(status="ok", validation_loss=loss,
                      validation_accuracy=float(np.mean(np.where(zv[:, 0] >= 0, 1, -1) == yv)),
                      state_sha256=hashlib.sha256(theta.tobytes()).hexdigest())
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        record["failure"] = str(exc)
    record.update(updates=steps, lower_ns=elapsed,
                  pullback_updates=min(steps, 100) if method == "warmup" else 0)
    return record


def tune(out):
    rng = random.Random(9001)
    cells = [(seed, method, rate) for seed in range(1, 5) for method in METHODS for rate in RATES]
    rng.shuffle(cells)
    rows = [run(seed, "tuning", i, 1, method, rate) for i, (seed, method, rate) in enumerate(cells, 1)]
    save_csv(out / "tuning_runs.csv", rows)
    grid = []
    selected = {}
    for method in METHODS:
        for rate in RATES:
            subset = [r for r in rows if r["method"] == method and r["gd_rate"] == rate]
            complete = len(subset) == 4 and all(r["status"] == "ok" for r in subset)
            grid.append(dict(method=method, rate=rate, complete=complete,
                             validation_loss=mean(r["validation_loss"] for r in subset) if complete else "",
                             validation_accuracy=mean(r["validation_accuracy"] for r in subset) if complete else ""))
        viable = [r for r in grid if r["method"] == method and r["complete"]]
        if viable:
            selected[method] = min(viable, key=lambda r: (r["validation_loss"], r["rate"]))["rate"]
    save_csv(out / "tuning_grid.csv", grid)
    save_json(out / "selection.json", dict(rates=selected, tuning_sha256=sha(out / "tuning_runs.csv"),
                                          grid_sha256=sha(out / "tuning_grid.csv"),
                                          plan_sha256=sha(Path(__file__).with_name("PLAN.md"))))
    if set(selected) != set(METHODS):
        raise RuntimeError("At least one method has no viable tuning rate.")


def evaluate(out):
    selected = json.loads((out / "selection.json").read_text())["rates"]
    rng = random.Random(9002)
    # Each repeat has four first positions per method. Shuffle the eight pair blocks.
    schedule = []
    for repeat in range(1, 4):
        seeds = list(range(1, 9))
        rng.shuffle(seeds)
        for index, seed in enumerate(seeds):
            order = METHODS if index % 2 == 0 else METHODS[::-1]
            schedule.extend((seed, repeat, method) for method in order)
    rows = [run(seed, "evaluation", i, repeat, method, selected[method])
            for i, (seed, repeat, method) in enumerate(schedule, 1)]
    save_csv(out / "evaluation_runs.csv", rows)
    if any(r["status"] != "ok" for r in rows):
        save_json(out / "evaluation_summary.json", dict(complete=False, failures=[r for r in rows if r["status"] != "ok"]))
        return
    fields = ("validation_loss", "validation_accuracy", "updates", "lower_ns", "crossing_ns", "pullback_updates")
    seed_means, paired = [], []
    for seed in range(1, 9):
        grouped = {}
        for method in METHODS:
            subset = [r for r in rows if r["seed"] == seed and r["method"] == method]
            assert sorted(r["repeat"] for r in subset) == [1, 2, 3]
            grouped[method] = {k: mean(r[k] for r in subset) for k in fields}
            seed_means.append(dict(seed=seed, method=method, **grouped[method]))
        paired.append(dict(seed=seed, **{k: grouped["warmup"][k] - grouped["gd"][k]
                                         for k in ("validation_loss", "validation_accuracy")}))
    save_csv(out / "evaluation_seed_means.csv", seed_means)
    save_csv(out / "evaluation_paired.csv", paired)
    result = dict(complete=True, seed_count=8, repeats_per_seed=3, arms={}, differences={})
    for method in METHODS:
        group = [r for r in seed_means if r["method"] == method]
        result["arms"][method] = {k: {"mean": mean(r[k] for r in group), "sd": stdev(r[k] for r in group)}
                                  for k in fields}
    for k, smaller_better in (("validation_loss", True), ("validation_accuracy", False)):
        vals = [r[k] for r in paired]
        result["differences"][k] = dict(mean=mean(vals), sd=stdev(vals),
                                       warmup_wins=sum(v < 0 if smaller_better else v > 0 for v in vals),
                                       ties=vals.count(0))
    save_json(out / "evaluation_summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1), np.errstate(over="raise", invalid="raise", divide="raise"):
        warm_x, warm_y = rings(80123)
        for method in ("gd", "pullback"):
            theta = core.initialize_parameters(80124)
            for _ in range(10):
                delta, _ = core.direction(theta, warm_x, warm_y, method, damping=0.1)
                theta = theta + (8.0 if method == "pullback" else 0.125) * delta
        save_json(out / "metadata.json",
                  dict(generator_sha256=sha(generator.__file__), core_sha256=sha(core.__file__),
                       driver_sha256=sha(__file__), plan_sha256=sha(Path(__file__).with_name("PLAN.md")),
                       platform=platform.platform(), cpu=platform.processor(), numpy=np.__version__,
                       python=platform.python_version(), libraries=threadpool_info(),
                       tuning_schedule_seed=9001, evaluation_schedule_seed=9002, budget_ns=BUDGET))
        tune(out)
        evaluate(out)
    print(f"Recorded tuning and fresh evaluation in {out}")


if __name__ == "__main__":
    main()
