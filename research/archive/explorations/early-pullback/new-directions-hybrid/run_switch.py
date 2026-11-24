"""Evaluate fixed optimizer switches on paired development draws. Python 3.10+."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from pullback_feature_learning import core
from pullback_feature_learning.data import two_moons


METHODS = ("gd", "pullback")
SWITCH = 100
ENDPOINTS = (100, 500, 2000)
GD_BUDGET = {500: 1240, 2000: 2740}
RATES = {"gd": 0.125, "pullback": 8.0}
DAMPING = 0.1


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_hash(vector):
    return hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest()


def advance(theta, steps, method, x, y):
    for _ in range(steps):
        delta, _ = core.direction(theta, x, y, method, DAMPING)
        theta = theta + RATES[method] * delta
    if not np.isfinite(theta).all():
        raise ValueError("A trajectory contains nonfinite weights.")
    return theta


def checkpoint(theta, train, validation, first, second, step, pair):
    before = min(SWITCH, step)
    pullback_steps = before * int(first == "pullback") + (step - before) * int(second == "pullback")
    row = dict(pair=pair, first=first, second=second, step=step,
               gd_steps=step - pullback_steps, pullback_steps=pullback_steps,
               state_sha256=vector_hash(theta))
    for label, (x, y) in (("train", train), ("validation", validation)):
        features = core.representation(theta, x)
        predictions = np.where(features[:, 0] >= 0, 1, -1)
        row[f"{label}_loss"] = core.logistic_loss(features, y)
        row[f"{label}_accuracy"] = float(np.mean(predictions == y))
    return row


def run_pair(pair):
    train = two_moons(256, 0.15, 61000 + pair)
    validation = two_moons(256, 0.15, 63000 + pair)
    initial = core.initialize_parameters(62000 + pair)
    rows = []
    for first in METHODS:
        prefix = advance(initial, SWITCH, first, *train)
        for second in METHODS:
            theta, previous = prefix.copy(), SWITCH
            endpoints = sorted(set(ENDPOINTS) | (set(GD_BUDGET.values()) if first == second == "gd" else set()))
            for step in endpoints:
                theta = advance(theta, step - previous, second, *train)
                rows.append(checkpoint(theta, train, validation, first, second, step, pair))
                previous = step
    return rows


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["first"], row["second"], row["step"]), []).append(row)
    means = []
    for (first, second, step), group in grouped.items():
        if len(group) != 8 or len({row["pair"] for row in group}) != 8:
            raise ValueError("Incomplete pair group.")
        entry = dict(first=first, second=second, step=step, pairs=len(group))
        for key in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy"):
            vals = [row[key] for row in group]
            entry[f"{key}_mean"] = float(np.mean(vals))
            entry[f"{key}_sd"] = float(np.std(vals, ddof=1))
        means.append(entry)
    row_by_key = {(r["first"], r["second"], r["step"], r["pair"]): r for r in rows}
    diffs = []
    for step in (500, 2000):
        for first, second, comparator_step in (("gd", "gd", step), ("pullback", "pullback", step),
                                               ("gd", "pullback", step), ("gd", "gd", GD_BUDGET[step])):
            losses, accuracies = [], []
            for pair in range(1, 9):
                hybrid = row_by_key["pullback", "gd", step, pair]
                comparator = row_by_key[first, second, comparator_step, pair]
                losses.append(hybrid["validation_loss"] - comparator["validation_loss"])
                accuracies.append(hybrid["validation_accuracy"] - comparator["validation_accuracy"])
            diffs.append(dict(hybrid_step=step, comparator_first=first, comparator_second=second,
                              comparator_step=comparator_step, mean_validation_loss_diff=float(np.mean(losses)),
                              sd_validation_loss_diff=float(np.std(losses, ddof=1)),
                              pairs_lower_loss=sum(d < 0 for d in losses),
                              mean_validation_accuracy_diff_pp=float(100 * np.mean(accuracies)),
                              sd_validation_accuracy_diff_pp=float(100 * np.std(accuracies, ddof=1)),
                              pairs_higher_accuracy=sum(d > 0 for d in accuracies),
                              pairs_tied_accuracy=sum(d == 0 for d in accuracies)))
    return means, diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory.")
    parser.add_argument("--repository", type=Path, required=True, help="Installed source checkout.")
    args = parser.parse_args()
    repo = args.repository.resolve()
    config_path = repo / "configs" / "formal.json"
    config = json.loads(config_path.read_text())
    if config["methods"] != {"gd": {"learning_rate": RATES["gd"]},
                             "pullback": {"learning_rate": RATES["pullback"], "damping": DAMPING}}:
        raise ValueError("Fixed settings disagree with the formal configuration.")
    if (config["train_size"], config["validation_size"], config["noise"]) != (256, 256, 0.15):
        raise ValueError("Dataset settings disagree with the formal configuration.")
    if Path(core.__file__).resolve() != (repo / "src" / "pullback_feature_learning" / "core.py").resolve():
        raise ValueError("The installed core does not match the given checkout.")
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        rows = [row for pair in range(1, 9) for row in run_pair(pair)]
        libraries = threadpool_info()
    means, differences = aggregate(rows)
    write_csv(args.output / "checkpoints.csv", rows)
    write_csv(args.output / "means.csv", means)
    write_csv(args.output / "hybrid_paired_differences.csv", differences)
    metadata = dict(pairs=[dict(pair=i, train_seed=61000+i, validation_seed=63000+i, initialization_seed=62000+i)
                           for i in range(1, 9)], training_examples=256, validation_examples=256,
                    noise=0.15, learning_rates=RATES, damping=DAMPING, switch_step=SWITCH,
                    gd_proxy_steps=GD_BUDGET, python=platform.python_version(), numpy=np.__version__,
                    source_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                    threadpools=libraries, config_sha256=file_hash(config_path), core_sha256=file_hash(Path(core.__file__)),
                    driver_sha256=file_hash(Path(__file__)),
                    data_sha256=file_hash(repo / "src" / "pullback_feature_learning" / "data.py"))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} checkpoints from eight paired draws.")


if __name__ == "__main__":
    main()

