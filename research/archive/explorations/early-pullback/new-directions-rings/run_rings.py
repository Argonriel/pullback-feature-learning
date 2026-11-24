"""Compare three fixed training schedules on independent concentric-ring draws."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess

import numpy as np
from threadpoolctl import threadpool_limits

from pullback_feature_learning import core


SCHEDULES = {"gd": ("gd", "gd"), "pullback": ("pullback", "pullback"), "warmup": ("pullback", "gd")}
RATES = {"gd": 0.125, "pullback": 8.0}
CHECKPOINTS = (100, 500, 2000)


def rings(seed):
    rng = np.random.default_rng(seed)
    labels = np.repeat([-1.0, 1.0], 128)
    angle = rng.uniform(0, 2 * np.pi, len(labels))
    radius = np.where(labels < 0, 0.5, 1.2)
    xy = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    xy += rng.normal(0, 0.15, size=xy.shape)
    order = rng.permutation(len(labels))
    return xy[order], labels[order]


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def advance(theta, method, steps, train):
    for _ in range(steps):
        delta, _ = core.direction(theta, *train, method=method, damping=0.1)
        theta = theta + RATES[method] * delta
    if not np.isfinite(theta).all():
        raise ValueError("Non-finite trajectory.")
    return theta


def score(theta, datasets, record):
    record["state_sha256"] = digest_bytes(theta.tobytes())
    for split, (x, y) in datasets.items():
        features = core.representation(theta, x)
        record[f"{split}_loss"] = core.logistic_loss(features, y)
        record[f"{split}_accuracy"] = float(np.mean(np.where(features[:, 0] >= 0, 1, -1) == y))
    return record


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if (args.repository / "src/pullback_feature_learning/core.py").resolve() != Path(core.__file__).resolve():
        raise ValueError("Package installation differs from source directory.")
    config_file = args.repository / "configs/formal.json"
    config = json.loads(config_file.read_text())
    if config["methods"] != {"gd": {"learning_rate": 0.125}, "pullback": {"learning_rate": 8, "damping": 0.1}}:
        raise ValueError("Original settings have changed.")
    records, seeds = [], []
    with threadpool_limits(limits=1):
        for pair in range(1, 9):
            datasets = {"train": rings(71000 + pair), "validation": rings(73000 + pair)}
            initial = core.initialize_parameters(72000 + pair)
            seed = dict(pair=pair, train_seed=71000 + pair, validation_seed=73000 + pair,
                        initialization_seed=72000 + pair, initial_sha256=digest_bytes(initial.tobytes()))
            for name, (xy, y) in datasets.items():
                assert xy.shape == (256, 2) and np.all(np.unique(y, return_counts=True)[1] == [128, 128])
                seed[f"{name}_sha256"] = digest_bytes(xy.tobytes() + y.tobytes())
            assert seed["train_sha256"] != seed["validation_sha256"]
            seeds.append(seed)
            prefixes = {m: advance(initial, m, 100, datasets["train"]) for m in RATES}
            for schedule, (first, second) in SCHEDULES.items():
                theta = prefixes[first].copy()
                previous = 100
                for step in CHECKPOINTS:
                    theta = advance(theta, second, step - previous, datasets["train"])
                    r = score(theta, datasets, dict(pair=pair, schedule=schedule, step=step))
                    records.append(r)
                    previous = step
    summaries, paired = [], []
    for step in CHECKPOINTS:
        groups = {}
        for arm in SCHEDULES:
            subset = sorted([r for r in records if r["schedule"] == arm and r["step"] == step], key=lambda r: r["pair"])
            assert [r["pair"] for r in subset] == list(range(1, 9))
            groups[arm] = subset
            summary = dict(schedule=arm, step=step)
            for metric in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy"):
                vals = [r[metric] for r in subset]
                summary[metric + "_mean"] = mean(vals)
                summary[metric + "_sd"] = stdev(vals)
            summaries.append(summary)
        for base in ("gd", "pullback"):
            comparison = dict(reference=base, step=step)
            for metric in ("validation_loss", "validation_accuracy"):
                ds = [h[metric] - b[metric] for h, b in zip(groups["warmup"], groups[base])]
                comparison[metric + "_diff_mean"] = mean(ds)
                comparison[metric + "_diff_sd"] = stdev(ds)
                comparison[metric + "_hybrid_wins"] = sum(x < 0 for x in ds) if metric.endswith("loss") else sum(x > 0 for x in ds)
                comparison[metric + "_ties"] = sum(x == 0 for x in ds)
            paired.append(comparison)
    write_rows(args.output / "checkpoints.csv", records)
    write_rows(args.output / "means.csv", summaries)
    write_rows(args.output / "paired.csv", paired)
    write_rows(args.output / "seeds.csv", seeds)
    meta = dict(seed_pairs=8, train_size=256, validation_size=256, noise=0.15, centers=[0, 0], radii=[0.5, 1.2],
                switch_at=100, settings=config["methods"], checkpoints=CHECKPOINTS, numpy=np.__version__, threads=1,
                source_commit=subprocess.check_output(["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True).strip(),
                core_sha256=digest_bytes(Path(core.__file__).read_bytes()), driver_sha256=digest_bytes(Path(__file__).read_bytes()),
                plan_sha256=digest_bytes(Path(__file__).with_name("PLAN.md").read_bytes()))
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} ring checkpoints and {len(paired)} paired contrasts.")


if __name__ == "__main__":
    main()

