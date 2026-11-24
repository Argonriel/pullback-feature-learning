"""Evaluate and contrast population-risk components for the replayed rings states."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from polar import evaluate, verify_controls


ARMS = ("gd", "pullback", "warmup")
STEPS = (0, 100, 500)
METRICS = ("radial_excess", "angular_excess", "excess_loss", "population_loss")


def write_rows(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(groups, out):
    colors = {"gd":"#4b6997", "pullback":"#007b71", "warmup":"#b85927"}
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.5))
    for ax, measure, title in zip(axes, METRICS[:2], ["Radial fit", "Sensitivity to angle"]):
        ax.set_title(title, loc="left", fontsize=13)
        for arm in ARMS:
            steps = (100, 500) if arm == "warmup" else STEPS
            values = np.array([[groups[i, arm, j][measure] for i in range(1, 9)] for j in steps])
            avg, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
            ax.fill_between(steps, np.maximum(0,avg-sd), avg+sd, color=colors[arm], alpha=.095,
                            linewidth=0)
            ax.plot(steps, avg, color=colors[arm], marker={"gd":"o","pullback":"s","warmup":"D"}[arm],
                    linestyle="--" if arm == "warmup" else "-", linewidth=1.8, markersize=5)
        ax.set(xlim=(-15, 520), ylim=(0, None), xticks=STEPS,
               xlabel="Training updates", ylabel="Population excess loss (nats)")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Which part of population loss falls during training?", fontsize=15, y=.96)
    fig.subplots_adjust(left=.083, right=.985, top=.81, bottom=.27, wspace=.29)
    fig.legend(axes[0].lines, ["GD throughout", "Pullback throughout", "100 pullback, then GD"],
               loc="lower center", bbox_to_anchor=(.5,.080), ncols=3, frameon=False)
    fig.text(.5,.047, "Means over eight paired draws. Shading shows one sample SD, clipped at zero. Panels use separate vertical scales.",
             ha="center", fontsize=8.5)
    fig.savefig(out / "population_components.png", dpi=185)
    fig.savefig(out / "population_components.svg")
    plt.close(fig)


def main(source, out):
    out.mkdir(parents=True, exist_ok=False)
    with (source / "replay.csv").open(newline="", encoding="utf-8") as file:
        index = list(csv.DictReader(file))
    key_hash = {r["state_key"]: r["state_sha256"] for r in index}
    controls, evaluated = {}, {}
    with threadpool_limits(limits=1):
        controls = verify_controls()
        with np.load(source / "states.npz", allow_pickle=False) as states:
            assert set(states.files) == set(key_hash) and len(states.files) == 48
            for key in states.files:
                assert hashlib.sha256(states[key].tobytes()).hexdigest() == key_hash[key]
                metrics = evaluate(states[key])
                if not metrics["grid_converged"]:
                    raise ArithmeticError(f"Polar quadrature did not converge at {key}: {metrics}")
                assert abs(metrics["identity_error"]) < 1e-9
                assert min(metrics["radial_excess"],metrics["angular_excess"]) >= -1e-12
                evaluated[key] = metrics
    rows = [dict(pair=int(r["pair"]), schedule=r["schedule"], step=int(r["step"]),
                 state_key=r["state_key"], state_sha256=r["state_sha256"], **evaluated[r["state_key"]])
            for r in index]
    groups = {(r["pair"],r["schedule"],r["step"]): r for r in rows}
    means = []
    for step in STEPS:
        for arm in ARMS:
            row = dict(schedule=arm,step=step)
            for metric in METRICS:
                values = [groups[i,arm,step][metric] for i in range(1,9)]
                row[metric+"_mean"] = mean(values)
                row[metric+"_sd"] = stdev(values)
            means.append(row)
    contrasts = []
    for step in (100,500):
        for left,right in (("gd","pullback"),("gd","warmup"),("pullback","warmup")):
            contrasts.append(("between_schedules",left,step,right,step))
    for arm in ARMS:
        for first,last in ((0,100),(100,500)):
            contrasts.append(("within_schedule",arm,first,arm,last))
    by_seed, contrast_means = [], []
    for kind,a,s,b,t in contrasts:
        c = dict(comparison=kind,reference=a,reference_step=s,comparator=b,comparator_step=t)
        items = []
        for i in range(1,9):
            diffs = {metric:groups[i,a,s][metric]-groups[i,b,t][metric] for metric in METRICS}
            assert abs(diffs["radial_excess"]+diffs["angular_excess"]-diffs["excess_loss"]) < 1e-10
            items.append(dict(pair=i,**c,**diffs))
        by_seed.extend(items)
        row=c.copy()
        for metric in METRICS:
            vals=[r[metric] for r in items]
            row[metric+"_mean"]=mean(vals)
            row[metric+"_sd"]=stdev(vals)
            row[metric+"_positive"]=sum(v>1e-12 for v in vals)
        contrast_means.append(row)
    write_rows(out / "states.csv", rows)
    write_rows(out / "means.csv", means)
    write_rows(out / "paired_changes.csv", by_seed)
    write_rows(out / "mean_changes.csv", contrast_means)
    audit = dict(unique_quadratures=len(evaluated), methods_times_seeds=len(rows),
                 max_identity_error=max(abs(v["identity_error"]) for v in evaluated.values()),
                 max_refinement_difference=max(v["grid_error"] for v in evaluated.values()),
                 min_bayes_loss=min(v["bayes_loss"] for v in evaluated.values()),
                 max_bayes_loss=max(v["bayes_loss"] for v in evaluated.values()),
                 paired_changes=len(by_seed), controls=controls)
    (out / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    figure(groups,out)
    print(json.dumps(audit))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    main(args.replay,args.out)
