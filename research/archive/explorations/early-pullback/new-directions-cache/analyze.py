"""Validate and summarize the serial timing round and render the costs figure."""

import csv
import json
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run01"
METHODS = ["gd", "pullback", "pb100_then_gd", "refresh10", "refresh50"]
LABELS = ["GD", "Full pullback", "Pullback 100 → GD", "Refresh every 10", "Refresh every 50"]


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():
    rows = read_rows(OUT / "paired_endpoints.csv")
    by_key = {(int(r["pair"]), r["method"], r["endpoint"]): r for r in rows}
    assert len(rows) == len(by_key) == 160 and not json.loads((OUT / "run.json").read_text())["failures"]
    endpoints = ["step_100", "step_500", "step_2000", "time_100ms"]
    summary = {}
    for endpoint in endpoints:
        summary[endpoint] = {}
        for m in METHODS:
            sample = [by_key[i, m, endpoint] for i in range(1, 9)]
            losses = [float(r["val_loss"]) for r in sample]
            gd = [float(by_key[i, "gd", endpoint]["val_loss"]) for i in range(1, 9)]
            summary[endpoint][m] = dict(
                accuracy_percent=mean(100*float(r["val_accuracy"]) for r in sample),
                loss=mean(losses), loss_sd=stdev(losses), loss_median=median(losses),
                mean_updates=mean(int(r["updates"]) for r in sample),
                mean_ms=mean(1000*float(r["core_seconds"]) for r in sample),
                lower_loss_than_gd=sum(a < b for a, b in zip(losses, gd)),
            )
            for r in sample:
                n = int(r["updates"])
                if endpoint == "time_100ms":
                    assert float(r["core_seconds"]) <= 0.1 < float(r["next_core_seconds"])
                else:
                    assert n == int(endpoint.split("_")[1])
                builds = ({"gd": 0, "pullback": n, "pb100_then_gd": min(n, 100)}.get(m))
                if builds is None:
                    interval = int(m[7:])
                    builds = (n + interval - 1) // interval
                assert int(r["metric_builds"]) == builds

    # Independent drivers used the same seeds. Cross-check the overlapping fixed endpoints.
    hybrid_rows = read_rows(ROOT.parent / "new-directions-hybrid" / "results" / "checkpoints.csv")
    mapping = {("gd", "gd"): "gd", ("pullback", "pullback"): "pullback",
               ("pullback", "gd"): "pb100_then_gd"}
    matched = 0
    for r in hybrid_rows:
        m = mapping.get((r["first"], r["second"]))
        if m and int(r["step"]) in (100, 500, 2000):
            other = by_key[int(r["pair"]), m, f"step_{r['step']}"]
            assert abs(float(other["val_loss"]) - float(r["validation_loss"])) < 1e-11
            assert float(other["val_accuracy"]) == float(r["validation_accuracy"])
            matched += 1
    assert matched == 72, matched
    geo_rows = read_rows(ROOT.parent / "new-directions-geometry" / "results" / "checkpoints.csv")
    geometry_matches = 0
    for r in geo_rows:
        if r["arm"] in ("gd", "pb") and int(r["step"]) in (100, 500):
            m = "gd" if r["arm"] == "gd" else "pullback"
            other = by_key[int(r["seed"]), m, f"step_{r['step']}"]
            assert abs(float(other["val_loss"]) - float(r["validation_loss"])) < 1e-9
            assert float(other["val_accuracy"]) == float(r["validation_accuracy"])
            geometry_matches += 1
    assert geometry_matches == 32
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = ["#426181", "#087c78", "#b05c30", "#8461a2", "#8461a2"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4), layout="constrained")
    for ax, ep, title in zip(axes, ["step_500", "time_100ms"],
                             ["After 500 updates", "After 100 ms of core training"]):
        for j, m in enumerate(METHODS):
            losses = [float(by_key[i, m, ep]["val_loss"]) for i in range(1, 9)]
            ax.scatter(losses, [j + 0.034*(i-3.5) for i in range(8)],
                       s=18, c=colors[j], alpha=.5, linewidths=0)
            ax.scatter(mean(losses), j, marker="D", c=colors[j], edgecolor="#222222", s=45)
        ax.set(xscale="log", xlim=(.015, 25), ylim=(4.6, -.6),
               yticks=range(5), yticklabels=LABELS if ax is axes[0] else [],
               xlabel="Validation logistic loss (log scale)", title=title)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].legend(handles=[Line2D([], [], marker="o", color="gray", lw=0, label="One pair"),
                            Line2D([], [], marker="D", color="black", lw=0, label="Mean")],
                   loc="upper left", bbox_to_anchor=(0, -.2), ncol=2, frameon=False)
    fig.savefig(OUT / "validation_cost.png", dpi=180)
    fig.savefig(OUT / "validation_cost.svg")
    plt.close(fig)
    print(f"Verified all {len(rows)} endpoints, {matched} switch and {geometry_matches} geometry comparisons.")


if __name__ == "__main__":
    main()
