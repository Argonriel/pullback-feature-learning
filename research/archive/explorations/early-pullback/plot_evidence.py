"""Render direction and transfer comparisons using the archived development CSV files."""

import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "new-directions-rings").is_dir() else HERE.parent
GD, PB, SWITCH = "#435c85", "#007e78", "#ab552e"


def read(folder):
    path = ROOT / f"new-directions-{folder}" / "results" / "checkpoints.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def draw_pair(ax, first, second, pair_index, colors):
    assert len(first) == len(second) == 8
    center = pair_index * 1.45
    for i, (a, b) in enumerate(zip(first, second)):
        dx = (i - 3.5) * .012
        ax.plot([center + dx - .22, center + dx + .22], [100*a, 100*b],
                color="#c8cbd0", linewidth=.8, zorder=1)
        ax.scatter([center + dx - .22, center + dx + .22], [100*a, 100*b],
                   c=colors, s=13, alpha=.45, linewidths=0)
    for offset, values, color in zip((-.22, .22), (first, second), colors):
        score = mean(values) * 100
        ax.scatter(center+offset, score, s=48, marker="D", c=color, edgecolor="white", zorder=3)
        ax.annotate(f"{score:.1f}", (center+offset, score),
                    xytext=(0, 8 if offset > 0 else -15), textcoords="offset points",
                    color=color, fontsize=9, weight="bold", ha="center")


def legend(ax, labels, colors):
    handles = [Line2D([], [], marker="D", color=c, linestyle="", label=l, markersize=6)
               for c, l in zip(colors, labels)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.30),
              frameon=False, ncol=2, columnspacing=.9, handletextpad=.3)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.1), layout="constrained")
    geo = read("geometry")
    for k, (gd, pb) in enumerate([("gd", "pb_at_gd_size"), ("gd_at_pb_size", "pb")]):
        def arm(name):
            rows = sorted((r for r in geo if r["arm"] == name and r["step"] == "100"),
                          key=lambda r: int(r["seed"]))
            return [float(r["validation_accuracy"]) for r in rows]
        draw_pair(a, arm(gd), arm(pb), k, (GD, PB))
    a.set_title("Changing the direction at 100 updates", fontsize=12, loc="left")
    a.set_xticks([0, 1.45], ["GD size rule", "Pullback size rule"])
    legend(a, ["GD direction", "Pullback direction"], [GD, PB])

    moons = read("hybrid")
    rings = read("rings")
    for k, data in enumerate([moons, rings]):
        def arm(switch):
            if k == 0:
                names = ("pullback", "gd") if switch else ("gd", "gd")
                rows = [r for r in data if (r["first"], r["second"]) == names and r["step"] == "500"]
            else:
                name = "warmup" if switch else "gd"
                rows = [r for r in data if r["schedule"] == name and r["step"] == "500"]
            return [float(r["validation_accuracy"]) for r in sorted(rows, key=lambda r: int(r["pair"]))]
        draw_pair(b, arm(False), arm(True), k, (GD, SWITCH))
    b.set_title("Using pullback for the first 100 of 500 updates", fontsize=12, loc="left")
    b.set_xticks([0, 1.45], ["Two moons", "Concentric rings"])
    legend(b, ["GD throughout", "Pullback → GD"], [GD, SWITCH])
    for ax in (a, b):
        ax.set(ylim=(47, 105), xlim=(-.62, 2.06), ylabel="Validation accuracy (%)")
        ax.grid(axis="y", alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"early_geometry.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
