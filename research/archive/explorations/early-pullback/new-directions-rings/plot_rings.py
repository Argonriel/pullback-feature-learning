"""Plot the first of the eight rings seed pairs and verify every archived state."""
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from threadpoolctl import threadpool_limits

from pullback_feature_learning import core
from run_rings import advance, rings


BASE = Path(__file__).resolve().parent
COLORS = ("#137185", "#BD5A22")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    with (BASE / "results" / "seeds.csv").open() as stream:
        first, *_ = csv.DictReader(stream)
    assert first["pair"] == "1"
    x, y = rings(int(first["validation_seed"]))
    train = rings(int(first["train_seed"]))
    initial = core.initialize_parameters(int(first["initialization_seed"]))
    assert sha(x.tobytes() + y.tobytes()) == first["validation_sha256"]
    assert sha(train[0].tobytes() + train[1].tobytes()) == first["train_sha256"]
    assert sha(initial.tobytes()) == first["initial_sha256"]
    with (BASE / "results" / "checkpoints.csv").open() as stream:
        checkpoints = {(r["schedule"], int(r["step"])): r for r in csv.DictReader(stream) if r["pair"] == "1"}
    with threadpool_limits(limits=1):
        gd100 = advance(initial, "gd", 100, train)
        pb100 = advance(initial, "pullback", 100, train)
        gd500 = advance(gd100, "gd", 400, train)
        warm500 = advance(pb100, "gd", 400, train)
        states = [("Initialization", initial, None),
                  ("100 GD updates", gd100, ("gd", 100)),
                  ("500 GD updates", gd500, ("gd", 500)),
                  ("100 pullback updates", pb100, ("pullback", 100)),
                  ("100 pullback + 400 GD", warm500, ("warmup", 500))]
        fitted = []
        for title, theta, key in states:
            z = core.representation(theta, x)
            acc = np.mean(np.where(z[:, 0] >= 0, 1, -1) == y)
            if key:
                assert sha(theta.tobytes()) == checkpoints[key]["state_sha256"]
                assert float(checkpoints[key]["validation_accuracy"]) == acc
            fitted.append((title, z, acc))
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": "#8799A1", "text.color": "#1D323D",
                         "axes.labelcolor": "#314752", "xtick.color": "#455A64", "ytick.color": "#455A64"})
    fig = plt.figure(figsize=(10.4, 6.7), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.085, right=0.975, bottom=0.16, top=0.80, hspace=0.55, wspace=0.34)
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])]
    ymax = max(np.abs(z[:, 1]).max() for _, z, _ in fitted) * 1.07
    for ax, (title, z, acc) in zip(axes, fitted):
        xmax = 1.07 * np.abs(z[:, 0]).max()
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-ymax, ymax)
        ax.set_box_aspect(0.60)
        ax.axvline(0, color="#32434A", lw=1.0, ls="--", zorder=1)
        for label, color, marker in ((-1, COLORS[0], "o"), (1, COLORS[1], "^")):
            points = z[y == label]
            ax.scatter(points[:, 0], points[:, 1], c=color, marker=marker, s=18, alpha=0.72,
                       edgecolors="white", linewidths=0.25, zorder=2)
        ax.set_title(f"{title}\n{acc * 100:.1f}% correct", loc="left", fontsize=10.6, pad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        ax.tick_params(length=3, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Second feature, $z_2$")
    fig.text(0.078, 0.938, "How a pullback warm-up separates two rings", fontsize=19, weight="bold")
    fig.text(0.078, 0.885, "Same 256 validation examples in every panel. The classifier assigns points to classes by the sign of the first feature.",
             fontsize=10)
    fig.text(0.53, 0.098, "First feature, $z_1$   ·   Horizontal scale varies by panel", ha="center", fontsize=10)
    handles = [Line2D([], [], ls="none", marker="o", color=COLORS[0], label="Inner ring (−1)"),
               Line2D([], [], ls="none", marker="^", color=COLORS[1], label="Outer ring (+1)"),
               Line2D([], [], ls="--", color="#32434A", lw=1, label="Fixed decision boundary ($z_1=0$)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.011), ncol=3, frameon=False, fontsize=9)
    for ext in ("svg", "png"):
        fig.savefig(BASE / f"rings_features_pair1.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    for title, z, acc in fitted:
        print(f"{title}: {acc:.6f}; z1 range {z[:, 0].min():.3f} to {z[:, 0].max():.3f}; z2 range {z[:, 1].min():.3f} to {z[:, 1].max():.3f}")


if __name__ == "__main__":
    main()

