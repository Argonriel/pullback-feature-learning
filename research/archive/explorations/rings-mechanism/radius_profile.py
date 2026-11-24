"""Visualize the score around the known decision radius on the replayed rings."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core


ROOT = Path(__file__).resolve().parent


def probability(h):
    return np.exp(-np.logaddexp(0, -h))


def oracle(r):
    sigma2 = .15**2
    return (.5**2 - 1.2**2) / (2*sigma2) + np.log(np.i0(r*1.2/sigma2)) - np.log(np.i0(r*.5/sigma2))


def main():
    r = np.unique(np.r_[np.linspace(.30, 1.40, 111), .86419688])
    phi = (np.arange(256)+.5)*2*np.pi/256
    x = (r[:, None, None] * np.c_[np.cos(phi), np.sin(phi)][None, :, :]).reshape(-1, 2)
    colors = ["#56677a", "#127a75", "#b85a31"]
    states = np.load(ROOT / "stage1" / "states.npz")
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.5), sharex=True, layout="constrained")
    rows = []
    with threadpool_limits(limits=1):
        for col, steps in enumerate((100, 500)):
            axes[0, col].plot(r, probability(oracle(r)), "--", color="black", lw=1.4, label="True posterior")
            curves = [("gd", "GD"), ("pullback", "Pullback" if steps == 500 else "Pullback / hybrid")]
            if steps == 500:
                curves.append(("warmup", "Pullback → GD"))
            for (key, name), color in zip(curves, colors):
                radial, angle_penalty = [], []
                for pair in range(1, 9):
                    h = core.representation(states[f"pair{pair}_{key}{steps}"], x)[:, 0].reshape(len(r), -1)
                    avg = h.mean(axis=1)
                    radial.append(probability(avg))
                    angle_penalty.append(np.logaddexp(0, h).mean(axis=1) - np.logaddexp(0, avg))
                avg_p, avg_penalty = np.mean(radial, axis=0), np.mean(angle_penalty, axis=0)
                axes[0, col].plot(r, avg_p, color=color, lw=2, label=name)
                axes[1, col].plot(r, avg_penalty, color=color, lw=2)
                rows.extend(dict(step=steps, method=key, radius=radius, true_p=p,
                                 model_p_from_angle_averaged_logit=m, angular_penalty_at_radius=a)
                            for radius, p, m, a in zip(r, probability(oracle(r)), avg_p, avg_penalty))
            axes[0, col].set_title(f"{steps} updates")
            axes[0, col].legend(fontsize=8, frameon=False, loc="upper left")
            axes[1, col].set_xlabel("Observed radius")
            for ax in axes[:, col]:
                ax.axvline(.86419688, color="#888", linestyle=":", lw=1)
                ax.spines[["top","right"]].set_visible(False)
            axes[0,col].set_ylim(-.03,1.04)
    for row, label in enumerate(["Outer-class probability", "Angular penalty (nats)"]):
        axes[row, 0].set_ylabel(label)
    top = max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(-.002*top, top)
    fig.savefig(ROOT / "stage1" / "radius_profile.png", dpi=185)
    fig.savefig(ROOT / "stage1" / "radius_profile.svg")
    plt.close(fig)
    with (ROOT / "stage1" / "radius_profile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
