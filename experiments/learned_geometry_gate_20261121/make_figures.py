"""Render paired primary gate contrasts without inferential intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runner import digest, utcnow, write_json


def make_figures(scores, output):
    scores, output = Path(scores), Path(output)
    rows = json.loads((scores / "contrasts.json").read_text(encoding="utf-8"))
    selected = [r for r in rows if r["fraction"] == .25 and r["damping"] == .1]
    if any(r.get("B") is None or r.get("D") is None for r in selected):
        raise ValueError("Primary paired figures require complete contrast values.")
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True)
    colors = ("#2563a6", "#1d887a", "#9566ac", "#c0792b")
    titles = {"B": "B: tangent PB − nonlinear PB",
              "D": "D: PB advantage, nonlinear − tangent"}
    source_points = []
    for row_index, task in enumerate(("moons", "rings")):
        task_rows = [r for r in selected if r["task"] == task]
        ids = sorted({r["seed"] for r in task_rows})
        if len(ids) != 4:
            raise ValueError("The frozen figure expects four paired discovery seeds.")
        for col_index, estimand in enumerate(("B", "D")):
            ax = axes[row_index, col_index]
            ax.axhline(0., color="#667085", lw=.9, zorder=0)
            for value in (-.01, .01):
                ax.axhline(value, color="#b65235", ls=(0, (5, 3)), lw=1.1, zorder=0)
            means = []
            all_values = []
            for i, seed in enumerate(ids):
                values = []
                for h in (.01, .005):
                    matches = [r for r in task_rows if r["seed"] == seed and r["step_size"] == h]
                    if len(matches) != 1:
                        raise ValueError(f"Missing or duplicate point: {task}, {seed}, {h}")
                    val = matches[0][estimand]
                    values.append(val)
                    source_points.append(dict(task=task, estimand=estimand, seed=seed,
                                              step_size=h, value=val))
                all_values.extend(values)
                # Small fixed offsets separate paired seed points; x denotes
                # one of two explicit controllers, never a continuous rate.
                offset = (i-1.5)*.045
                ax.plot(np.arange(2)+offset, values, color=colors[i], lw=1.1,
                        marker="o", ms=5, alpha=.85, label=seed, zorder=2)
            for h in (.01,.005):
                means.append(float(np.mean([r[estimand] for r in task_rows if r["step_size"] == h])))
            ax.plot((0.,1.), means, color="#111827", lw=2.2, marker="D", ms=7,
                    label="Mean", zorder=4)
            lo, hi = min(min(all_values), -.01), max(max(all_values), .01)
            padding = max(.006, (hi-lo)*.20)
            ax.set_ylim(lo-padding, hi+padding)
            for x, value in enumerate(means):
                ax.annotate(f"mean {value:+.4f}", (x,value), xytext=(0,12),
                            textcoords="offset points", ha="center", fontsize=9,
                            color="#111827", bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=1))
            ax.set_xlim(-.22,1.22)
            ax.set_xticks((0,1), ("Coarse\nlogit RMS 0.01\nparameter cap 0.10",
                                  "Refined\nlogit RMS 0.005\nparameter cap 0.05"))
            ax.tick_params(axis="x", labelsize=8.5, labelbottom=True)
            ax.set_ylabel("Risk contrast (nats / example)")
            ax.set_title(f"{task.capitalize()} · {titles[estimand]}", fontsize=10.5, loc="left",pad=12)
            ax.grid(axis="y", color="#e5e7eb", lw=.6, alpha=.6)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.045),ncol=5,frameon=False)
    fig.suptitle("Matched-loss contrasts do not pass the prespecified screening gates",x=.06,y=.98,
                 ha="left",fontsize=15,fontweight="bold")
    fig.text(.06,.937,"Primary endpoint: training loss = 0.25 × initial loss. Each line is one paired seed; diamonds are arithmetic means.",
             ha="left",fontsize=10,color="#475467")
    fig.text(.06,.025,"Dashed lines: ±0.01-nat effect floor. Positive B favors nonlinear PB; positive D favors its comparative nonlinear gain.\n"
             "Four discovery pairs, descriptive only. Failure of the screening rule does not establish equivalence or absence of nonlinearity.",
             ha="left",va="bottom",fontsize=9,color="#475467")
    fig.subplots_adjust(left=.08,right=.98,top=.865,bottom=.20,wspace=.27,hspace=.63)
    for extension in ("png","svg"):
        fig.savefig(output/f"primary_paired_contrasts.{extension}",dpi=180,facecolor="white")
    plt.close(fig)
    write_json(output/"plot_data.json",source_points)
    write_json(output/"figure_manifest.json",dict(created_utc=utcnow(),
               source_file=str((scores/"contrasts.json").resolve()),
               source_sha256=digest(scores/"contrasts.json"),script_sha256=digest(__file__),
               role="descriptive_paired_primary_contrasts_no_inferential_intervals",
               output_sha256={p.name:digest(p) for p in output.iterdir()}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    make_figures(args.scores,args.output)
