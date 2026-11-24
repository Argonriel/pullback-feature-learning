"""Produce the switch-time figure and paired summaries of the learning-rate check."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
COLORS = {"Original 16 seeds": "#176B78", "Independent 16 seeds": "#B34E2D"}
COHORTS = (("Original 16 seeds", "results", "trajectory_results", "rate_results"),
           ("Independent 16 seeds", "validation_results", "trajectory_validation_results", "rate_validation_results"))


def read(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(rows, field, draws):
    a = np.array([float(row[field]) for row in rows])
    lo, hi = np.quantile(a[draws].mean(axis=1), [.025, .975])
    return a.mean(), lo, hi


def wins(rows, key):
    settings = np.array(sorted({int(r[key]) for r in rows}))
    counts = [sum(float(r["gap"]) < 0 for r in rows if int(r[key]) == v) for v in settings]
    return settings, counts


def headings(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=23, color="#1F353B")
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color="#50636A", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A2AFB2")
    ax.grid(axis="y", color="#E3EBEA", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#3D555D", labelsize=9)


def main():
    draws = np.random.default_rng(291031).integers(0, 16, (20000, 16))
    rates, summaries, rate_changes = [.5, 1, 2], [], []
    fig, ((a,b),(c,d)) = plt.subplots(2,2,figsize=(11.5,8.15))
    fig.subplots_adjust(top=.84, bottom=.115, left=.085, right=.975, wspace=.29, hspace=.68)
    fig.text(.085,.957,"When does a second view help more?", fontsize=21, fontweight="bold", color="#19383D")
    fig.text(.085,.918,"Concentric rings  |  64 extra predictions per batch of 256  |  32 paired training seeds", fontsize=10.5, color="#51676E")
    headings(a, "A   Change when GD starts", "At update 500, how many models favor the region rule?")
    headings(b, "B   Follow one switch through training", "Switch at update 100 and follow the two rules on the same models.")
    headings(c, "C   Compare the sources of benefit", "At update 500. The larger benefit determines the winning rule on average.")
    headings(d, "D   Change only the GD learning rate", "400 GD updates after the switch. Bars show paired 95% intervals.")
    for i, (label, end, trajectory, rate) in enumerate(COHORTS):
        color = COLORS[label]
        models = read(ROOT / end / "endpoints.csv")
        path = [r for r in read(ROOT / trajectory / "by_seed.csv")
                if r["switch_step"] == "100" and r["branch"] == "gd_after_switch"]
        for ax, records, key in ((a, models,"pb_steps"), (b,path,"gd_elapsed")):
            positions, counts = wins(records,key)
            ax.plot(positions,counts, color=color, linewidth=1.5, marker=("o","s")[i], markersize=4,
                    markerfacecolor=(color,"white")[i],label=label)
        for j,switch in enumerate((100,500)):
            group = [r for r in models if int(r["pb_steps"]) == switch]
            xa = j*3 + i*.37
            sym = 1000 * np.mean([float(r["invariant_advantage"]) for r in group])
            orient = 1000 * np.mean([float(r["orientation_term"]) for r in group])
            c.plot([xa,xa],[sym,orient],color=color,lw=1.6)
            c.scatter(xa,orient,color=color,s=37,zorder=3,marker="o")
            c.scatter(xa,sym,facecolor="white",edgecolor=color,s=43,zorder=3,marker="s",lw=1.6)
        rrows = read(ROOT / rate / "models.csv")
        final_rates = {(int(r["pair"]),float(r["gd_rate"])):float(r["gap"]) for r in rrows if r["gd_steps"]=="400"}
        for low_rate, high_rate in ((.5,1),(1,2),(.5,2)):
            pairs = sorted({p for p, _ in final_rates})
            diff = np.array([final_rates[p,high_rate]-final_rates[p,low_rate] for p in pairs])
            lo, hi = np.quantile(diff[draws].mean(axis=1), [.025,.975])
            rate_changes.append(dict(cohort=label,from_rate=low_rate,to_rate=high_rate,
                                     mean_gap_change=diff.mean(),lower=lo,upper=hi,
                                     shifts_toward_region=int((diff<0).sum()),n=len(diff)))
        for j,lr in enumerate(rates):
            for age in (25,100,200,400):
                group = [r for r in rrows if float(r["gd_rate"])==lr and int(r["gd_steps"])==age]
                assert len(group)==16 and len({r["pair"] for r in group})==16
                mean,lo,hi = summarize(group,"gap",draws)
                summaries.append(dict(cohort=label,rate=lr,gd_steps=age,mean_gap=mean,lower=lo,upper=hi,
                                      region_wins=sum(float(r["gap"])<0 for r in group),
                                      orientation_term=np.mean([float(r["orientation_term"]) for r in group]),
                                      region_symmetric_edge=np.mean([float(r["region_symmetric_edge"]) for r in group])))
                if age == 400:
                    anchor = {r["pair"]:r for r in models if r["pb_steps"]=="100"}
                    if lr==2:
                        assert max(abs(float(r["gap"]) - float(anchor[r["pair"]]["gap"])) for r in group)<1e-12
                    d.errorbar(j+(i-.5)*.14,1000*mean,
                               yerr=np.array([[mean-lo],[hi-mean]])*1000,fmt=("o","s")[i],
                               markerfacecolor=(color,"white")[i],color=color,capsize=3,ms=5)
    for ax in (a,b):
        ax.axhline(8,linewidth=1,color="#6B787B",ls="--")
        ax.set_ylim(-.7,16.8)
        ax.set_yticks([0,4,8,12,16])
        ax.set_ylabel("Seeds favoring the region rule (of 16)",fontsize=9)
    a.set_xlabel("Pullback updates before switching to GD",fontsize=9)
    b.set_xlabel("GD updates since the switch",fontsize=9)
    a.legend(loc="lower left", fontsize=8, frameon=True, edgecolor="none", framealpha=.86)
    b.set_xticks([0,50,100,150,200,250,300,350,400])
    c.set(xlim=(-.7,4.1),ylim=(0,6.65),xticks=[.18,3.18],xticklabels=["Switch at 100", "All pullback"])
    c.set_ylabel("Log-loss reduction (× 0.001 nats)",fontsize=9)
    c.scatter([],[],marker="o",color="#586A70",s=29,label="Confidence: orientation benefit")
    c.scatter([],[],marker="s",facecolor="white",edgecolor="#586A70",s=31,label="Region: extra symmetric benefit")
    c.legend(loc="upper right",fontsize=8,frameon=False)
    d.axhline(0,linewidth=1,color="#6B787B",ls="--")
    d.set(xticks=range(3),xticklabels=["0.5","1","2"], xlim=(-.45,2.45))
    d.set_xlabel("GD learning rate",fontsize=9)
    d.set_ylabel("Region loss − confidence loss (× 0.001 nats)",fontsize=9)
    fig.text(.085,.035,"A–B: counts give each seed equal weight.  C–D: loss differences retain the size of each outcome.  Positive differences in D favor confidence.",
             fontsize=9,color="#50636A")
    fig.savefig(ROOT/"switch_time.png",dpi=180,facecolor="white")
    fig.savefig(ROOT/"switch_time.pdf",facecolor="white")
    with (ROOT/"rate_summary.csv").open("w",newline="",encoding="utf8") as stream:
        w=csv.DictWriter(stream, summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)
    with (ROOT/"paired_rate_contrasts.csv").open("w",newline="",encoding="utf8") as stream:
        w=csv.DictWriter(stream, rate_changes[0].keys())
        w.writeheader()
        w.writerows(rate_changes)


if __name__=="__main__":
    main()
