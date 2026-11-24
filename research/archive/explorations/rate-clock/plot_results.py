"""Publication figure for the routing comparisons with different training clocks."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parent
RATES=(.5,1.,2.)
COLORS=("#14685F","#C55F36","#334E77")


def read(path):
    with path.open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def main():
    timed=read(ROOT/"clock_results"/"clock_summary.csv")
    wins={(float(r["rate"]),int(r["tau"])):int(r["n_negative"]) for r in timed
          if r["cohort"]=="pooled" and r["metric"]=="gap"}
    fitted=read(ROOT/"match_results"/"reference_first_all_by_rate.csv")
    loss_wins={float(r["gd_rate"]):int(r["region_wins"]) for r in fitted if
               (r["target_key"],r["matching"],r["variant"],r["cohort"])==
               ("reference_first_hit_all_rates","first","equal_loss","combined")}
    clock=sorted({t for rate,t in wins})
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    fig,(a,b)=plt.subplots(1,2,figsize=(11.5,5.35),gridspec_kw={"width_ratios":[1.04,1]})
    fig.subplots_adjust(left=.075,right=.975,top=.73,bottom=.22,wspace=.20)
    fig.text(.075,.925,"Does the smaller GD learning rate catch up?",fontsize=20,fontweight="bold",color="#1D3739")
    fig.text(.075,.86,"Models in which region-based routing gives lower log loss than first-view confidence, out of 32 paired seeds",
             fontsize=10.6,color="#3E5559")
    for ax,title in ((a,"A   Three ways to compare the rates"),(b,"B   Follow each rate as training continues")):
        ax.set_title(title,loc="left",fontweight="bold",fontsize=11.4,pad=14,color="#223F43")
        ax.set_ylim(-.8,36)
        ax.set_yticks([0,8,16,24,32])
        ax.grid(axis="y",color="#DBE4E4",lw=.8)
        ax.axhline(16,ls="--",color="#879394",lw=.9)
        ax.set_axisbelow(True)
        ax.spines[["left","bottom"]].set_color("#AAB9B7")
        ax.tick_params(colors="#415659",labelsize=9)
    a.set_ylabel("Number of models",fontsize=10)
    x=np.arange(3)
    for i,(rate,color,marker) in enumerate(zip(RATES,COLORS,("o","s","^"))):
        counts=[wins[rate,int(400*rate)],wins[rate,800],loss_wins[rate]]
        bars=a.bar(x+(i-1)*.23,counts,.21,label=f"GD rate {rate:g}",color=color)
        a.bar_label(bars,padding=2,fontsize=9,color="#263D3E")
        b.plot(clock,[wins[rate,t] for t in clock],color=color,lw=2,marker=marker,
               markerfacecolor=("white" if i==1 else color),ms=5,label=f"GD rate {rate:g}")
    a.set_xticks(x,["400 updates\nat each rate","Same nominal\ntime (800)","Same training loss\nas rate 2 at 400"])
    b.set_xticks([0,400,800,1200,1600])
    b.set_xlabel("Nominal GD time (learning rate × updates)",fontsize=10)
    b.legend(loc="lower right",frameon=True,framealpha=.92,edgecolor="none",fontsize=8.7)
    fig.text(.075,.071,"The loss target is the rate-2 training loss at update 400 within each seed. Each rate is scored at its first crossing of that target.",
             fontsize=9,color="#4F6567")
    fig.savefig(ROOT/"rate_clock.png",dpi=180,facecolor="white")
    fig.savefig(ROOT/"rate_clock.pdf",facecolor="white")


if __name__=="__main__":
    main()
