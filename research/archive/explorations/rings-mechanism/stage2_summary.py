"""Paired contrasts, cohort summaries, and endpoint figure for feature metric weights."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


WEIGHTS = (0., 1./9, 1., 9.)
LABELS = ("0", "1/9", "1", "9")
GROUPS = (("full", "native"), ("full", "matched"), ("hybrid", "native"))
TITLES = ("Full pullback", "Pullback, matched logit step", "100 pullback updates → GD")
STEPS = (100, 500)
FIELDS = ("population_loss","radial_excess","angular_excess","validation_loss","train_loss",
          "validation_accuracy","cum_z1","cum_z2","matched_scale_max","match_max_error",
          "max_step_z1","max_step_z2","loss_increases")
PAIR_IDS = list(range(1, 17))
TOL = 1e-11


def weight_index(x):
    matches = np.flatnonzero(np.isclose(float(x), WEIGHTS, rtol=1e-12, atol=1e-15))
    if len(matches) != 1:
        raise ValueError(f"Unknown weight {x}")
    return int(matches[0])


def load_grid(root):
    with (root / "rows.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records, cohorts = {}, {}
    for r in rows:
        i, s, rule, k, t = int(r["pair"]), r["schedule"], r["size_rule"], weight_index(r["weight"]), int(r["step"])
        key = (i,s,rule,k,t)
        assert i in PAIR_IDS and (s,rule) in GROUPS and t in STEPS and key not in records, key
        assert cohorts.setdefault(i,r["cohort"]) == r["cohort"] and r["cohort"] in ("original","fresh")
        assert str(r["grid_converged"]).lower() in ("true","1") and 0 <= float(r["identity_error"]) < 1e-9
        digest = r["state_sha256"]
        assert len(digest) == 64 and int(digest,16) >= 0, key
        values = {f: float(r[f]) for f in (*FIELDS,"grid_error","bayes_loss","identity_error")}
        assert np.all(np.isfinite(list(values.values()))) and values["grid_error"]>=0, key
        assert values["match_max_error"]<1e-9 and min(values["matched_scale_max"],values["match_max_error"])>=0, key
        assert 0<=values["loss_increases"]<=t, key
        assert min(values[f] for f in ("radial_excess","angular_excess","cum_z1","cum_z2")) >= -TOL, key
        residual = abs(values["population_loss"]-values["bayes_loss"]-
                       values["radial_excess"]-values["angular_excess"])
        assert residual <= TOL * max(1, values["population_loss"]), (key,residual)
        assert 0<=values["validation_accuracy"]<=1 and abs(values["bayes_loss"]-.028295290274933)<1e-8
        assert abs(round(values["validation_accuracy"]*256)-values["validation_accuracy"]*256) < 1e-7
        records[key] = dict(state_sha256=digest, **values)
    reconstructed=0
    for i in PAIR_IDS:
        for t in STEPS:
            ref=records[i,"full","native",2,t]
            key=(i,"full","matched",2,t)
            if key not in records:
                records[key]=ref
                reconstructed+=1
            assert ref["state_sha256"] == records[key]["state_sha256"], (i,t,"baseline")
            for f in ("population_loss","train_loss","validation_loss","cum_z1","cum_z2"):
                assert abs(ref[f]-records[key][f])<=TOL*max(1,abs(ref[f]))
        for k in range(4):
            a,b=records[i,"full","native",k,100],records[i,"hybrid","native",k,100]
            assert a["state_sha256"]==b["state_sha256"], (i,k,"shared prefix")
            for f in ("population_loss","train_loss","validation_loss","cum_z1","cum_z2"):
                assert a[f]==b[f], (i,k,f)
            for schedule,rule in GROUPS:
                lo,hi=records[i,schedule,rule,k,100],records[i,schedule,rule,k,500]
                assert hi["cum_z1"]>=lo["cum_z1"]-TOL and hi["cum_z2"]>=lo["cum_z2"]-TOL
    assert len(records)==384 and len(rows)+reconstructed == 384
    assert [list(cohorts.values()).count(c) for c in ("original","fresh")]==[8,8]
    return records,cohorts,len(rows),reconstructed


def resampling(cohorts):
    idx = {k: np.array([i-1 for i in PAIR_IDS if cohorts[i]==k]) for k in ("original","fresh")}
    rng=np.random.default_rng(970021)
    resampled=np.concatenate([rng.choice(indices,size=(20000,len(indices)),replace=True)
                              for indices in idx.values()],axis=1)
    return idx,resampled


def stats(v, idx, resampled):
    means=v[resampled].mean(axis=1)
    return dict(mean=float(v.mean()), seed_sd=float(v.std(ddof=1)), median=float(np.median(v)),
                interval95=list(map(float,np.quantile(means,[.025,.975]))),
                original_mean=float(v[idx["original"]].mean()), fresh_mean=float(v[idx["fresh"]].mean()),
                n_lower=int(sum(v < -TOL)), n_tied=int(sum(abs(v)<=TOL)), n_higher=int(sum(v>TOL)))


def armname(s,rule,t,k):
    return f"{s}_{rule}_step{t}_weight{LABELS[k]}"


def chart(root,summary):
    colors={"bayes_loss":"#c6cbd1","radial_excess":"#13877e","angular_excess":"#dd9655"}
    fig,axes=plt.subplots(2,3,figsize=(11.1,7.2),sharex=True,sharey="row")
    fig.subplots_adjust(left=.09,right=.985,bottom=.17,top=.85,wspace=.08,hspace=.10)
    x=np.arange(4)
    for col,(s,rule) in enumerate(GROUPS):
        top,bot=axes[:,col]
        arms=[summary["arms"][armname(s,rule,500,k)] for k in range(4)]
        delta=[summary["versus_weight1"][armname(s,rule,500,k)]["population_loss"] for k in range(4)]
        base=np.zeros(4)
        for q in colors:
            values=np.array([arm[q]["mean"] for arm in arms])
            top.bar(x,values,bottom=base,color=colors[q],width=.55)
            base+=values
        for j,arm in enumerate(arms):
            lo,hi=arm["population_loss"]["interval95"]
            top.vlines(j,lo,hi,color="black",lw=1.2)
            top.hlines([lo,hi],j-.06,j+.06,color="black",lw=1.2)
        top.set_title(TITLES[col],fontsize=11)
        bot.axhline(0,lw=.9,ls="--",color="#696969")
        for j,d in enumerate(delta):
            lo,hi=d["interval95"]
            bot.vlines(j,lo,hi,color="#366487",lw=1.3)
            bot.hlines([lo,hi],j-.06,j+.06,color="#366487",lw=1.2)
            bot.scatter(j,d["mean"],s=25,color="#244e72",zorder=3)
        for ax in (top,bot):
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.15)
            ax.set_axisbelow(True)
        bot.set(xticks=x,xticklabels=LABELS,xlabel="Second-coordinate weight")
    axes[0,0].set_ylabel("Population logistic loss (nats)")
    axes[1,0].set_ylabel("Paired change versus weight 1")
    fig.suptitle("Does restricting the second feature improve the classifier?\n"
                 "Population loss after 500 updates on concentric rings",fontsize=14)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in
                       zip(colors.values(),("Bayes loss","Radial fit","Angular variation"))],
               loc="lower center",bbox_to_anchor=(.5,.054),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.031,"Means and pointwise 95% bootstrap intervals across 16 seeds, resampled within cohorts.",
             ha="center",fontsize=8)
    for ext in ("png","svg"):
        fig.savefig(root/f"weight_effects.{ext}",dpi=175)
    plt.close(fig)


def main(root):
    records,cohorts,n,reconstructed=load_grid(root)
    idx,draws=resampling(cohorts)
    out=dict(seed_count=16,rows_read=n,reconstructed_matched1_rows=reconstructed,
             logical_cells=len(records),original_pairs=[i+1 for i in idx["original"].tolist()],
             fresh_pairs=[i+1 for i in idx["fresh"].tolist()],bootstrap_replicates=len(draws),
             bootstrap_seed=970021,interval="Pointwise 95% percentile bootstrap, paired seeds resampled within cohorts.",
             contrast="Current weight minus weight 1 within the same schedule, size rule, checkpoint and pair. Lower loss favors current weight.",
             arms={},versus_weight1={},matched_minus_native_at_same_weight={},
             maximum_grid_difference=max(r["grid_error"] for r in records.values()),
             maximum_identity_error=max(r["identity_error"] for r in records.values()),
             maximum_matched_logit_step_error=max(r["match_max_error"] for r in records.values()))
    paired=[]
    for s,rule in GROUPS:
        for t in STEPS:
            for k in range(4):
                name=armname(s,rule,t,k)
                vals=[records[i,s,rule,k,t] for i in PAIR_IDS]
                baseline=[records[i,s,rule,2,t] for i in PAIR_IDS]
                out["arms"][name]={}
                out["versus_weight1"][name]={}
                rows=[dict(pair=i, cohort=cohorts[i], schedule=s, size_rule=rule,step=t,
                           weight=WEIGHTS[k], reference_weight=1) for i in PAIR_IDS]
                for f in (*FIELDS,"bayes_loss"):
                    v=np.array([r[f] for r in vals])
                    d=v-np.array([r[f] for r in baseline])
                    out["arms"][name][f]=stats(v,idx,draws)
                    out["versus_weight1"][name][f]=stats(d,idx,draws)
                    for row,value in zip(rows,d):
                        row[f+"_change"]=float(value)
                paired.extend(rows)
    for t in STEPS:
        for k in range(4):
            name=armname("full","matched",t,k)
            out["matched_minus_native_at_same_weight"][name]={}
            for field in FIELDS:
                delta=np.array([records[i,"full","matched",k,t][field]-
                                records[i,"full","native",k,t][field] for i in PAIR_IDS])
                out["matched_minus_native_at_same_weight"][name][field]=stats(delta,idx,draws)
    with (root/"paired_weight_effects.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    chart(root,out)
    (root/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Checked {n} source rows, {len(records)} logical cells, {len(out['arms'])} arms.")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent/"stage2")
