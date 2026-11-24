"""Paired estimates for budgeted half-turn averaging, using frozen evaluation rows."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


MODELS = ("gd", "pb", "hybrid")
NAMES = {"gd": "GD", "pb": "Pullback", "hybrid": "Pullback → GD"}
RULES = ("raw", "random", "boundary", "radius", "folded", "full")
BOOTSTRAP_SEED = 291031
N_BOOT = 20_000


def read_models(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    groups = {m: {} for m in MODELS}
    for row in records:
        model, pair = row["model"], int(row["pair"])
        assert model in groups and pair not in groups[model], (model, pair)
        values = {r: float(row[r + "_loss"]) for r in RULES}
        assert all(np.isfinite(v) and v >= 0 for v in values.values()), row
        assert int(float(row["n_cal"])) == 8192 and int(float(row["n_eval"])) == 32768
        frac = float(row["budget"])
        assert frac == .25 and int(row['executed_extra_per_limited_policy']) == 8192, row
        assert max(float(row['max_identity_error']),float(row['max_deploy_error'])) < 1e-10, row
        assert np.isclose(values["random"], .75 * values["raw"] + .25 * values["full"],
                          atol=1e-10, rtol=1e-10), row
        # Under the half-turn invariant gate and antithetic evaluation, the
        # budgeted gain is a subset of the gain available from treating all pairs.
        assert all(values["full"] - 1e-10 <= v <= values["raw"] + 1e-10 for v in values.values()), row
        groups[model][pair] = values
    pairs = sorted(groups["gd"])
    assert len(pairs) == 16 and all(set(groups[m]) == set(pairs) for m in MODELS)
    return pairs, groups


def verify_batches(path, groups):
    batches = {(m, p): {} for m in MODELS for p in groups[m]}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row['model'], int(row['pair'])
            batch = int(row['batch'])
            assert batch not in batches[key], (key, batch)
            batches[key][batch] = [float(row[rule+'_loss']) for rule in RULES]
    for (m, p), values in batches.items():
        assert set(values) == set(range(1,129)), (m,p)
        observed = np.mean(list(values.values()), axis=0)
        recorded = [groups[m][p][rule] for rule in RULES]
        np.testing.assert_allclose(observed, recorded, rtol=1e-12, atol=1e-12)
    return sum(len(values) for values in batches.values())


def estimate(values, draws):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1)),
                median=float(np.median(values)),
                ci95=list(map(float, np.quantile(values[draws].mean(axis=1), [.025, .975]))))


def contrast(values, draws):
    return dict(**estimate(values, draws), positive_pairs=int(np.sum(values > 1e-12)),
                negative_pairs=int(np.sum(values < -1e-12)), tied_pairs=int(np.sum(np.abs(values) <= 1e-12)))


def ratio(numer, denom, draws):
    # Each arm retains equal weight in both means; seeds with near-zero available
    # gain do not acquire excessive weight through individual percentage division.
    n, d = float(np.mean(numer)), float(np.mean(denom))
    d_boot = denom[draws].mean(axis=1)
    if d <= 0 or np.any(d_boot <= 0):
        return dict(estimate=None, ci95=None)
    boot = numer[draws].mean(axis=1) / d_boot
    return dict(estimate=n/d, ci95=list(map(float, np.quantile(boot, [.025, .975]))))


def summarize(groups, pairs):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(pairs), size=(N_BOOT, len(pairs)))
    summary = dict(pairs=pairs, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=N_BOOT,
                   ci_method="Pointwise 95% percentile interval. Resample full paired model/data seeds.",
                   primary="random_loss minus folded_loss on full pullback at step 500",
                   units="Population logistic loss per example, in nats.",
                   fraction_definition="Mean reduction relative to raw divided by mean reduction when all inputs receive the second view.",
                   models={})
    paired = []
    for m in MODELS:
        loss = {k: np.asarray([groups[m][p][k] for p in pairs]) for k in RULES}
        available = loss["raw"] - loss["full"]
        arms = {}
        for rule in RULES:
            gain = loss["raw"] - loss[rule]
            extra = loss["random"] - loss[rule]
            arms[rule] = dict(loss=estimate(loss[rule], draws), gain_over_raw=contrast(gain, draws),
                              relative_gain_over_raw=ratio(gain, loss["raw"], draws),
                              fraction_of_full_gain=ratio(gain, available, draws),
                              additional_gain_over_random=contrast(extra, draws))
            for i, p in enumerate(pairs):
                paired.append(dict(pair=p, model=m, rule=rule, population_loss=loss[rule][i],
                                   absolute_gain_over_raw=gain[i],
                                   relative_gain_over_raw=gain[i]/loss["raw"][i] if loss["raw"][i] else "",
                                   full_gain_available=available[i],
                                   fraction_of_full_gain=gain[i]/available[i] if available[i] > 1e-12 else "",
                                   additional_gain_over_random=extra[i]))
        compared = {f"folded_vs_{other}": contrast(loss[other]-loss["folded"], draws)
                    for other in ("random", "radius", "boundary")}
        calibration_cost = 2*8192
        deployment_examples = 32768
        matched_random_quota = .25 + calibration_cost/deployment_examples
        matched_random = loss["raw"] - matched_random_quota * available
        share = arms["folded"]["fraction_of_full_gain"]
        surplus = share["estimate"] - .25
        amortization = dict(
            pricing="Counts each network example-forward, including both calibration views. Assumes the same deployment horizon per model. Lookup and batch routing are outside the forward count.",
            calibration_forwards=calibration_cost,
            illustrative_horizon=deployment_examples,
            uniform_extra_view_fraction_at_equal_total_cost=matched_random_quota,
            folded_advantage_over_equal_total_cost_uniform=contrast(matched_random-loss["folded"], draws),
            break_even_future_examples_for_mean_gains=calibration_cost/surplus if surplus>0 else None,
            break_even_ci95=[calibration_cost/(f-.25) for f in reversed(share["ci95"])] if share["ci95"][0]>.25 else None)
        summary["models"][m] = dict(arms=arms, folded_advantage=compared, amortization=amortization)
    return summary, paired


def figure(root, summary, paired):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.7, 4.8),
                                      gridspec_kw={"width_ratios": [1.0, 1.15]})
    fig.subplots_adjust(left=.095, right=.985, top=.78, bottom=.26, wspace=.37)
    fig.suptitle("Allocating a second view to 25% of inputs", fontsize=17, y=.97)
    model_x = np.arange(3)
    blues = ["#586e8b", "#097c77", "#b65635"]
    for k,(m,c) in enumerate(zip(MODELS,blues)):
        vals = [r["additional_gain_over_random"] for r in paired if r["model"] == m and r["rule"] == "folded"]
        jitter = np.linspace(-.115,.115,len(vals))
        left.scatter(k+jitter, vals, s=16, color=c, alpha=.43, linewidth=0, zorder=2)
        metrics = summary["models"][m]["folded_advantage"]["folded_vs_random"]
        lo,hi = metrics["ci95"]
        left.vlines(k,lo,hi,color=c,lw=2,zorder=3)
        left.scatter(k,metrics["mean"],marker="D",s=48,color=c,edgecolors="white",linewidths=.7,zorder=4)
    left.axhline(0,color="#777",lw=.9,linestyle="--")
    left.set(xlim=(-.45,2.45), xticks=model_x, xticklabels=[NAMES[m] for m in MODELS],
             title="Extra gain over random allocation", ylabel="Population loss saved (nats)")
    right.axhline(.25, label="Random baseline (25%)", color="#555", lw=1.1, linestyle="--")
    for shift,rule,color,label,marker in [(-.19,"boundary","#949eaa","Boundary","s"),
                                          (0.,"radius","#a0613e","Radius","o"),
                                          (.19,"folded","#13786f","Folded angle + radius","D")]:
        for k,m in enumerate(MODELS):
            val = summary["models"][m]["arms"][rule]["fraction_of_full_gain"]
            if val["estimate"] is None:
                continue
            x = k+shift
            right.vlines(x,*val["ci95"],color=color,lw=1.6)
            right.scatter(x,val["estimate"],s=40,color=color,marker=marker,
                          label=label if k==0 else None,zorder=3)
    right.set(xlim=(-.50,2.50),ylim=(-.04,1.04),xticks=model_x,xticklabels=[NAMES[m] for m in MODELS],
              title="Share of the all-input gain captured",ylabel="Fraction of full two-view gain")
    right.yaxis.set_major_formatter(PercentFormatter(1.))
    for axis in (left, right):
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y",alpha=.15)
        axis.tick_params(axis="x",labelsize=9.2)
    fig.legend(*right.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.065),
               ncols=4, frameon=False, fontsize=9)
    fig.text(.5,.025,"Left dots show 16 seed pairs. Vertical bars give 95% seed-bootstrap intervals. The right panel compares pooled mean gains.",
             ha="center", fontsize=8.5)
    for ext in ("png","svg"):
        fig.savefig(root/f"allocation_results.{ext}",dpi=185)
    plt.close(fig)


def main(root):
    pairs, groups = read_models(root / "models.csv")
    summary, paired = summarize(groups, pairs)
    summary['verified_batches'] = verify_batches(root / 'batches.csv', groups)
    (root/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    with (root/"paired_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    figure(root,summary,paired)
    print(f"Verified {len(pairs)*len(MODELS)} model records and wrote {len(paired)} paired metrics.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path(__file__).resolve().parent/"results")
    main(parser.parse_args().results)
