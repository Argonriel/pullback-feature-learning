"""Paired bootstrap summaries of the fixed 500-update switch-time experiments."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESAMPLES = 20000
BOOTSTRAP_SEED = 472103
METRICS = ("gap", "orientation_term", "invariant_advantage", "symmetric_gain", "folded_gain",
           "mask_disagreement", "raw_loss", "full_loss", "folded_loss", "confidence_loss")


def estimates(values, indices):
    lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return tuple(map(float, (values.mean(), lo, hi)))


def main():
    cohort_rows = {}
    for label, folder in (("original", "results"), ("validation", "validation_results")):
        with (ROOT / folder / "endpoints.csv").open(encoding="utf8", newline="") as f:
            cohort_rows[label] = list(csv.DictReader(f))
    cohort_rows["pooled"] = cohort_rows["original"] + cohort_rows["validation"]
    summaries, changes, anchor_changes = [], [], []
    for group, rows in cohort_rows.items():
        seeds = sorted({int(r["pair"]) for r in rows})
        switches = sorted({int(r["pb_steps"]) for r in rows})
        lookup = {(int(r["pair"]), int(r["pb_steps"])): r for r in rows}
        indices = np.random.default_rng(BOOTSTRAP_SEED).integers(len(seeds), size=(RESAMPLES,len(seeds)))
        assert len(lookup) == len(rows) == len(seeds)*len(switches)
        gaps = np.array([[float(lookup[n,s]["gap"]) for s in switches] for n in seeds])
        for i,switch in enumerate(switches):
            row = dict(cohort=group, pb_steps=switch, gd_steps=500-switch, n_pairs=len(seeds),
                       gap_median=float(np.median(gaps[:,i])),
                       confidence_wins=int((gaps[:,i]>0).sum()), folded_wins=int((gaps[:,i]<0).sum()))
            for metric in METRICS:
                v = np.array([float(lookup[n,switch][metric]) for n in seeds])
                row[metric], row[metric+"_lo"], row[metric+"_hi"] = estimates(v, indices)
            summaries.append(row)
        for j,k in zip(range(len(switches)-1), range(1,len(switches))):
            delta = gaps[:,k]-gaps[:,j]
            mean,lo,hi = estimates(delta, indices)
            changes.append(dict(cohort=group, from_pb_steps=switches[j], to_pb_steps=switches[k],
                                gap_change=mean, pointwise_lo=lo, pointwise_hi=hi,
                                confidence_to_folded=int(((gaps[:,j]>0)&(gaps[:,k]<0)).sum()),
                                folded_to_confidence=int(((gaps[:,j]<0)&(gaps[:,k]>0)).sum())))
        for metric in ("gap", "orientation_term", "invariant_advantage", "mask_disagreement"):
            differences = np.array([float(lookup[n,500][metric])-float(lookup[n,100][metric]) for n in seeds])
            mean,lo,hi = estimates(differences, indices)
            anchor_changes.append(dict(cohort=group, metric=metric, from_pb_steps=100, to_pb_steps=500,
                                       mean_within_pair_change=mean, pointwise_lo=lo, pointwise_hi=hi,
                                       positive_changes=int(np.sum(differences>0)), n_pairs=len(seeds)))
    for name,rows in (("endpoint_curve.csv", summaries), ("adjacent_endpoint_changes.csv", changes),
                      ("paired_anchor_contrasts.csv", anchor_changes)):
        with (ROOT/name).open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f,list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (ROOT/"endpoint_summary_metadata.json").open("w", encoding="utf8") as f:
        json.dump(dict(bootstrap_seed=BOOTSTRAP_SEED, resamples=RESAMPLES,
                       ci="Pointwise 95% paired percentile bootstrap, resampling full seed vectors.",
                       units="Log losses and gains in nats per example. Mask disagreement is a fraction.",
                       signed_gap="Folded loss minus confidence loss. Negative favors the folded rule.",
                       groups={"original":"1-16","validation":"17-32, independent of the original study",
                               "pooled":"1-32, descriptive combined estimate"}), f, indent=2)
    for r in summaries:
        print(f"{r['cohort']:>10} {r['pb_steps']:4} {r['gap']:+.6f} [{r['gap_lo']:+.6f}, {r['gap_hi']:+.6f}] "
              f"folded {r['folded_wins']}/{r['n_pairs']} orientation={r['orientation_term']:.6f} "
              f"invariant advantage={r['invariant_advantage']:.6f}")


if __name__ == "__main__":
    main()
