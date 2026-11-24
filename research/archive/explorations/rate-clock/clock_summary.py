"""Summarize nominal-time and equal-update contrasts over matched training seeds."""
from itertools import combinations
import csv
from pathlib import Path
import numpy as np
from clock_run import RATES, ROOT, TIMES, write_rows


FIELDS = ("gap", "train_loss", "raw_loss", "full_loss", "folded_loss", "confidence_loss",
          "orientation_term", "invariant_advantage", "symmetric_gain", "folded_gain",
          "mask_disagreement")
COHORTS = {"original":range(1,17), "independent":range(17,33), "pooled":range(1,33)}


def stats(v, draws):
    lo, hi = np.quantile(v[draws].mean(axis=1), [.025,.975])
    return dict(mean=float(v.mean()), per_seed_median=float(np.median(v)), lower=float(lo),
                upper=float(hi), n_positive=int((v>0).sum()), n_negative=int((v<0).sum()))


def main(out):
    with (out/"by_seed.csv").open(encoding="utf8") as f:
        values={(int(r["pair"]), float(r["rate"]), int(r["tau"])): r for r in csv.DictReader(f)}
    rows, comps, recovery = [],[],[]
    for group, seeds in COHORTS.items():
        draws=np.random.default_rng(291031).integers(0,len(seeds),size=(20000,len(seeds)))
        def vector(rate,t,field):
            return np.array([float(values[i,rate,t][field]) for i in seeds])
        for t in TIMES:
            for field in FIELDS:
                for rate in RATES:
                    v=vector(rate,t,field)
                    rows.append(dict(cohort=group,metric=field,rate=rate,tau=t,steps=int(t/rate),
                                     **stats(v,draws)))
                for r0,r1 in combinations(RATES,2):
                    v0,v1=vector(r0,t,field),vector(r1,t,field)
                    delta=v1-v0
                    comps.append(dict(cohort=group,metric=field,low_rate=r0,high_rate=r1,tau=t,
                                      **stats(delta,draws),
                                      mean_absolute_change=float(abs(delta).mean()),
                                      same_metric_sign=int((np.sign(v0)==np.sign(v1)).sum())))
        for field in FIELDS:
            fixed=vector(2,800,field)-vector(.5,200,field)
            after=vector(2,800,field)-vector(.5,800,field)
            catch=fixed-after
            for name,value in (("400_steps_high_minus_low",fixed),("tau800_high_minus_low",after),
                               ("low_rate_1600_minus_400_steps",catch)):
                recovery.append(dict(cohort=group,metric=field,comparison=name,**stats(value,draws)))
    write_rows(out/"clock_summary.csv",rows)
    write_rows(out/"clock_contrasts.csv",comps)
    write_rows(out/"recovery_contrasts.csv",recovery)


if __name__=="__main__":
    main(ROOT/"clock_results")
