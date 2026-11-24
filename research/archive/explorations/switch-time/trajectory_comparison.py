"""Summarize typical versus extreme seed effects for the two trajectory cohorts."""
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load(group):
    with (ROOT/group/"by_seed.csv").open(newline="",encoding="utf-8") as stream:
        return {(int(row["pair"]),int(row["switch_step"]),int(row["gd_elapsed"]),row["branch"]):row
                for row in csv.DictReader(stream)}


def main():
    results, joins = [], []
    for cohort, path, endpoint_path in (("discovery","trajectory_results","results"),
                                        ("replication","trajectory_validation_results","validation_results")):
        table = load(path)
        with (ROOT/endpoint_path/"endpoints.csv").open(newline="",encoding="utf-8") as stream:
            endpoints = {(int(row["pair"]),int(row["pb_steps"])):row for row in csv.DictReader(stream)}
        fields = {key:key for key in ("raw_loss","full_loss","folded_loss","confidence_loss",
                                       "gap","symmetric_gain","orientation_term","folded_gain","mask_disagreement")}
        fields["folded_symmetric_advantage"] = "invariant_advantage"
        for (pair,switch,elapsed,branch),row in table.items():
            if switch+elapsed != 500:
                continue
            endpoint = endpoints[pair,500 if branch=="continued_pullback" else switch]
            err = max(abs(float(row[k])-float(endpoint[v])) for k,v in fields.items())
            if err>1e-10:
                raise AssertionError((pair,switch,branch,err))
            joins.append(dict(cohort=cohort,pair=pair,switch_step=switch,branch=branch,
                              endpoint_pb_steps=endpoint["pb_steps"],max_absolute_error=err))
        pairs = sorted({key[0] for key in table})
        checkpoints = sorted({(key[1],key[2]) for key in table})
        for switch, elapsed in checkpoints:
            def values(branch, metric, d=elapsed):
                return np.array([float(table[p,switch,d,branch][metric]) for p in pairs])

            metric = values("gd_after_switch","gap")
            own_risk = values("gd_after_switch","raw_loss")
            pb = values("continued_pullback","raw_loss")
            change = own_risk-values("gd_after_switch","raw_loss",0)
            worse = int(np.argmax(change))
            neg = int(np.argmin(metric))
            pos = int(np.argmax(metric))
            seed_draws = np.random.default_rng(291031).integers(0,len(pairs),size=(20000,len(pairs)))
            ci_low,ci_high=np.quantile(metric[seed_draws].mean(axis=1),[.025,.975])
            results.append(dict(cohort=cohort,switch_step=switch,gd_elapsed=elapsed,total_step=switch+elapsed,
                                mean_gap=metric.mean(),median_gap=np.median(metric),ci_low=ci_low,
                                ci_high=ci_high,confidence_wins=int((metric>0).sum()),
                                folded_wins=int((metric<0).sum()),min_gap=metric[neg],
                                min_gap_pair=pairs[neg],max_gap=metric[pos],max_gap_pair=pairs[pos],
                                mean_orientation_term=values("gd_after_switch","orientation_term").mean(),
                                mean_folded_symmetric_advantage=values("gd_after_switch","folded_symmetric_advantage").mean(),
                                pb_control_mean_gap=values("continued_pullback","gap").mean(),
                                mean_raw=own_risk.mean(),median_raw=np.median(own_risk),mean_control_raw=pb.mean(),
                                mean_raw_change_since_switch=change.mean(),median_raw_change_since_switch=np.median(change),
                                increased_raw_pairs=int((change>0).sum()),max_raw_increase=change[worse],
                                max_raw_increase_pair=pairs[worse],outlying_pair_start_risk=values("gd_after_switch","raw_loss",0)[worse],
                                outlying_pair_current_risk=own_risk[worse]))
    with (ROOT/"trajectory_validation_results"/"cohort_comparison.csv").open("w",encoding="utf-8",newline="") as fp:
        wr=csv.DictWriter(fp,fieldnames=results[0].keys())
        wr.writeheader()
        wr.writerows(results)
    with (ROOT/"trajectory_validation_results"/"endpoint_joins.csv").open("w",encoding="utf-8",newline="") as fp:
        wr=csv.DictWriter(fp,fieldnames=joins[0].keys())
        wr.writeheader()
        wr.writerows(joins)


if __name__=="__main__":
    main()
