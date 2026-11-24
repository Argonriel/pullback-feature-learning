"""Sensitivity check using the same first-hit rule at all three learning rates."""
import csv

import numpy as np
from threadpoolctl import threadpool_limits
from matched_loss import HERE, dump, first_hit, posterior, record_match, rings, sample_inputs, summarize


def main():
    directory = HERE/"match_results"
    rows = []
    differences = []
    with (directory/"by_seed.csv").open(encoding="utf8",newline="") as f:
        for source in csv.DictReader(f):
            if source["target_key"]=="rate2_at400" and float(source["gd_rate"])<2.:
                numeric = {k:float(v) for k,v in source.items() if k not in ("target_key","matching","variant")}
                numeric["pair"] = int(numeric["pair"])
                rows.append(dict(**numeric, matching=source["matching"], variant=source["variant"],
                                 target_key="reference_first_hit_all_rates"))
    with threadpool_limits(1), np.load(HERE/"clock_results"/"training_paths.npz") as saved:
        for pair in range(1,33):
            train = rings(121000+pair)
            calibration = sample_inputs(123000+pair,8192)
            x = sample_inputs(131000+pair,32768)
            tie = np.random.default_rng(132000+pair).random(len(x))
            p = posterior(np.linalg.norm(x,axis=1))
            loss, path = saved[f"loss_p{pair}_r20"], saved[f"weights_p{pair}_r20"]
            target = float(loss[400])
            cache = {}
            for name, hold in (("first",0),("sustained",5)):
                step = first_hit(loss,target,hold)
                differences.append(dict(pair=pair,rule=name,gd2_reference_update=400,
                                        gd2_matched_update=step,target_cross_entropy=target))
                if step is not None:
                    record_match(pair,"reference_first_hit_all_rates",target,2.,name,step,
                                 path,loss,train,calibration,x,tie,p,cache,rows)
    summary, contrasts=summarize(rows)
    dump(directory/"reference_first_all_by_rate.csv",summary)
    dump(directory/"reference_first_all_contrasts.csv",contrasts)
    dump(directory/"reference_high_rate_revisits.csv",differences)


if __name__=="__main__":
    main()
