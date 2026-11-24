"""Check every matched training loss and rebuild a sample of policy scores."""
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from clock_run import libraries

HERE = Path(__file__).resolve().parent


def read(path):
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    prior, switch = libraries()
    out = HERE/"match_results"
    rows = read(out/"by_seed.csv")
    clock = {(int(r["pair"]),float(r["rate"])):r for r in read(HERE/"clock_results"/"by_seed.csv")
             if r["gd_steps"]=="400"}
    residuals, anchors, rebuilt = [], [], []
    max_loss_error = 0.
    with threadpool_limits(1), np.load(HERE/"clock_results"/"training_paths.npz") as paths:
        for pair in range(1,33):
            xy,y = prior.rings(121000+pair)
            cal = prior.sample_inputs(123000+pair,8192)
            inputs = prior.sample_inputs(131000+pair,32768)
            tie = np.random.default_rng(132000+pair).random(len(inputs))
            p = prior.posterior(np.linalg.norm(inputs,axis=1))
            for rate in (.5,1.,2.):
                states = paths[f"weights_p{pair}_r{int(10*rate)}"]
                losses = paths[f"loss_p{pair}_r{int(10*rate)}"]
                for row in rows:
                    if (int(row["pair"]),float(row["gd_rate"]))!=(pair,rate):
                        continue
                    step = int(row["first_eligible_step"])
                    alpha = float(row["segment_fraction"])
                    assert 0 <= alpha <= 1
                    target = float(row["target_loss"])
                    assert abs(losses[step]-float(row["actual_crossing_loss"]))<1e-14
                    assert float(row["overshoot"])>=-1e-13
                    t = states[step] if alpha==1. else (1-alpha)*states[step-1]+alpha*states[step]
                    achieved = prior.core.logistic_loss(prior.core.representation(t,xy),y)
                    loss_error = abs(achieved - float(row["achieved_loss"]))
                    if row["variant"]=="equal_loss":
                        loss_error=max(loss_error,abs(target-achieved))
                    assert loss_error<2e-12
                    max_loss_error=max(max_loss_error,loss_error)
                    gap = float(row["gap"])
                    residuals.append(abs(gap-(float(row["orientation_term"])-float(row["region_symmetric_edge"]))))
                    if row["target_key"]=="rate2_at400" and rate==2 and row["variant"]=="actual" and row["matching"]=="first":
                        anchors.append(abs(gap - float(clock[pair,rate]["gap"])))
                    if row["target_key"]=="0.050" and row["matching"]=="first" and pair in (1,17) and row["variant"]=="equal_loss":
                        gate=prior.Gate.fit(cal,prior.with_model(t))
                        values,_=switch.evaluate(t,gate,inputs,p,tie,prior)
                        rebuilt.append(max(abs(float(row[k])-values[k]) for k in
                                           ("raw_loss","confidence_loss","folded_loss","gap")))
    assert len(anchors)==32 and max(anchors)<1e-12
    assert len(rebuilt)==6 and max(rebuilt)<1e-12 and max(residuals)<1e-12
    report=dict(scored_rows=len(rows),maximum_training_loss_reconstruction_error=max_loss_error,
                maximum_gap_identity_error=max(residuals),reference_models_checked=len(anchors),
                reference_gap_max_difference=max(anchors),
                independently_rerouted_sample=len(rebuilt),rerouting_max_loss_difference=max(rebuilt))
    (out/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(report)


if __name__=="__main__":
    main()
