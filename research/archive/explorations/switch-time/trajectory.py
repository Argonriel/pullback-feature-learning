"""Follow two optimizer switches against same-age uninterrupted pullback controls."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core


ROOT = Path(__file__).resolve().parent
SWITCHES = (100, 350)
OFFSETS = (0, 1, 2, 5, 10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400)
LIMIT = 500
MEMBERS = ("raw_loss", "full_loss", "folded_loss", "confidence_loss", "gap",
           "folded_gain", "symmetric_gain", "orientation_term",
           "folded_symmetric_advantage", "mask_disagreement")


def prior_default():
    for name in ("budgeted-rotation", "budgeted-rotation-final"):
        path = ROOT.parent / name
        if (path / "supplemental_confidence.py").is_file():
            return path
    raise FileNotFoundError("Pass --prior with the path to the preceding allocation study.")


def csv_rows(path, rows):
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(values, draws):
    means = values[draws].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def summary(rows, pairs):
    n = len(pairs)
    draws = np.random.default_rng(291031).integers(0, n, size=(20000, n))
    by_group = {}
    for row in rows:
        key = row["switch_step"], row["gd_elapsed"], row["branch"]
        by_group.setdefault(key, []).append(row)
    groups, comparisons = [], []
    for (switch, delay, branch), group in by_group.items():
        assert [r["pair"] for r in group] == list(pairs)
        margin = np.array([r["gap"] for r in group])
        low, high = paired_interval(margin, draws)
        groups.append(dict(switch_step=switch, gd_elapsed=delay, total_step=switch+delay,
                           branch=branch, **{metric: float(np.mean([r[metric] for r in group]))
                                             for metric in MEMBERS},
                           ci_low=low, ci_high=high, median_gap=float(np.median(margin)),
                           median_raw_loss=float(np.median([r["raw_loss"] for r in group])),
                           confidence_wins=int((margin>0).sum()),
                           folded_wins=int((margin<0).sum())))
        if branch == "gd_after_switch":
            matched = by_group[switch, delay, "continued_pullback"]
            effect = margin - [r["gap"] for r in matched]
            effect_low, effect_high = paired_interval(effect, draws)
            comparisons.append(dict(switch_step=switch,gd_elapsed=delay,total_step=switch+delay,
                                    switched_minus_continued_gap=float(effect.mean()),
                                    ci_low=effect_low,ci_high=effect_high,
                                    smaller_gap_on_gd_pairs=int((effect<0).sum())))
    return groups, comparisons


def audit_prior(rows, prior):
    with (prior / "supplemental_results" / "models.csv").open(encoding="utf-8", newline="") as stream:
        prior_rows = {(int(r["pair"]),r["model"]):r for r in csv.DictReader(stream)}
    checks = []
    for row in rows:
        if row["switch_step"] != 100 or row["total_step"] != 500:
            continue
        key = (row["pair"], "hybrid" if row["branch"] == "gd_after_switch" else "pb")
        err = max(abs(row[k] - float(prior_rows[key][k])) for k in
                  ("raw_loss","full_loss","folded_loss","confidence_loss"))
        checks.append(err)
    if len(checks) != 32 or max(checks)>1e-10:
        raise AssertionError("Shared step-500 endpoints failed to reproduce the prior study.")
    return max(checks)


def run(prior, out, pairs):
    sys.path.insert(0, str(prior))
    from gate import Gate, gap
    from generator import rings
    from run import CAL, EVAL, advance, posterior, sample_inputs, nll, with_model
    from supplemental_confidence import route

    out.mkdir(parents=True,exist_ok=False)
    ages = {s: [d for d in OFFSETS if s+d<=LIMIT] for s in SWITCHES}
    pb_ages = sorted({s+d for s,v in ages.items() for d in v})
    rows = []
    for pair in pairs:
        data = rings(121000 + pair)
        cal = sample_inputs(123000+pair, CAL)
        inputs = sample_inputs(131000+pair, EVAL)
        tie = np.random.default_rng(132000+pair).random(EVAL)
        label_probability = posterior(np.linalg.norm(inputs, axis=1))

        def evaluate(state):
            predict = with_model(state)
            table = Gate.fit(cal,predict)
            first, opposite = predict(inputs), predict(-inputs)
            mid = (first + opposite) / 2
            left = nll(first,label_probability)
            right = nll(opposite,label_probability)
            averaged = nll(mid,label_probability)
            single = (left+right)/2
            benefit = gap(first,opposite)
            folded = route(table.priority(inputs,"folded"),tie)
            plus = route(-abs(first),tie)
            minus = route(-abs(opposite),tie)
            both = (plus.astype(int) + minus) / 2
            difference = plus.astype(int) - minus
            symmetric = both * benefit
            signed = difference * (left-right) / 4
            folded_benefit = folded * benefit
            folded_risk = np.where(folded,averaged,single).mean()
            selective = (np.where(plus,averaged,left)+np.where(minus,averaged,right))/2
            confidence_risk = selective.mean()
            np.testing.assert_allclose(single-selective,symmetric+signed,atol=2e-12,rtol=1e-10)
            loss_gap = folded_risk-confidence_risk
            np.testing.assert_allclose(loss_gap, np.mean(symmetric+signed-folded_benefit),atol=2e-12)
            return dict(raw_loss=single.mean(),full_loss=averaged.mean(),folded_loss=folded_risk,
                        confidence_loss=confidence_risk,gap=loss_gap,folded_gain=folded_benefit.mean(),
                        symmetric_gain=symmetric.mean(),orientation_term=signed.mean(),
                        folded_symmetric_advantage=np.mean(folded_benefit-symmetric),
                        mask_disagreement=np.mean(plus!=minus))

        theta = core.initialize_parameters(122000+pair)
        pb_scores, switch_weights = {}, {}
        previous = 0
        for age in pb_ages:
            theta = advance(theta,data,"pullback",8,age-previous)
            pb_scores[age] = evaluate(theta)
            if age in SWITCHES:
                switch_weights[age] = theta.copy()
            previous = age
        for switch in SWITCHES:
            continuation = switch_weights[switch]
            last_delay = 0
            for delay in ages[switch]:
                pullback = pb_scores[switch+delay]
                continuation = advance(continuation,data,"gd",2,delay-last_delay)
                after = evaluate(continuation) if delay else pullback
                for branch, result in (("continued_pullback", pullback),("gd_after_switch", after)):
                    rows.append(dict(pair=pair,switch_step=switch,gd_elapsed=delay,total_step=switch+delay,
                                     branch=branch,**result))
                last_delay = delay
        print(f"Finished trajectory pair {pair} ({pair-pairs.start+1} of {len(pairs)})",flush=True)
    max_difference = audit_prior(rows,prior) if list(pairs)==list(range(1,17)) else None
    grouped, comparisons = summary(rows,pairs)
    csv_rows(out/"by_seed.csv",rows)
    csv_rows(out/"by_age.csv",grouped)
    csv_rows(out/"paired_branch_comparisons.csv",comparisons)
    with (out/"metadata.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(pairs=list(pairs),switch_steps=SWITCHES,offsets_by_switch=ages,
                       calibration_inputs_per_checkpoint=CAL,test_inputs_per_pair=EVAL,
                       matching="Same initialized training trajectory to the fork, training samples, test samples, calibration samples, ties, and evaluation budget in each pair.",
                       interval="Pointwise paired 95% percentile bootstrap, 20000 replicates, seed 291031; each resample draws the full set of training pairs.",
                       interpretation="Positive folded-minus-confidence loss favors confidence. GD elapsed is also used for the matched pullback branch and denotes updates since the fork.",
                       prior_endpoint_max_difference=max_difference),stream,indent=2)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior",type=Path,help="Folder for the preceding allocation experiment")
    parser.add_argument("--out",type=Path,default=ROOT/"trajectory_results")
    parser.add_argument("--first-pair",type=int,default=1)
    parser.add_argument("--last-pair",type=int,default=16)
    args=parser.parse_args()
    if args.first_pair<1 or args.first_pair>args.last_pair:
        parser.error("Pair indices must start at one or greater and form a nonempty interval.")
    with threadpool_limits(1):
        run(args.prior or prior_default(),args.out,range(args.first_pair,args.last_pair+1))
