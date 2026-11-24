"""Compare post-switch second-view routing at matched training cross-entropy."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PRIOR = next(p for p in (HERE.parent / "switch-time", HERE.parent / "switch-time-final")
             if (p / "rate_control.py").exists())
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))
from rate_control import core, measure, posterior, rings, sample_inputs

RATES = (.5, 1., 2.)
LEVELS = (.2, .1, .075, .05, .04, .03, .02)


def dump(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        wr = csv.DictWriter(stream, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def first_hit(loss, target, hold):
    if loss[0] < target:
        return None
    for step in np.flatnonzero((loss[1:] <= target) & (loss[:-1] > target)) + 1:
        if step + hold < len(loss) and np.all(loss[step:step + hold + 1] <= target):
            return int(step)
    return None


def train_loss(theta, x, y):
    return core.logistic_loss(core.representation(theta, x), y)


def interpolate(path, step, target, x, y):
    lo, hi = 0., 1.
    prev, end = path[step-1], path[step]
    actual = train_loss(end, x, y)
    if abs(actual - target) < 1e-13:
        return end, actual, 1.
    before = train_loss(prev, x, y)
    assert before > target and actual < target
    for _ in range(55):
        mid = (lo + hi) / 2
        state = prev + mid * (end - prev)
        loss = train_loss(state, x, y)
        if abs(loss-target) < 2e-14:
            return state, loss, mid
        if loss > target:
            lo = mid
        else:
            hi = mid
    assert abs(loss - target) < 1e-12
    return state, loss, mid


def record_match(pair, target_key, target, rate, rule, step, path, losses, data,
                 calibration, inputs, tie, p, cache, rows):
    base = dict(pair=pair, target_key=target_key, target_loss=target, gd_rate=rate,
                matching=rule, first_eligible_step=step, achieved_nominal_time=rate*step,
                actual_crossing_loss=losses[step], overshoot=target-losses[step])
    for variant in ("actual", "equal_loss"):
        key = (rate, step, target if variant=="equal_loss" else None)
        if key not in cache:
            if variant == "actual" or (target_key=="rate2_at400" and rate==2):
                theta, loss, portion = path[step], train_loss(path[step], *data), 1.
                assert abs(loss - losses[step]) < 3e-14
            else:
                theta, loss, portion = interpolate(path,step,target,*data)
            scored = measure(theta, calibration, inputs, tie, p)
            cache[key] = dict(achieved_loss=loss, segment_fraction=portion,
                              **{k:scored[k] for k in ("raw_loss","folded_loss","confidence_loss",
                                                      "gap","orientation_term","region_symmetric_edge")})
        result = cache[key]
        if variant == "equal_loss":
            assert abs(result["achieved_loss"]-target) < 1e-12
        rows.append(dict(**base, variant=variant, **result))


def summarize(rows):
    summary, contrasts = [], []
    rng = np.random.default_rng(501111)
    boot = {}
    for level in dict.fromkeys(row["target_key"] for row in rows):
        for rule in ("first","sustained"):
            for variant in ("actual","equal_loss"):
                lookup = {(r["pair"],r["gd_rate"]):r for r in rows if
                          (r["target_key"],r["matching"],r["variant"]) == (level,rule,variant)}
                for cohort, seed_range in (("original",range(1,17)),("independent",range(17,33)),("combined",range(1,33))):
                    complete = [s for s in seed_range if all((s,rate) in lookup for rate in RATES)]
                    common = dict(target_key=level, matching=rule,variant=variant,cohort=cohort,
                                  total_seeds=len(seed_range),complete_seeds=len(complete))
                    if not complete:
                        continue
                    n = len(complete)
                    if n not in boot:
                        boot[n] = rng.integers(0,n,(20000,n))
                    gaps_by_rate = {}
                    for rate in RATES:
                        grp = [lookup[s,rate] for s in complete]
                        gap = np.array([r["gap"] for r in grp])
                        gaps_by_rate[rate] = gap
                        lc,uc = np.quantile(gap[boot[n]].mean(axis=1), [.025,.975])
                        summary.append(dict(**common,gd_rate=rate,region_wins=int((gap<0).sum()),
                                            mean_gap=gap.mean(),median_gap=np.median(gap),gap_lo=lc,gap_hi=uc,
                                            mean_loss=np.mean([r["achieved_loss"] for r in grp]),
                                            mean_overshoot=np.mean([r["overshoot"] for r in grp]),
                                            maximum_overshoot=max(r["overshoot"] for r in grp),
                                            median_steps=np.median([r["first_eligible_step"] for r in grp]),
                                            max_absolute_matching_error=max(abs(r["achieved_loss"]-r["target_loss"]) for r in grp)))
                    for lo,hi in ((.5,1.),(1.,2.),(.5,2.)):
                        delta = gaps_by_rate[hi] - gaps_by_rate[lo]
                        lc,uc = np.quantile(delta[boot[n]].mean(axis=1), [.025,.975])
                        contrasts.append(dict(**common,low_rate=lo,high_rate=hi,mean_gap_change=delta.mean(),
                                              median_gap_change=np.median(delta),ci_lo=lc,ci_hi=uc,
                                              toward_region=int((delta<0).sum())))
    return summary, contrasts


def main(paths_file, out):
    if not paths_file.exists():
        raise FileNotFoundError(f"Run the nominal-time study first: {paths_file}.")
    out.mkdir(parents=True,exist_ok=False)
    rows, coverage = [], []
    with threadpool_limits(1), np.load(paths_file) as data_arrays:
        for pair in range(1,33):
            train = rings(121000+pair)
            calibration = sample_inputs(123000+pair,8192)
            inputs = sample_inputs(131000+pair,32768)
            tie = np.random.default_rng(132000+pair).random(len(inputs))
            p = posterior(np.linalg.norm(inputs,axis=1))
            losses = {lr: data_arrays[f"loss_p{pair}_r{int(10*lr)}"] for lr in RATES}
            paths = {lr: data_arrays[f"weights_p{pair}_r{int(10*lr)}"] for lr in RATES}
            for lr in RATES:
                assert len(paths[lr])==len(losses[lr])==1+int(1600/lr)
                assert np.array_equal(paths[lr][0],paths[2.][0])
            cache = {}
            levels = [(f"{t:.3f}",t) for t in LEVELS]+[("rate2_at400",float(losses[2.][400]))]
            for level,target in levels:
                for rule in ("first","sustained"):
                    for lr in RATES:
                        fixed = level=="rate2_at400" and lr==2.
                        hold = int(10/lr) if rule=="sustained" else 0
                        step = (400 if losses[lr][0]>=target else None) if fixed else first_hit(losses[lr],target,hold)
                        coverage.append(dict(pair=pair,gd_rate=lr,target_key=level,target_loss=target,
                                             matching=rule,reached=int(step is not None),
                                             first_eligible_step=step if step is not None else ""))
                        if step is not None:
                            record_match(pair,level,target,lr,rule,step,paths[lr],losses[lr],
                                         train,calibration,inputs,tie,p,cache,rows)
            print(f"Matched training loss for pair {pair}/32",flush=True)
    dump(out/"by_seed.csv",rows)
    dump(out/"coverage.csv",coverage)
    summaries, contrasts = summarize(rows)
    dump(out/"by_rate.csv",summaries)
    dump(out/"contrasts.csv",contrasts)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths",type=Path,default=HERE/"clock_results"/"training_paths.npz")
    parser.add_argument("--out",type=Path,default=HERE/"match_results")
    args = parser.parse_args()
    main(args.paths,args.out)
