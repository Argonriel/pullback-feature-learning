"""Rebuild selective risks and paired bands independently of the experiment scripts."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
STEPS = (0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 475, 500)
FIELDS = ("raw_loss", "full_loss", "folded_loss", "confidence_loss", "gap",
          "folded_gain", "symmetric_gain", "orientation_term", "mask_disagreement")
EVAL = 32768
TOL = 3e-11


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def draw(seed, n):
    rng = np.random.default_rng(seed)
    r = rng.choice([0.5, 1.2], n)
    theta = rng.uniform(0, 2*np.pi, n)
    mean = r[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    return mean + rng.normal(0, 0.15, (n, 2))


def predict(state, x):
    h = x
    for layer in state.reshape(2, 40):
        weights = layer[:16].reshape(8, 2)
        readout = layer[16:32].reshape(2, 8)
        hidden = np.tanh(h @ weights.T + layer[32:])
        h = h + (0.5 * hidden) @ readout.T
    return h[:, 0]


def cells(x, radial_cuts):
    radius = np.hypot(x[:, 0], x[:, 1])
    orientation = np.arctan2(x[:, 1], x[:, 0]) % np.pi
    angle_bin = np.minimum((orientation * (8 / np.pi)).astype(int), 7)
    return 8 * np.searchsorted(radial_cuts, radius) + angle_bin


def select(values, keys):
    mask = np.zeros((len(keys)//256, 256), bool)
    for row, group, tiebreak in zip(mask, values.reshape(-1, 256), keys.reshape(-1, 256)):
        threshold = np.partition(group, -64)[-64]
        row[:] = group > threshold
        tied = np.flatnonzero(group == threshold)
        row[tied[np.argsort(tiebreak[tied])[:64-row.sum()]]] = True
    assert np.all(mask.sum(axis=1) == 64)
    return mask.ravel()


def rebuild(state, pair):
    # The logistic Jensen gap is also minus the log Bhattacharyya coefficient.
    cal = draw(123000 + pair, 8192)
    ea, eb = predict(state, cal), predict(state, -cal)
    midpoint = 0.5*(ea+eb)
    exp_tail = np.exp(-np.abs(midpoint))
    bernoulli_variance = exp_tail/(1+exp_tail)**2
    gap = 0.5*np.log1p(4*bernoulli_variance*np.sinh(0.25*(ea-eb))**2)
    edges = np.quantile(np.hypot(cal[:,0], cal[:,1]), np.arange(1,16)/16)
    bins = cells(cal, edges)
    assert np.all(np.bincount(bins, minlength=128) > 0)
    priority = np.array([gap[bins == j].mean() for j in range(128)])
    x = draw(131000+pair, EVAL)
    tie = np.random.default_rng(132000+pair).random(EVAL)
    radius = np.hypot(x[:,0], x[:,1])
    # With a uniform underlying angle the radial mixture likelihood contains I0.
    # The common radial prefactors cancel in the two-class posterior odds.
    odds = ((.5**2-1.2**2)/(2*.15**2) + np.log(np.i0(radius*1.2/.15**2))
            - np.log(np.i0(radius*.5/.15**2)))
    posterior = 1/(1+np.exp(-odds))
    a, b = predict(state,x), predict(state,-x)
    m = (a+b)/2
    risk = lambda s: np.logaddexp(0,s)-posterior*s
    la, lb, lm = risk(a), risk(b), risk(m)
    raw = (la+lb)/2
    delta = raw-lm
    f = select(priority[cells(x,edges)],tie)
    ma, mb = select(-abs(a),tie).astype(int), select(-abs(b),tie).astype(int)
    lf = np.where(f,lm,raw)
    lc = (np.where(ma,lm,la)+np.where(mb,lm,lb))/2
    orient = (ma-mb)*(la-lb)/4
    sym = (ma+mb)*delta/2
    np.testing.assert_allclose(lf-lc, sym+orient-f*delta, atol=TOL, rtol=0)
    terms = (raw, lm, lf, lc, lf-lc, f*delta, sym, orient, ma!=mb)
    return {label: float(values.mean()) for label,values in zip(FIELDS,terms)}


def group_by(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def difference(left, right, fields):
    return max(abs(float(left[k])-float(right[k])) for k in fields)


def identity(row, edge):
    return max(abs(float(row["folded_loss"])-float(row["confidence_loss"])-float(row["gap"])),
               abs(float(row["orientation_term"])-float(row[edge])-float(row["gap"])),
               abs(float(row["folded_gain"])+float(row["folded_loss"])-float(row["raw_loss"])),
               abs(float(row["symmetric_gain"])+float(row["orientation_term"])
                   +float(row["confidence_loss"])-float(row["raw_loss"])))


def bands(matrix, times, label):
    assert matrix.shape == (16, len(times))
    mean = matrix.mean(axis=0)
    # The same seed is reselected with all its ages, schedules and paired policies.
    index = np.random.default_rng(291031).integers(0, 16, (20000,16))
    boot = matrix[index].mean(axis=1)
    point = np.quantile(boot,[.025,.975],axis=0)
    width = float(np.quantile(np.abs(boot-mean).max(axis=1),.95))
    return [dict(series=label,step_or_lag=t,mean=float(mean[j]),
                 n_positive=int(np.sum(matrix[:,j]>0)), n_negative=int(np.sum(matrix[:,j]<0)),
                 point_low=float(point[0,j]),point_high=float(point[1,j]),
                 band_low=float(mean[j]-width),band_high=float(mean[j]+width))
            for j,t in enumerate(times)]


def audit_endpoints(folder, ids, prior):
    rows = read(folder / "endpoints.csv")
    records = {(int(r["pair"]),int(r["pb_steps"])): r for r in rows}
    assert len(records) == len(rows) == len(ids)*len(STEPS)
    groups = group_by(read(folder/"endpoint_batches.csv"), ("pair","pb_steps"))
    assert len(groups) == len(records)
    batch_err, anchor_err, id_err, raw_err = 0., 0., 0., 0.
    raw_checks = {(ids[0],0),(ids[0],50),(ids[0],400),(ids[6],250),(ids[-1],475)}
    largest_absolute_gap = max(records, key=lambda k:abs(float(records[k]["gap"])))
    raw_checks.add(largest_absolute_gap)
    with np.load(folder/"states.npz") as states:
        for pair in ids:
            for step in STEPS:
                r = records[pair,step]
                assert int(r["step"]) == 500 and int(r["gd_steps_after"]) == 500-step
                assert all(int(r[k]) == 8192 for k in
                           ("folded_extra","confidence_extra","confidence_swapped_extra"))
                state = states[f"p{pair}_s{step}_t500"]
                assert sha(state) == r["state_hash"]
                rr = groups[str(pair),str(step)]
                assert len(rr) == 128 and {int(b["batch"]) for b in rr} == set(range(1,129))
                means = {k:np.mean([float(b[k]) for b in rr]) for k in FIELDS}
                batch_err = max(batch_err, difference(means,r,FIELDS))
                id_err = max(id_err, identity(r,"invariant_advantage"))
                if (pair,step) in raw_checks:
                    raw_err = max(raw_err, difference(rebuild(state,pair),r,FIELDS))
                if pair <= 16 and step in (100,500):
                    name = "pb" if step == 500 else "hybrid"
                    old = prior[pair,name]
                    assert r["eval_hash"] == old["eval_hash"]
                    anchor_err = max(anchor_err,difference(old,r,FIELDS[:4]))
    assert max(batch_err,anchor_err,id_err,raw_err) < TOL
    matrix = np.array([[float(records[p,t]["gap"]) for t in STEPS] for p in ids])
    result = dict(seed_min=ids[0], seed_max=ids[-1], states=len(rows),
                  verified_batch_records=len(records)*128, max_batch_mean_error=batch_err,
                  max_anchor_loss_error=anchor_err if ids[0]==1 else None,
                  max_decomposition_error=id_err, rebuilt_from_weights=len(raw_checks),
                  worst_gap_row_rebuilt=dict(pair=largest_absolute_gap[0],switch=largest_absolute_gap[1],
                                             gap=float(records[largest_absolute_gap]["gap"])),
                  max_independent_rebuild_error=raw_err)
    return result,records,bands(matrix,STEPS,folder.name)


def audit_trajectory(folder, endpoints, ids, series_prefix):
    rows = read(folder/"by_seed.csv")
    records = {(int(r["pair"]),int(r["switch_step"]),int(r["gd_elapsed"]),r["branch"]):r
               for r in rows}
    assert len(records) == len(rows)
    id_err = max(identity(r,"folded_symmetric_advantage") for r in rows)
    end_err, zero_err, pb_err = 0.,0.,0.
    pb_values = {}
    for (pair,s,d,branch),r in records.items():
        age = s+d
        assert int(r["total_step"]) == age and age <= 500
        if d==0:
            zero_err = max(zero_err, difference(r,records[pair,s,d,"continued_pullback"],FIELDS))
        if age==500:
            pb = s if branch=="gd_after_switch" else 500
            end_err = max(end_err, difference(r,endpoints[pair,pb],FIELDS))
        if branch=="continued_pullback":
            key = (pair,age)
            if key in pb_values:
                pb_err = max(pb_err, difference(r,pb_values[key],FIELDS))
            pb_values[key]=r
    assert max(id_err,end_err,zero_err,pb_err) < TOL
    output=[]
    for s in sorted({k[1] for k in records}):
        delays=sorted({k[2] for k in records if k[1]==s})
        paths={}
        for arm in ("gd_after_switch","continued_pullback"):
            paths[arm]=np.array([[float(records[p,s,d,arm]["gap"]) for d in delays] for p in ids])
            output.extend(bands(paths[arm],delays,f"{series_prefix}_switch_{s}_{arm}"))
        output.extend(bands(paths["gd_after_switch"]-paths["continued_pullback"],delays,
                            f"{series_prefix}_switch_{s}_paired_effect"))
    return dict(records=len(rows),max_decomposition_error=id_err,max_zero_lag_error=zero_err,
                max_shared_age_pullback_error=pb_err,max_shared_endpoint_error=end_err),output


def audit_rates(folder, trajectory_folder, ids):
    expected = {(p,r,s) for p in ids for r in (0.5,1.,2.) for s in (25,100,200,400)}
    source = read(folder/"models.csv")
    rows = {(int(r["pair"]),float(r["gd_rate"]),int(r["gd_steps"])):r for r in source}
    assert set(rows) == expected and len(source) == len(expected)
    trajectories = {(int(r["pair"]),int(r["gd_elapsed"])):r for r in read(trajectory_folder/"by_seed.csv")
                    if r["branch"] == "gd_after_switch" and int(r["switch_step"])==100}
    error = max(difference(r, trajectories[p,d], ("raw_loss","folded_loss","confidence_loss",
                                                  "gap","orientation_term"))
                for (p,rate,d),r in rows.items() if rate==2)
    arithmetic = max(abs(float(r["gap"])-float(r["orientation_term"])+float(r["region_symmetric_edge"]))
                     for r in rows.values())
    assert max(error,arithmetic) < TOL
    return dict(records=len(rows), shared_trajectory_rows=len(ids)*4, max_common_path_error=error,
                max_decomposition_error=arithmetic)


def main(args):
    root = args.study
    prior = {(int(r["pair"]),r["model"]):r for r in read(args.prior/"supplemental_results"/"models.csv")}
    orig, records, rows = audit_endpoints(root/"results",list(range(1,17)),prior)
    validation, new_records, new = audit_endpoints(root/"validation_results",list(range(17,33)),prior)
    traj, paths = audit_trajectory(root/"trajectory_results",records,range(1,17),"original")
    new_traj, new_paths = audit_trajectory(root/"trajectory_validation_results",new_records,range(17,33),"new")
    rates = [audit_rates(root/name,root/target,ids) for name,target,ids in
             (("rate_results","trajectory_results",range(1,17)),
              ("rate_validation_results","trajectory_validation_results",range(17,33)))]
    write(root/"paired_audit_bands.csv", rows+new+paths+new_paths)
    summary = {"original_cohort":orig, "validation_cohort":validation, "matched_age_trajectories":traj,
               "targeted_validation_trajectories":new_traj,
               "learning_rate_controls":rates,
               "inference":"Pointwise percentile intervals and unstudentized simultaneous maximum-deviation bands. 20000 paired seed resamples, RNG seed 291031. Each series has its own band."}
    (root/"independent_audit.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT)
    parser.add_argument("--prior",type=Path,default=ROOT.parent/"budgeted-rotation")
    with threadpool_limits(1):
        main(parser.parse_args())
