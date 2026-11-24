"""Fresh-draw experiment in allocating second views on concentric rings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from pullback_feature_learning import core
from generator import rings
from gate import allocate, deploy, gap, Gate

ROOT = Path(__file__).resolve().parent
CAL, EVAL, BATCH, QUOTA = 8192, 32768, 256, 64
DECISION_RADIUS = .86419687750821
POLICIES = ("raw", "full", "random", "radius", "folded", "boundary")


def sample_inputs(seed, n):
    rng = np.random.default_rng(seed)
    radius = rng.choice([.5, 1.2], size=n)
    phi = rng.uniform(0, 2*np.pi, n)
    return radius[:, None] * np.c_[np.cos(phi), np.sin(phi)] + rng.normal(0, .15, (n,2))


def posterior(r):
    logit = ((.5**2 - 1.2**2) / (2*.15**2)
             + np.log(np.i0(1.2*r/.15**2)) - np.log(np.i0(.5*r/.15**2)))
    return np.exp(-np.logaddexp(0, -logit))


def nll(h, p):
    return (1-p)*np.logaddexp(0., h) + p*np.logaddexp(0., -h)


def csvout(path, rows):
    with path.open("x", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def jsonout(path, value):
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sha(a):
    return hashlib.sha256(a).hexdigest()


def advance(theta, data, method, rate, n):
    for _ in range(n):
        d, _ = core.direction(theta, *data, method=method, damping=.1)
        theta = theta + rate*d
    if not np.isfinite(theta).all():
        raise ArithmeticError("Nonfinite trajectory")
    return theta


def with_model(theta):
    return lambda x: core.representation(theta, x)[:,0]


def fit_all():
    states, gates, provenance = {}, {}, []
    for pair in range(1,17):
        x,y = rings(121000+pair)
        theta = core.initialize_parameters(122000+pair)
        prefix = advance(theta, (x,y), "pullback", 8, 100)
        checkpoints = dict(gd=advance(theta, (x,y), "gd", .5, 500),
                           pb=advance(prefix, (x,y), "pullback", 8, 400),
                           hybrid=advance(prefix, (x,y), "gd", 2, 400))
        cal = sample_inputs(123000+pair, CAL)
        for name, v in checkpoints.items():
            key = f"{pair}_{name}"
            states[key] = v
            gates[key] = Gate.fit(cal, with_model(v))
            provenance.append(dict(pair=pair, model=name, state_hash=sha(v.tobytes()),
                                   train_hash=sha(x.tobytes()+y.tobytes()), cal_hash=sha(cal.tobytes()),
                                   cal_views=2*CAL, cal_seed=123000+pair, train_seed=121000+pair,
                                   init_seed=122000+pair))
    return states, gates, provenance


def score(pair, name, theta, gate, x, tie, batch_rows):
    predict = with_model(theta)
    h, b = predict(x), predict(-x)
    m = .5*(h+b)
    r = np.linalg.norm(x,axis=1)
    p = posterior(r)
    raw, full = .5*(nll(h,p)+nll(b,p)), nll(m,p)
    d = gap(h,b)
    err = float(np.max(np.abs((raw-full)-d)))
    assert err < 1e-11 and d.min() >= -1e-12
    w = QUOTA/BATCH
    losses = dict(raw=raw, full=full, random=(1-w)*raw+w*full)
    counts, max_deployment_diff = {}, 0.
    for policy in ("radius","folded","boundary"):
        if policy == "boundary":
            scores = -abs(r - DECISION_RADIUS)
        else:
            scores = gate.priority(x, policy)
            assert np.array_equal(scores, gate.priority(-x, policy))
        mask = allocate(scores.reshape(-1,BATCH), tie.reshape(-1,BATCH), QUOTA)
        assert (mask.sum(axis=1)==QUOTA).all()
        flat = mask.ravel()
        counts[policy] = int(flat.sum())
        losses[policy] = np.where(flat, full, raw)
        if policy != "boundary":
            # Audit the actual two-pass deployment path on a full batch.
            calls = []
            def counted(z):
                calls.append(len(z))
                return predict(z)
            result, keep = deploy(x[:BATCH],counted,gate,policy,tie[:BATCH],QUOTA)
            ref = np.where(flat[:BATCH],m[:BATCH],h[:BATCH])
            discrepancy = float(np.max(np.abs(result-ref)))
            assert np.array_equal(keep,flat[:BATCH]) and calls==[BATCH,QUOTA] and discrepancy < 1e-10
            max_deployment_diff=max(max_deployment_diff,discrepancy)
    grouped={k:v.reshape(-1,BATCH).mean(axis=1) for k,v in losses.items()}
    for i in range(len(x)//BATCH):
        batch_rows.append(dict(pair=pair,model=name,batch=i+1,
                               **{k+"_loss":float(v[i]) for k,v in grouped.items()}))
    return dict(pair=pair, model=name, **{k+"_loss":float(v.mean()) for k,v in losses.items()},
                budget=w, n_cal=CAL, n_eval=len(x), executed_extra_per_limited_policy=counts["folded"],
                eval_seed=124000+pair, eval_hash=sha(x.tobytes()), max_identity_error=err,
                max_deploy_error=max_deployment_diff)


def benchmark(states,gates):
    rng=np.random.default_rng(127000)
    x=sample_inputs(127001, BATCH*16).reshape(16,BATCH,2)
    keys=rng.random((16,BATCH))
    records=[]
    for name in ("gd","pb","hybrid"):
        k=f"1_{name}"
        gate=gates[k]
        pred=with_model(states[k])
        for rep in range(7):
            for policy in rng.permutation(["raw","full","radius","folded"]):
                start=perf_counter_ns()
                for block, tie in zip(x,keys):
                    if policy=="raw":
                        pred(block)
                    elif policy=="full":
                        .5*(pred(block)+pred(-block))
                    else:
                        deploy(block,pred,gate,policy,tie,QUOTA)
                duration=perf_counter_ns()-start
                if rep>0:
                    records.append(dict(model=name,repeat=rep,policy=policy,
                        ms_per_256=duration/len(x)/1e6))
    return records


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        states,gates,provenance=fit_all()
        np.savez_compressed(out/"models.npz",**states)
        np.savez_compressed(out/"calibrated_gates.npz", **{
            k+"_"+field:getattr(v,field) for k,v in gates.items()
            for field in ("edges","radius_scores","folded_scores")})
        csvout(out/"training_and_calibration.csv",provenance)
        rows, batch_rows = [], []
        for pair in range(1,17):
            x=sample_inputs(124000+pair,EVAL)
            tie=np.random.default_rng(125000+pair).random(EVAL)
            for name in ("gd","pb","hybrid"):
                k=f"{pair}_{name}"
                rows.append(score(pair,name,states[k],gates[k],x,tie,batch_rows))
        csvout(out/"batches.csv",batch_rows)
        csvout(out/"models.csv",rows)
        csvout(out/"timing.csv",benchmark(states,gates))
        jsonout(out/"metadata.json",dict(python=platform.python_version(),numpy=np.__version__,
                platform=platform.platform(),libraries=threadpool_info(),clock="perf_counter_ns",
                calibration_examples=CAL,evaluation_examples=EVAL,batch_size=BATCH,quota=QUOTA,
                heldout_design="iid unlabeled inputs with antithetic population-risk evaluation",
                hashes={p.name:sha(p.read_bytes()) for p in (Path(__file__),ROOT/"gate.py",
                    ROOT/"generator.py",ROOT/"DESIGN.md",Path(core.__file__))}))
    print(f"Completed {len(rows)} frozen models and {len(batch_rows)} evaluation batches.")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=ROOT/"results")
    main(ap.parse_args().out)
