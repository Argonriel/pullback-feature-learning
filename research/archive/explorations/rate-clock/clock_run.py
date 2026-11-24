"""Re-evaluate three GD rates on the same elapsed gradient-flow time grid."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
RATES = (0.5, 1.0, 2.0)
TIMES = (0, 50, 100, 200, 400, 600, 800, 1200, 1600)


def sibling(name):
    return next(p for p in (ROOT.parent / name, ROOT.parent / (name + "-final"))
                if p.is_dir())


def libraries():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sibling("budgeted-rotation")))
    sys.path.insert(0, str(sibling("switch-time")))
    import run as prior
    import run_endpoints as previous
    return prior, previous


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def archive_anchors():
    anchors = {}
    for cohort in ("rate_results", "rate_validation_results"):
        with (sibling("switch-time") / cohort / "models.csv").open(encoding="utf8") as file:
            for row in csv.DictReader(file):
                anchors[int(row["pair"]), float(row["gd_rate"]), int(row["gd_steps"])] = row
    return anchors


def run(out):
    prior, previous = libraries()
    anchors = archive_anchors()
    out.mkdir(exist_ok=False)
    rows, checks, arrays = [], [], {}
    begin = perf_counter()
    with threadpool_limits(1):
        for pair in range(1, 33):
            data = prior.rings(121000 + pair)
            x, y = data
            fork = prior.advance(prior.core.initialize_parameters(122000 + pair),
                                 data, "pullback", 8, 100)
            cal = prior.sample_inputs(123000 + pair, 8192)
            test = prior.sample_inputs(131000 + pair, 32768)
            p = prior.posterior(np.linalg.norm(test, axis=1))
            tie = np.random.default_rng(132000 + pair).random(len(test))
            initial_score = None
            for rate in RATES:
                indices = {int(t / rate): t for t in TIMES}
                size = max(indices) + 1
                weights = np.empty((size, fork.size))
                loss = np.empty(size)
                weights[0] = fork
                for step in range(size):
                    theta = weights[step]
                    loss[step] = prior.core.logistic_loss(prior.core.representation(theta, x), y)
                    if step in indices:
                        if step > 0 or initial_score is None:
                            gate = prior.Gate.fit(cal, prior.with_model(theta))
                            values, _ = previous.evaluate(theta, gate, test, p, tie, prior)
                            if step == 0:
                                initial_score = values
                        else:
                            values = initial_score
                        key = (pair, rate, step)
                        checksum = hashlib.sha256(theta.tobytes()).hexdigest()
                        rows.append(dict(pair=pair, rate=rate, gd_steps=step, tau=indices[step],
                                         train_loss=loss[step], state_sha256=checksum, **values))
                        if key in anchors:
                            a = anchors[key]
                            err = max(abs(values[k] - float(a[k])) for k in
                                      ("raw_loss","folded_loss","confidence_loss","gap","orientation_term"))
                            if err > 2e-12:
                                raise AssertionError(f"Prior rate-control anchor mismatch for {key}: {err}")
                            checks.append(dict(pair=pair,rate=rate,gd_steps=step,max_loss_difference=err))
                    if step < size - 1:
                        direction, _ = prior.core.direction(theta, x, y, method="gd", damping=.1)
                        weights[step+1] = theta + rate*direction
                assert np.all(np.isfinite(loss)) and np.all(np.isfinite(weights))
                tag = f"p{pair}_r{int(10*rate)}"
                arrays["weights_"+tag] = weights
                arrays["loss_"+tag] = loss
            print(f"Completed seed {pair}/32 in {perf_counter()-begin:.1f} seconds",flush=True)
    assert len(rows)==len(TIMES)*3*32
    assert sum(r["gd_steps"]==400 for r in checks)==32*3
    write_rows(out/"by_seed.csv", rows)
    write_rows(out/"archive_anchors.csv", checks)
    np.savez_compressed(out/"training_paths.npz", **arrays)
    metadata = dict(pairs=list(range(1,33)), rates=RATES, times=TIMES, loss="Full-batch, pre-update training BCE at index k, after precisely k GD updates.",
                    fork="100 pullback steps at rate 8 and damping 0.1 from the existing switch-time experiment.",
                    models_per_state=1, n_cal=8192, n_eval=32768, per_batch=256, extra_budget=64,
                    elapsed_seconds=perf_counter()-begin, archive_anchor_rows=len(checks),
                    max_archive_overlap=max(r["max_loss_difference"] for r in checks),
                    path_arrays="loss_p{i}_r{10*rate} and weights_p{i}_r{10*rate}, indexed by GD update count including zero.",
                    digest={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in
                            (Path(__file__),Path(prior.__file__),Path(previous.__file__),Path(prior.core.__file__))})
    with (out/"metadata.json").open("x",encoding="utf8") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"clock_results")
    run(parser.parse_args().out)
