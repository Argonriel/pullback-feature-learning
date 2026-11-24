"""Weighted-feature intervention on the paired rings tasks. Run train, then score."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from pullback_feature_learning import core
from generator import rings
import polar


ROOT = Path(__file__).resolve().parent
WEIGHTS = (0., 1/9, 1., 9.)
MOVES = ("cum_z1", "cum_z2", "max_step_z1", "max_step_z2", "loss_increases", "matched_scale_max", "match_max_error")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def log_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def save_rows(path, rows):
    tmp = path.with_suffix(".part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def seeds(pair):
    start, i = (71000, pair) if pair <= 8 else (101000, pair-8)
    return start+i, start+1000+i, start+2000+i


def solve(jac, grad, weight):
    if weight == 1:
        h = jac.T @ jac
    else:
        a, b = jac[0::2], jac[1::2]
        h = a.T @ a
        if weight:
            h += weight * (b.T @ b)
    h.flat[::core.PARAMETER_COUNT+1] += .1
    return np.linalg.solve(h, -grad)


def fastloss(z, y):
    return core.logistic_loss(z, y)


def train_one(pair, rule, weight, train, valid, init, bank, original):
    x, y = train
    xv, yv = valid
    rows = []
    ids = dict(pair=pair, cohort="original" if pair <= 8 else "fresh",
               size_rule=rule, weight=weight)

    def snapshot(theta, z, step, schedule, diag):
        key = digest(theta.tobytes())
        if key not in bank:
            bank[key] = theta.copy()
        if pair <= 8 and weight == 1:
            old_key = f"pair{pair}_" + ("warmup500" if schedule == "hybrid" and step == 500 else f"pullback{step}")
            if not np.array_equal(theta, original[old_key]):
                raise AssertionError(f"Baseline differs from archived state {old_key}")
        zv = core.representation(theta, xv)
        rows.append(dict(**ids, schedule=schedule, step=step, state_sha256=key,
                         train_loss=fastloss(z, y), validation_loss=fastloss(zv, yv),
                         validation_accuracy=float(np.mean(np.where(zv[:,0]>=0, 1, -1)==yv)),
                         **diag))

    theta = init.copy()
    z = core.representation(theta, x)
    diag = dict.fromkeys(MOVES, 0.)
    loss = fastloss(z, y)

    def record_motion(old_z, new_z, new_loss, previous_loss, d):
        rms = np.sqrt(np.mean((new_z-old_z)**2, axis=0))
        d["cum_z1"] += rms[0]
        d["cum_z2"] += rms[1]
        d["max_step_z1"] = max(d["max_step_z1"], rms[0])
        d["max_step_z2"] = max(d["max_step_z2"], rms[1])
        d["loss_increases"] += int(new_loss > previous_loss + 1e-12)

    for step in range(1, 501):
        jac = core.representation_jacobian(theta, x)
        g = jac.T @ core.representation_gradient(z, y).ravel()
        update = solve(jac, g, weight)
        scale = 1.
        if rule == "matched":
            a = jac[0::2]
            base = solve(jac, g, 1.)
            target = np.linalg.norm(a @ base)
            raw = np.linalg.norm(a @ update)
            if not raw and target:
                raise ArithmeticError("A nonzero target cannot be matched.")
            scale = target/raw if target else 0.
            err = abs(np.linalg.norm(a @ (scale*update)) - target)*8/np.sqrt(len(x))
            if err > 1e-10*(1+target):
                raise ArithmeticError("Target first-coordinate motion missed.")
            diag["match_max_error"] = max(diag["match_max_error"], err)
        diag["matched_scale_max"] = max(diag["matched_scale_max"], scale)
        theta = theta + (8.*scale) * update
        new_z = core.representation(theta, x)
        new_loss = fastloss(new_z, y)
        if not np.isfinite(new_loss) or not np.isfinite(theta).all():
            raise FloatingPointError(f"Nonfinite native/weighted state {step}")
        record_motion(z, new_z, new_loss, loss, diag)
        z, loss = new_z, new_loss
        if step in (100, 500):
            snapshot(theta, z, step, "full", diag)
        if step == 100 and rule == "native":
            snapshot(theta, z, step, "hybrid", diag)
            switch = (theta.copy(), z.copy(), loss, diag.copy())
    if rule == "native":
        theta, z, loss, diag = switch
        for i in range(400):
            delta, _ = core.direction(theta, x, y, "gd", .1)
            theta = theta + .125 * delta
            new_z = core.representation(theta, x)
            new_loss = fastloss(new_z, y)
            if not np.isfinite(new_loss) or not np.isfinite(theta).all():
                raise FloatingPointError(f"Nonfinite switch state at update {101+i}")
            record_motion(z, new_z, new_loss, loss, diag)
            z, loss = new_z, new_loss
        snapshot(theta, z, 500, "hybrid", diag)
    return rows


def train(out):
    out.mkdir(parents=True, exist_ok=False)
    originals = np.load(ROOT / "stage1" / "states.npz")
    bank, rows, failures = {}, [], []
    provenance = []
    with threadpool_limits(limits=1), np.errstate(invalid="raise", over="raise", divide="raise"):
        info = threadpool_info()
        for pair in range(1, 17):
            tr, ini, va = seeds(pair)
            train_xy, valid_xy = rings(tr), rings(va)
            init = core.initialize_parameters(ini)
            provenance.append(dict(pair=pair, train_seed=tr, init_seed=ini, validation_seed=va,
                                   train_hash=digest(b"".join(i.tobytes() for i in train_xy)),
                                   validation_hash=digest(b"".join(i.tobytes() for i in valid_xy)),
                                   initial_hash=digest(init.tobytes())))
            for weight in WEIGHTS:
                for rule in (("native",) if weight == 1 else ("native","matched")):
                    try:
                        group = train_one(pair, rule, weight, train_xy, valid_xy, init, bank, originals)
                    except (ArithmeticError, np.linalg.LinAlgError) as exc:
                        failures.append(dict(pair=pair, weight=weight, rule=rule, reason=repr(exc)))
                        continue
                    rows.extend(group)
                    if weight == 1:
                        rows.extend(dict(row, size_rule="matched") for row in group if row["schedule"] == "full")
            print(f"Finished pair {pair}/16", flush=True)
    originals.close()
    np.savez_compressed(out/"states.npz", **bank)
    save_rows(out/"training_rows.csv", rows)
    log_json(out/"training_record.json", dict(seeds=provenance, unique_states=len(bank), logical_rows=len(rows),
             failures=failures, threadpools=info, python=platform.python_version(), numpy=np.__version__,
             hashes={p.name:digest(p.read_bytes()) for p in (Path(__file__), Path(core.__file__),ROOT/"generator.py",
                          ROOT/"stage2_plan.md", ROOT/"stage1"/"population"/"audit.json")}))
    if failures or len(rows)!=384:
        raise RuntimeError(f"Incomplete batch: {failures}")


def score(out):
    if (out/"rows.csv").exists():
        raise FileExistsError("Population scores already exist")
    with (out/"training_rows.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert not json.loads((out/"training_record.json").read_text())["failures"]
    with threadpool_limits(limits=1), np.load(out/"states.npz") as states:
        results = {}
        for i,key in enumerate(states.files):
            theta = states[key]
            assert digest(theta.tobytes()) == key
            results[key] = polar.evaluate(theta)
            if (i+1)%32 == 0:
                print(f"Scored {i+1}/{len(states.files)} distinct states", flush=True)
    save_rows(out/"rows.csv", [dict(row, **results[row["state_sha256"]]) for row in rows])
    print(f"Scored {len(results)} unique states in {len(rows)} logical records", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train","score"])
    parser.add_argument("--out", type=Path, default=ROOT/"stage2")
    args = parser.parse_args()
    (train if args.action=="train" else score)(args.out)
