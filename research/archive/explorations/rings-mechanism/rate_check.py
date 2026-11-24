"""Evaluate previously selected GD rates on the original eight rings pairs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy
from statistics import mean, stdev

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core
import polar
from polar import evaluate


HERE = Path(__file__).resolve().parent
METRICS = ("population_loss", "radial_excess", "angular_excess")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def gd(theta, x, y, eta, count):
    for _ in range(count):
        d, _ = core.direction(theta, x, y, method="gd")
        theta = theta + eta*d
    if not np.isfinite(theta).all():
        raise ArithmeticError("Nonfinite trajectory")
    return theta


def main(archive, selection, primary, out):
    chosen = json.loads(selection.read_text(encoding="utf-8-sig"))["rates"]
    assert chosen == {"gd": .5, "warmup": 2.}
    out.mkdir(parents=True, exist_ok=False)
    primary_rows = read_rows(primary / "population/states.csv")
    base = {(int(r["pair"]), r["schedule"], int(r["step"])):r for r in primary_rows}
    seed_records = read_rows(archive / "results/seeds.csv")
    driver = archive / "run_rings.py"
    recorded = json.loads((primary / "replay_audit.json").read_text())
    assert digest(driver.read_bytes()) == recorded["original_driver_sha256"]
    assert digest(Path(core.__file__).read_bytes()) == recorded["core_sha256"]
    assert digest((primary / "states.npz").read_bytes()) == recorded["state_bank_sha256"]
    rings = runpy.run_path(str(driver))["rings"]
    rows, paired, new_states = [], [], {}
    with np.load(primary / "states.npz", allow_pickle=False) as bank, threadpool_limits(limits=1):
        for seed_row in seed_records:
            pair = int(seed_row["pair"])
            x, y = rings(int(seed_row["train_seed"]))
            assert digest(x.tobytes()+y.tobytes()) == seed_row["train_sha256"]
            init = bank[f"pair{pair}_initial"]
            prefix = bank[f"pair{pair}_pullback100"]
            assert digest(init.tobytes()) == seed_row["initial_sha256"]
            assert digest(prefix.tobytes()) == base[pair,"warmup",100]["state_sha256"]
            theta = gd(init, x, y, chosen["gd"], 100)
            candidates = [("gd",100,theta), ("gd",500,gd(theta,x,y,chosen["gd"],400)),
                          ("warmup",500,gd(prefix,x,y,chosen["warmup"],400))]
            for method, step, state in candidates:
                metrics = evaluate(state)
                assert metrics["grid_converged"] and metrics["identity_error"] < 1e-12
                rows.append(dict(pair=pair, method=method, step=step, gd_rate=chosen[method],
                                 state_sha256=digest(state.tobytes()), **metrics))
                new_states[f"pair{pair}_{method}{step}"] = state
                for ref in (method, "pullback"):
                    r = base[pair,ref,step]
                    diffs = {k: float(r[k])-metrics[k] for k in METRICS}
                    assert abs(diffs["population_loss"]-diffs["radial_excess"]-diffs["angular_excess"]) < 1e-11
                    paired.append(dict(pair=pair, candidate=method, step=step, reference=ref, **diffs))
    assert len(rows) == 24 and len(paired) == 48 and len(new_states) == 24
    write_rows(out / "states.csv", rows)
    write_rows(out / "paired.csv", paired)
    np.savez_compressed(out / "states.npz", **new_states)
    by_group = []
    for arm, step in (("gd",100),("gd",500),("warmup",500)):
        subset = [r for r in rows if (r["method"], r["step"]) == (arm,step)]
        assert sorted(r["pair"] for r in subset) == list(range(1,9))
        item = dict(method=arm, step=step)
        for k in METRICS:
            vals=[r[k] for r in subset]
            item[k+"_mean"] = mean(vals)
            item[k+"_sd"] = stdev(vals)
        by_group.append(item)
    write_rows(out / "means.csv", by_group)
    audit = dict(selection_sha256=digest(selection.read_bytes()), rates=chosen,
                 states_evaluated=len(rows), contrasts=len(paired), old_prefix_hashes_verified=8,
                 max_grid_delta=max(r["grid_error"] for r in rows),
                 max_identity_error=max(r["identity_error"] for r in rows),
                 polar_sha256=digest(Path(polar.__file__).read_bytes()),
                 plan_sha256=digest((HERE / "rate_check_plan.md").read_bytes()))
    (out/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(by_group))
    print(json.dumps(audit))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True,help="Original rings study")
    p.add_argument("--selection",type=Path,required=True,help="Original rings learning-rate selection JSON")
    p.add_argument("--primary",type=Path,required=True,help="Primary stage1 outputs")
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    main(args.archive,args.selection,args.primary,args.out)
