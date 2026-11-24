"""Restore fixed rings checkpoints from the original study and verify byte identity."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import runpy

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(archive, output):
    meta = json.loads((archive / "results/metadata.json").read_text(encoding="utf-8"))
    driver = archive / "run_rings.py"
    assert digest(driver.read_bytes()) == meta["driver_sha256"]
    assert digest(Path(core.__file__).read_bytes()) == meta["core_sha256"]
    original = runpy.run_path(str(driver))
    expected = {(int(r["pair"]), r["schedule"], int(r["step"])): r
                for r in load_rows(archive / "results/checkpoints.csv")}
    seeds = load_rows(archive / "results/seeds.csv")
    assert [int(s["pair"]) for s in seeds] == list(range(1, 9))
    output.mkdir(parents=True, exist_ok=False)
    bank, rows = {}, []
    max_error, hashes_checked = 0., 0
    with threadpool_limits(limits=1):
        for seed in seeds:
            pair = int(seed["pair"])
            data = {k: original["rings"](int(seed[f"{k}_seed"])) for k in ("train", "validation")}
            init = core.initialize_parameters(int(seed["initialization_seed"]))
            assert digest(init.tobytes()) == seed["initial_sha256"]
            for split, (x, y) in data.items():
                assert digest(x.tobytes() + y.tobytes()) == seed[f"{split}_sha256"]
            states = {"initial": init}
            for method in ("gd", "pullback"):
                states[method+"100"] = original["advance"](init.copy(), method, 100, data["train"])
                states[method+"500"] = original["advance"](states[method+"100"].copy(), method, 400, data["train"])
            states["warmup500"] = original["advance"](states["pullback100"].copy(), "gd", 400, data["train"])
            for name, theta in states.items():
                bank[f"pair{pair}_{name}"] = theta
            for arm in ("gd", "pullback", "warmup"):
                for step in (0, 100, 500):
                    name = "initial" if step == 0 else f"{arm}{step}"
                    if arm == "warmup" and step == 100:
                        name = "pullback100"
                    state = states[name]
                    row = original["score"](state, data, dict(pair=pair, schedule=arm, step=step,
                                                            state_key=f"pair{pair}_{name}"))
                    if step:
                        reference = expected[pair, arm, step]
                        assert row["state_sha256"] == reference["state_sha256"]
                        hashes_checked += 1
                        for metric in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy"):
                            max_error = max(max_error, abs(float(reference[metric]) - row[metric]))
                    rows.append(row)
    assert len(bank) == 48 and len(rows) == 72 and hashes_checked == 48 and max_error < 1e-12
    np.savez_compressed(output / "states.npz", **bank)
    original["write_rows"](output / "replay.csv", rows)
    audit = dict(unique_states=len(bank), schedule_states=len(rows), initial_hashes=8, split_hashes=16,
                 archived_trained_checkpoint_hashes=hashes_checked, max_metric_error=max_error,
                 original_driver_sha256=meta["driver_sha256"], core_sha256=meta["core_sha256"],
                 archived_checkpoints_sha256=digest((archive / "results/checkpoints.csv").read_bytes()),
                 plan_sha256=digest(Path(__file__).with_name("stage1_plan.md").read_bytes()),
                 state_bank_sha256=digest((output / "states.npz").read_bytes()),
                 numpy_version=np.__version__, python_version=platform.python_version())
    (output / "replay_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="The original new-directions-rings folder")
    parser.add_argument("--out", type=Path, required=True, help="A fresh directory")
    args = parser.parse_args()
    main(args.archive, args.out)
