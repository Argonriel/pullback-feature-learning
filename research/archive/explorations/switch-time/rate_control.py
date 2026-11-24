"""Compare post-switch GD learning rates from the same 100-update pullback checkpoint."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
PRIOR = next(p for p in (ROOT.parent / "budgeted-rotation", ROOT.parent / "budgeted-rotation-final")
             if (p / "run.py").is_file())
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))
from run import advance, core, nll, posterior, rings, sample_inputs, with_model
from gate import Gate
from supplemental_confidence import route


def measure(theta, calibration, x, tie, p):
    predict = with_model(theta)
    gate = Gate.fit(calibration, predict)
    a, b = predict(x), predict(-x)
    la, lb = nll(a, p), nll(b, p)
    raw = (la + lb) / 2
    both = nll((a + b) / 2, p)
    d = raw - both
    regional = route(gate.priority(x, "folded"), tie)
    ma, mb = route(-abs(a), tie), route(-abs(b), tie)
    orientation = (ma.astype(int) - mb.astype(int)) * (la - lb) / 4
    confidence_loss = (np.where(ma, both, la) + np.where(mb, both, lb)).mean() / 2
    region_loss = np.where(regional, both, raw).mean()
    region_edge = np.mean(regional * d) - np.mean((ma.astype(int) + mb.astype(int)) * d) / 2
    assert abs((region_loss - confidence_loss) - (orientation.mean() - region_edge)) < 1e-12
    diff = ma != mb
    return dict(raw_loss=raw.mean(), folded_loss=region_loss, confidence_loss=confidence_loss,
                gap=region_loss-confidence_loss, orientation_term=orientation.mean(),
                region_symmetric_edge=region_edge, fraction_routed_differently=diff.mean(),
                worse_orientation_selected_given_different=np.mean(orientation[diff] > 0)
                if diff.any() else float("nan"))


def main(pairs, out):
    out.mkdir(exist_ok=False)
    rows = []
    with threadpool_limits(limits=1):
        for pair in pairs:
            data = rings(121000 + pair)
            initial = core.initialize_parameters(122000 + pair)
            shared = advance(initial, data, "pullback", 8, 100)
            calibration = sample_inputs(123000 + pair, 8192)
            x = sample_inputs(131000 + pair, 32768)
            tie = np.random.default_rng(132000 + pair).random(len(x))
            p = posterior(np.linalg.norm(x, axis=1))
            for rate in (0.5, 1, 2):
                theta = shared.copy()
                previous = 0
                for steps in (25, 100, 200, 400):
                    theta = advance(theta, data, "gd", rate, steps - previous)
                    previous = steps
                    rows.append(dict(pair=pair, gd_rate=rate, gd_steps=steps,
                                     **measure(theta, calibration, x, tie, p)))
            print(f"Learning-rate control completed pair {pair}", flush=True)
    with (out / "models.csv").open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (PRIOR / "supplemental_results" / "models.csv").open() as stream:
        original = {int(row["pair"]): row for row in csv.DictReader(stream) if row["model"] == "hybrid"}
    for row in rows:
        if row["pair"] in original and row["gd_rate"] == 2 and row["gd_steps"] == 400:
            for field in ("folded_loss", "confidence_loss", "raw_loss"):
                assert abs(row[field] - float(original[row["pair"]][field])) < 1e-12
    print("Checked all available original-study overlaps.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pair", type=int, default=1)
    parser.add_argument("--last-pair", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ROOT / "rate_results")
    args = parser.parse_args()
    if not 1 <= args.first_pair <= args.last_pair <= 32:
        parser.error("The study uses seed indices 1 through 32.")
    main(range(args.first_pair, args.last_pair + 1), args.out)
