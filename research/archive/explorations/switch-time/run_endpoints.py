"""Sweep the switch from pullback to GD while holding total training at 500 steps."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core

HERE = Path(__file__).resolve().parent
SWITCHES = (0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 475, 500)
STEPS = 500
PULLBACK_RATE, GD_RATE, DAMPING = 8., 2., 0.1
PAIRED_COHORT = tuple(range(1, 17))
VALIDATION_COHORT = tuple(range(17, 33))
N_CAL, N_EVAL, BATCH, BUDGET = 8192, 32768, 256, 64
ANCHORS = {100: "hybrid", 500: "pb"}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def find_prior():
    candidates = (HERE.parent / "budgeted-rotation", HERE.parent / "budgeted-rotation-final",
                  Path.home() / "Documents/Research/pullback-feature-learning/explorations/budgeted-rotation")
    return next((p for p in candidates if (p / "gate.py").is_file()), candidates[0])


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trajectories(pair, rings, advance):
    data = rings(121000 + pair)
    theta = core.initialize_parameters(122000 + pair)
    end = {}
    previous = 0
    for switch in SWITCHES:
        theta = advance(theta, data, "pullback", PULLBACK_RATE, switch - previous)
        end[switch] = advance(theta, data, "gd", GD_RATE, STEPS - switch)
        previous = switch
    return end, digest(data[0].tobytes() + data[1].tobytes())


def quota_mask(scores, tie, allocate):
    mask = allocate(scores.reshape(-1, BATCH), tie.reshape(-1, BATCH), BUDGET)
    np.testing.assert_array_equal(mask.sum(axis=1), np.full(len(mask), BUDGET))
    return mask.ravel()


def evaluate(model, gate, x, p, tie, prior):
    predict = prior.with_model(model)
    a, b = predict(x), predict(-x)
    m = (a + b) / 2
    la, lb, lm = prior.nll(a, p), prior.nll(b, p), prior.nll(m, p)
    raw = (la + lb) / 2
    d = prior.gap(a, b)
    folded_priority = gate.priority(x, "folded")
    np.testing.assert_array_equal(folded_priority, gate.priority(-x, "folded"))
    f = quota_mask(folded_priority, tie, prior.allocate)
    ma = quota_mask(-abs(a), tie, prior.allocate)
    mb = quota_mask(-abs(b), tie, prior.allocate)
    folded = np.where(f, lm, raw)
    confidence = (np.where(ma, lm, la) + np.where(mb, lm, lb)) / 2
    avgmask = (ma.astype(int) + mb) / 2
    signmask = ma.astype(int) - mb.astype(int)
    symmetric = avgmask * d
    signed = signmask * (la - lb) / 4
    folded_gain = f * d
    point_resid = float(np.max(np.abs((folded - confidence) - (symmetric + signed - folded_gain))))
    if point_resid > 1e-11 or np.min(d) < -1e-12:
        raise AssertionError("The pointwise routing identity or Jensen inequality failed")
    # Independent actual-call audits on the first batch for both folded and confidence.
    for z, first, mask, gate_name in ((x, a, f, "folded"), (x, a, ma, "confidence"), (-x, b, mb, "confidence")):
        calls = []
        def counted(inputs):
            calls.append(len(inputs))
            return predict(inputs)
        if gate_name == "folded":
            output, selected = prior.deploy(z[:BATCH], counted, gate, gate_name, tie[:BATCH], BUDGET)
        else:
            initial = counted(z[:BATCH])
            selected = quota_mask(-abs(initial), tie[:BATCH], prior.allocate)
            output = initial.copy()
            output[selected] = (initial[selected] + counted(-z[:BATCH][selected])) / 2
        assert calls == [BATCH, BUDGET]
        np.testing.assert_array_equal(selected, mask[:BATCH])
        np.testing.assert_allclose(output, np.where(selected, m[:BATCH], first[:BATCH]), atol=1e-11, rtol=1e-11)
    values = dict(raw_loss=raw, full_loss=lm, folded_loss=folded, confidence_loss=confidence,
                  gap=folded - confidence, symmetric_gain=symmetric, orientation_term=signed,
                  folded_gain=folded_gain, invariant_advantage=folded_gain - symmetric,
                  mask_disagreement=ma != mb)
    summary = {k: float(v.mean()) for k, v in values.items()}
    summary.update({k: int(v.sum()) for k, v in {"folded_extra": f, "confidence_extra": ma,
                                                  "confidence_swapped_extra": mb}.items()})
    summary["max_identity_residual"] = point_resid
    return summary, {k: v.reshape(-1, BATCH).mean(axis=1) for k, v in values.items()}


def main(prior_dir, out, pairs):
    sys.path.insert(0, str(prior_dir.resolve()))
    import run as prior
    assert (prior.CAL, prior.EVAL, prior.BATCH, prior.QUOTA) == (N_CAL, N_EVAL, BATCH, BUDGET)
    with (prior_dir / "supplemental_results" / "models.csv").open(encoding="utf-8", newline="") as f:
        expected = {(int(r["pair"]), r["model"]): r for r in csv.DictReader(f)}
    with np.load(prior_dir / "results" / "models.npz") as old:
        anchor_hash = {key: digest(old[key].tobytes()) for key in old.files}
    out.mkdir(exist_ok=False, parents=True)
    start = perf_counter()
    all_states, tables, rows, batches, anchors = {}, {}, [], [], []
    for number, pair in enumerate(pairs, start=1):
        end, training_hash = trajectories(pair, prior.rings, prior.advance)
        cal = prior.sample_inputs(123000 + pair, N_CAL)
        x = prior.sample_inputs(131000 + pair, N_EVAL)
        tie = np.random.default_rng(132000 + pair).random(N_EVAL)
        p = prior.posterior(np.linalg.norm(x, axis=1))
        hashes = dict(train_hash=training_hash, cal_hash=digest(cal.tobytes()),
                      eval_hash=digest(x.tobytes()), tie_hash=digest(tie.tobytes()))
        for switch, theta in end.items():
            key = f"p{pair}_s{switch}_t{STEPS}"
            all_states[key] = theta
            gate = prior.Gate.fit(cal, prior.with_model(theta))
            for field in ("edges", "radius_scores", "folded_scores"):
                tables[f"{key}_{field}"] = getattr(gate, field)
            result, grouped = evaluate(theta, gate, x, p, tie, prior)
            labels = dict(pair=pair, pb_steps=switch, step=STEPS, gd_steps_after=STEPS - switch)
            state_hash = digest(theta.tobytes())
            rows.append(dict(**labels, **result, **hashes, state_hash=state_hash,
                             train_seed=121000+pair, init_seed=122000+pair, cal_seed=123000+pair,
                             eval_seed=131000+pair, tie_seed=132000+pair))
            for idx in range(N_EVAL // BATCH):
                batches.append(dict(**labels, batch=idx+1, **{k: float(v[idx]) for k, v in grouped.items()}))
            if switch in ANCHORS and pair in PAIRED_COHORT:
                name = ANCHORS[switch]
                archived = expected[pair, name]
                assert state_hash == anchor_hash[f"{pair}_{name}"]
                assert archived["eval_hash"] == hashes["eval_hash"]
                error = max(abs(float(archived[k]) - result[k]) for k in ("raw_loss", "full_loss",
                                                                          "folded_loss", "confidence_loss"))
                if error > 1e-12:
                    raise AssertionError(f"Anchor discrepancy {pair}/{name}: {error}")
                anchors.append(dict(**labels, model=name, exact_state=True, max_policy_loss_difference=error))
        print(f"Finished seed {pair} ({number}/{len(pairs)}) in {perf_counter()-start:.1f}s cumulative", flush=True)
    save_csv(out / "endpoints.csv", rows)
    save_csv(out / "endpoint_batches.csv", batches)
    if anchors:
        save_csv(out / "anchor_audit.csv", anchors)
    np.savez_compressed(out / "states.npz", **all_states)
    np.savez_compressed(out / "calibrated_gates.npz", **tables)
    meta = dict(switches=SWITCHES, pairs=pairs, total_steps=STEPS, pb_lr=PULLBACK_RATE, damping=DAMPING,
                gd_lr=GD_RATE, cal_examples=N_CAL, eval_examples=N_EVAL, batch=BATCH, quota=BUDGET,
                evaluation="Seeds 1-16 use the prior confidence holdout; seeds 17-32 use independent inputs. Schedules use common random numbers within each pair.",
                gap_definition="folded_loss minus confidence_loss, positive favors first-view confidence",
                identity="gap = orientation_term - invariant_advantage",
                elapsed_seconds=perf_counter()-start, fresh_training="All states recomputed from initial weights",
                input_hashes={path.name:digest(path.read_bytes()) for path in (
                    Path(__file__), prior_dir/"run.py", prior_dir/"gate.py", prior_dir/"generator.py", Path(core.__file__))})
    with (out / "endpoint_metadata.json").open("x", encoding="utf8") as stream:
        json.dump(meta, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=find_prior())
    parser.add_argument("--out", type=Path, default=HERE/"results")
    parser.add_argument("--first-pair", type=int, default=1)
    parser.add_argument("--last-pair", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.first_pair <= args.last_pair <= max(VALIDATION_COHORT):
        parser.error("Choose seed indices between 1 and 32, inclusive.")
    with threadpool_limits(1):
        main(args.prior, args.out, tuple(range(args.first_pair, args.last_pair + 1)))
