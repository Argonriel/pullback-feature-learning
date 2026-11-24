"""Score a fixed first-view uncertainty rule with separately routed orientations."""
import argparse
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from gate import Gate, allocate
from run import BATCH, EVAL, QUOTA, ROOT, csvout, jsonout, nll, posterior, sample_inputs, sha, with_model


def route(priority, tie):
    selected = allocate(priority.reshape(-1, BATCH), tie.reshape(-1, BATCH), QUOTA)
    if not np.all(selected.sum(axis=1) == QUOTA):
        raise AssertionError("A batch missed the deployment quota")
    return selected.ravel()


def deploy_confidence(x, predict, tie):
    logits = predict(x).copy()
    mask = route(-np.abs(logits), tie)
    logits[mask] = 0.5 * (logits[mask] + predict(-x[mask]))
    return logits, mask


def audit_deployment(x, pred, tie, expected_scores, expected_mask):
    calls = []

    def counted(z):
        calls.append(len(z))
        return pred(z)

    actual, selected = deploy_confidence(x[:BATCH], counted, tie[:BATCH])
    assert calls == [BATCH, QUOTA]
    np.testing.assert_array_equal(selected, expected_mask[:BATCH])
    np.testing.assert_allclose(actual, expected_scores[:BATCH], rtol=1e-11, atol=1e-11)


def evaluate(theta, gate, x, tie):
    pred = with_model(theta)
    first, swapped = pred(x), pred(-x)
    averaged = 0.5 * (first + swapped)
    p = posterior(np.linalg.norm(x, axis=1))
    raw_risk = 0.5 * (nll(first, p) + nll(swapped, p))
    full_risk = nll(averaged, p)
    quota_fraction = QUOTA / BATCH
    losses = {"raw": raw_risk, "full": full_risk,
              "random": (1 - quota_fraction) * raw_risk + quota_fraction * full_risk}
    for policy in ("radius", "folded"):
        priorities = gate.priority(x, policy)
        np.testing.assert_array_equal(priorities, gate.priority(-x, policy))
        chosen = route(priorities, tie)
        losses[policy] = np.where(chosen, full_risk, raw_risk)

    first_mask = route(-np.abs(first), tie)
    swapped_mask = route(-np.abs(swapped), tie)
    first_output = np.where(first_mask, averaged, first)
    swapped_output = np.where(swapped_mask, averaged, swapped)
    losses["confidence"] = 0.5 * (nll(first_output, p) + nll(swapped_output, p))
    audit_deployment(x, pred, tie, first_output, first_mask)
    audit_deployment(-x, pred, tie, swapped_output, swapped_mask)
    batched = {k: a.reshape(-1, BATCH).mean(axis=1) for k, a in losses.items()}
    assert all(np.all(np.isfinite(a)) for a in batched.values())
    return batched, np.mean(first_mask != swapped_mask)


def main(archives, out):
    with (archives / "training_and_calibration.csv").open(newline="", encoding="utf-8") as src:
        provenance = {(int(row["pair"]), row["model"]): row for row in csv.DictReader(src)}
    models = ("gd", "pb", "hybrid")
    fields = ("edges", "radius_scores", "folded_scores")
    batch_rows, rows = [], []
    with np.load(archives / "models.npz") as states, np.load(archives / "calibrated_gates.npz") as tables:
        assert len(states.files) == 48 and len(tables.files) == 144
        for pair in range(1, 17):
            x = sample_inputs(131000 + pair, EVAL)
            tie = np.random.default_rng(132000 + pair).random(EVAL)
            for model in models:
                key = f"{pair}_{model}"
                theta = states[key]
                assert sha(theta.tobytes()) == provenance[(pair, model)]["state_hash"]
                gate = Gate(*(tables[f"{key}_{field}"] for field in fields))
                losses, disagreement = evaluate(theta, gate, x, tie)
                rows.append(dict(pair=pair, model=model, eval_seed=131000 + pair,
                                 tie_seed=132000 + pair, eval_hash=sha(x.tobytes()),
                                 **{f"{k}_loss": float(a.mean()) for k, a in losses.items()},
                                 confidence_orientation_mask_disagreement=float(disagreement),
                                 extra_per_orientation=EVAL // BATCH * QUOTA))
                for batch in range(EVAL // BATCH):
                    batch_rows.append(dict(pair=pair, model=model, batch=batch + 1,
                                           **{f"{k}_loss": float(a[batch]) for k, a in losses.items()}))

    draws = np.random.default_rng(291031).integers(0, 16, size=(20000, 16))
    stats = {}
    for model in models:
        group = [r for r in rows if r["model"] == model]
        diffs = np.array([r["folded_loss"] - r["confidence_loss"] for r in group])
        assert len(diffs) == 16
        stats[model] = dict(mean_folded_minus_confidence=float(diffs.mean()),
                            ci95=np.quantile(diffs[draws].mean(axis=1), [.025, .975]).tolist(),
                            folded_lower_pairs=int(np.sum(diffs < 0)),
                            confidence_lower_pairs=int(np.sum(diffs > 0)),
                            confidence_mask_disagreement=float(np.mean(
                                [r["confidence_orientation_mask_disagreement"] for r in group])),
                            mean_loss={name: float(np.mean([r[f"{name}_loss"] for r in group]))
                                       for name in ("raw", "random", "radius", "folded", "confidence", "full")})
    out.mkdir(parents=True, exist_ok=False)
    csvout(out / "models.csv", rows)
    csvout(out / "batches.csv", batch_rows)
    hashes = {p.name: sha(p.read_bytes()) for p in
              (archives / "models.npz", archives / "calibrated_gates.npz",
               ROOT / "SUPPLEMENTAL_DESIGN.md", Path(__file__))}
    jsonout(out / "summary.json", dict(bootstrap_seed=291031, bootstrap_resamples=20000,
                                       n_pairs=16, n_eval_per_pair=EVAL, quota=QUOTA,
                                       batch_size=BATCH, by_model=stats, hashes=hashes))
    print(f"Verified and saved {len(rows)} checkpoints, {len(batch_rows)} batches and both deployment audits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "supplemental_results")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.archives, args.out)
