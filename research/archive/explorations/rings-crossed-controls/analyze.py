"""Verify the crossed rings experiment and summarize contrasts across paired seeds."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("gd", "hybrid")
POLICIES = ("original", "selected")
ENDPOINTS = ("step500", "time100ms")
CELLS = [f"{ep}.{policy}" for ep in ENDPOINTS for policy in POLICIES]
BUDGET_NS = 100_000_000
EXPECTED_RATES = {("gd", "original"): 0.125, ("hybrid", "original"): 0.125,
                  ("gd", "selected"): 0.5, ("hybrid", "selected"): 2.0}
BOOT_SEED = 20261120
BOOT_REPS = 20_000
MEASURES = ("validation_loss", "validation_accuracy_pp", "updates", "core_ms")


def check_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = {}
    seeds = {}
    repeats_by_trial = defaultdict(list)
    for row in rows:
        ep, policy, method = row["endpoint"], row["policy"], row["method"]
        pair, repeat = int(row["pair"]), int(row["repeat"])
        if ep not in ENDPOINTS or (method, policy) not in EXPECTED_RATES:
            raise ValueError(f"Unexpected cell in {row}")
        key = (pair, repeat, method, policy, ep)
        assert key not in records and float(row["gd_rate"]) == EXPECTED_RATES[method, policy], key
        seed_tuple = tuple(int(row[k]) for k in ("train_seed", "init_seed", "validation_seed"))
        assert seeds.setdefault(pair, seed_tuple) == seed_tuple, key
        state = row["state_sha256"]
        assert len(state) == 64 and int(state, 16) >= 0, key
        n, elapsed = int(row["updates"]), int(row["core_ns"])
        assert n >= 0 and elapsed >= 0 and (n == 0) == (elapsed == 0), key
        assert int(row["pullback_updates"]) == (min(n, 100) if method == "hybrid" else 0), key
        if ep == "step500":
            assert n == 500 and not row["cross_ns"], key
            repeats_by_trial[pair, method, policy].append(
                (state, float(row["validation_loss"]), float(row["validation_accuracy"])))
        else:
            assert elapsed <= BUDGET_NS < int(row["cross_ns"]), key
        loss, accuracy = float(row["validation_loss"]), float(row["validation_accuracy"])
        assert np.isfinite(loss) and loss >= 0 and 0 <= accuracy <= 1, key
        assert abs(256 * accuracy - round(256 * accuracy)) < 1e-8, key
        records[key] = row

    pairs = sorted(seeds)
    repeats = sorted({key[1] for key in records})
    assert len(pairs) == 16 and len(repeats) == 3 and len(records) == 384
    assert len({s for triplet in seeds.values() for s in triplet}) == 3 * len(pairs)
    order_by_trial = {}
    for (pair, repeat, method, policy, _), row in records.items():
        order = int(row["run_order"])
        assert order_by_trial.setdefault((pair, repeat, method, policy), order) == order
    assert sorted(order_by_trial.values()) == list(range(1, 193))
    for repeat in repeats:
        for method in METHODS:
            for policy in POLICIES:
                positions = [(order_by_trial[pair, repeat, method, policy] - 1) % 4 for pair in pairs]
                assert [positions.count(i) for i in range(4)] == [4, 4, 4, 4]
    for values in repeats_by_trial.values():
        assert len(values) == 3 and len(set(values)) == 1, "Fixed-step repeats differ"
    for pair in pairs:
        for repeat in repeats:
            for method in METHODS:
                for policy in POLICIES:
                    key = (pair, repeat, method, policy)
                    fixed, timed = (records[(*key, ep)] for ep in ENDPOINTS)
                    n = int(timed["updates"])
                    fixed_ns, timed_ns = int(fixed["core_ns"]), int(timed["core_ns"])
                    if n == 500:
                        assert fixed_ns == timed_ns and fixed["state_sha256"] == timed["state_sha256"], key
                    elif n > 500:
                        assert fixed_ns < timed_ns, key
                    else:
                        assert fixed_ns >= int(timed["cross_ns"]), key
    return records, seeds, pairs, repeats


def arm_means(records, pairs, repeats):
    means = {}
    for ep in ENDPOINTS:
        for policy in POLICIES:
            for method in METHODS:
                values = []
                for pair in pairs:
                    rows = [records[pair, r, method, policy, ep] for r in repeats]
                    values.append(np.mean([[float(row["validation_loss"]),
                                            100 * float(row["validation_accuracy"]),
                                            int(row["updates"]), int(row["core_ns"]) / 1e6]
                                           for row in rows], axis=0))
                means[ep, policy, method] = np.array(values)
    return means


def contrasts(cell):
    so, ss, to, ts = (cell[key] for key in CELLS)
    result = dict(rate_at_step500=ss-so, rate_at_time100ms=ts-to,
                  budget_at_original=to-so, budget_at_selected=ts-ss,
                  interaction=(ts-to)-(ss-so), total_change=ts-so,
                  rate_only_closure_minus_budget_only_closure=to-ss)
    np.testing.assert_allclose(result["interaction"], (ts-ss)-(to-so), atol=1e-12)
    np.testing.assert_allclose(result["total_change"],
                               result["rate_at_step500"] + result["budget_at_selected"], atol=1e-12)
    np.testing.assert_allclose(result["total_change"],
                               result["budget_at_original"] + result["rate_at_time100ms"], atol=1e-12)
    return result


def estimate(vec, indices):
    bounds = np.quantile(np.mean(vec[indices], axis=1), [0.025, 0.975])
    return dict(mean=float(vec.mean()), sd_across_pairs=float(vec.std(ddof=1)),
                median=float(np.median(vec)), percentile_95=bounds.tolist(),
                n_positive=int(np.sum(vec > 1e-12)), n_negative=int(np.sum(vec < -1e-12)),
                n_tied=int(np.sum(np.abs(vec) <= 1e-12)))


def plot_summary(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))
    fig.subplots_adjust(left=.095, right=.98, bottom=.31, top=.80, wspace=.30)
    fig.suptitle("How much does the pullback warm-up help on rings?", fontsize=15, y=.965)
    colors = ["#447699", "#bc642b"]
    for ax, measure, title in zip(axes, ["loss_advantage", "accuracy_advantage_pp"],
                                  ["Validation loss advantage", "Accuracy advantage (percentage points)"]):
        ax.set_title(title, fontsize=11, loc="left", pad=9)
        ax.axhline(0, color="#60666c", linestyle="--", linewidth=.8, zorder=0)
        for k, (ep, label) in enumerate(zip(ENDPOINTS, ["500 updates", "100 ms of training"])):
            vals = [summary["cell_margins"][f"{ep}.{p}"][measure] for p in POLICIES]
            x = np.arange(2) + (k-.5)*.07
            y = [v["mean"] for v in vals]
            lower, upper = np.array([v["percentile_95"] for v in vals]).T
            ax.plot(x, y, color=colors[k], marker=("o", "D")[k], ms=6, label=label)
            ax.vlines(x, lower, upper, color=colors[k], linewidth=1.2)
            ax.hlines(lower, x-.017, x+.017, color=colors[k], linewidth=1.2)
            ax.hlines(upper, x-.017, x+.017, color=colors[k], linewidth=1.2)
        ax.set(xlim=(-.20, 1.20), xticks=[0, 1], xticklabels=["Inherited rate", "Selected rates"])
        ax.margins(y=.13)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center",
               ncols=2, bbox_to_anchor=(.5, .115))
    fig.text(.5, .075, "Positive values favor the 100-step pullback warm-up over ordinary GD.",
             ha="center", fontsize=9)
    fig.text(.5, .030, "Means and 95% paired-seed bootstrap intervals across 16 pairs; three timing repeats averaged per pair.",
             ha="center", fontsize=8.3)
    fig.savefig(out / "crossed_rings.png", dpi=180)
    fig.savefig(out / "crossed_rings.svg")
    plt.close(fig)


def main(out):
    records, seeds, pairs, repeats = check_rows(out / "rows.csv")
    means = arm_means(records, pairs, repeats)
    indices = np.random.default_rng(BOOT_SEED).integers(0, len(pairs), size=(BOOT_REPS, len(pairs)))
    summary = dict(pair_count=len(pairs), timing_repeats=len(repeats), endpoint_records=len(records),
                   bootstrap=dict(replicates=BOOT_REPS, seed=BOOT_SEED, unit="paired seed triplet",
                                  interval="pointwise 95% percentile interval of the paired mean"),
                   conventions=dict(loss_advantage="GD loss minus hybrid loss",
                                    accuracy_advantage_pp="hybrid accuracy minus GD accuracy in percentage points",
                                    rate_change="selected minus original",
                                    budget_change="100 ms minus 500 steps",
                                    interaction="rate change at 100 ms minus rate change at 500 steps"),
                   arm_means={}, cell_margins={}, contrasts={}, arm_changes={})
    timed = [row for row in records.values() if row["endpoint"] == "time100ms"]
    summary["timing_checks"] = dict(
        minimum_retained_ms=min(int(row["core_ns"]) for row in timed) / 1e6,
        maximum_crossing_ms=max(int(row["cross_ns"]) for row in timed) / 1e6,
        warmups_completed=sum(int(row["pullback_updates"]) == 100 for row in timed if row["method"] == "hybrid"),
        timed_hybrid_rows=sum(row["method"] == "hybrid" for row in timed),
    )
    margins = {}
    for ep in ENDPOINTS:
        for policy in POLICIES:
            label = f"{ep}.{policy}"
            summary["arm_means"][label] = {
                method: {name: estimate(values, indices)
                         for name, values in zip(MEASURES, means[ep, policy, method].T)}
                for method in METHODS}
            gd, hybrid = (means[ep, policy, m] for m in METHODS)
            margins[label] = np.column_stack([gd[:, 0]-hybrid[:, 0], hybrid[:, 1]-gd[:, 1]])

    measures = ("loss_advantage", "accuracy_advantage_pp")
    changes = contrasts(margins)
    paired_rows = []
    for target, vals in (("cell_margins", margins), ("contrasts", changes)):
        for label, values in vals.items():
            summary[target][label] = {name: estimate(vec, indices) for name, vec in zip(measures, values.T)}
            for i, pair in enumerate(pairs):
                paired_rows.append(dict(pair=pair, section=target, quantity=label,
                                        loss_advantage=float(values[i, 0]),
                                        accuracy_advantage_pp=float(values[i, 1])))
    for method in METHODS:
        cells = {f"{e}.{p}": means[e, p, method][:, :2] for e in ENDPOINTS for p in POLICIES}
        summary["arm_changes"][method] = {
            name: {measure: estimate(vec, indices) for measure, vec in zip(MEASURES[:2], value.T)}
            for name, value in contrasts(cells).items()}
    with (out / "paired_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot_summary(summary, out)
    print(f"Verified {len(records)} endpoint rows, {len(pairs)} paired draws and {len(paired_rows)} paired summaries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    main(parser.parse_args().results)
