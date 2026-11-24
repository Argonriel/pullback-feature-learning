"""Independent training-artifact audit; never reads predictive evaluation files.

This does not import the study runner. Seed generation, forward evaluation,
loss calculations, controller, and replay logic are implemented here. The
archived core Jacobian is reused for replays, and independently checked by
complex-step differentiation of this file's forward implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pullback_feature_learning.core import representation_jacobian


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def forward(theta, x):
    h = x
    for offset in (0, 40):
        weights_in = theta[offset:offset + 16].reshape(8, 2)
        weights_out = theta[offset + 16:offset + 32].reshape(2, 8)
        bias = theta[offset + 32:offset + 40]
        h = h + 0.5 * np.tanh(h @ weights_in.T + bias) @ weights_out.T
    return h


def complex_jacobian(theta, x):
    jac = np.empty((2 * len(x), 80))
    for k in range(80):
        perturbed = theta.astype(complex)
        perturbed[k] += 1e-25j
        jac[:, k] = np.imag(forward(perturbed, x)).ravel() / 1e-25
    return jac


def loss(z, y):
    return float(np.mean(np.logaddexp(0.0, -y * z[:, 0])))


def initial_parameters(seed):
    gen = np.random.default_rng(seed)
    pieces = []
    for _ in range(2):
        pieces.append(gen.normal(0.0, 1.0 / np.sqrt(2.0), (8, 2)).reshape(-1))
        pieces.append(gen.normal(0.0, 0.1 / np.sqrt(8.0), (2, 8)).reshape(-1))
        pieces.append(np.zeros(8))
    return np.concatenate(pieces)


def data_from_seed(task, seed, count):
    gen = np.random.default_rng(seed)
    label = np.concatenate((-np.ones(count // 2), np.ones(count // 2)))
    if task == "moons":
        angle = gen.uniform(0.0, np.pi, count)
        x = np.stack((np.cos(angle), np.sin(angle)), axis=1)
        x[count // 2:, 0] = 1.0 - x[count // 2:, 0]
        x[count // 2:, 1] = 0.5 - x[count // 2:, 1]
    else:
        angle = gen.uniform(0.0, 2.0 * np.pi, count)
        radius = np.where(label < 0, 0.5, 1.2)
        x = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)
    x += gen.normal(0.0, 0.15, x.shape)
    permutation = gen.permutation(count)
    return x[permutation], label[permutation]


def identity(row, include_fraction=True):
    damping = row["damping"]
    values = (row["task"], row["seed"], row["arm"],
              None if damping is None or damping == "" else float(damping),
              float(row["step_size"]))
    return values + (float(row["fraction"]),) if include_fraction else values


def audit_archive(folder):
    cfg, provenance = read_json(folder / "config.json"), read_json(folder / "provenance.json")
    assert "finished_utc" in provenance, "Do not audit an actively written run"
    checked_hashes = []
    for path, expected in provenance["source_sha256"].items():
        assert sha(path) == expected, f"Source mismatch: {path}"
        checked_hashes.append(path)
    assert sha(provenance["config_source"]) == provenance["config_sha256"]
    assert read_json(provenance["config_source"]) == cfg
    for name, expected in provenance["output_sha256"].items():
        assert sha(folder / name) == expected, f"Artifact mismatch: {folder / name}"
        checked_hashes.append(str(folder / name))
    data = dict(np.load(folder / "data.npz", allow_pickle=False))
    states = dict(np.load(folder / "states.npz", allow_pickle=False))
    rows = read_json(folder / "snapshots.json")
    summaries = read_csv(folder / "summaries.csv")
    actual = read_json(folder / "actual_seeds.json")
    regenerated = []
    for seed in actual:
        expected_seed = next(s for s in cfg["seeds"] if s["id"] == seed["id"])
        offset = 10000 if seed["task"] == "rings" else 0
        for field in ("train_seed", "init_seed", "test_seed"):
            assert seed[field] == expected_seed[field] + offset
        key = seed["task"] + "__" + seed["id"]
        xx, yy = data_from_seed(seed["task"], seed["train_seed"], cfg["n_train"])
        theta0 = initial_parameters(seed["init_seed"])
        assert np.array_equal(xx, data[key + "__x"])
        assert np.array_equal(yy, data[key + "__y"])
        assert np.array_equal(theta0, data[key + "__theta0"])
        regenerated.append(key)
    tangent_cache = {}
    max_loss_error, max_target_error = 0.0, 0.0
    max_jac_error = 0.0
    for row in rows:
        key = row["task"] + "__" + row["seed"]
        xx, yy, theta0 = (data[key + "__" + suffix] for suffix in ("x", "y", "theta0"))
        theta = states[row["state_key"]]
        assert theta.shape == (80,) and np.all(np.isfinite(theta))
        if key not in tangent_cache:
            independent_jac = complex_jacobian(theta0, xx)
            native_jac = representation_jacobian(theta0, xx)
            max_jac_error = max(max_jac_error, float(np.max(np.abs(independent_jac - native_jac))))
            tangent_cache[key] = (forward(theta0, xx), independent_jac)
        zz0, j0 = tangent_cache[key]
        if row["arm"].startswith("tangent"):
            zz = (zz0.ravel() + j0 @ (theta - theta0)).reshape(-1, 2)
        else:
            zz = forward(theta, xx)
        computed_loss = loss(zz, yy)
        max_loss_error = max(max_loss_error, abs(computed_loss - row["train_loss"]))
        assert abs(computed_loss - row["train_loss"]) < 2e-11
        assert abs(loss(zz0, yy) - row["initial_loss"]) < 1e-14
        if row["status"] == "initial":
            assert row["step"] == 0 and np.array_equal(theta, theta0)
        elif row["status"] == "target":
            error = computed_loss - row["fraction"] * row["initial_loss"]
            max_target_error = max(max_target_error, abs(error))
            assert abs(error) < cfg["target_atol"] + 2e-11
            assert abs(error - row["target_error"]) < 2e-11
        assert row["max_parameter_step"] <= cfg["param_step_cap"] + 1e-12
        assert row["max_loss_increase"] <= 1e-12
    assert max_jac_error < 1e-12
    assert len(states) == len(rows) == provenance["state_count"]
    assert len(summaries) == provenance["run_count"]
    summary_counts = {}
    for row in summaries:
        summary_counts[row["status"]] = summary_counts.get(row["status"], 0) + 1
        if row["status"] == "complete":
            assert int(row["completed_targets"]) == len(cfg["loss_fractions"])
        else:
            assert row["status"] == "max_steps"
            assert int(row["step"]) == cfg["max_steps"]
        assert float(row["max_parameter_step"]) <= cfg["param_step_cap"] + 1e-12
        assert float(row["max_taylor_ratio"]) <= cfg["remainder_rtol"] + 1e-6
    report = {"folder": str(folder), "hashes_checked": len(checked_hashes) + 1,
              "seed_pairs_exactly_regenerated": regenerated, "state_count": len(rows),
              "summary_status_counts": summary_counts,
              "independent_forward_loss_max_error": max_loss_error,
              "independent_target_max_error": max_target_error,
              "complex_step_initial_jacobian_max_error": max_jac_error}
    return cfg, data, states, rows, report


def replay(cfg, data, task, arm, step_size, reference_rows, states):
    key = task + "__d1"
    xx, yy, theta0 = (data[key + "__" + suffix] for suffix in ("x", "y", "theta0"))
    j_initial, z_initial = representation_jacobian(theta0, xx), forward(theta0, xx)
    theta = theta0.copy()
    zz = z_initial.copy()
    current_loss = loss(zz, yy)
    target_losses = [f * current_loss for f in cfg["loss_fractions"]]
    target_index = 0
    references = {int(row["step"]): row for row in reference_rows if row["task"] == task
                  and row["seed"] == "d1" and row["arm"] == arm
                  and row["step_size"] == step_size and row["status"] != "initial"}
    max_param = max_rem = max_difference = max_loss_difference = 0.0
    cap_count = backtrack_count = 0
    matched = []
    max_predicted_logit_rms = 0.0
    maximum_jacobian_discrepancy = 0.0

    def model(parameters):
        if arm.startswith("tangent"):
            return (z_initial.ravel() + j_initial @ (parameters - theta0)).reshape(-1, 2)
        return forward(parameters, xx)

    for step in range(1, cfg["max_steps"] + 1):
        jac = j_initial if arm.startswith("tangent") else representation_jacobian(theta, xx)
        if step in (1, 50) and arm == "nonlinear_pb":
            maximum_jacobian_discrepancy = max(maximum_jacobian_discrepancy,
                float(np.max(np.abs(jac - complex_jacobian(theta, xx)))))
        readout_derivative = -yy * np.exp(-np.logaddexp(0.0, yy * zz[:, 0])) / len(yy)
        gg = np.zeros_like(zz)
        gg[:, 0] = readout_derivative
        gradient = jac.T @ gg.ravel()
        metric = jac.T @ jac
        metric.flat[::81] += 0.1
        update = np.linalg.solve(metric, -gradient)
        slope = float(gradient @ update)
        velocity = jac[::2] @ update
        velocity_rms = float(np.sqrt(np.mean(velocity ** 2)))
        direction_norm = float(np.linalg.norm(update))
        logit_rate = step_size / velocity_rms
        cap_rate = cfg["param_step_cap"] / direction_norm
        cap_count += cap_rate < logit_rate
        rate = min(logit_rate, cap_rate)

        def proposal(rate_value):
            parameters = theta + rate_value * update
            outputs = model(parameters)
            objective = loss(outputs, yy)
            remainder = float(np.sqrt(np.mean((outputs[:, 0] - zz[:, 0]
                                               - rate_value * velocity) ** 2)))
            valid = (np.isfinite(objective) and np.all(np.isfinite(parameters))
                     and objective <= current_loss + cfg["armijo"] * rate_value * slope
                     and remainder <= cfg["remainder_rtol"] * rate_value * velocity_rms + 1e-12)
            return parameters, outputs, objective, remainder, valid

        for backtracks in range(61):
            candidate, output, value, remainder, accepted = proposal(rate)
            if accepted:
                backtrack_count += backtracks
                break
            rate /= 2.0
        else:
            raise AssertionError("Independent replay line search failed")
        is_crossing = value <= target_losses[target_index]
        if is_crossing:
            lo, hi = 0.0, rate
            for _ in range(80):
                middle = (lo + hi) / 2.0
                middle_loss = loss(model(theta + middle * update), yy)
                if middle_loss <= target_losses[target_index]:
                    hi = middle
                else:
                    lo = middle
                if (middle_loss <= target_losses[target_index]
                        and abs(middle_loss - target_losses[target_index]) <= cfg["target_atol"]):
                    break
            rate = hi
            candidate, output, value, remainder, accepted = proposal(rate)
            assert accepted
            assert abs(value - target_losses[target_index]) <= cfg["target_atol"]
        assert rate * direction_norm <= cfg["param_step_cap"] + 1e-12
        assert rate * velocity_rms <= step_size + 1e-12
        max_param = max(max_param, rate * direction_norm)
        max_rem = max(max_rem, remainder / max(rate * velocity_rms, 1e-300))
        max_predicted_logit_rms = max(max_predicted_logit_rms, rate * velocity_rms)
        theta, zz, current_loss = candidate, output, value
        if step in references:
            row = references[step]
            difference = float(np.max(np.abs(theta - states[row["state_key"]])))
            max_difference = max(max_difference, difference)
            max_loss_difference = max(max_loss_difference, abs(value - row["train_loss"]))
            assert difference < 2e-7, (task, arm, step, difference)
            matched.append({"step": step, "fraction": row["fraction"],
                            "state_max_absolute_error": difference,
                            "state_exactly_equal": bool(np.array_equal(theta, states[row["state_key"]]))})
        if is_crossing:
            target_index += 1
            if target_index == len(target_losses):
                break
    assert len(matched) == len(references), (matched, references)
    assert maximum_jacobian_discrepancy < 1e-12
    return {"task": task, "seed": "d1", "arm": arm, "step_size": step_size,
            "steps_replayed": step, "matched_states": matched,
            "max_parameter_error": max_difference, "max_train_loss_error": max_loss_difference,
            "max_parameter_step": max_param, "max_taylor_ratio": max_rem,
            "max_predicted_logit_rms": max_predicted_logit_rms,
            "parameter_cap_active_before_backtracking_steps": int(cap_count),
            "parameter_cap_active_fraction": float(cap_count / step),
            "backtracks": backtrack_count,
            "complex_step_nonlinear_jacobian_error": maximum_jacobian_discrepancy,
            "independence_limit": "Native Jacobian reused; checked separately by complex-step differentiation. No runner imported."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    started = time.perf_counter()
    with threadpool_limits(1):
        discovery = audit_archive(args.archive / "discovery")
        extension = audit_archive(args.archive / "coverage_extension")
        cfg_d, data_d, states_d, rows_d, report_d = discovery
        cfg_e, data_e, states_e, rows_e, report_e = extension
        differences = [key for key in set(cfg_d) | set(cfg_e) if cfg_d.get(key) != cfg_e.get(key)]
        assert set(differences) == {"tasks", "arms", "max_steps"}
        assert cfg_d["max_steps"] == 6000 and cfg_e["max_steps"] == 60000
        assert cfg_e["tasks"] == ["rings"] and cfg_e["arms"] == ["tangent_gd", "tangent_pb"]
        ext_lookup = {identity(row): row for row in rows_e if row["status"] in ("initial", "target")}
        overlap = []
        for row in rows_d:
            if row["task"] != "rings" or not row["arm"].startswith("tangent"):
                continue
            if row["status"] not in ("initial", "target"):
                continue
            other = ext_lookup[identity(row)]
            assert np.array_equal(states_d[row["state_key"]], states_e[other["state_key"]])
            assert row["step"] == other["step"] and row["train_loss"] == other["train_loss"]
            overlap.append({"arm": row["arm"], "seed": row["seed"], "step_size": row["step_size"],
                            "fraction": row["fraction"], "step": row["step"]})
        for key in data_e:
            assert np.array_equal(data_e[key], data_d[key])
        replays = []
        for task, arm in (("moons", "nonlinear_pb"), ("moons", "tangent_pb"),
                          ("rings", "nonlinear_pb"), ("rings", "tangent_pb")):
            bundle = extension if task == "rings" and arm == "tangent_pb" else discovery
            cfg, data, states, rows, _ = bundle
            replays.append(replay(cfg, data, task, arm, .005, rows, states))
        # Quantify whether nominal logit-step halving actually changed saved
        # states. This flags a parameter-cap-dominated comparison, not a bug.
        nominal_refinement = []
        for name, bundle in (("discovery", discovery), ("coverage_extension", extension)):
            _, _, states, rows, _ = bundle
            lookup = {identity(row): row for row in rows if row["status"] == "target"}
            for key, row in lookup.items():
                if row["step_size"] != .01:
                    continue
                smaller = key[:4] + (.005,) + key[5:]
                if smaller in lookup:
                    other = lookup[smaller]
                    equal = np.array_equal(states[row["state_key"]], states[other["state_key"]])
                    if equal:
                        nominal_refinement.append({"archive": name, "task": row["task"],
                            "seed": row["seed"], "arm": row["arm"], "fraction": row["fraction"],
                            "identical_state_despite_nominal_step_halving": True})
    report = {"status": "passed with stated scope limitations", "evaluation_files_read": [],
              "archives": [report_d, report_e], "coverage_config_changed_fields": differences,
              "exact_shared_target_and_initial_states": overlap,
              "exact_shared_state_count": len(overlap), "representative_independent_replays": replays,
              "identical_nominal_refinement_targets": nominal_refinement,
              "caveats": [
                  "No held-out or population result is audited or inferred here.",
                  "The original 6000-step gate retains its censored outcomes. The 60000-step extension is supplemental, as declared.",
                  "Halving the logit-RMS target need not halve actual steps when the shared parameter cap dominates. Identical states are not independent flow-convergence evidence.",
                  "All saved states receive independent forward/loss checks; only four representative full trajectories receive step-by-step acceptance replay.",
                  "The native Jacobian is reused for replay but checked against independent complex-step derivatives at all initial datasets and representative nonlinear states.",
                  "The 1e-12 Taylor acceptance allowance means a recorded ratio can slightly exceed the nominal 0.05 threshold at extremely small steps; audit evaluates the actual additive condition.",
                  "Matched scalar training loss does not establish identical predictive paths or isolate metric adaptation; the planned fixed-metric intervention remains necessary.",
                  "The shared controller includes full Jacobian work and diagnostic searches, so these timings cannot fairly rank production GD versus PB costs."],
              "audit_source_sha256": sha(__file__), "elapsed_seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "saved_states_checked": report_d["state_count"] + report_e["state_count"],
                      "exact_overlap_states": len(overlap),
                      "replayed_steps": sum(r["steps_replayed"] for r in replays),
                      "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
