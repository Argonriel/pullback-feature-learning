"""Audit the explicit controller-refinement supplement without reading scores.

The earlier audit script and its results are preserved. This extension reuses
its independent checks and controller implementation, then verifies assembly
and actual training-prediction sensitivity to the two declared controllers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import audit_gate as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--previous-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    started = time.perf_counter()
    old = audit.read_json(args.previous_audit)
    assert old["audit_source_sha256"] == audit.sha(audit.__file__)
    with threadpool_limits(1):
        fine = audit.audit_archive(args.archive / "refinement")
        cfg, data, states, rows, fine_report = fine
        coarse_cfg = audit.read_json(args.archive / "discovery/config.json")
        changed_fields = [key for key in set(cfg) | set(coarse_cfg)
                          if cfg.get(key) != coarse_cfg.get(key)]
        assert set(changed_fields) == {"step_sizes", "param_step_cap", "max_steps"}
        assert cfg["step_sizes"] == [0.005]
        assert cfg["param_step_cap"] == 0.05 and cfg["max_steps"] == 120000
        assert fine_report["state_count"] == 128
        assert fine_report["summary_status_counts"] == {"complete": 32}
        for folder in ("discovery", "coverage_extension"):
            bank = dict(np.load(args.archive / folder / "data.npz", allow_pickle=False))
            for key in bank:
                assert np.array_equal(bank[key], data[key])
        replays = [audit.replay(cfg, data, task, arm, .005, rows, states)
                   for task, arm in (("rings", "tangent_pb"), ("moons", "nonlinear_pb"))]

        # Check all assembled states against the correct untouched component.
        assembled = args.archive / "discovery_refined"
        provenance = audit.read_json(assembled / "provenance.json")
        component_banks, component_rows = {}, {}
        for component in provenance["components"]:
            folder = Path(component["path"])
            assert audit.sha(folder / "provenance.json") == component["provenance_sha256"]
            component_banks[folder.name] = dict(np.load(folder / "states.npz", allow_pickle=False))
            component_rows[folder.name] = {
                audit.identity(row): row for row in audit.read_json(folder / "snapshots.json")
                if row["status"] in ("initial", "target")}
        for name, expected in provenance["output_sha256"].items():
            assert audit.sha(assembled / name) == expected
        for path, expected in provenance["source_sha256"].items():
            assert audit.sha(path) == expected
        assembled_states = dict(np.load(assembled / "states.npz", allow_pickle=False))
        assembled_rows = audit.read_json(assembled / "snapshots.json")
        assert len(assembled_rows) == len(assembled_states) == 256
        counts = {}
        for row in assembled_rows:
            source_name = ("refinement" if row["step_size"] == .005 else
                           "coverage_extension" if row["task"] == "rings"
                           and row["arm"].startswith("tangent") else "discovery")
            source = component_rows[source_name][audit.identity(row)]
            assert np.array_equal(assembled_states[row["state_key"]],
                                  component_banks[source_name][source["state_key"]])
            for field in ("train_loss", "step", "target_error", "status", "initial_loss"):
                assert row[field] == source[field]
            expected_cap = .05 if source_name == "refinement" else .1
            assert row["param_step_cap"] == expected_cap
            expected_ceiling = {"discovery": 6000, "coverage_extension": 60000,
                                "refinement": 120000}[source_name]
            assert row["component_max_steps"] == expected_ceiling
            counts[source_name] = counts.get(source_name, 0) + 1

        # Quantify actual prediction sensitivity, using only training inputs.
        primary = {audit.identity(row): row for row in assembled_rows
                   if row["status"] == "target" and row["fraction"] == .25}
        sensitivities = []
        cache = {}
        for key, coarse in primary.items():
            if coarse["step_size"] != .01:
                continue
            refined_key = key[:4] + (.005,) + key[5:]
            refined = primary[refined_key]
            data_key = coarse["task"] + "__" + coarse["seed"]
            xx, theta0 = (data[data_key + "__" + suffix] for suffix in ("x", "theta0"))
            if data_key not in cache:
                cache[data_key] = (audit.forward(theta0, xx), audit.complex_jacobian(theta0, xx))
            z0, j0 = cache[data_key]
            theta_coarse = assembled_states[coarse["state_key"]]
            theta_fine = assembled_states[refined["state_key"]]
            if coarse["arm"].startswith("tangent"):
                z_coarse = (z0.ravel() + j0 @ (theta_coarse - theta0)).reshape(-1, 2)
                z_fine = (z0.ravel() + j0 @ (theta_fine - theta0)).reshape(-1, 2)
            else:
                z_coarse = audit.forward(theta_coarse, xx)
                z_fine = audit.forward(theta_fine, xx)
            sensitivities.append({"task": coarse["task"], "seed": coarse["seed"],
                "arm": coarse["arm"], "fraction": .25,
                "coarse_steps": coarse["step"], "refined_steps": refined["step"],
                "fine_to_coarse_step_count_ratio": refined["step"] / coarse["step"],
                "state_identical": bool(np.array_equal(theta_coarse, theta_fine)),
                "parameter_distance": float(np.linalg.norm(theta_coarse - theta_fine)),
                "training_logit_rms_difference": float(np.sqrt(np.mean((z_coarse[:, 0] - z_fine[:, 0]) ** 2))),
                "training_loss_difference": refined["train_loss"] - coarse["train_loss"]})
        assert len(sensitivities) == 32
        assert not any(row["state_identical"] for row in sensitivities)
        grouped = []
        for task in ("moons", "rings"):
            for arm in cfg["arms"]:
                subset = [row for row in sensitivities if row["task"] == task and row["arm"] == arm]
                grouped.append({"task": task, "arm": arm,
                    "mean_training_logit_rms_difference": float(np.mean([r["training_logit_rms_difference"] for r in subset])),
                    "max_training_logit_rms_difference": max(r["training_logit_rms_difference"] for r in subset),
                    "mean_step_count_ratio": float(np.mean([r["fine_to_coarse_step_count_ratio"] for r in subset]))})
    report = {"status": "passed; actual controller refinement verified",
        "evaluation_files_read": [], "refinement_archive_audit": fine_report,
        "changed_config_fields": changed_fields,
        "source_data_matches_original_and_extension": True,
        "representative_independent_replays": replays,
        "assembled_states_exactly_matched_to_components": len(assembled_rows),
        "assembled_component_counts": counts,
        "primary_training_prediction_sensitivity": sensitivities,
        "primary_sensitivity_summary": grouped,
        "caveats": [
            "The report uses training artifacts only; it makes no held-out-risk conclusion or gate decision.",
            "Both scalar upper bounds are halved at a common state. Backtracking can alter actual accepted-step ratios.",
            "All 32 primary parameter states change under the repaired refinement. Two resolutions quantify sensitivity, not a convergence theorem.",
            "Small matched-training prediction changes do not bound population-risk changes without additional assumptions.",
            "The original protocol and both amendments retain their distinct status; the assembled run is supplemental.",
            "Replay reuses the native analytic Jacobian, separately checked by independent complex-step differentiation. No study runner is imported."],
        "previous_audit_sha256": audit.sha(args.previous_audit),
        "preserved_previous_audit_code_sha256": audit.sha(audit.__file__),
        "audit_source_sha256": audit.sha(__file__),
        "refinement_amendment_sha256": audit.sha(args.archive / "REFINEMENT_AMENDMENT.md"),
        "elapsed_seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
        "states_checked": fine_report["state_count"], "assembled_states_checked": len(assembled_rows),
        "independent_replay_steps": sum(r["steps_replayed"] for r in replays),
        "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
