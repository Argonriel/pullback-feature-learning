"""Read archived rings states on their original training inputs; never train/score holdouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from threadpoolctl import threadpool_limits


def sha(data):
    return hashlib.sha256(data).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def ratio(a, b):
    return float(a / b) if b else None


def cosine(a, b):
    return ratio(float(np.dot(a, b)), float(np.linalg.norm(a) * np.linalg.norm(b)))


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def summarize(rows, fields):
    result = {}
    for key in fields:
        v = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        result[key] = dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                          minimum=float(v.min()), maximum=float(v.max()))
    return result


def run(args):
    archive = args.archive.resolve()
    repo = args.repository.resolve()
    source = Path(__file__).resolve()
    core_path = repo / "src/pullback_feature_learning/core.py"
    generator_path = archive / "generator.py"
    record_path = archive / "stage2/training_record.json"
    rows_path = archive / "stage2/training_rows.csv"
    states_path = archive / "stage2/states.npz"
    tracked = [core_path, generator_path, record_path, rows_path, states_path,
               archive / "stage2_run.py", archive / "stage2_plan.md"]
    before = {str(p): sha(p.read_bytes()) for p in tracked}
    record = json.loads(record_path.read_text())
    assert record["hashes"]["core.py"] == before[str(core_path)]
    assert record["hashes"]["generator.py"] == before[str(generator_path)]
    core = module(core_path, "audit_core")
    generator = module(generator_path, "audit_generator")
    assert core.PARAMETER_COUNT == 80
    with rows_path.open(newline="", encoding="utf-8") as f:
        old_rows = list(csv.DictReader(f))
    for r in old_rows:
        r["pair"], r["step"] = int(r["pair"]), int(r["step"])
        r["weight"] = float(r["weight"])
        for key in ("train_loss", "cum_z1", "cum_z2"):
            r[key] = float(r[key])
    with np.load(states_path, allow_pickle=False) as bank:
        states = {k: bank[k].copy() for k in bank.files}
    assert len(old_rows) == 384 and len(states) == 288
    for k, theta in states.items():
        assert sha(theta.tobytes()) == k

    args.out.mkdir(parents=True, exist_ok=False)
    xys, initial, outputs, raw_arrays = {}, {}, {}, {}
    # Validation seeds in the historical record are deliberately never passed to the generator.
    for item in record["seeds"]:
        pair = int(item["pair"])
        x, y = generator.rings(item["train_seed"])
        theta0 = core.initialize_parameters(item["init_seed"])
        assert sha(x.tobytes() + y.tobytes()) == item["train_hash"]
        assert sha(theta0.tobytes()) == item["initial_hash"]
        xys[pair] = (x, y)
        initial[pair] = (theta0, core.representation(theta0, x))
        raw_arrays[f"pair{pair}_x"] = x
        raw_arrays[f"pair{pair}_y"] = y
        raw_arrays[f"pair{pair}_initial_z"] = initial[pair][1]

    max_loss_error = 0.
    for row in old_rows:
        key = row["state_sha256"]
        if key not in outputs:
            x, y = xys[row["pair"]]
            outputs[key] = core.representation(states[key], x)
            raw_arrays[f"z_{key}"] = outputs[key]
        loss = core.logistic_loss(outputs[key], xys[row["pair"]][1])
        max_loss_error = max(max_loss_error, abs(loss-row["train_loss"]))
    assert max_loss_error < 1e-12

    endpoint_rows = []
    groups = {}
    for row in old_rows:
        if row["schedule"] == "hybrid" and row["step"] == 100:
            continue  # Identical to full/native/100, already included there.
        key = (row["pair"], row["size_rule"], row["schedule"], row["step"])
        groups.setdefault(key, []).append(row)
    head = np.arange(64, 72)  # Final block's second output row: exactly readout-null.
    active = np.array([i for i in range(80) if i not in head])
    for group, members in sorted(groups.items()):
        pair, rule, schedule, step = group
        z0 = initial[pair][1]
        for left, right in itertools.combinations(sorted(members, key=lambda r:r["weight"]), 2):
            zl, zr = outputs[left["state_sha256"]], outputs[right["state_sha256"]]
            dt = states[right["state_sha256"]] - states[left["state_sha256"]]
            diff, pl, pr = zr[:,0]-zl[:,0], sigmoid(zl[:,0]), sigmoid(zr[:,0])
            lmove, rmove = zl[:,0]-z0[:,0], zr[:,0]-z0[:,0]
            row = dict(pair=pair, cohort=left["cohort"], size_rule=rule,
                       schedule=schedule, step=step, weight_left=left["weight"],
                       weight_right=right["weight"], state_left=left["state_sha256"],
                       state_right=right["state_sha256"],
                       logit_rms_difference=rms(diff), logit_max_difference=float(abs(diff).max()),
                       logit_relative_rms=ratio(rms(diff), rms(zl[:,0])),
                       probability_mean_absolute_difference=float(abs(pr-pl).mean()),
                       probability_max_difference=float(abs(pr-pl).max()),
                       training_class_disagreement=float(np.mean((zl[:,0]>=0)!=(zr[:,0]>=0))),
                       training_loss_difference=right["train_loss"]-left["train_loss"],
                       logit_cosine=cosine(zl[:,0], zr[:,0]),
                       displacement_cosine=cosine(lmove, rmove),
                       first_path_left=left["cum_z1"], first_path_right=right["cum_z1"],
                       first_path_relative_difference=ratio(right["cum_z1"]-left["cum_z1"], left["cum_z1"]),
                       first_chord_left=rms(lmove), first_chord_right=rms(rmove),
                       first_tortuosity_left=ratio(left["cum_z1"], rms(lmove)),
                       first_tortuosity_right=ratio(right["cum_z1"], rms(rmove)),
                       second_path_left=left["cum_z2"], second_path_right=right["cum_z2"],
                       second_path_reduction_ratio=ratio(left["cum_z2"], right["cum_z2"]),
                       second_endpoint_rms_difference=rms(zr[:,1]-zl[:,1]),
                       parameter_difference_norm=float(np.linalg.norm(dt)),
                       parameter_difference_head_energy_fraction=ratio(float(np.dot(dt[head],dt[head])),float(np.dot(dt,dt))),
                       parameter_difference_active_norm=float(np.linalg.norm(dt[active])))
            endpoint_rows.append(row)

    # Fixed-state linear-algebra diagnostics. No candidate is applied to any state.
    state_rows = [r for r in old_rows if r["schedule"]=="full" and r["size_rule"]=="native"]
    state_rows += [dict(pair=p, cohort="original" if p<=8 else "fresh", step=0,
                        weight="initial", state_sha256=f"initial_{p}") for p in sorted(xys)]
    local_rows, spectral_arrays = [], {}
    for row in state_rows:
        pair, key = row["pair"], row["state_sha256"]
        theta, z = initial[pair] if row["step"]==0 else (states[key],outputs[key])
        x, y = xys[pair]
        jac = core.representation_jacobian(theta, x)
        a, b = jac[0::2], jac[1::2]
        assert np.max(abs(a[:,head])) == 0.
        r = core.representation_gradient(z,y)[:,0]
        grad = a.T @ r
        aa, bb = a.T @ a, b.T @ b
        _, sb, vhb = np.linalg.svd(b, full_matrices=False)
        _, sa, vha = np.linalg.svd(a, full_matrices=False)
        tol_b = max(b.shape)*np.finfo(float).eps*sb[0]
        tol_a = max(a.shape)*np.finfo(float).eps*sa[0]
        rank_b, rank_a = int(sum(sb>tol_b)), int(sum(sa>tol_a))
        kernel_b = vhb[rank_b:].T
        kernel_a = vha[rank_a:].T
        au = a @ kernel_b
        u_au, s_au, _ = np.linalg.svd(au, full_matrices=False)
        au_tol = max(au.shape)*np.finfo(float).eps*(s_au[0] if len(s_au) else 0.)
        au_rank = int(sum(s_au>au_tol))
        u_span = u_au[:,:au_rank]
        r_accessible = u_span @ (u_span.T @ r)
        bhead = b[:,head]
        u_h, s_h, _ = np.linalg.svd(bhead, full_matrices=False)
        h_rank = int(sum(s_h>max(bhead.shape)*np.finfo(float).eps*s_h[0]))
        h_span = u_h[:,:h_rank]
        solutions = {}
        for gamma in (0., 1/9, 1., 9.):
            m = aa + gamma*bb + .1*np.eye(80)
            solutions[gamma] = np.linalg.solve(m,grad)
        reference = solutions[0.]
        for gamma, s in solutions.items():
            ds = s-reference
            v1, v2 = a@s, b@s
            vb_body, vb_head = b[:,active]@s[active], bhead@s[head]
            norm_s2 = float(np.dot(s,s))
            norm_ds2 = float(np.dot(ds,ds))
            norm_body2 = float(np.dot(vb_body,vb_body))
            local_rows.append(dict(pair=pair, cohort=row["cohort"], step=row["step"],
                source_weight=row["weight"], state_sha256=key, candidate_weight=gamma,
                rank_a=rank_a, rank_b=rank_b, rank_b_relative_1e10=int(sum(sb>1e-10*sb[0])),
                rank_readout_in_kernel_b=au_rank,
                gradient_energy_outside_readout_kernel_b=ratio(float(np.sum((r-r_accessible)**2)),float(np.dot(r,r))),
                parameter_direction_norm=float(np.linalg.norm(s)),
                readout_motion_rms=rms(v1), unused_motion_rms=rms(v2),
                readout_direction_cosine_vs_weight0=cosine(v1,a@reference),
                readout_direction_relative_change_vs_weight0=ratio(float(np.linalg.norm(v1-a@reference)),float(np.linalg.norm(a@reference))),
                unused_motion_ratio_vs_weight0=ratio(float(np.linalg.norm(v2)),float(np.linalg.norm(b@reference))),
                head_parameter_energy_fraction=ratio(float(np.dot(s[head],s[head])),norm_s2),
                change_head_energy_fraction_vs_weight0=ratio(float(np.dot(ds[head],ds[head])),norm_ds2),
                change_linear_readout_null_energy_fraction_vs_weight0=ratio(float(np.sum((kernel_a.T@ds)**2)),norm_ds2),
                unused_body_energy_cancelable_by_head=ratio(float(np.sum((h_span.T@vb_body)**2)),norm_body2),
                unused_head_body_cosine=cosine(vb_body,vb_head),
                unused_total_to_body_motion_ratio=ratio(float(np.linalg.norm(v2)),float(np.linalg.norm(vb_body)))))
        spectral_arrays[f"{key}_a_singular_values"] = sa
        spectral_arrays[f"{key}_b_singular_values"] = sb
        spectral_arrays[f"{key}_directions"] = np.stack([solutions[g] for g in (0.,1/9,1.,9.)])
        spectral_arrays[f"{key}_a"] = a
        spectral_arrays[f"{key}_b"] = b
        spectral_arrays[f"{key}_loss_gradient"] = r

    write_csv(args.out/"endpoint_pairs.csv",endpoint_rows)
    write_csv(args.out/"fixed_state_diagnostics.csv",local_rows)
    np.savez_compressed(args.out/"training_inputs_and_outputs.npz",**raw_arrays)
    np.savez_compressed(args.out/"fixed_state_arrays.npz",**spectral_arrays)
    main=[r for r in endpoint_rows if r["size_rule"]=="native" and r["schedule"]=="full"
          and r["step"]==500 and r["weight_left"]==0 and r["weight_right"]==9]
    stats_fields=[k for k in main[0] if k not in ("pair","cohort","size_rule","schedule","step",
                 "weight_left","weight_right","state_left","state_right")]
    summaries={}
    for rule,schedule,step in sorted({(r["size_rule"],r["schedule"],r["step"]) for r in endpoint_rows}):
        rr=[r for r in endpoint_rows if (r["size_rule"],r["schedule"],r["step"])==(rule,schedule,step)
            and r["weight_left"]==0 and r["weight_right"]==9]
        summaries[f"{rule}_{schedule}_{step}_weight0_vs9"] = summarize(rr,stats_fields)
    diagnostics={}
    local_fields=[k for k in local_rows[0] if k not in ("pair","cohort","step","source_weight","state_sha256","candidate_weight")]
    for step in (0,100,500):
        rr=[r for r in local_rows if r["step"]==step and r["candidate_weight"]==9 and
            (r["source_weight"]==1. or r["source_weight"]=="initial")]
        diagnostics[f"baseline_state_step{step}_candidate9_vs0"] = summarize(rr,local_fields)
    after={str(p):sha(p.read_bytes()) for p in tracked}
    assert before==after
    write_json(args.out/"summary.json",dict(endpoint=summaries, fixed_state=diagnostics,
        main_ratio_of_mean_second_paths=float(np.mean([r["second_path_left"] for r in main])/np.mean([r["second_path_right"] for r in main])),
        main_relative_difference_of_mean_first_paths=float(np.mean([r["first_path_right"] for r in main])/np.mean([r["first_path_left"] for r in main])-1)))
    write_json(args.out/"audit_record.json",dict(scope="Archived training inputs and stored model states only; no training, validation, holdout, or population scoring.",
        source_sha256=sha(source.read_bytes()), historical_files_sha256=before, historical_files_unchanged=True,
        max_training_loss_reconstruction_error=max_loss_error, reconstructed_training_sets=len(xys),
        checked_archived_state_hashes=len(states), logical_rows_checked=len(old_rows), endpoint_pair_rows=len(endpoint_rows),
        fixed_states=len(state_rows), diagnostic_rows=len(local_rows), numpy=np.__version__,python=platform.python_version(),
        damping=.1, candidate_weights=[0.,1/9,1.,9.], exact_readout_null_head_indices=head.tolist(),
        source_archive=str(archive), repository=str(repo), prescribed_rank_tolerance="max(shape)*machine_epsilon*largest_singular_value; B also reported at relative1e-10",
        note="A/B nullspaces are local linear objects. Endpoint parameter differences need not be linearized accurately. All summaries are descriptive retrospective diagnostics."))
    (args.out/"audit_archived_states.py").write_bytes(source.read_bytes())
    print(json.dumps(dict(out=str(args.out),main=summaries["native_full_500_weight0_vs9"],
                          baseline_step500=diagnostics["baseline_state_step500_candidate9_vs0"]),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--repository",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
