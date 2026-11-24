"""Measure hypothetical, same-state pullback steps at two normalization rules."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core
from generator import rings


HERE = Path(__file__).resolve().parent
WEIGHTS = (0., 1/9, 1., 9.)
STATES = ("initial", "gd100", "pullback100", "gd500", "pullback500", "warmup500")
RULES = ("scalar8", "matched_logit")
FIELDS = ("rate", "pred_loss_decrease", "actual_loss_decrease", "pred_z1", "actual_z1", "pred_z2", "actual_z2")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(file, rows):
    with file.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(v):
    return float(np.linalg.norm(v)/np.sqrt(len(v)))


def slope_check(t, hessian, penalty, gradient):
    root = np.linalg.cholesky(hessian)
    left = np.linalg.solve(root, penalty)
    whitened = np.linalg.solve(root, left.T)
    scale = np.linalg.eigvalsh((whitened + whitened.T)*.5)[-1]
    # h*scale bounds the relative matrix perturbation; the measurement
    # never requires negative weights, including at the t=0 boundary.
    h = min(1.e-2*(1+t), .005/max(float(scale), 1.e-12))
    def q(j):
        return gradient @ np.linalg.solve(hessian + (j*h)*penalty, gradient)
    if t < 2*h:
        return float((-25*q(0)+48*q(1)-36*q(2)+16*q(3)-3*q(4))/(12*h)), h
    return float((q(-2)-8*q(-1)+8*q(1)-q(2))/(12*h)), h


def interventions(theta, x, y, pair, name):
    z = core.representation(theta, x)
    jac = core.representation_jacobian(theta, x)
    a, b = jac[0::2], jac[1::2]
    g = core.representation_gradient(z, y)[:, 0]
    grad = a.T @ g
    base = a.T @ a + .1*np.eye(len(theta))
    penalty = b.T @ b
    loss = core.logistic_loss(z, y)
    center = np.linalg.solve(base + penalty, grad)
    legacy, _ = core.direction(theta, x, y, "pullback", .1)
    referr = float(np.linalg.norm(center+legacy)/np.linalg.norm(legacy))
    target = 8*rms(a @ center)
    rows, derivatives = [], []
    for t in WEIGHTS:
        matrix = base+t*penalty
        s = np.linalg.solve(matrix, grad)
        v1, v2 = a@s, b@s
        q = float(grad@s)
        analytic = -float(v2@v2)
        measured, h = slope_check(t, matrix, penalty, grad)
        absolute = abs(analytic-measured)
        normalized = absolute/(abs(analytic)+q/(1+t))
        derivatives.append(dict(pair=pair, state=name, weight=t, q=q, slope=analytic,
                                fd_slope=measured, fd_h=h, scaled_fd_error=normalized,
                                solve_relative_error_t1=referr))
        for rule in RULES:
            rate = 8. if rule == "scalar8" else target/rms(v1)
            after = core.representation(theta-rate*s, x)
            after_loss = core.logistic_loss(after, y)
            delta = after-z
            rows.append(dict(pair=pair, state=name, rule=rule, weight=t, rate=rate,
                             state_sha256=digest(theta.tobytes()), train_loss_before=loss,
                             train_loss_after=after_loss, pred_loss_decrease=rate*q,
                             actual_loss_decrease=loss-after_loss, pred_z1=rate*rms(v1),
                             actual_z1=rms(delta[:, 0]), pred_z2=rate*rms(v2), actual_z2=rms(delta[:, 1]),
                             logit_alignment=q/(np.linalg.norm(g)*np.linalg.norm(v1))))
    return rows, derivatives


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    src = HERE / "stage1"
    audit = json.loads((src/"replay_audit.json").read_text())
    assert audit["state_bank_sha256"] == digest((src/"states.npz").read_bytes())
    reference = {}
    with (src/"replay.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            reference[r["state_key"]] = float(r["train_loss"]), r["state_sha256"]
    rows, slopes = [], []
    with np.load(src/"states.npz") as bank, threadpool_limits(limits=1):
        assert set(bank.files) == {f"pair{i}_{state}" for i in range(1, 9) for state in STATES}
        for pair in range(1, 9):
            x, y = rings(71000+pair)
            for state in STATES:
                key = f"pair{pair}_{state}"
                local, derivs = interventions(bank[key], x, y, pair, state)
                ref_loss, ref_hash = reference[key]
                assert abs(local[0]["train_loss_before"]-ref_loss) < 1e-12 and local[0]["state_sha256"] == ref_hash
                rows.extend(local)
                slopes.extend(derivs)
    assert len(rows) == 384 and len(slopes) == 192
    write_csv(output/"branches.csv", rows)
    write_csv(output/"derivatives.csv", slopes)
    cells = {(r["pair"], r["state"],r["rule"],r["weight"]):r for r in rows}
    q_sequences, z2_sequences, max_motion_error = [], [], 0.
    for pair in range(1, 9):
        for state in STATES:
            scalar = [cells[pair, state, RULES[0], t] for t in WEIGHTS]
            matched = [cells[pair, state, RULES[1], t] for t in WEIGHTS]
            q_sequences.append(all(r["pred_loss_decrease"] >= s["pred_loss_decrease"]-1e-15
                                   for r,s in zip(scalar,scalar[1:])))
            z2_sequences.append(all(r["pred_z2"] >= s["pred_z2"]-1e-15 for r,s in zip(scalar,scalar[1:])))
            target = scalar[2]["pred_z1"]
            max_motion_error=max(max_motion_error,max(abs(r["pred_z1"]/target-1) for r in matched))
    summaries=[]
    for level, labels in (("seed",range(1,9)), ("state",STATES)):
        for label in labels:
            for rule in RULES:
                for t in WEIGHTS:
                    group=[r for r in rows if r["rule"]==rule and r["weight"]==t and r["pair" if level=="seed" else "state"]==label]
                    summaries.append(dict(level=level,label=label, rule=rule,weight=t,n_states=len(group),
                                          **{f:mean(float(r[f]) for r in group) for f in FIELDS}))
    write_csv(output/"group_means.csv", summaries)
    orderings = []
    for rule in RULES:
        for metric in ("pred_loss_decrease","actual_loss_decrease"):
            for state in STATES:
                vectors=np.array([[cells[i,state,rule,t][metric] for t in WEIGHTS] for i in range(1,9)])
                orderings.append(dict(rule=rule,metric=metric,state=state,
                                      nonincreasing_states=int(np.all(np.diff(vectors,axis=1)<=1e-15,axis=1).sum()),
                                      zero_greater_than_one=int((vectors[:,0]>vectors[:,2]+1e-15).sum()),
                                      nine_greater_than_one=int((vectors[:,3]>vectors[:,2]+1e-15).sum()),
                                      finite_loss_increases=int((vectors<0.).sum()) if metric=="actual_loss_decrease" else 0))
    write_csv(output/"orderings.csv",orderings)
    result=dict(unique_states=48,seed_pairs=8, branches=384, derivative_points=192,
                q_monotone_count=sum(q_sequences), unused_motion_monotone_count=sum(z2_sequences),
                max_normalized_fd_error=max(r["scaled_fd_error"] for r in slopes),
                max_logit_match_relative_error=max_motion_error,
                max_t1_parameter_relative_error=max(r["solve_relative_error_t1"] for r in slopes),
                max_fd_absolute_error=max(abs(r["fd_slope"]-r["slope"]) for r in slopes),
                fixed_rate_z2_at_t9_over_t0_median=float(np.median([
                    cells[i,s,"scalar8",9.]["pred_z2"]/cells[i,s,"scalar8",0.]["pred_z2"]
                    for i in range(1,9) for s in STATES])),
                plan_sha256=digest((HERE/"stage2_local_plan.md").read_bytes()),
                generator_sha256=digest((HERE/"generator.py").read_bytes()),
                stage1_bank_sha256=audit["state_bank_sha256"])
    (output/"checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    assert result["q_monotone_count"] == result["unused_motion_monotone_count"] == 48
    assert result["max_normalized_fd_error"]<1e-4 and max_motion_error<1e-12
    assert result["max_t1_parameter_relative_error"]<1e-8
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
