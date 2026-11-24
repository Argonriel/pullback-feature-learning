"""Separate held-out/population scoring and prespecified gate decisions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import runner
from pullback_feature_learning import core

SIGMA, INNER, OUTER, RMAX = .15, .5, 1.2, 3.
GRIDS = (48, 96, 192, 384, 768, 1536, 3072)
COMPONENTS = ("risk", "radial_excess", "angular_excess")


def oracle_logit(r):
    return ((INNER**2 - OUTER**2)/(2*SIGMA**2)
            + np.log(np.i0(r * OUTER/SIGMA**2))
            - np.log(np.i0(r * INNER/SIGMA**2)))


@lru_cache(maxsize=1)
def decision_radius():
    lo, hi = INNER, OUTER
    for _ in range(54):
        mid = (lo+hi)/2
        if oracle_logit(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def density(r, radius):
    return r/SIGMA**2 * np.exp(-(r*r+radius**2)/(2*SIGMA**2)) * np.i0(r*radius/SIGMA**2)


@lru_cache(maxsize=len(GRIDS))
def polar_grid(n):
    nodes, weights = np.polynomial.legendre.leggauss(n//2)
    intervals = ((0., decision_radius()), (decision_radius(), RMAX))
    r = np.concatenate([(a+b)/2 + nodes*(b-a)/2 for a,b in intervals])
    dr = np.concatenate([weights*(b-a)/2 for a,b in intervals])
    wm, wp = dr*density(r, INNER)/2, dr*density(r, OUTER)/2
    angles = (np.arange(n)+.5)*2*np.pi/n
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    return r, wm, wp, unit


def circle_logits(theta0, theta, linear, r, unit):
    batch = max(1, 4096//len(unit))
    values = np.empty((len(r), len(unit)))
    for i in range(0, len(r), batch):
        points = (r[i:i+batch, None, None]*unit[None, :, :]).reshape(-1, 2)
        values[i:i+batch] = runner.predict(theta0, theta, points, linear)[:, 0].reshape(-1, len(unit))
    return values


def integrate(h, r, wm, wp):
    avg = h.mean(axis=1)
    soft = np.logaddexp(0., h)
    risk = wm @ soft.mean(axis=1) + wp @ np.logaddexp(0., -h).mean(axis=1)
    oracle = oracle_logit(r)
    bayes = wm @ np.logaddexp(0., oracle) + wp @ np.logaddexp(0., -oracle)
    radial = wm @ np.logaddexp(0., avg) + wp @ np.logaddexp(0., -avg) - bayes
    angular = (wm+wp) @ (soft.mean(axis=1)-np.logaddexp(0., avg))
    residual = abs((risk-bayes)-(radial+angular))
    accuracy = wm @ np.mean(h < 0, axis=1) + wp @ np.mean(h >= 0, axis=1)
    if not np.all(np.isfinite([risk,bayes,radial,angular])) or min(radial,angular) < -2e-12 or residual > 2e-12*max(1.,risk):
        raise ArithmeticError("Population decomposition identity/positivity failed.")
    return dict(risk=float(risk), bayes_risk=float(bayes), radial_excess=float(radial),
                angular_excess=float(angular), accuracy=float(accuracy), identity_error=float(residual))


def affine_growth_bound(theta0, theta, linear):
    """Return C,L with |first-coordinate logit(x)| <= C+L*||x||.

    For the tangent map, propagate a parameter directional derivative through
    both blocks using ||tanh||_2<=sqrt(8) and ||diag(tanh')||_2<=1.
    """
    base = theta0 if linear else theta
    blocks = core.unpack(base)
    c = sum(.5*np.abs(b[0]).sum() for _,b,_ in blocks)
    if not linear:
        return float(c), 1.
    changes = core.unpack(theta-theta0)
    c_h, l_h = 0., 1.
    c_d, l_d = 0., 0.
    for (a,b,_), (da,db,dbias) in zip(blocks, changes):
        bnorm, anorm = np.linalg.norm(b,2), np.linalg.norm(a,2)
        danorm = np.linalg.norm(da,2)
        growth = 1+.5*bnorm*anorm
        c_d, l_d = (growth*c_d+.5*np.sqrt(8)*np.linalg.norm(db,2)
                     +.5*bnorm*(danorm*c_h+np.linalg.norm(dbias)),
                     growth*l_d+.5*bnorm*danorm*l_h)
        c_h += .5*np.sqrt(8)*bnorm
    return float(c+c_d), float(1+l_d)


def risk_tail_bound(theta0, theta, linear):
    c,l = affine_growth_bound(theta0,theta,linear)
    mass_bound = np.exp(-(RMAX-OUTER)**2/(2*SIGMA**2))
    return float((np.log(2.)+c+l*(RMAX+SIGMA**2/(RMAX-OUTER)))*mass_bound)


def population(theta0, theta, linear):
    previous = None
    for n in GRIDS:
        r, wm, wp, unit = polar_grid(n)
        value = integrate(circle_logits(theta0,theta,linear,r,unit),r,wm,wp)
        if previous is not None:
            differences = {k:abs(value[k]-previous[k]) for k in COMPONENTS}
            if all(differences[k] <= 2e-7+2e-6*abs(value[k]) for k in COMPONENTS):
                return dict(**value, grid_nodes=n, grid_error=max(differences.values()),
                            grid_converged=True, tail_risk_bound=risk_tail_bound(theta0,theta,linear),
                            accuracy_role="descriptive_discontinuous_not_refinement_target")
        previous = value
    raise ArithmeticError(f"Population integration did not converge: {differences}")


def independent_population(theta0,theta,linear):
    """Uniform-radius Simpson and independently shifted prime angle grid."""
    r = np.linspace(0.,RMAX,769)
    angle = (np.arange(521)+np.sqrt(2)/2)*2*np.pi/521
    unit = np.column_stack((np.cos(angle),np.sin(angle)))
    w = np.ones(len(r))
    w[1:-1:2],w[2:-1:2] = 4,2
    w *= (r[1]-r[0])/3
    wm,wp = .5*w*density(r,INNER),.5*w*density(r,OUTER)
    return integrate(circle_logits(theta0,theta,linear,r,unit),r,wm,wp)


def read_rows(path):
    with Path(path).open(encoding="utf-8",newline="") as handle:
        return list(csv.DictReader(handle))


def contrasts(rows,data,states):
    groups = defaultdict(dict)
    for row in rows:
        if row["status"] != "target" or row.get("evaluation_status") != "complete":
            continue
        # GD is reused across damping contrasts. The main complete four-arm
        # contrast is lambda .1; sensitivity outputs retain their own PB values.
        key = (row["task"],row["seed"],row["cohort"],row["step_size"],row["fraction"])
        groups[key][(row["arm"],row["damping"])] = row
    result=[]
    for key,arms in sorted(groups.items()):
        task,seed,cohort,step_size,q=key
        dampings=sorted({d for a,d in arms if d is not None})
        for damp in dampings:
            row=dict(task=task,seed=seed,cohort=cohort,step_size=step_size,fraction=q,damping=damp)
            selected={a:arms.get((a,None if a.endswith("gd") else damp)) for a in runner.ARMS}
            for a,v in selected.items():
                row[a+"_risk"] = None if v is None else v["risk"]
            row["all_four_present"] = all(selected[a] is not None for a in runner.ARMS[:4])
            npb,tpb=selected["nonlinear_pb"],selected["tangent_pb"]
            ngd,tgd=selected["nonlinear_gd"],selected["tangent_gd"]
            fixed=selected["nonlinear_fixed_pb"]
            row["B"] = tpb["risk"]-npb["risk"] if npb and tpb else None
            row["D"] = (ngd["risk"]-npb["risk"])-(tgd["risk"]-tpb["risk"]) if row["all_four_present"] else None
            row["F"] = fixed["risk"]-npb["risk"] if fixed and npb else None
            if npb and tpb:
                stem=f"{task}__{seed}"
                x,theta0=data[stem+"__x"],data[stem+"__theta0"]
                pn=runner.predict(theta0,states[npb["state_key"]],x,False)[:,0]
                pt=runner.predict(theta0,states[tpb["state_key"]],x,True)[:,0]
                row["nonlinear_vs_tangent_pb_train_logit_rms"]=float(np.sqrt(np.mean((pn-pt)**2)))
            result.append(row)
    return result


def group_means(contrasts):
    grouped=defaultdict(list)
    for r in contrasts:
        grouped[(r["task"],r["cohort"],r["step_size"],r["fraction"],r["damping"])].append(r)
    out=[]
    for key,rs in sorted(grouped.items()):
        row=dict(zip(("task","cohort","step_size","fraction","damping"),key))
        row["pairs"]=len(rs)
        for field in ("B","D","F","nonlinear_vs_tangent_pb_train_logit_rms")+tuple(a+"_risk" for a in runner.ARMS):
            vals=[r[field] for r in rs if r.get(field) is not None]
            row[field+"_n"]=len(vals)
            row[field+"_mean"]=float(np.mean(vals)) if vals else None
            row[field+"_sd"]=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        out.append(row)
    return out


def gate_decisions(config, contrasts):
    decisions=[]
    cohorts=sorted(set(s.get("cohort","unspecified") for s in config["seeds"]))
    for task in config["tasks"]:
        for cohort in cohorts:
            ids=[str(s["id"]) for s in config["seeds"] if s.get("cohort","unspecified")==cohort]
            for estimand in ("B","D"):
                checks=[]
                for step_size in (.01,.005):
                    rs=[r for r in contrasts if r["task"]==task and r["cohort"]==cohort
                        and r["fraction"]==.25 and r["damping"]==.1 and r["step_size"]==step_size]
                    complete=(len(rs)==len(ids)==4 and all(r["all_four_present"] for r in rs)
                              and {r["seed"] for r in rs}==set(ids))
                    vals=[r[estimand] for r in rs if r.get(estimand) is not None]
                    mean=float(np.mean(vals)) if vals else None
                    agree=sum(v*mean>0 for v in vals) if mean is not None else 0
                    checks.append(dict(step_size=step_size,all_four_arms_all_four_pairs=complete,
                                       mean=mean,sign_agreements=agree,values=vals,
                                       passes=bool(complete and mean is not None and abs(mean)>=.01 and agree>=3)))
                diff=abs(checks[0]["mean"]-checks[1]["mean"]) if all(c["mean"] is not None for c in checks) else None
                same_sign=all(c["mean"] is not None for c in checks) and checks[0]["mean"]*checks[1]["mean"]>0
                passed=bool(all(c["passes"] for c in checks) and diff is not None and diff<=.003 and same_sign)
                direction="benefit" if passed and checks[1]["mean"]>0 else "harm" if passed else "not_retained"
                reason=("passes_screen" if passed else "insufficient_primary_target_coverage"
                        if not all(c["all_four_arms_all_four_pairs"] for c in checks)
                        else "effect_size_sign_or_refinement_gate_failed")
                decisions.append(dict(task=task,cohort=cohort,estimand=estimand,passes=passed,reason=reason,
                                      direction=direction,step_checks=checks,refinement_difference=diff,
                                      gate_role="necessary_screen_not_novelty_or_causal_attribution"))
    return decisions


def score(run_path,output):
    run_path,output=Path(run_path),Path(output)
    config=json.loads((run_path/"config.json").read_text(encoding="utf-8-sig"))
    training=json.loads((run_path/"provenance.json").read_text(encoding="utf-8-sig"))
    if "finished_utc" not in training:
        raise RuntimeError("Scoring requires a finished training pass.")
    output.mkdir(parents=True,exist_ok=False)
    audit=dict(started_utc=runner.utcnow(),run=str(run_path.resolve()),source_sha256=runner.digest(__file__),
               training_finished_utc=training["finished_utc"],input_sha256={},data_regeneration={},
               training_loss_max_error=0.,independent_quadrature=[],failures=[])
    for name in ("config.json","provenance.json","data.npz","states.npz","snapshots.json","actual_seeds.json"):
        audit["input_sha256"][name]=runner.digest(run_path/name)
        if name in training.get("output_sha256",{}):
            if audit["input_sha256"][name]!=training["output_sha256"][name]:
                raise ArithmeticError(f"Training output hash changed: {name}")
    for path,sha in training["source_sha256"].items():
        if runner.digest(path)!=sha:
            raise ArithmeticError(f"Training source hash changed: {path}")
    snapshots=json.loads((run_path/"snapshots.json").read_text(encoding="utf-8"))
    actual=json.loads((run_path/"actual_seeds.json").read_text(encoding="utf-8"))
    evaluation_data,cache,rows={},{},[]
    with np.load(run_path/"data.npz") as data,np.load(run_path/"states.npz") as states,threadpool_limits(1):
        for seed in actual:
            stem=f"{seed['task']}__{seed['id']}"
            x,y=runner.make_data(seed["task"],seed["train_seed"],config.get("n_train",256))
            theta0=core.initialize_parameters(seed["init_seed"])
            for suffix,val in (("x",x),("y",y),("theta0",theta0)):
                if not np.array_equal(val,data[stem+"__"+suffix]):
                    raise ArithmeticError(f"Data regeneration mismatch: {stem} {suffix}")
            audit["data_regeneration"][stem]="exact"
            if seed["task"]=="moons":
                xv,yv=runner.make_data("moons",seed["test_seed"],20000)
                evaluation_data[stem+"__x"],evaluation_data[stem+"__y"]=xv,yv
        np.savez_compressed(output/"evaluation_data.npz",**evaluation_data)
        for snapshot in snapshots:
            row=dict(snapshot)
            stem=f"{row['task']}__{row['seed']}"
            theta0,theta=data[stem+"__theta0"],states[row["state_key"]]
            linear=row["arm"].startswith("tangent")
            if not np.all(np.isfinite(theta)):
                raise ArithmeticError("Nonfinite saved parameters.")
            loss=core.logistic_loss(runner.predict(theta0,theta,data[stem+"__x"],linear),data[stem+"__y"])
            error=abs(loss-row["train_loss"])
            audit["training_loss_max_error"]=max(error,audit["training_loss_max_error"])
            if error>2e-12:
                raise ArithmeticError(f"Training loss differs at {row['state_key']}")
            cache_key=(stem,linear,hashlib.sha256(theta.tobytes()).hexdigest())
            try:
                if cache_key not in cache:
                    if row["task"]=="moons":
                        z=runner.predict(theta0,theta,evaluation_data[stem+"__x"],linear)
                        y=evaluation_data[stem+"__y"]
                        val=dict(risk=core.logistic_loss(z,y),accuracy=float(np.mean(np.where(z[:,0]>=0,1.,-1.)==y)),
                                 evaluation_size=20000,evaluation_role="fresh_balanced_held_out")
                    else:
                        val=population(theta0,theta,linear)
                        val["evaluation_role"]="known_law_polar_quadrature_with_tail_bound"
                    cache[cache_key]=val
                row.update(cache[cache_key])
                row["evaluation_status"]="complete"
            except (ArithmeticError,FloatingPointError) as exc:
                row["evaluation_status"]="failed"
                audit["failures"].append(dict(state_key=row["state_key"],error=str(exc)))
            rows.append(row)
            runner.write_json(output/"metrics.json",rows)
            runner.write_csv(output/"metrics.csv",rows)
        # Audit a prespecified first pair in each cohort, all available primary
        # .005, lambda .1 arms. GD is damping-independent.
        first_ids={}
        for seed in config["seeds"]:
            first_ids.setdefault(seed.get("cohort","unspecified"),str(seed["id"]))
        for row in rows:
            if not (row["task"]=="rings" and row["seed"]==first_ids[row["cohort"]]
                    and row["fraction"]==.25 and row["step_size"]==.005
                    and row["damping"] in (None,.1) and row["evaluation_status"]=="complete"):
                continue
            stem=f"rings__{row['seed']}"
            value=independent_population(data[stem+"__theta0"],states[row["state_key"]],row["arm"].startswith("tangent"))
            diffs={k:abs(value[k]-row[k]) for k in COMPONENTS}
            passed=max(diffs.values())<=2e-6
            audit["independent_quadrature"].append(dict(state_key=row["state_key"],arm=row["arm"],
                                                        differences=diffs,passes=passed))
            if not passed:
                audit["failures"].append(dict(state_key=row["state_key"],error="independent_quadrature_tolerance"))
        paired=contrasts(rows,data,states)
        means=group_means(paired)
        decisions=gate_decisions(config,paired)
        if audit["failures"]:
            for decision in decisions:
                decision["passes"]=False
                decision["direction"]="evaluation_audit_failed"
        runner.write_json(output/"contrasts.json",paired)
        runner.write_csv(output/"contrasts.csv",paired)
        runner.write_json(output/"group_means.json",means)
        runner.write_csv(output/"group_means.csv",means)
        runner.write_json(output/"gate_decisions.json",decisions)
    audit["finished_utc"]=runner.utcnow()
    audit["status"]="passed" if not audit["failures"] else "failed"
    audit["scored_rows"]=len(rows)
    audit["unique_evaluations"]=len(cache)
    audit["output_sha256"]={p.name:runner.digest(p) for p in output.iterdir() if p.name!="audit.json"}
    runner.write_json(output/"audit.json",audit)
    print(json.dumps(dict(audit_status=audit["status"],rows=len(rows),decisions=decisions),indent=2))
    return audit


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    score(args.run,args.output)
