"""Measure the training Hessian at the shared fork and the next 25 true GD losses."""
import csv
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core


ROOT = Path(__file__).resolve().parent
PRIOR = next(p for p in (ROOT.parent / "budgeted-rotation", ROOT.parent / "budgeted-rotation-final")
             if (p / "run.py").is_file())
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))
from run import advance, rings


def train_loss(theta, x, y):
    return core.logistic_loss(core.representation(theta, x), y)


def eigenvalue(theta, x, y, delta):
    hessian = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        change = np.zeros_like(theta)
        change[j] = delta
        plus = core.parameter_gradient(theta + change, x, y)
        minus = core.parameter_gradient(theta - change, x, y)
        hessian[:, j] = (plus - minus) / (2*delta)
    if np.max(np.abs(hessian-hessian.T)) > 2e-6:
        raise ValueError("Finite-difference Hessian exceeds the symmetry tolerance.")
    eigs, vecs = np.linalg.eigh((hessian + hessian.T)/2)
    v = vecs[:, -1]
    step = .0002
    curvature = (train_loss(theta+step*v,x,y) + train_loss(theta-step*v,x,y)
                 - 2*train_loss(theta,x,y)) / step**2
    if not np.isclose(eigs[-1], curvature, atol=2.e-5, rtol=2.e-5):
        raise ValueError("Second difference of the actual training objective failed.")
    return eigs[-1], abs(curvature-eigs[-1])


def main():
    out = ROOT / "curvature_results"
    out.mkdir(exist_ok=False)
    rows = []
    with threadpool_limits(1):
        for pair in range(1,33):
            x,y = rings(121000+pair)
            state = advance(core.initialize_parameters(122000+pair),(x,y),"pullback",8,100)
            lam, loss_check = eigenvalue(state,x,y,1.e-5)
            fine, _ = eigenvalue(state,x,y,5.e-6)
            if abs(lam-fine) > 1.e-6:
                raise ValueError("Hessian refinement failed.")
            for rate in (.5,1.,2.):
                theta = state.copy()
                losses = [train_loss(theta,x,y)]
                for _ in range(25):
                    theta = theta - rate*core.parameter_gradient(theta,x,y)
                    losses.append(train_loss(theta,x,y))
                dv = np.diff(losses)
                rows.append(dict(pair=pair,rate=rate,lambda_max=lam,
                                 eta_times_lambda=lam*rate,above_local_scale=lam*rate>2,
                                 initial_loss=losses[0],first_loss=losses[1],second_loss=losses[2],
                                 loss_after_25=losses[-1],upward_steps=int((dv>1e-12).sum()),
                                 largest_upward_step=dv.max(),highest_loss=max(losses),
                                 directional_loss_check=loss_check))
    with (out/"by_seed.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for rate in (.5,1.,2.):
        group=[r for r in rows if r["rate"]==rate]
        scale=[r["eta_times_lambda"] for r in group]
        print(rate, "median eta*lambda", np.median(scale),
              "outside local scale", sum(r["above_local_scale"] for r in group),
              "any loss increases",sum(r["upward_steps"]>0 for r in group),
              "median number of increases", np.median([r["upward_steps"] for r in group]))


if __name__=="__main__":
    main()
