"""Independent quadrature cross-check on the six distinct first-pair checkpoints."""

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning.core import representation

from polar import evaluate, verify_controls


def independent(theta):
    # Composite Simpson on a uniform radial grid, and a prime count of angular
    # samples with an independent origin. No adaptive grid nodes are reused.
    radii = np.linspace(0., 3., 769)
    angles = (np.arange(521) + np.sqrt(2)/2) * (2*np.pi / 521)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    weights = np.ones(len(radii))
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    weights *= (radii[1] - radii[0])/3
    totals = np.zeros(3)
    for i in range(0, len(radii), 24):
        r = radii[i:i+24]
        points = (r[:, None, None]*unit).reshape(-1, 2)
        logits = representation(theta, points)[:, 0].reshape(len(r), len(angles))
        hbar = logits.mean(axis=1)
        coefficients = []
        for radius in (.5, 1.2):
            coefficients.append(.5*r/.15**2 * np.i0(r*radius/.15**2)
                                * np.exp(-(r**2+radius**2)/(2*.15**2)))
        fm, fp = coefficients
        posterior_logit = ((.5**2-1.2**2)/(2*.15**2)
                           + np.log(np.i0(r*1.2/.15**2))-np.log(np.i0(r*.5/.15**2)))
        full = fm*np.logaddexp(0., logits).mean(axis=1) + fp*np.logaddexp(0., -logits).mean(axis=1)
        radial_model = fm*np.logaddexp(0., hbar) + fp*np.logaddexp(0., -hbar)
        optimal = fm*np.logaddexp(0., posterior_logit) + fp*np.logaddexp(0., -posterior_logit)
        totals += weights[i:i+len(r)] @ np.column_stack((full, radial_model-optimal, full-radial_model))
    return dict(zip(("population_loss", "radial_excess", "angular_excess"), map(float, totals)))


def main():
    folder = Path(__file__).resolve().parent / "stage1"
    report = {"controls": verify_controls(), "independent_grid": {"radial_simpson_nodes": 769,
              "angular_equispaced_nodes": 521, "angular_phase_in_steps": np.sqrt(2)/2}, "states": {}}
    with threadpool_limits(1), np.load(folder / "states.npz") as states:
        for key in sorted(k for k in states if k.startswith("pair1_")):
            main_grid = evaluate(states[key])
            other = independent(states[key])
            differences = {k: abs(v-main_grid[k]) for k,v in other.items()}
            if max(differences.values()) > 2e-6:
                raise ArithmeticError(f"Independent polar grid discrepancy: {key} {differences}")
            report["states"][key] = {"adaptive_grid_nodes": main_grid["radial_nodes"],
                "independent_values": other, "absolute_discrepancies": differences}
    assert len(report["states"]) == 6
    report["maximum_discrepancy"] = max(max(v["absolute_discrepancies"].values())
                                        for v in report["states"].values())
    (folder / "polar_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(report["maximum_discrepancy"])


if __name__ == "__main__":
    main()
