"""A standard two-parameter linear model reproduces the path-length headline.

This is an analytical screening countermodel, not a fit to the rings data.
No dataset, neural-network training, or population-risk evaluation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def simulate(gamma, steps=500, eta=.05, damping=.1, a=1., b=np.sqrt(8/15)):
    """Direct damped pullback updates for f=u, h=a*u+b*v, L=(u-1)^2/2."""
    predictive = np.array([1., 0.])
    unused = np.array([a, b])
    metric = (np.outer(predictive, predictive)
              + gamma*np.outer(unused, unused) + damping*np.eye(2))
    denominator = (1+damping)*damping + gamma*((1+damping)*b*b+damping*a*a)
    rate = (damping+gamma*b*b)/denominator
    slope = -gamma*a*b/(damping+gamma*b*b)
    unused_slope = a*damping/(damping+gamma*b*b)
    assert 0 < eta*rate < 1, "Monotone discrete dynamics required for path formula"
    theta = np.zeros(2)
    path = np.zeros(2)
    records = []
    max_formula_error = 0.
    for step in range(steps+1):
        readout, second = float(predictive@theta), float(unused@theta)
        expected_u = 1-(1-eta*rate)**step
        expected = np.array([expected_u, slope*expected_u])
        max_formula_error = max(max_formula_error, float(np.max(abs(theta-expected))),
                                abs(second-unused_slope*expected_u))
        records.append(dict(step=step, u=float(theta[0]), v=float(theta[1]),
                            readout=readout, unused=second,
                            predictive_path=float(path[0]), unused_path=float(path[1]),
                            loss=.5*(readout-1)**2))
        if step < steps:
            update = -eta*np.linalg.solve(metric, predictive*(readout-1))
            theta += update
            path += np.abs([predictive@update, unused@update])
    assert max_formula_error < 1e-12
    assert abs(path[0]-records[-1]['readout']) < 1e-12
    assert abs(path[1]-records[-1]['unused']) < 1e-12
    return dict(gamma=gamma, rate=rate, parameter_slope=slope,
                unused_vs_readout_slope=unused_slope,
                formula_max_error=max_formula_error, endpoint=records[-1], records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = [simulate(gamma) for gamma in (0., 1/9, 1., 9.)]
    first, last = runs[0]['endpoint'], runs[-1]['endpoint']
    # Continuous time can be rescaled exactly by each rate. At common tau=k*t,
    # all readouts equal 1-exp(-tau), but the unused feature slopes still differ.
    matched_clock = []
    for tau in np.linspace(0., 10., 51):
        values = [float(1-np.exp(-run['rate']*(tau/run['rate']))) for run in runs]
        matched_clock.append(dict(tau=float(tau), readouts=values,
                                  max_difference=max(values)-min(values)))
    clock_error = max(row['max_difference'] for row in matched_clock)
    assert clock_error < 1e-14
    summary = dict(
        status='PASS', example='Two-parameter linear weighted damped least squares',
        parameters=dict(a=1., b=float(np.sqrt(8/15)), damping=.1, eta=.05, steps=500),
        expected_infinite_time_unused_path_ratio=49.,
        observed_finite_unused_path_ratio=first['unused_path']/last['unused_path'],
        predictive_path_relative_difference=abs(last['predictive_path']/first['predictive_path']-1),
        matched_continuous_clock_max_readout_error=clock_error,
        closed_form_max_error=max(run['formula_max_error'] for run in runs),
        endpoints=[dict(gamma=run['gamma'], rate=run['rate'], **run['endpoint']) for run in runs],
        meaning='The path-length headline alone does not identify a new nonlinear mechanism.',
        limitation='Illustrative scalar-output model; not a causal explanation fitted to archived rings models.',
        python=platform.python_version(), numpy=np.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'data.json').write_text(json.dumps(dict(runs=runs, matched_clock=matched_clock), indent=2),
                                       encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
