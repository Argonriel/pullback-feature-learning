"""Exact fixed-state conditional-label-noise screen, not an innovation claim.

A nonlinear two-parameter representation is held at one fixed state. Its
current-batch pullback metric depends on the inputs but not the current labels.
All input/label events are enumerated for a batch of size two.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


THETA = np.array([.3, -.2])
INPUTS = np.array([[1., .4], [.2, 2.]])
PROBS = np.array([.8, .2])
TARGETS = np.array([.6, -.9])
DAMPING = .2
BATCH = 2


def representation(x):
    a, b = THETA
    first, second = x
    # z=(a*x1+a*b*x2,b*x2); prediction uses z[0].
    z = np.array([a * first + a * b * second, b * second])
    jac = np.array([[first + b * second, a * second], [0., second]])
    return z, jac


def enumerate_noise(amplitudes):
    events = []
    for inputs in itertools.product(range(2), repeat=BATCH):
        xprob = float(np.prod(PROBS[list(inputs)]))
        zj = [representation(INPUTS[i]) for i in inputs]
        metric = sum(j.T @ j for _, j in zj) / BATCH
        metric += DAMPING * np.eye(2)
        for signs in itertools.product((-1., 1.), repeat=BATCH):
            grad = sum(j[0] * (z[0] - TARGETS[i] - amplitudes[i] * s)
                       for (z, j), i, s in zip(zj, inputs, signs)) / BATCH
            events.append((xprob / (2 ** BATCH), -np.linalg.solve(metric, grad)))
    probability = np.array([p for p, _ in events])
    directions = np.array([u for _, u in events])
    mean = probability @ directions
    centered = directions - mean
    covariance = np.einsum('n,ni,nj->ij', probability, centered, centered)
    return dict(event_count=len(events), probability_mass=float(probability.sum()),
                mean_update=mean.tolist(), covariance=covariance.tolist())


def verify():
    results = {name: enumerate_noise(np.array(amps)) for name, amps in {
        'no_label_noise': [0., 0.],
        'homoskedastic_label_noise': [.5, .5],
        'heteroskedastic_label_noise': [.1, 3.],
    }.items()}
    base = np.array(results['no_label_noise']['mean_update'])
    error = max(float(np.max(np.abs(np.array(r['mean_update']) - base)))
                for r in results.values())
    assert error < 1e-13
    hetero_cov = np.array(results['heteroskedastic_label_noise']['covariance'])
    base_cov = np.array(results['no_label_noise']['covariance'])
    covariance_change = float(np.linalg.norm(hetero_cov - base_cov))
    assert covariance_change > .1
    return dict(status='passed', role='Exact conditional-expectation reduction screen',
                state=THETA.tolist(), inputs=INPUTS.tolist(), probabilities=PROBS.tolist(),
                conditional_target_means=TARGETS.tolist(), batch_size=BATCH,
                damping=DAMPING, loss='half squared error on z[0]',
                representation='z=(theta1*x1+theta1*theta2*x2,theta2*x2)',
                cases=results, mean_update_max_absolute_difference=error,
                heteroskedastic_covariance_change_frobenius=covariance_change,
                scope='Fixed state, fresh zero-mean label noise conditional on current inputs; this does not assert invariance of future parameter distributions.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    value = verify()
    out.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(value, indent=2))
