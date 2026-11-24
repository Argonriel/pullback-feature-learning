"""Finite-look empirical Bernstein bounds on independent batch statistics."""
import math
import numpy as np

BATCH = 256
QUOTA = 64
LOOKS = (128, 512, 2048)
CLIPS = (.01, .025, .05, .1, .25, .5)


def jensen_gap(first, partner):
    """Convex logistic log-partition gap; no labels enter this computation."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(partner, dtype=float)
    value = .5 * (np.logaddexp(0., a) + np.logaddexp(0., b)) - np.logaddexp(0., .5 * (a + b))
    if np.min(value, initial=0.) < -1e-12:
        raise ArithmeticError('Negative Jensen gap exceeds numerical tolerance')
    return np.maximum(value, 0.)


def jensen_bound(d_bound):
    """If |(h-hT)/2| <= D, then J <= log cosh(D/2)."""
    return float(np.logaddexp(.5 * d_bound, -.5 * d_bound) - np.log(2.))


def empirical_bernstein_lower(values, range_width, delta):
    """Maurer--Pontil (2009), Theorem 4, applied to 1-rescaled values.

    The random units are complete independent batches. Sample variance uses n-1.
    This is a single-look one-sided bound. Caller allocates delta across all looks.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 or not 0. < delta < 1. or range_width < 0.:
        raise ValueError('Invalid empirical Bernstein arguments')
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    log_factor = math.log(2. / delta)
    stochastic = math.sqrt(2. * variance * log_factor / n)
    range_penalty = 7. * range_width * log_factor / (3. * (n - 1.))
    return dict(mean=mean, variance=variance, n_batches=n, range_width=float(range_width),
                delta=delta, stochastic_penalty=stochastic, range_penalty=range_penalty,
                lower=mean-stochastic-range_penalty)


def rings_tail_bound(C, cutoff=3., sigma=.15, radii=(.5, 1.2)):
    """Upper bound E[J 1{r>cutoff}] under the known Gaussian-noisy ring law.

    J <= (r+C)/2, r <= r0+S, S Rayleigh(sigma). The normal-tail integral
    is bounded by sigma^2/t exp(-t^2/(2 sigma^2)), t=cutoff-r0>0.
    """
    terms = []
    for r0 in radii:
        t = cutoff-r0
        if t <= 0:
            raise ValueError('Cutoff must exceed all noiseless ring radii')
        terms.append(.5 * (cutoff+C+sigma*sigma/t) * math.exp(-t*t/(2*sigma*sigma)))
    return float(np.mean(terms))


def statistics(gaps, masks, region=None):
    """Return batch means. Masks must be chosen from the entire original batch."""
    gaps = np.asarray(gaps).reshape(-1, BATCH)
    masks = [np.asarray(mask).reshape(-1, BATCH) for mask in masks]
    if any(not np.all(mask.sum(axis=1) == QUOTA) for mask in masks):
        raise AssertionError('Quota violation')
    result = {}
    for index, mask in enumerate(masks):
        result[f'policy{index}_unclipped'] = (mask*gaps).mean(axis=1)
        for clip in CLIPS:
            result[f'policy{index}_clip_{clip:g}'] = (mask*np.minimum(gaps, clip)).mean(axis=1)
    keep = np.ones_like(gaps) if region is None else np.asarray(region).reshape(gaps.shape)
    result['relative_truncated'] = ((masks[0].astype(float)-masks[1])*gaps*keep).mean(axis=1)
    result['relative_unclipped'] = ((masks[0].astype(float)-masks[1])*gaps).mean(axis=1)
    result['union_selected'] = np.logical_or.reduce(masks).sum(axis=1)
    return result
