"""Risk under the specified Gaussian-noisy rings, in nats per example."""

from functools import lru_cache
import json

import numpy as np
from pullback_feature_learning.core import representation, unpack


INNER, OUTER, SIGMA, R_MAX = 0.5, 1.2, 0.15, 3.0
GRID_ATOL = 2e-7
GRID_RTOL = 2e-6
# (total radial Gauss nodes, equispaced midpoint angles). The radial interval
# is split at the Bayes decision boundary for the discontinuous Bayes rule.
# The first-stage states converge by 768. A fresh intervention checkpoint
# required a denser grid at the same tolerance; lower levels are identical.
GRIDS = tuple((n, n) for n in (48, 96, 192, 384, 768, 1536, 3072))
COMPONENTS = ("population_loss", "radial_excess", "angular_excess")


def bayes_logit(radii):
    r = np.asarray(radii)
    return ((INNER**2 - OUTER**2)/(2 * SIGMA**2)
            + np.log(np.i0(r * OUTER / SIGMA**2))
            - np.log(np.i0(r * INNER / SIGMA**2)))


@lru_cache(maxsize=1)
def decision_radius():
    lo, hi = INNER, OUTER
    for _ in range(54):
        mid = (lo + hi) / 2
        if bayes_logit(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (hi + lo) / 2


def class_density(r, radius):
    return (r / SIGMA**2 * np.exp(-(r*r + radius**2)/(2 * SIGMA**2))
            * np.i0(r * radius / SIGMA**2))


@lru_cache(maxsize=len(GRIDS))
def polar_grid(radial_nodes, angular_nodes):
    nodes, weights = np.polynomial.legendre.leggauss(radial_nodes // 2)
    intervals = [(0., decision_radius()), (decision_radius(), R_MAX)]
    radii = np.concatenate([(b+a)/2 + nodes * ((b-a)/2) for a, b in intervals])
    dr = np.concatenate([weights * ((b-a)/2) for a, b in intervals])
    w0 = dr * class_density(radii, INNER) / 2
    w1 = dr * class_density(radii, OUTER) / 2
    angles = (np.arange(angular_nodes) + .5) * (2*np.pi / angular_nodes)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    return radii, w0, w1, unit


def logits_on_circles(theta, radii, unit):
    # Block the forward pass to limit peak RAM on refined grids.
    n_angle = len(unit)
    radial_batch = max(1, 16384 // n_angle)
    h = np.empty((len(radii), n_angle))
    for i in range(0, len(radii), radial_batch):
        rs = radii[i:i+radial_batch]
        points = (rs[:, None, None] * unit[None, :, :]).reshape(-1, 2)
        h[i:i+len(rs)] = representation(theta, points)[:, 0].reshape(len(rs), -1)
    return h


def integrate(h, radii, w0, w1):
    avg = h.mean(axis=1)
    soft = np.logaddexp(0., h)
    # Independent class-conditional direct evaluation of the population loss.
    loss = w0 @ soft.mean(axis=1) + w1 @ np.logaddexp(0., -h).mean(axis=1)
    oracle = bayes_logit(radii)
    bayes = w0 @ np.logaddexp(0., oracle) + w1 @ np.logaddexp(0., -oracle)
    radial = w0 @ np.logaddexp(0., avg) + w1 @ np.logaddexp(0., -avg) - bayes
    angular = (w0+w1) @ (soft.mean(axis=1) - np.logaddexp(0., avg))
    residual = abs((loss - bayes) - (radial + angular))
    tol = 2e-12 * max(1., loss)
    if not np.all(np.isfinite([loss, bayes, radial, angular])) or min(radial, angular) < -tol or residual > tol:
        raise ArithmeticError("Polar loss decomposition failed its identity or positivity check.")
    return dict(population_loss=float(loss), bayes_loss=float(bayes),
                radial_excess=float(radial), angular_excess=float(angular),
                excess_loss=float(loss - bayes), identity_error=float(residual))


def evaluate(theta):
    """Refine quadrature until the three smooth reported outcomes stabilize.

    The grid difference estimates integration accuracy. A separate analytic
    upper bound covers the population logistic loss beyond radius R_MAX.
    """
    previous = None
    for nr, na in GRIDS:
        radii, w0, w1, unit = polar_grid(nr, na)
        value = integrate(logits_on_circles(theta, radii, unit), radii, w0, w1)
        if previous is not None:
            delta = {k: abs(value[k]-previous[k]) for k in COMPONENTS}
            converged = all(delta[k] <= GRID_ATOL + GRID_RTOL * abs(value[k]) for k in COMPONENTS)
            if converged:
                # |h| <= r + sum_l 0.5 ||B_l[first row]||_1, and
                # P(r>x) <= exp(-(x-OUTER)^2/(2 sigma^2)) for x>=OUTER.
                residual_bound = sum(.5*np.abs(b[0]).sum() for _, b, _ in unpack(theta))
                tail = (np.log(2.) + residual_bound + R_MAX + SIGMA**2/(R_MAX-OUTER)
                        ) * np.exp(-(R_MAX-OUTER)**2/(2*SIGMA**2))
                return dict(**value, grid_error=max(delta.values()), radial_nodes=nr,
                            angular_nodes=na, grid_converged=True, tail_loss_bound=float(tail))
        previous = value
    raise ArithmeticError(f"Polar quadrature failed to converge at grid {(nr, na)}; differences {delta}.")


def verify_controls():
    """Independent oracle, radial-only and angular-only controls with known answers."""
    radii, w0, w1, unit = polar_grid(96, 96)
    oracle = bayes_logit(radii)
    h = np.broadcast_to(oracle[:, None], (len(radii), len(unit)))
    predicted = integrate(h, radii, w0, w1)
    zero = integrate(np.zeros_like(h), radii, w0, w1)
    offset = integrate(h + .7, radii, w0, w1)
    varying = integrate(h + 2*unit[:, 0][None, :], radii, w0, w1)
    accuracy = float(np.maximum(w0,w1).sum())
    # Independent trapezoidal oracle integration from the earlier rings note.
    p = np.linspace(0., R_MAX, 120001)
    inner, outer = class_density(p, INNER), class_density(p, OUTER)
    b = bayes_logit(p)
    # Explicit equal-spacing trapezoid works across NumPy 1.26 and 2.x.
    def trap(v):
        return float((v.sum() - (v[0] + v[-1])/2) * (R_MAX/(len(v)-1)))
    dense_loss = .5*trap(inner*np.logaddexp(0., b) + outer*np.logaddexp(0., -b))
    dense_acc = .5*trap(np.maximum(inner, outer))
    mass = float((w0+w1).sum())
    assert abs(1-mass)<1e-12 and abs(predicted["bayes_loss"]-dense_loss)<1e-9
    assert abs(accuracy-dense_acc)<1e-8
    assert max(abs(predicted[k]) for k in ("radial_excess","angular_excess","excess_loss")) < 1e-12
    assert abs(zero["population_loss"]-np.log(2.))<1e-12 and abs(zero["angular_excess"])<1e-12
    assert offset["radial_excess"] > 0 and abs(offset["angular_excess"])<1e-12
    assert varying["angular_excess"] > 0 and abs(varying["radial_excess"])<1e-12
    return dict(decision_radius=decision_radius(), bayes_accuracy=accuracy, bayes_loss=predicted["bayes_loss"],
                gaussian_polar_mass=mass, dense_oracle_loss=dense_loss, dense_oracle_accuracy=dense_acc,
                zero_logit_loss=zero["population_loss"], radial_only_excess=offset["radial_excess"],
                angular_only_excess=varying["angular_excess"], grid_atol=GRID_ATOL, grid_rtol=GRID_RTOL,
                truncation_radius=R_MAX)


if __name__ == "__main__":
    print(json.dumps(verify_controls(), indent=2, allow_nan=False))
