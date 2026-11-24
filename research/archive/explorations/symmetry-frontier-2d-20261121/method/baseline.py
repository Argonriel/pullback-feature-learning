"""Input-only, exactly half-turn-invariant Jensen allocation.

All fitting functions use coordinates and model predictions only.  Hidden
posteriors belong exclusively to independent evaluators.
"""
from itertools import product
import time
import numpy as np

BATCH = 256
QUOTA = 64
FORWARD_BUDGET = 65536


def quotient_matrix(adapter, x):
    z = np.asarray(adapter.quotient(x), dtype=float)
    return z[:, None] if z.ndim == 1 else z


def folded_angle(x):
    """The same floating-point representative for x and its half-turn."""
    pair = np.asarray(x)[:, :2]
    flip = (pair[:, 1] < 0) | ((pair[:, 1] == 0) & (pair[:, 0] < 0))
    canonical = np.where(flip[:, None], -pair, pair)
    # Normalizing signed zero makes the canonical axis representatives equal.
    canonical = np.where(canonical == 0, 0., canonical)
    return np.arctan2(canonical[:, 1], canonical[:, 0]) % np.pi


def jensen(first, partner):
    first = np.asarray(first)
    partner = np.asarray(partner)
    gain = .5 * (np.logaddexp(0., first) + np.logaddexp(0., partner))
    gain -= np.logaddexp(0., .5 * (first + partner))
    return np.maximum(gain, 0.)


def default_axes(adapter):
    if adapter.name == 'rings':
        return [np.linspace(0., 3., 128)], 512
    if adapter.name in ('transfer', 'waveform', 'harmonic'):
        return [np.linspace(.4, 1.4, 128)], 512
    return [np.linspace(.4, 1.4, 16), np.linspace(.4, 1.4, 16)], 256


def fit_grid(adapter, theta, axes=None, angle_count=None):
    """Charge exactly one model evaluation per stored full-orbit grid point."""
    start = time.perf_counter()
    if axes is None:
        axes, default_angles = default_axes(adapter)
    else:
        axes = [np.asarray(a, dtype=float) for a in axes]
        default_angles = 512 if len(axes) == 1 else 256
    angle_count = default_angles if angle_count is None else int(angle_count)
    if angle_count % 2:
        raise ValueError('An even angle count is required.')
    phi = 2 * np.pi * (np.arange(angle_count) + .5) / angle_count
    mesh = np.meshgrid(*axes, phi, indexing='ij')
    coords = adapter.coordinates(*mesh).reshape(-1, adapter.dimension)
    h = np.empty(len(coords))
    calls = []
    for begin in range(0, len(coords), 4096):
        end = min(begin + 4096, len(coords))
        h[begin:end] = adapter.predict(theta, coords[begin:end])
        calls.append(end - begin)
    h = h.reshape(*[len(a) for a in axes], angle_count)
    paired = np.roll(h, angle_count // 2, axis=-1)
    full_gain = jensen(h, paired)
    # Reuse a single half-orbit field, so invariance is structural rather than
    # dependent on approximate equality of trigonometric coordinates.
    half_gain = .5 * (full_gain[..., :angle_count // 2] +
                      full_gain[..., angle_count // 2:])
    table = dict(theta=np.asarray(theta), h=h, folded_jensen=half_gain,
                 ndim=np.array(len(axes)), angle_count=np.array(angle_count),
                 forward_examples=np.array(len(coords)),
                 forward_call_sizes=np.asarray(calls),
                 fit_seconds=np.array(time.perf_counter() - start))
    for i, axis in enumerate(axes):
        table[f'axis_{i}'] = axis
    if len(axes) == 1:
        table['radii'] = axes[0]
        table['gain_profile'] = full_gain.mean(axis=-1)
    return table


def scores(adapter, x, table):
    """Multilinear quotient interpolation and periodic folded-angle interpolation."""
    z = quotient_matrix(adapter, x)
    ndim = int(table['ndim'])
    if z.shape[1] != ndim:
        raise ValueError('Quotient dimension does not match the table.')
    lower, fractions = [], []
    for j in range(ndim):
        axis = table[f'axis_{j}']
        value = np.clip(z[:, j], axis[0], axis[-1])
        lo = np.clip(np.searchsorted(axis, value, side='right') - 1,
                     0, len(axis) - 2)
        lower.append(lo)
        fractions.append((value - axis[lo]) / (axis[lo + 1] - axis[lo]))
    field = table['folded_jensen']
    count = field.shape[-1]
    position = folded_angle(x) * count / np.pi - .5
    angle_floor = np.floor(position).astype(np.int64)
    angle_fraction = position - angle_floor
    answer = np.zeros(len(x))
    for corners in product((0, 1), repeat=ndim):
        indices = tuple(lo + c for lo, c in zip(lower, corners))
        weight = np.ones(len(x))
        for fraction, corner in zip(fractions, corners):
            weight *= fraction if corner else (1 - fraction)
        left = field[indices + (angle_floor % count,)]
        right = field[indices + ((angle_floor + 1) % count,)]
        answer += weight * ((1 - angle_fraction) * left + angle_fraction * right)
    return answer


def allocate(score, ties):
    score = np.asarray(score).reshape(-1, BATCH)
    ties = np.asarray(ties).reshape(-1, BATCH)
    order = np.lexsort((ties, -score), axis=1)
    mask = np.zeros_like(score, dtype=bool)
    np.put_along_axis(mask, order[:, :QUOTA], True, axis=1)
    assert np.all(mask.sum(axis=1) == QUOTA)
    return mask.ravel()


def select(adapter, x, table, ties):
    return allocate(scores(adapter, x, table), ties)


def nearest_scores(adapter, x, table):
    z = quotient_matrix(adapter, x)
    indices = []
    for j in range(int(table['ndim'])):
        axis = table[f'axis_{j}']
        midpoints = .5 * (axis[1:] + axis[:-1])
        indices.append(np.searchsorted(midpoints, z[:, j], side='right'))
    count = table['folded_jensen'].shape[-1]
    indices.append(np.floor(folded_angle(x) * count / np.pi).astype(int) % count)
    return table['folded_jensen'][tuple(indices)]


def select_nearest(adapter, x, table, ties):
    return allocate(nearest_scores(adapter, x, table), ties)


def histogram_configs(ndim):
    return ((16, 8), (32, 16), (64, 32)) if ndim == 1 else ((4, 4, 8), (8, 8, 16), (16, 16, 16))


def fit_folded(adapter, theta, x, config=None, prepared=None):
    """Histogram baseline with frozen development-selected resolution.

    ``prepared`` reuses the same model logits across candidate histograms.
    All resolutions and the empty-cell pooled-mean fallback are prespecified.
    """
    start = time.perf_counter()
    h, partner = prepared if prepared is not None else (
        adapter.predict(theta, x), adapter.predict(theta, adapter.partner(x)))
    gain = jensen(h, partner)
    z = quotient_matrix(adapter, x)
    config = histogram_configs(z.shape[1])[0] if config is None else tuple(config)
    if z.shape[1] not in (1, 2):
        raise ValueError('Only the frozen one- and two-invariant tasks are supported.')
    cell = np.zeros(len(x), dtype=int)
    if len(config) != z.shape[1] + 1:
        raise ValueError('Histogram config must give each quotient axis and angle bins.')
    count_cells = int(np.prod(config))
    table = dict(ndim=np.array(z.shape[1]), config=np.asarray(config))
    for j in range(z.shape[1]):
        side = config[j]
        edges = np.quantile(z[:, j], np.arange(1, side) / side)
        table[f'edges_{j}'] = edges
        cell = side * cell + np.searchsorted(edges, z[:, j])
    angle_bins = config[-1]
    angle_bin = np.minimum((angle_bins * folded_angle(x) / np.pi).astype(int), angle_bins - 1)
    index = angle_bins * cell + angle_bin
    count = np.bincount(index, minlength=count_cells)
    score = np.divide(np.bincount(index, weights=gain, minlength=count_cells), count,
                      out=np.full(count_cells, gain.mean()), where=count > 0)
    table.update(scores=score, counts=count, forward_examples=np.array(2 * len(x)),
                 forward_call_sizes=np.array([len(x), len(x)]) if prepared is None else np.array([], dtype=int),
                 additional_forward_examples=np.array(2 * len(x) if prepared is None else 0),
                 calibration_x=x, calibration_first=h, calibration_partner=partner,
                 fit_seconds=np.array(time.perf_counter() - start))
    return table


def folded_scores(adapter, x, table):
    z = quotient_matrix(adapter, x)
    config = table['config']
    cell = np.zeros(len(x), dtype=int)
    for j in range(int(table['ndim'])):
        cell = int(config[j]) * cell + np.searchsorted(table[f'edges_{j}'], z[:, j])
    angle_bins = int(config[-1])
    angle_bin = np.minimum((angle_bins * folded_angle(x) / np.pi).astype(int), angle_bins - 1)
    return table['scores'][angle_bins * cell + angle_bin]


def select_folded(adapter, x, table, ties):
    return allocate(folded_scores(adapter, x, table), ties)


def deploy(adapter, theta, x, table, ties, name='symmetric_jensen', predict=None):
    """Actual 256+64 examples per batch; no unselected partner calls."""
    if name == 'symmetric_jensen':
        mask = select(adapter, x, table, ties)
    elif name == 'folded':
        mask = select_folded(adapter, x, table, ties)
    elif name == 'grid_nearest':
        mask = select_nearest(adapter, x, table, ties)
    else:
        raise ValueError(name)
    predictor = (lambda a: adapter.predict(theta, a)) if predict is None else predict
    first = predictor(x)
    second = predictor(adapter.partner(x[mask]))
    output = first.copy()
    output[mask] = .5 * (first[mask] + second)
    return output, mask
