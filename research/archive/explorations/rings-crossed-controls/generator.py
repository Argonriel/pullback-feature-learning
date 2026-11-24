"""Generator copied verbatim from the first rings experiment."""
import numpy as np


def rings(seed):
    rng = np.random.default_rng(seed)
    labels = np.repeat([-1.0, 1.0], 128)
    angle = rng.uniform(0, 2 * np.pi, len(labels))
    radius = np.where(labels < 0, 0.5, 1.2)
    xy = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    xy += rng.normal(0, 0.15, size=xy.shape)
    order = rng.permutation(len(labels))
    return xy[order], labels[order]
