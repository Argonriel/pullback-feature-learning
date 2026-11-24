"""Independent, balanced two-moons samples with continuous random angles."""

import numpy as np


def two_moons(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2 or n % 2:
        raise ValueError("The sample count must be a positive even number.")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("Noise must be finite and nonnegative.")
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, np.pi, n)
    x = np.column_stack((np.cos(angle), np.sin(angle)))
    x[n // 2:, 0] = 1 - x[n // 2:, 0]
    x[n // 2:, 1] = 0.5 - x[n // 2:, 1]
    x += rng.normal(0, noise, x.shape)
    y = np.concatenate((-np.ones(n // 2), np.ones(n // 2)))
    order = rng.permutation(n)
    return x[order], y[order]
