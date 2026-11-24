"""Two residual blocks, independent backpropagation, and a full Jacobian.

Each block stores A (8,2), B (2,8), b (8,) in row-major order.
Representations and Jacobian rows use sample-major (sample, coordinate) order.
The representation metric is a SUM; the logistic objective is a MEAN.
"""

from __future__ import annotations

import numpy as np

PARAMETER_COUNT = 80


def initialize_parameters(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(2):
        blocks.extend((rng.normal(0, 1 / np.sqrt(2), (8, 2)).ravel(),
                       rng.normal(0, 0.1 / np.sqrt(8), (2, 8)).ravel(),
                       np.zeros(8)))
    return np.concatenate(blocks).astype(np.float64)


def unpack(theta: np.ndarray):
    theta = np.asarray(theta)
    if theta.shape != (PARAMETER_COUNT,):
        raise ValueError("Expected exactly 80 parameters.")
    return [(theta[k:k + 16].reshape(8, 2),
             theta[k + 16:k + 32].reshape(2, 8), theta[k + 32:k + 40])
            for k in (0, 40)]


def representation(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    h = np.asarray(x)
    if h.ndim != 2 or h.shape[1] != 2:
        raise ValueError("Inputs must have shape (n, 2).")
    for a, b_matrix, bias in unpack(theta):
        h = h + 0.5 * np.tanh(h @ a.T + bias) @ b_matrix.T
    return h


def representation_jacobian(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute the analytic full Jacobian, including both output coordinates."""
    h = np.asarray(x)
    if h.ndim != 2 or h.shape[1] != 2:
        raise ValueError("Inputs must have shape (n, 2).")
    jac = np.zeros((len(h), 2, PARAMETER_COUNT), dtype=np.result_type(theta, x))
    eye = np.eye(2)
    for block, (a, b_matrix, bias) in enumerate(unpack(theta)):
        t = np.tanh(h @ a.T + bias)
        derivative = 1 - t * t
        state_jac = eye + 0.5 * np.einsum("oi,ni,ij->noj", b_matrix, derivative, a)
        jac = np.einsum("noj,njp->nop", state_jac, jac)
        offset = 40 * block
        jac[:, :, offset:offset + 16] += (
            0.5 * np.einsum("oi,ni,nj->noij", b_matrix, derivative, h)
        ).reshape(len(h), 2, 16)
        jac[:, :, offset + 16:offset + 32] += (
            0.5 * np.einsum("ok,nj->nokj", eye, t)
        ).reshape(len(h), 2, 16)
        jac[:, :, offset + 32:offset + 40] += (
            0.5 * np.einsum("oi,ni->noi", b_matrix, derivative)
        )
        h = h + 0.5 * t @ b_matrix.T
    return jac.reshape(2 * len(h), PARAMETER_COUNT)


def logistic_loss(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.logaddexp(0.0, -y * z[:, 0]).mean())


def representation_gradient(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(z)
    # exp(-logaddexp(0, margin)) is sigmoid(-margin), without overflow.
    gradient[:, 0] = -y * np.exp(-np.logaddexp(0.0, y * z[:, 0])) / len(y)
    return gradient


def parameter_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reverse-mode derivative, independent of the explicit Jacobian code.

GD uses this path so its training time does not include a full Jacobian.
"""
    h = x
    cache = []
    blocks = unpack(theta)
    for a, b_matrix, bias in blocks:
        t = np.tanh(h @ a.T + bias)
        cache.append((h, t))
        h = h + 0.5 * t @ b_matrix.T
    v = representation_gradient(h, y)
    gradients = []
    for (a, b_matrix, _), (h_previous, t) in zip(reversed(blocks), reversed(cache)):
        d_b = 0.5 * v.T @ t
        d_pre = 0.5 * (v @ b_matrix) * (1 - t * t)
        d_a = d_pre.T @ h_previous
        d_bias = d_pre.sum(axis=0)
        gradients.append(np.concatenate((d_a.ravel(), d_b.ravel(), d_bias)))
        v = v + d_pre @ a
    return np.concatenate(gradients[::-1])


def direction(theta: np.ndarray, x: np.ndarray, y: np.ndarray,
              method: str = "gd", damping: float = 0.01) -> tuple[np.ndarray, dict]:
    """Return a descent direction; the caller applies its own learning rate.

No inverse is formed and no normalization of J.T @ J is introduced.
Extra scientific diagnostics are performed separately from timed training.
"""
    if method == "gd":
        return -parameter_gradient(theta, x, y), {}
    if method != "pullback":
        raise ValueError("Method must be 'gd' or 'pullback'.")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("Damping must be finite and strictly positive.")
    z = representation(theta, x)
    jac = representation_jacobian(theta, x)
    gradient = jac.T @ representation_gradient(z, y).ravel()
    gram = jac.T @ jac
    gram.flat[::PARAMETER_COUNT + 1] += damping
    return np.linalg.solve(gram, -gradient), {}
