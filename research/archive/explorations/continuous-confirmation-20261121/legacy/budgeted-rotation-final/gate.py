"""Fit on unlabeled model calls and allocate a batch of extra half-turn views."""
from dataclasses import dataclass
import numpy as np


def gap(a, b):
    soft = lambda s: np.logaddexp(0., s)
    return .5*(soft(a)+soft(b))-soft(.5*(a+b))


def angle_bin(x):
    # The same canonical half-plane representative is selected for x and -x.
    flip = (x[:, 1] < 0) | ((x[:, 1] == 0) & (x[:, 0] < 0))
    c = np.where(flip[:, None], -x, x)
    angle = np.arctan2(c[:, 1], c[:, 0]) % np.pi
    return np.minimum((8*angle/np.pi).astype(int), 7)


@dataclass
class Gate:
    edges: np.ndarray
    radius_scores: np.ndarray
    folded_scores: np.ndarray

    @classmethod
    def fit(cls, x, predict):
        r = np.linalg.norm(x, axis=1)
        benefit = gap(predict(x), predict(-x))
        edges = np.quantile(r, np.arange(1,16)/16)
        idx = np.searchsorted(edges, r)
        counts = np.bincount(idx, minlength=16)
        if min(counts)==0 or np.min(benefit)<-1e-12:
            raise ValueError("Calibration coverage or nonnegativity check failed")
        radial = np.bincount(idx, weights=benefit, minlength=16)/counts
        cell = 8*idx+angle_bin(x)
        counts = np.bincount(cell, minlength=128)
        sums = np.bincount(cell, weights=benefit, minlength=128)
        folded = np.divide(sums, counts, out=np.repeat(radial,8), where=counts>0)
        return cls(edges, radial, folded)

    def priority(self, x, name):
        idx = np.searchsorted(self.edges, np.linalg.norm(x, axis=1))
        if name == "radius":
            return self.radius_scores[idx]
        if name == "folded":
            return self.folded_scores[8*idx+angle_bin(x)]
        raise ValueError(name)


def allocate(scores, tie_key, quota):
    """Rank independently down the last axis, taking exactly quota per batch."""
    if not 0 <= quota <= scores.shape[-1] or scores.shape != tie_key.shape:
        raise ValueError("Inconsistent budget or tie keys")
    order = np.lexsort((tie_key, -scores), axis=-1)
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, order[..., :quota], True, axis=-1)
    return mask


def deploy(x, predict, gate, policy, tie_key, quota=64):
    """No labels, generator parameters, or second-view values enter routing."""
    logits = predict(x).copy()
    selected = allocate(gate.priority(x, policy), tie_key, quota)
    if selected.any():
        logits[selected] = .5*(logits[selected]+predict(-x[selected]))
    return logits, selected
