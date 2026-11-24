"""Independent finite examples for the conditional projection identity and routing rule."""
from itertools import combinations
import numpy as np


def softplus(a):
    return np.logaddexp(0., a)


def test_symmetry_and_radial_preservation():
    count = 80
    angles = np.arange(count) * 2*np.pi/count
    radius = np.array([.5, .86, 1.2])[:, None]
    h = 7*(radius-.86) + (.1+radius)*np.sin(angles) + .8*np.cos(2*angles)
    other = np.roll(h, count//2, axis=1)
    avg = (h+other)/2
    d = (softplus(h)+softplus(other))/2-softplus(avg)
    assert d.min() > -1e-14
    # The folded cell is identical for the two orientations of an observation.
    cells = np.broadcast_to(np.repeat(np.arange(8), 5).tolist()*2, h.shape)
    cells = cells + 8*np.arange(len(radius))[:, None]
    assert np.array_equal(cells, np.roll(cells,count//2,axis=1))
    max_identity = 0.
    for p in (0., 1., 1/(1+np.exp(-7*(radius-.864)))):
        delta = softplus(h)-p*h - (softplus(avg)-p*avg)
        for cell in np.unique(cells):
            include = cells == cell
            max_identity = max(max_identity, abs(delta[include].mean()-d[include].mean()))
    assert max_identity < 1e-13
    gate = (cells % 3 == 1) | (cells == 18)
    gated = np.where(gate,avg,h)
    np.testing.assert_allclose(h.mean(axis=1), gated.mean(axis=1), rtol=0, atol=1e-13)
    # The drop in angular Jensen penalty equals the routed Jensen gains.
    original_angle_penalty = softplus(h).mean(axis=1)-softplus(h.mean(axis=1))
    gated_angle_penalty = softplus(gated).mean(axis=1)-softplus(gated.mean(axis=1))
    np.testing.assert_allclose(original_angle_penalty-gated_angle_penalty, (gate*d).mean(axis=1), atol=1e-13)
    sig = 1/(1+np.exp(-avg))
    alternative = .5*np.log1p(4*sig*(1-sig)*np.sinh((h-other)/4)**2)
    np.testing.assert_allclose(d,alternative,rtol=1e-12,atol=1e-14)
    return float(max_identity)


def test_ties_and_budget():
    # With one slot left at score 3, three tied items each have probability 1/3.
    scores=np.array([1.,3.,4.,3.,0.,3.,2.])
    k=2
    alternatives = list(combinations(range(len(scores)),k))
    totals = [sum(scores[list(subset)]) for subset in alternatives]
    best = [s for s,v in zip(alternatives,totals) if v==max(totals)]
    assert len(best)==3 and max(totals)==7
    probs=np.array([sum(i in s for s in best)/len(best) for i in range(len(scores))])
    np.testing.assert_allclose(probs,[0,1/3,1,1/3,0,1/3,0])
    # Tie priorities are attached to input indices independently of the sign of x.
    priorities=np.random.default_rng(9312).random(len(scores))
    choices=np.lexsort((priorities,-scores))[:k]
    assert tuple(sorted(choices)) in best and len(choices)==k
    return len(alternatives)


def test_angular_remainder_and_wrong_gate():
    phi=2*np.pi*np.arange(64)/64
    even=np.cos(2*phi)
    opposite=np.roll(even,32)
    np.testing.assert_allclose(even,opposite,atol=2e-15)
    d=(softplus(even)+softplus(opposite))/2-softplus((even+opposite)/2)
    assert np.max(np.abs(d))<1e-14 and (softplus(even).mean()-softplus(even.mean()))>.05
    # y=1 on both orientations. A gate acting only on the correct orientation
    # degrades it and leaves the confidently wrong orientation uncorrected.
    h=np.array([4.,-4.])
    m=np.zeros(2)
    q=h>0
    gain=(softplus(-h)-softplus(-np.where(q,m,h))).mean()
    assert gain<-.3
    return float(gain)


if __name__=="__main__":
    residual=test_symmetry_and_radial_preservation()
    candidates=test_ties_and_budget()
    counter=test_angular_remainder_and_wrong_gate()
    print(f"Conditional loss identity residual {residual:.3g}; all {candidates} subsets checked; "
          f"orientation-sensitive counterexample gain {counter:.6f} nats.")
