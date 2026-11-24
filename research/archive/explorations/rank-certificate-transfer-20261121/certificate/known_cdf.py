"""Known radial-law brackets for the unchanged rank selector's population CDF.

All enclosure statements are in real arithmetic. Float64 checks and padding are
numerical safeguards, not a directed-rounding proof.
"""
from pathlib import Path
import sys, math
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'reference'))
import rank_policy as policy
import local_jets as jets


def radial_cdf(r, task):
    if task == 'rings':
        return jets.rice_cdf(r)
    u = np.clip(np.asarray(r)-.4, 0, 1)
    # Integral of 15 u (1-u) [u^3+(1-u)^3].
    return 7.5*u*u-20*u**3+22.5*u**4-9*u**5, 0.


def distribution(table, task='rings', budget=131072):
    edges = table['radii']; a = table['gain_profile']
    variation = abs(np.diff(a))
    n = np.maximum(1, np.ceil((budget-len(variation))*variation/max(variation.sum(), 1e-300)).astype(int))
    r = np.concatenate([np.linspace(edges[i], edges[i+1], n[i], endpoint=False)
                        for i in range(len(n))] + [edges[-1:]])
    value = np.interp(r, edges, a)
    cdf, tail = radial_cdf(r, task)
    mass = np.diff(cdf)
    assert mass.min() > -1e-13
    mass = np.maximum(mass, 0.)
    amin = np.minimum(value[:-1], value[1:])
    amax = np.maximum(value[:-1], value[1:])
    # Outside support, np.interp uses the endpoint location. The truncated Rice
    # mixture's missing mass can be assigned here with at most `tail` CDF error.
    amin = np.r_[a[0], amin, a[-1]]; amax = np.r_[a[0], amax, a[-1]]
    mass = np.r_[cdf[0], mass, 1-cdf[-1]]
    assert abs(mass.sum()-1) < 1e-10
    result = dict(r=r, amin=amin, amax=amax, mass=mass,
                  width=np.array(float(table['width'])), tail=np.array(tail),
                  task=np.array(task), coordinate_bins=np.array(len(mass)))
    for name, location in [('lo', amin), ('hi', amax)]:
        order = np.argsort(location)
        result[name+'_a'] = location[order]
        result[name+'_mass'] = mass[order]
        result[name+'_cum'] = np.r_[0., np.cumsum(mass[order])]
        result[name+'_moment'] = np.r_[0., np.cumsum(mass[order]*location[order])]
    return result


def empirical_distribution(table, radii, alpha):
    """DKW band from independent verification-only radii, not fit radii.

    The policy must already be fixed independently of this sample. The DKW
    event is uniform in score, so subsequent local numerical integration and
    radius evaluation need no additional pointwise confidence union bound.
    """
    if not 0 < alpha < 1:raise ValueError('alpha must be in (0,1)')
    a=np.sort(np.interp(np.asarray(radii),table['radii'],table['gain_profile']))
    mass=np.full(len(a),1/len(a));cum=np.r_[0.,np.arange(1,len(a)+1)/len(a)]
    moment=np.r_[0.,np.cumsum(a)/len(a)]
    epsilon=np.sqrt(np.log(2/alpha)/(2*len(a)))
    result=dict(amin=a,amax=a,mass=mass,width=np.array(float(table['width'])),
                tail=np.array(epsilon),task=np.array('empirical_DKW'),
                coordinate_bins=np.array(len(a)),alpha=np.array(alpha),
                independent_verification_radii=np.asarray(radii))
    for side in ('lo','hi'):
        result[side+'_a']=a;result[side+'_mass']=mass
        result[side+'_cum']=cum;result[side+'_moment']=moment
    return result


def mixture(dist, score, side):
    a, mass, moment = (dist[side+'_a'], dist[side+'_cum'], dist[side+'_moment'])
    score = np.asarray(score); w = float(dist['width'])
    lo = np.searchsorted(a, score-w, side='right')
    hi = np.searchsorted(a, score, side='right')
    return np.clip(mass[lo]+(score*(mass[hi]-mass[lo])-(moment[hi]-moment[lo]))/w, 0, 1)


def bracket(dist, score):
    # Larger conditional-uniform locations produce the smaller score CDF.
    pad = float(dist['tail'])+2e-12
    return np.maximum(0, mixture(dist, score, 'hi')-pad), np.minimum(1, mixture(dist, score, 'lo')+pad)


def cdf_a(dist, value, side):
    index = np.searchsorted(dist[side+'_a'], value, side='right')
    return dist[side+'_cum'][index]


def derivative_bracket(dist, low, high):
    w = float(dist['width']); pad = float(dist['tail'])+2e-12
    lower = np.maximum(0, cdf_a(dist, low, 'hi')-cdf_a(dist, high-w, 'lo')-2*pad)/w
    upper = np.minimum(1, cdf_a(dist, high, 'lo')-cdf_a(dist, low-w, 'hi')+2*pad)/w
    return lower, upper


def q_table(n=262145):
    u = np.linspace(0, 1, n); q = policy.rank_probability(u)
    c = math.lgamma(256)-math.lgamma(192)-math.lgamma(64)
    second = (191*np.exp(c+190*np.log(190/253)+63*np.log(63/253))+
              63*np.exp(c+191*np.log(191/253)+62*np.log(62/253)))
    return u, q, second/(8*(n-1)**2)+2e-12


def q_bracket(dist, score, qt):
    lo, hi = bracket(dist, score)
    u, q, error = qt
    return np.maximum(0, np.interp(lo, u, q)-error), np.minimum(1, np.interp(hi, u, q)+error)


def beta_density(u):
    u = np.asarray(u); clipped = np.clip(u, 1e-300, 1-1e-16)
    c = math.lgamma(256)-math.lgamma(192)-math.lgamma(64)
    out = np.exp(c+191*np.log(clipped)+63*np.log1p(-clipped))
    return np.where((u <= 0)|(u >= 1), 0, out)


def q_score_derivative(dist, low, high):
    fl = bracket(dist, low)[0]; fh = bracket(dist, high)[1]
    a, b = beta_density(fl), beta_density(fh)
    lower = np.minimum(a, b)
    upper = np.where((fl <= 191/254)&(fh >= 191/254), policy.Q_LIPSCHITZ, np.maximum(a,b))
    dl, dh = derivative_bracket(dist, low, high)
    return np.maximum(0, lower*dl-2e-12), upper*dh+2e-12
