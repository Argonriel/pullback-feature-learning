"""Label-free two-view rotation calibration; no labels or risk evaluator imported.

The deployment rule alpha(r) depends only on radius. Offline angular orbits
make every candidate's first and second marginal identical, including on the
finite calibration set. All deployed policies use two model evaluations.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from pullback_feature_learning import core


def phi(h):
    return np.logaddexp(h / 2, -h / 2)


def radii_sample(seed, n):
    rng = np.random.default_rng(seed)
    radii = rng.choice([.5, 1.2], n)
    angle = rng.uniform(0, 2*np.pi, n)
    x = radii[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    x += rng.normal(0, .15, (n, 2))
    return np.linalg.norm(x, axis=1), rng.uniform(0, 2*np.pi, n)


def orbit_logits(theta, radii, phases, n_angles=128, logit_velocity=None):
    angles = phases[:, None] + 2*np.pi*np.arange(n_angles)[None, :]/n_angles
    x = (radii[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=-1)).reshape(-1, 2)
    values=[]
    for i in range(0, len(x), 4096):
        z=x[i:i+4096]
        v=core.representation(theta, z)[:, 0]
        if logit_velocity is not None:
            v=v+logit_velocity(z)
        values.append(v)
    return np.concatenate(values).reshape(len(radii), n_angles)


@dataclass
class RotationSelection:
    names: list
    edges: np.ndarray
    angles: np.ndarray
    predicted_gains: np.ndarray
    gain_curves: np.ndarray
    radial_gain_curves: np.ndarray
    radii: np.ndarray
    phases: np.ndarray
    h: np.ndarray
    power: np.ndarray
    calibration_seconds: float
    forward_examples: int
    velocity_examples: int

    def alpha(self, radius, name):
        bins=np.searchsorted(self.edges[1:-1], radius, side='right')
        return self.angles[self.names.index(name), bins]

    def save(self, path):
        np.savez_compressed(path, names=np.array(self.names), edges=self.edges,
            angles=self.angles, predicted_gains=self.predicted_gains,
            gain_curves=self.gain_curves, radial_gain_curves=self.radial_gain_curves,
            radii=self.radii, phases=self.phases, h=self.h, power=self.power,
            calibration_seconds=self.calibration_seconds,
            forward_examples=self.forward_examples, velocity_examples=self.velocity_examples)


def selection_from_logits(h, radii, phases=None, bins=8, truncations=(4, 8, 16, 64)):
    start=perf_counter()
    nr, na=h.shape
    assert na%2 == 0 and max(truncations)<=na//2
    candidate=np.arange(1, na//2+1)*2*np.pi/na
    edges=np.r_[0., np.quantile(radii, np.arange(1, bins)/bins), np.inf]
    membership=np.searchsorted(edges[1:-1], radii, side='right')
    counts=np.bincount(membership, minlength=bins)
    assert counts.min()>0
    # DFT Parseval energy per positive-frequency mode, including Nyquist once.
    fft=np.fft.rfft(h,axis=1)/na
    power=2*np.abs(fft[:,1:])**2
    power[:,-1]/=2
    assert np.max(np.abs(power.sum(1)-h.var(1)))<1e-9
    k=np.arange(1, na//2+1)
    removal=np.sin(k[:,None]*candidate[None,:]/2)**2
    avg=h.mean(1)
    sig=np.exp(-np.logaddexp(0,-avg))
    curvature=sig*(1-sig)
    convex=phi(h)
    exact=np.array([(convex-phi(.5*(h+np.roll(h,-j,axis=1)))).mean(1)
                    for j in range(1,na//2+1)]).T
    assert exact.min()>-1e-12
    score_matrices={'exact':exact}
    for trunc in truncations:
        score_matrices[f'fourier_K{trunc}']=.5*(power[:,:trunc]*curvature[:,None])@removal[:trunc]
        score_matrices[f'energy_K{trunc}']=power[:,:trunc]@removal[:trunc]
    names=[]; angles=[]; predictions=[]; curves=[]; radial_curves=[]
    for label, score in score_matrices.items():
        global_curve=score.mean(0)
        radial_curve=np.array([score[membership==b].mean(0) for b in range(bins)])
        for policy in ('global','radial'):
            names.append(label+'_'+policy)
            choices=np.repeat(global_curve.argmax(),bins) if policy=='global' else radial_curve.argmax(1)
            angles.append(candidate[choices])
            predictions.append(float(np.sum(counts*radial_curve[np.arange(bins),choices])/nr))
            curves.append(global_curve)
            radial_curves.append(radial_curve)
    return RotationSelection(names,edges,np.array(angles),np.array(predictions),
        np.array(curves),np.array(radial_curves),radii,
        np.zeros(nr) if phases is None else phases,h,power,perf_counter()-start,0,0)


def calibrated_angle_selection(theta, calibration_radii, phases=None,
        logit_velocity=None, n_angles=128, bins=8, truncations=(4,8,16,64)):
    """Fit immutable alpha(r) policies without evaluation labels or risk.

    logit_velocity is an optional callable x -> predicted logit increment.
    For a tangent step delta, use lambda x: J_first(theta,x) @ delta.
    The predicted h+increment is calibrated before observing the new state.
    """
    start=perf_counter()
    phases=np.zeros(len(calibration_radii)) if phases is None else phases
    h=orbit_logits(theta,calibration_radii,phases,n_angles,logit_velocity)
    result=selection_from_logits(h,calibration_radii,phases,bins,truncations)
    result.calibration_seconds=perf_counter()-start
    result.forward_examples=len(calibration_radii)*n_angles
    result.velocity_examples=0 if logit_velocity is None else result.forward_examples
    return result


def rotate(x, alpha):
    c,s=np.cos(alpha),np.sin(alpha)
    return np.column_stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]))


def deploy(theta, x, selection, name):
    alpha=selection.alpha(np.linalg.norm(x,axis=1),name)
    return .5*(core.representation(theta,x)[:,0]+
               core.representation(theta,rotate(x,alpha))[:,0])
