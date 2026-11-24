"""Independent development-only normalization and multiclass-TV checks."""
from pathlib import Path
import datetime
import importlib.util
import itertools
import json
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'calibration-symmetry-realdata-20261121'
spec = importlib.util.spec_from_file_location('root_previous_task', OLD / 'training/reference/task.py')
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
rng = np.random.default_rng(2181001)

with threadpool_limits(limits=1):
    x, y, _ = task.sample_test(2181002, 32)
    theta = task.initialize(2181003)
    base = (4 * len(x) / 256) * task.direction(theta, x, y, damping=.1 * len(x) / 256)
    replication_rows = []
    for copies in (2, 3, 9):
        xx = np.repeat(x, copies, axis=0)
        yy = np.repeat(y, copies)
        update = (4 * len(xx) / 256) * task.direction(theta, xx, yy, damping=.1 * len(xx) / 256)
        error = float(np.max(abs(update-base)))
        assert error < 1e-10
        replication_rows.append(dict(copies=copies, max_update_difference=error))
    gradient = task.parameter_gradient(theta, x, y)
    derivative_errors = []
    for _ in range(12):
        direction = rng.normal(size=144)
        direction /= np.linalg.norm(direction)
        fd = (task.loss(theta+1e-5*direction,x,y)-task.loss(theta-1e-5*direction,x,y))/(2e-5)
        derivative_errors.append(abs(float(fd-direction@gradient)))
    assert max(derivative_errors) < 1e-8

def lse(z):
    return np.logaddexp.reduce(z,axis=-1)

def posterior(z):
    z = z-z.max(axis=-1,keepdims=True)
    p = np.exp(z)
    return p/p.sum(axis=-1,keepdims=True)

def ce(z,p):
    return lse(z)-np.sum(z*p,axis=-1)

transform = np.array([1,0,3,2])
records = np.array(list(itertools.product(range(4),repeat=3)))
tie = np.array([.1,.2,.3])
checks = []
for classes in (2,3,10):
    for replicate in range(20):
        z = rng.normal(size=(4,classes))*3
        p = posterior(rng.normal(size=(4,classes)))
        prob = rng.dirichlet(np.ones(4))
        eta = .5*np.sum(abs(prob-prob[transform]))
        epsilon = np.max(.5*np.sum(abs(p-p[transform]),axis=1))
        d = .5*(z-z[transform])
        m = .5*(z+z[transform])
        J = .5*(lse(z)+lse(z[transform]))-lse(m)
        span = np.ptp(d,axis=1)
        D = span.max()
        scores = np.repeat(rng.random(2),2)
        selected = np.argsort(scores[records]+tie*1e-12,axis=1)[:,-1]
        mask = np.zeros_like(records,dtype=float)
        mask[np.arange(len(records)),selected] = 1
        for coordinate in range(3):
            reflected = records.copy()
            reflected[:,coordinate] = transform[reflected[:,coordinate]]
            assert np.array_equal(selected,np.argsort(scores[reflected]+tie*1e-12,axis=1)[:,-1])
        f = ce(z,p)-ce(m,p)
        pairing = .5*(f+f[transform])
        identity = J-.5*np.sum((p-p[transform])*d,axis=1)
        pairing_error = float(np.max(abs(pairing-identity)))
        weights = np.prod(prob[records],axis=1)
        gain = np.sum(weights*np.mean(mask*f[records],axis=1))
        A = np.sum(weights*np.mean(mask*J[records],axis=1))
        B = np.sum(weights*np.mean(mask*span[records],axis=1))
        lower = A-.5*epsilon*B-D*eta
        assert gain >= lower-1e-12
        assert np.max(abs(f)) <= D+1e-12
        offsets = rng.normal(size=(4,1))*100
        shifted = z+offsets
        shifted_d = .5*(shifted-shifted[transform])
        offset_error = max(float(np.max(abs(np.ptp(shifted_d,axis=1)-span))),
                           float(np.max(abs(ce(shifted,p)-ce(z,p)))))
        assert max(pairing_error,offset_error)<1e-11
        checks.append(dict(classes=classes,replicate=replicate,eta=float(eta),epsilon=float(epsilon),
                           true_gain=float(gain),population_lower=float(lower),D=float(D),
                           identity_error=pairing_error,common_offset_error=offset_error))

result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',
    no_confirmation_streams_used=True,replication_normalization=replication_rows,
    gradient_checks=12,max_gradient_error=max(derivative_errors),
    finite_population_batch_checks=checks,
    maximum_identity_error=max(x['identity_error'] for x in checks),
    maximum_common_offset_error=max(x['common_offset_error'] for x in checks))
(ROOT/'root_audit/PREFLIGHT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(dict(status='PASS',normalization_checks=3,gradient_checks=12,
    multiclass_finite_population_checks=len(checks),max_identity_error=result['maximum_identity_error'])))
