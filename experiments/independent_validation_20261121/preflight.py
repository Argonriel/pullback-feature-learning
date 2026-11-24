"""Independent checks using development-only streams, before confirmation.

Finite enumeration is a numerical check, not a substitute for the written proof.
"""
from pathlib import Path
import datetime, importlib.util, itertools, json
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('audit_old_task', ROOT/'root_audit/reference/task.py')
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
rng = np.random.default_rng(2341001)

def lse(z):
    return np.logaddexp.reduce(z, axis=-1)

def ce(z, p):
    return lse(z) - np.sum(z*p, axis=-1)

with threadpool_limits(limits=1):
    x, y, _ = task.sample_test(2341002, 32)
    theta = task.initialize(2341003)
    step = len(x)*4/256*task.direction(theta,x,y,damping=len(x)*.1/256)
    duplication = []
    for k in (2,3,9):
        xx=np.repeat(x,k,axis=0); yy=np.repeat(y,k)
        rep=len(xx)*4/256*task.direction(theta,xx,yy,damping=len(xx)*.1/256)
        err=float(np.max(abs(step-rep))); assert err<1e-10
        duplication.append(dict(copies=k,max_error=err))
    grad=task.parameter_gradient(theta,x,y); errors=[]
    for _ in range(16):
        v=rng.normal(size=144);v/=np.linalg.norm(v)
        numerical=(task.loss(theta+1e-5*v,x,y)-task.loss(theta-1e-5*v,x,y))/2e-5
        errors.append(abs(float(numerical-v@grad)))
    assert max(errors)<1e-8

    T=np.array([1,0,3,2]); records=np.array(list(itertools.product(range(4),repeat=3)))
    cases=[]; max_extremal=0.
    for classes in (2,3,10):
        for rep in range(80):
            logits=rng.normal(size=(4,classes))*rng.choice([.1,1.,5.,20.])
            post=rng.dirichlet(np.ones(classes),size=4)
            probs=rng.dirichlet(np.ones(4))
            r=(probs-probs[T])/(probs+probs[T])
            eps=float(np.max(np.sum(abs(post-post[T]),axis=1)/2))
            m=(logits+logits[T])/2; d=(logits-logits[T])/2
            J=(lse(logits)+lse(logits[T]))/2-lse(m)
            S=np.ptp(d,axis=1); D=float(S.max())
            K=(lse(logits)-lse(logits[T]))/2
            t=K-(d.max(axis=1)+d.min(axis=1))/2
            assert np.max(abs(t)-S/2)<1e-12
            scores=np.repeat(rng.random(2),2)
            keys=np.array([.1,.2,.3])
            choice=np.argmax(scores[records]+1e-12*keys,axis=1)
            mask=np.zeros_like(records,float);mask[np.arange(64),choice]=1
            for i in range(3):
                changed=records.copy(); changed[:,i]=T[changed[:,i]]
                assert np.array_equal(choice,np.argmax(scores[changed]+1e-12*keys,axis=1))
            weights=np.prod(probs[records],axis=1)
            expectation=lambda a:float(np.sum(weights*np.mean(mask*a[records],axis=1)))
            gain=expectation(ce(logits,post)-ce(m,post))
            new=expectation(J+r*t-(abs(r)+(1-abs(r))*eps)*S/2)
            eta=float(np.sum(abs(probs-probs[T]))/2)
            old=expectation(J)-eps*expectation(S)/2-D*eta
            assert gain>=new-2e-12 and new>=old-2e-12
            f=ce(logits,post)-ce(m,post)
            orbit=(f+f[T])/2+r*(f-f[T])/2
            identity_error=abs(gain-expectation(orbit));assert identity_error<2e-12
            for e in (0,.01,.2,.7,1):
                for rho in (-1,-.6,0,.3,1):
                    v=d[0];lo=int(v.argmin());hi=int(v.argmax())
                    p=np.zeros(classes);pt=np.zeros(classes)
                    if rho>=0:
                        p[hi]=1;pt[hi]=1-e;pt[lo]+=e
                    else:
                        pt[lo]=1;p[lo]=1-e;p[hi]+=e
                    obj=((1+rho)*p-(1-rho)*pt)@v/2
                    bound=rho*(v.max()+v.min())/2+(abs(rho)+(1-abs(rho))*e)*np.ptp(v)/2
                    err=abs(float(obj-bound));max_extremal=max(max_extremal,err)
                    assert err<2e-12
            cases.append(dict(classes=classes,rep=rep,epsilon=eps,eta=eta,gain=gain,
                local_lower=new,old_lower=old,identity_error=identity_error))

out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    streams=[2341001,2341002,2341003],confirmation_streams_accessed=False,
    duplication_checks=duplication,gradient_directions=16,max_gradient_error=max(errors),
    finite_population_checks=len(cases),enumerated_batches_per_case=64,
    extremal_posterior_checks=len(cases)*25,max_extremal_error=max_extremal,
    max_orbit_identity_error=max(v['identity_error'] for v in cases),cases=cases,
    limitations='Floating-point independent algebra checks; the theorem requires a written proof.')
(ROOT/'root_audit/PREFLIGHT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k not in ['cases','duplication_checks']}))
