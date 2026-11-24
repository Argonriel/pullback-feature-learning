"""Independent finite-population and interval checks; no fresh branch data."""
from pathlib import Path
import datetime,itertools,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
rng=np.random.default_rng(2441001)
T=np.array([1,0,3,2])
records=np.array(list(itertools.product(range(4),repeat=3)))
lse=lambda z:np.logaddexp.reduce(z,axis=-1)
max_identity=max_endpoint=max_even=0.
min_slack=float('inf');cases=[]
for k in (2,3,10):
    for rep in range(100):
        z=rng.normal(size=(4,k))*rng.choice([.1,1.,5.,20.])
        p=rng.dirichlet(np.ones(k),size=4)
        prob=rng.dirichlet(np.ones(4));r=(prob-prob[T])/(prob+prob[T])
        epsilon=float(np.max(np.sum(abs(p-p[T]),axis=1)/2))
        m=(z+z[T])/2;d=(z-z[T])/2
        J=(lse(z)+lse(z[T]))/2-lse(m);B=np.ptp(d,axis=1)/2
        t=(lse(z)-lse(z[T]))/2-(d.max(1)+d.min(1))/2
        width=np.repeat(rng.uniform(0,1,2),2)
        low=np.maximum(-1,r-width);high=np.minimum(1,r+width)
        assert np.max(abs(low[T]+high))<1e-14
        def penalty(v):return v*t-(epsilon+(1-epsilon)*abs(v))*B
        endpoint=np.minimum(penalty(low),penalty(high))
        grid=low[None,:]+np.linspace(0,1,1001)[:,None]*(high-low)[None,:]
        exact=penalty(grid).min(0)
        max_endpoint=max(max_endpoint,float(np.max(abs(endpoint-exact))))
        assert np.max(abs(endpoint-exact))<1e-12
        W=np.maximum(abs(low)*B-low*t,abs(high)*B-high*t)
        closest=np.where(low>0,low,np.where(high<0,-high,0))
        R=(1-closest)*B
        separate=-W-epsilon*R
        assert np.all(separate<=endpoint+1e-12)
        max_even=max(max_even,float(np.max(abs(endpoint-endpoint[T]))))
        scores=np.repeat(rng.random(2),2)
        keys=np.array([.1,.2,.3]);choice=np.argmax(scores[records]+1e-12*keys,axis=1)
        mask=np.zeros_like(records,float);mask[np.arange(64),choice]=1
        for i in range(3):
            changed=records.copy();changed[:,i]=T[changed[:,i]]
            assert np.array_equal(choice,np.argmax(scores[changed]+1e-12*keys,axis=1))
        weights=np.prod(prob[records],axis=1)
        expectation=lambda a:float(np.sum(weights*np.mean(mask*a[records],axis=1)))
        f=(lse(z)-np.sum(z*p,axis=1))-(lse(m)-np.sum(m*p,axis=1))
        gain=expectation(f)
        known=expectation(J+penalty(r));robust=expectation(J+endpoint)
        conservative=expectation(J+separate)
        identity=expectation((f+f[T])/2+r*(f-f[T])/2)
        max_identity=max(max_identity,abs(gain-identity))
        assert gain>=known-1e-11 and known>=robust-1e-11 and robust>=conservative-1e-11
        min_slack=min(min_slack,gain-robust)
        cases.append(dict(classes=k,rep=rep,gain=gain,known=known,robust=robust,separate=conservative))

# A simultaneous bin-probability rectangle gives a posterior-TV upper bound
# only conditional on within-cell constancy. Verify endpoint propagation.
posterior_checks=0
for _ in range(10000):
    probabilities=rng.uniform(0,1,8);radius=rng.uniform(0,.5,8)
    lower=np.maximum(0,probabilities-radius);upper=np.minimum(1,probabilities+radius)
    partner=np.arange(8)^1
    tv=float(np.max(abs(probabilities-probabilities[partner])))
    bound=float(np.max(np.maximum(upper-lower[partner],upper[partner]-lower)))
    assert tv<=bound+1e-15
    posterior_checks+=1

out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    stream=2441001,branch_fresh_streams_opened=False,finite_populations=len(cases),
    enumerated_batches_per_population=64,endpoint_grid_points=1001,
    max_orbit_identity_error=max_identity,max_endpoint_error=max_endpoint,
    max_transform_even_error=max_even,min_population_bound_slack=min_slack,
    cell_rectangle_checks=posterior_checks,cases=cases,
    limits='Independent floating-point checks, not a formal proof or evidence that structural assumptions hold in real data.')
if '--record' in sys.argv:
    (ROOT/'root_audit/PREFLIGHT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
