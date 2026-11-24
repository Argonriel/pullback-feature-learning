"""Independent finite-difference check of regularized gradients and temperature."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'training'));import model
rng=np.random.default_rng(2141001);rows=[]
with threadpool_limits(limits=1):
 x,y=model.labeled_inputs(2141002,32);theta=model.task.initialize(2141003)
 for lam in (0.,.001,.01):
    gradient=-model.direction(theta,x,y,'gd',lam)
    z=model.task.representation(theta,x);jac=model.task.representation_jacobian(theta,x)
    for damping in (.1,.25):
        pb=model.direction(theta,x,y,'pb',lam,damping)
        residual=(jac.T@jac+damping*np.eye(144))@pb+gradient
        assert np.max(abs(residual))<2e-12
    maximum=0.
    for _ in range(12):
        v=rng.normal(size=144);v/=np.linalg.norm(v);eps=1e-5
        fd=(model.regularized_objective(theta+eps*v,x,y,lam)-model.regularized_objective(theta-eps*v,x,y,lam))/(2*eps)
        maximum=max(maximum,abs(fd-v@gradient))
    assert maximum<1e-8
    rows.append(dict(lam=lam,directional_checks=12,max_gradient_error=maximum,pb_normal_equation_max_error=float(np.max(abs(residual)))))
 temp_rows=[]
 for scale in (.1,1.,5.,20.):
    h=rng.normal(size=2048)*scale;p=1/(1+np.exp(-h/3));labels=np.where(rng.random(len(h))<p,1.,-1.)
    result=model.temperature(h,labels);beta=result['beta']
    assert result['calibrated_nll']<=result['raw_nll']+1e-14
    assert np.array_equal(h>=0,beta*h>=0)
    if result['status']=='interior':assert abs(result['derivative'])<1e-12
    # Independent dense beta grid can only be no better than the exact convex fit.
    grid=np.geomspace(model.BETA_MIN,model.BETA_MAX,1001)
    losses=np.logaddexp(0,-grid[:,None]*labels*h).mean(1)
    assert result['calibrated_nll']<=losses.min()+1e-12
    temp_rows.append(dict(logit_scale=scale,**result,grid_min=float(losses.min())))
result=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',
  development_streams=[2141001,2141002,2141003],no_fresh_streams=True,
  model_sha256=hashlib.sha256((ROOT/'training/model.py').read_bytes()).hexdigest(),gradient_checks=rows,temperature_checks=temp_rows)
(ROOT/'root_audit/TRAINING_PREFLIGHT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
