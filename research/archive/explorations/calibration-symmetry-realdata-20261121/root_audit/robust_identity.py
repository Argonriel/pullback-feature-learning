"""Independent adversarial label-defect identity and scaled-model range checks."""
from pathlib import Path
import datetime,json,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'training'));import model
sys.path.insert(0,str(ROOT.parent/'symmetry-frontier-2d-20261121/method'));import baseline
task=model.task
def loss(h,p):return (1-p)*np.logaddexp(0,h)+p*np.logaddexp(0,-h)
rows=[]
with threadpool_limits(limits=1):
 x=task.sample_inputs(2142001,4096);ties=np.random.default_rng(2142002).random(len(x));p0=task.posterior(x)
 theta=np.load(ROOT.parent/'symmetry-frontier-2d-20261121/validation/states/two_invariants_1.npz')['pb']
 for beta in (.25,1.,2.):
    h=beta*task.predict(theta,x);ht=beta*task.predict(theta,task.partner(x));d=(h-ht)/2;m=(h+ht)/2
    J=np.maximum((np.logaddexp(0,h)+np.logaddexp(0,ht))/2-np.logaddexp(0,m),0)
    L=1.;C=0.
    for A,B,b in task.unpack(theta):L*=1+.5*np.linalg.norm(A,2)*np.linalg.norm(B,2);C+=.5*abs(B[0]).sum()
    D=beta*min(1.4+C,1.4*L);assert max(abs(d))<=D+1e-12
    mask=baseline.allocate(J,ties)
    for world,q in [('fixed_odd',x[:,0]/np.linalg.norm(x[:,:2],axis=1)),('adversarial',np.sign(d))]:
      for eps in (0.,.025,.05,.1,.2,.4,.8,1.):
        deviation=np.minimum(eps/2,np.minimum(p0,1-p0))*q
        p=p0+deviation;pt=p0-deviation
        assert np.all((p>=0)&(p<=1)) and np.max(abs(p-pt))<=eps+1e-14
        out=np.where(mask,m,h);outT=np.where(mask,m,ht)
        direct=(loss(h,p)-loss(out,p)+loss(ht,pt)-loss(outT,pt))/2
        identity=mask*(J-(p-pt)*d/2)
        error=float(max(abs(direct-identity)));assert error<1e-12
        for clip in (.01,.5):
            lower=mask*(np.minimum(J,clip)-eps*abs(d)/2)
            assert np.all(lower<=direct+1e-12)
        rows.append(dict(beta=beta,world=world,epsilon=eps,identity_error=error,
            actual_label_gap=float(max(abs(p-pt))),mean_direct_gain=float(direct.mean()),
            global_D=float(D),observed_max_d=float(max(abs(d)))))
result=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',
  source='Prior completed model; new development streams2142001/2142002 only',
  cases=len(rows),max_identity_error=max(r['identity_error'] for r in rows),
  beta_skip_scaling_explicit=True,posterior_valid=True,lower_bound_pointwise_passed=True,rows=rows)
(ROOT/'root_audit/ROBUST_IDENTITY.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
