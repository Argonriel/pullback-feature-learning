"""Independent numerical task/geometry/model checks, not fresh evaluation."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import task
HERE=Path(__file__).resolve().parent

def main():
  rng=np.random.default_rng(1139001); errors={}; n=4096
  r=rng.uniform(.4,1.4,(n,2)); phi=rng.uniform(-np.pi,np.pi,n); a=rng.uniform(-np.pi,np.pi,n)
  x=task.coordinates(r[:,0],r[:,1],phi)
  errors['quotient_roundtrip']=float(np.max(np.abs(task.quotient(x)-r)))
  errors['group_coordinates']=float(np.max(np.abs(task.transform(x,a)-task.coordinates(r[:,0],r[:,1],phi+a))))
  errors['group_composition']=float(np.max(np.abs(task.transform(task.transform(x,a),.731)-task.transform(x,a+.731))))
  errors['partner_pi']=float(np.max(np.abs(task.partner(x)-task.transform(x,np.pi))))
  errors['partner_involution']=float(np.max(np.abs(task.partner(task.partner(x))-x)))
  errors['posterior_invariance']=float(np.max(np.abs(task.posterior(task.transform(x,a))-task.posterior(x))))
  errors['quotient_density']=float(np.max(np.abs(task.density_quotient(r)-1)))
  a1=np.array([[.5,1.3],[np.sqrt(.97),np.sqrt(.97)]])
  counter=dict(amplitudes=a1.tolist(),squared_total_norm=np.sum(a1*a1,axis=1).tolist(),posterior=task.posterior_quotient(a1).tolist())
  errors['counterexample_norm_match']=float(np.ptp(np.sum(a1*a1,axis=1)))
  assert np.ptp(task.posterior_quotient(a1))>.3
  t,w=np.polynomial.legendre.leggauss(256); axis=.9+.5*t; mass=w*.5
  rr=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1); pp=task.posterior_quotient(rr); ww=mass[:,None]*mass[None,:]
  prevalence=float(np.sum(ww*pp)); bayes_error=float(np.sum(ww*np.minimum(pp,1-pp)))
  bayes_error_exact=float(.5-.4*(np.log(np.cosh(4))/4)**2)
  bayes_entropy=float(np.sum(ww*(-pp*np.log(pp)-(1-pp)*np.log1p(-pp))))
  # Absolute value has a cusp at zero: integrate smooth positive quadrant independently.
  positive=.9+.25*(t+1); rp=np.stack(np.meshgrid(positive,positive,indexing='ij'),axis=-1)
  posw=(w*.25)[:,None]*(w*.25)[None,:]
  error_split=float(4*np.sum(posw*(1-task.posterior_quotient(rp))))
  errors['prevalence_half']=abs(prevalence-.5)
  errors['bayes_error_split_formula']=abs(error_split-bayes_error_exact)
  errors['cdf_support']=float(max(abs(task.cdf_quotient([1.4,1.4])-1),abs(task.cdf_quotient([.4,1.4]))))
  errors['rectangle_mass']=float(abs(task.rectangle_mass([.5,.7],[1.1,1.2])-.3))
  states=[('initial',task.initialize(1132001)),('pilot_pb4',np.load(HERE/'pilot_results/1_pb4_500.npz')['theta'])]
  xx=x[:17]; yy=np.where(rng.random(len(xx))<task.posterior(xx),1,-1)
  with threadpool_limits(limits=1):
    for name,theta in states:
      direction=rng.normal(size=len(theta)); direction/=np.linalg.norm(direction); eps=1e-5
      j=task.representation_jacobian(theta,xx).reshape(len(xx),4,-1)
      fd=(task.representation(theta+eps*direction,xx)-task.representation(theta-eps*direction,xx))/(2*eps)
      errors[name+'_representation_directional']=float(np.max(np.abs(fd-j@direction)))
      grad=task.parameter_gradient(theta,xx,yy)
      fdl=(task.loss(theta+eps*direction,xx,yy)-task.loss(theta-eps*direction,xx,yy))/(2*eps)
      errors[name+'_loss_directional']=float(abs(fdl-grad@direction))
      L,C=task.network_constants(theta)
      errors[name+'_residual_bound_violation']=float(max(0,np.max(np.abs(task.predict(theta,x)-x[:,0]))-C))
      actuald=(task.predict(theta,x)-task.predict(theta,task.partner(x)))/2
      errors[name+'_d_bound_violation']=float(max(0,np.max(np.abs(actuald))-min(1.4+C,1.4*L)))
  counts={}
  for seed in range(1,5):
      data=np.load(HERE/f'pilot_results/{seed}_inputs.npz'); xt,yt=task.sample_train(1131000+seed)
      assert np.array_equal(data['x'],xt) and np.array_equal(data['y'],yt)
      counts[seed]=[int(np.sum(yt==-1)),int(np.sum(yt==1))]
  failed={k:v for k,v in errors.items() if v>1e-7}
  payload=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='pass' if not failed else 'FAIL',
     geometry_n=n,model_checks=2,errors=errors,failed=failed,class_counts=counts,
     quotient_dimension=2,prevalence=prevalence,bayes_logloss=bayes_entropy,bayes_classification_error=bayes_error_exact,
     unsplit_bayes_error_quadrature=bayes_error,split_bayes_error_quadrature=error_split,equal_norm_counterexample=counter,
     core_sha256=hashlib.sha256((HERE/'core4d.py').read_bytes()).hexdigest())
  (HERE/'TASK_CHECKS.json').write_text(json.dumps(payload,indent=2)+'\n')
  print(json.dumps(payload,indent=2)); assert not failed

if __name__=='__main__':main()
