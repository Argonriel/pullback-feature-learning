"""Bounded transfer-task pilot; no confirmation streams are read."""
from pathlib import Path
import sys,json,csv,time,hashlib,datetime
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import task
RECIPES=(('gd',.1),('gd',.3),('pb',8.),('pb',16.))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run():
    out=ROOT/'pilot';out.mkdir(exist_ok=False)
    (out/'PROTOCOL.json').write_text(json.dumps(dict(timestamp_utc=now(),task='4D two-harmonic waveform; bounded Beta radial mixture',architecture='Two residual blocks,width8,4D,144parameters,fixedfirstcoordinate logit',seeds=list(range(1,5)),train_seed_base=1031000,init_seed_base=1032000,evaluation_seed_base=1033000,training_samples=256,training_exact_class_counts=[128,128],evaluation_samples=16384,recipes=RECIPES,steps=500,damping=.1,choice='For each optimizer, minimize mean pilot conditional logistic risk over its two predeclared rates; this training-rate choice explicitly uses pilot posterior supervision.',selector_tau=1.,new_confirmation_streams_read=False,source_sha256=sha(ROOT/'task.py')),indent=2))
    # Analytic Jacobian, independently implemented gradient, and task symmetry checks.
    theta=task.initialize(1032001);x,y=task.sample_train(1031001,256);rng=np.random.default_rng(1034001);direction=rng.normal(size=task.PARAMETER_COUNT);direction/=np.linalg.norm(direction);h=1e-5;jac=task.representation_jacobian(theta,x[:7]);fd=((task.representation(theta+h*direction,x[:7])-task.representation(theta-h*direction,x[:7]))/(2*h)).ravel();jac_error=float(abs(jac@direction-fd).max());g=task.parameter_gradient(theta,x,y);loss_fd=(task.loss(theta+h*direction,x,y)-task.loss(theta-h*direction,x,y))/(2*h);gradient_error=abs(float(g@direction-loss_fd));assert jac_error<1e-8 and gradient_error<1e-8
    assert np.array_equal(task.partner(task.partner(x)),x);assert np.allclose(task.transform(x,np.pi),task.partner(x),atol=1e-14);assert np.allclose(task.quotient(task.transform(x,.713)),task.quotient(x),atol=1e-14)
    (out/'checks.json').write_text(json.dumps(dict(jacobian_fd_max_error=jac_error,independent_gradient_fd_error=gradient_error,partner_involution=True,halfturn_matches_transform=True,quotient_invariant=True,training_class_counts=np.bincount((y+1)//2).tolist()),indent=2));rows=[]
    for seed in range(1,5):
        x,y=task.sample_train(1031000+seed,256);theta=task.initialize(1032000+seed);xt,yt,p=task.sample_test(1033000+seed,16384);np.savez_compressed(out/f'{seed}_data.npz',x=x,y=y,initial=theta,test_x=xt,test_y=yt,test_p=p)
        for method,rate in RECIPES:
            started=time.perf_counter()
            try:
                fitted,metadata=task.train(theta,x,y,500,method,rate,.1);h=task.predict(fitted,xt);hp=task.predict(fitted,task.partner(xt));m=(h+hp)/2;expected=float(np.mean(np.logaddexp(0,h)-p*h));two=float(np.mean(np.logaddexp(0,m)-p*m));sampled=float(np.logaddexp(0,-yt*h).mean());trainloss=task.loss(fitted,x,y)
                path=out/f'{seed}_{method}_{rate:g}.npz';np.savez_compressed(path,theta=fitted,loss_trace=np.asarray(metadata['loss_trace']));row=dict(seed=seed,method=method,rate=rate,status='success',conditional_risk=expected,full_partner_risk=two,sampled_risk=sampled,training_loss=trainloss,seconds=time.perf_counter()-started,state_sha256=sha(path),error='')
            except Exception as error:row=dict(seed=seed,method=method,rate=rate,status='failed',conditional_risk='',full_partner_risk='',sampled_risk='',training_loss='',seconds=time.perf_counter()-started,state_sha256='',error=repr(error))
            rows.append(row);print(json.dumps(row),flush=True)
    with (out/'results.csv').open('x',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    means={f'{method}_{rate:g}':float(np.mean([r['conditional_risk'] for r in rows if r['method']==method and r['rate']==rate])) for method,rate in RECIPES if all(r['status']=='success' for r in rows if r['method']==method and r['rate']==rate)}
    chosen={method:min([rate for meth,rate in RECIPES if meth==method and f'{meth}_{rate:g}' in means],key=lambda rate:(means[f'{method}_{rate:g}'],rate)) for method in ('gd','pb')}
    (ROOT/'TRAINING_CHOICE.json').write_text(json.dumps(dict(timestamp_utc=now(),chosen_rates=chosen,pilot_mean_conditional_risk=means,source_results_sha256=sha(out/'results.csv'),steps=500,damping=.1,task_models=4,supervised_training_recipe_choice=True,selector_tau_fixed=1.,new_confirmation_streams_read=False),indent=2));print('CHOICE',chosen,flush=True)
if __name__=='__main__':
    with threadpool_limits(1):run()
