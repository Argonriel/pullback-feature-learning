"""Development-state checks; no prospective streams are opened."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
import model, runner

def main():
    old=np.load(runner.SOURCE/'reference/DEVELOPMENT_STATE.npz');x=old['x'];y=old['y'];checks=[];rng=np.random.default_rng(2099901)
    for si,theta in enumerate((old['initial'],old['selected_pb'])):
        baseline=4*model.task.direction(theta,x,y,method='pb',damping=.1)
        for k in (2,3,9):
            xx=np.tile(x,(k,1));yy=np.tile(y,k)
            check=4*k*model.task.direction(theta,xx,yy,method='pb',damping=.1*k)
            checks.append(dict(test='PB_duplication_normalization',state=si,k=k,error=float(np.max(abs(check-baseline)))))
            gd=model.task.parameter_gradient(theta,x,y);gd2=model.task.parameter_gradient(theta,xx,yy)
            checks.append(dict(test='mean_gradient_duplication',state=si,k=k,error=float(np.max(abs(gd-gd2)))))
        jac=model.task.representation_jacobian(theta,x);grad=model.task.parameter_gradient(theta,x,y)
        normalized=-(4/256)*np.linalg.solve(jac.T@jac/len(y)+(.1/256)*np.eye(144),grad)
        checks.append(dict(test='normalized_metric_equivalence',state=si,error=float(np.max(abs(normalized-baseline)))))
        for index in range(8):
            d=rng.normal(size=144);d/=np.linalg.norm(d);eps=1e-5
            fd=(model.task.loss(theta+eps*d,x,y)-model.task.loss(theta-eps*d,x,y))/(2*eps)
            checks.append(dict(test='mean_loss_gradient_finite_difference',state=si,index=index,error=float(abs(fd-grad@d))))
    # Algebraic ledger checked from kernel-level contraction dimensions.
    P=144;D=4;H=8
    lu=sum(m+2*m*m for m in range(1,P));triangular=2*P*(P-1)+P
    constants=lu+triangular-P*P+2*P
    def counted(n):
        representation=2*n*(H*(2*D-1)+H+H+D*(2*H-1)+D)
        jacobian=2*n*(H*(2*D-1)+H+2*H+D*D*(3*H-1)+2*D*D+D*P*(2*D-1)+4*D*H*D+3*D*D*H+3*D*H+H+D*(2*H-1)+D)
        logistic=5*n;gram=P*P*(2*D*n-1)+P;rhs=D*n*P+P*(2*D*n-1)
        return representation+jacobian+logistic+gram+rhs+lu+triangular+2*P
    ledger=[];budget=500*counted(2304)
    for n in runner.NS:
        c=counted(n);assert c==179045*n+2001144
        updates=budget//c;slack=budget-updates*c;assert 0<=slack<c
        ledger.append(dict(n=n,cost_per_update=c,matched_updates=updates,slack=slack,relative_slack=slack/budget))
    assert runner.FLOP_STEPS=={r['n']:r['matched_updates'] for r in ledger}
    assert runner.work(256,4500)['training_example_updates']==runner.work(2304,500)['training_example_updates']==1152000
    cx,cy=model.labeled_inputs(2099902,256);theta=old['selected_pb'];raw=model.task.predict(theta,cx);full=.5*(raw+model.task.predict(theta,model.task.partner(cx)))
    fits={pipeline:model.temperature(h,cy) for pipeline,h in [('raw',raw),('full',full)]}
    for pipeline,h in [('raw',raw),('full',full)]:
        fit=fits[pipeline];assert fit['calibrated_nll']<=fit['raw_nll']+1e-12;assert np.array_equal(h>=0,fit['beta']*h>=0)
    assert all(p['n']+p['validation']+p['calibration']==4352 for p in runner.PROCEDURES)
    failed=[r for r in checks if r['error']>1e-8]
    result=dict(utc=runner.now(),status='PASS' if not failed else 'FAIL',fresh_streams_accessed=False,checks=checks,failures=failed,maximum_error=max(r['error'] for r in checks),
        scalar_flop_ledger=ledger,lu_flops=lu,triangular_flops=triangular,calibration_tests=fits,procedures=len(runner.PROCEDURES),baseline_training_work=runner.work(2304,500))
    runner.jo(runner.OUT/'PREFLIGHT.json',result);print(json.dumps(result,indent=2));assert not failed

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
