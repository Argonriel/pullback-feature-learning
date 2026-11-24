"""Historical development examples only; no new random streams opened."""
import numpy as np
from threadpoolctl import threadpool_limits
import model,runner

def main():
    old=np.load(runner.SOURCE/'reference/DEVELOPMENT_STATE.npz');x=old['x'];y=old['y'];checks=[];rng=np.random.default_rng(2099901)
    for state,theta in enumerate((old['initial'],old['selected_pb'])):
        base=4*model.task.direction(theta,x,y,method='pb',damping=.1)
        for k in (2,3,9,12,14,15):
            xx=np.tile(x,(k,1));yy=np.tile(y,k);update=4*k*model.task.direction(theta,xx,yy,method='pb',damping=.1*k)
            checks.append(dict(test='PB_duplication_normalization',state=state,k=k,error=float(np.max(abs(base-update)))))
        jac=model.task.representation_jacobian(theta,x);grad=model.task.parameter_gradient(theta,x,y)
        normalized=-(4/256)*np.linalg.solve(jac.T@jac/len(y)+(.1/256)*np.eye(144),grad)
        checks.append(dict(test='normalized_metric_equivalence',state=state,error=float(np.max(abs(base-normalized)))))
        for j in range(8):
            d=rng.normal(size=144);d/=np.linalg.norm(d);eps=1e-5;fd=(model.task.loss(theta+eps*d,x,y)-model.task.loss(theta-eps*d,x,y))/(2*eps)
            checks.append(dict(test='directional_gradient',state=state,index=j,error=float(abs(fd-grad@d))))
    P=144;D=4;H=8;lu=sum(m+2*m*m for m in range(1,P));solve=2*P*(P-1)+P
    ledger=[]
    for proc in runner.PROCEDURES:
        n=proc['n'];repr_cost=2*n*(H*(2*D-1)+2*H+D*(2*H-1)+D)
        jac_cost=2*n*(H*(2*D-1)+H+2*H+D*D*(3*H-1)+2*D*D+D*P*(2*D-1)+4*D*H*D+3*D*D*H+3*D*H+H+D*(2*H-1)+D)
        total=repr_cost+jac_cost+5*n+P*P*(2*D*n-1)+P+D*n*P+P*(2*D*n-1)+lu+solve+2*P
        assert total==179045*n+2001144
        slack=runner.BUDGET-total*proc['steps'];assert 0<=slack<total and n+proc['calibration']==4352
        ledger.append(dict(n=n,calibration=proc['calibration'],updates=proc['steps'],per_update_nominal_flops=total,nominal_budget=runner.BUDGET,unused_budget=slack,relative_shortfall=slack/runner.BUDGET,checkpoints=runner.checkpoints(proc['steps'])))
    # Enumerate every stochastic family used by both phases and check stream IDs.
    used={};streams=[]
    for phase,count,offset in [('development',8,0),('confirmation',16,50)]:
        for pair in range(1,count+1):
            sid=pair+offset
            ids=[('train_'+str(inc),2411000+inc+sid) for inc in (0,100,200,300)]+[('shuffle',2411800+sid),('init',2412000+sid),('calibration',2414000+sid),('evaluation_inputs',2416000+sid),('evaluation_labels',2417000+sid)]
            for purpose,seed in ids:
                assert seed not in used,(used.get(seed),phase,pair,purpose);used[seed]=(phase,pair,purpose);streams.append(dict(phase=phase,pair=pair,purpose=purpose,seed=seed))
    assert 2419050 not in used and 2419100 not in used
    cx,cy=model.labeled_inputs(2099902,256);theta=old['selected_pb'];raw=model.task.predict(theta,cx);full=.5*(raw+model.task.predict(theta,model.task.partner(cx)));fits={}
    for pipeline,h in [('raw',raw),('full',full)]:
        fit=model.temperature(h,cy);assert fit['calibrated_nll']<=fit['raw_nll']+1e-12;assert np.array_equal(h>=0,fit['beta']*h>=0);fits[pipeline]=fit
    failures=[r for r in checks if r['error']>1e-8]
    runner.jo(runner.OUT/'PREFLIGHT.json',dict(utc=runner.now(),status='PASS' if not failures else 'FAIL',fresh_streams_accessed=False,checks=checks,failures=failures,maximum_error=max(r['error'] for r in checks),
      arithmetic_ledger=ledger,calibration_checks=fits,stream_ids_unique=True,enumerated_streams=streams,development_state_sha256=runner.sha(runner.SOURCE/'reference/DEVELOPMENT_STATE.npz')))
    print('PASS',len(checks),'numerical checks; max error',max(r['error'] for r in checks));assert not failures

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
