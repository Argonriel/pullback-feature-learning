"""Development-only checks before the prospective source freeze."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
import model,study
HERE=Path(__file__).resolve().parent
def main():
    old=np.load(HERE/'reference/DEVELOPMENT_STATE.npz');states=[old['initial'],old['selected_pb']]
    x=old['x'];y=old['y'];rng=np.random.default_rng(2099901);checks=[]
    with threadpool_limits(limits=1):
      for index,theta in enumerate(states):
        update=4*model.task.direction(theta,x,y,method='pb',damping=.1)
        for k in (2,3,9):
            xx=np.tile(x,(k,1));yy=np.tile(y,k)
            duplicated=(4*k)*model.task.direction(theta,xx,yy,method='pb',damping=.1*k)
            gd=model.task.direction(theta,x,y,method='gd');gddup=model.task.direction(theta,xx,yy,method='gd')
            checks.append(dict(state=index,test='duplicate_pb',k=k,error=float(np.max(abs(update-duplicated)))))
            checks.append(dict(state=index,test='duplicate_gd_mean_gradient',k=k,error=float(np.max(abs(gd-gddup)))))
        jac=model.task.representation_jacobian(theta,x);grad=model.task.parameter_gradient(theta,x,y)
        normalized=-np.linalg.solve(jac.T@jac/len(y)+(.1/256)*np.eye(144),grad)*(4/256)
        checks.append(dict(state=index,test='normalized_metric_equivalence',k=1,error=float(np.max(abs(update-normalized)))))
        for j in range(8):
            d=rng.normal(size=144);d/=np.linalg.norm(d);eps=1e-5
            fd=(model.task.loss(theta+eps*d,x,y)-model.task.loss(theta-eps*d,x,y))/(2*eps)
            checks.append(dict(state=index,test='independent_loss_directional',k=j,error=float(abs(fd-grad@d))))
      theta=states[1];a=study.adapter();table=study.selector.fit_grid(a,theta)
      cx,cy=model.labeled_inputs(2099902,256);ties=np.random.default_rng(2099903).random(256)
      fitted={};masks={};calls={}
      for pipeline in ('raw','full','interp'):
        h,mask,ledger=study.deploy(theta,cx,ties,table,pipeline,1.);fit=model.temperature(h,cy)
        scaled,mask2,ledger2=study.deploy(theta,cx,ties,table,pipeline,fit['beta'])
        assert np.array_equal(mask,mask2) and np.array_equal(h>=0,scaled>=0)
        assert np.allclose(scaled,fit['beta']*h,rtol=0,atol=0) and ledger==ledger2
        assert fit['calibrated_nll']<=fit['raw_nll']+1e-12
        fitted[pipeline]=fit;masks[pipeline]=int(mask.sum());calls[pipeline]=ledger
      assert masks==dict(raw=0,full=256,interp=64)
    failed=[r for r in checks if r['error']>1e-8]
    result=dict(utc=datetime.now(timezone.utc).isoformat(),status='PASS' if not failed else 'FAIL',fresh_streams_accessed=False,
      development_source=study.sha(HERE/'reference/DEVELOPMENT_STATE.npz'),checks=checks,failures=failed,
      pipeline_fit_tests=fitted,pipeline_counts=calls,pipeline_selected=masks,
      normalized_extra_training=dict(n=2304,rate=36,damping=.9,normalized_rate=4/256,normalized_damping=.1/256),
      code_hashes={p:study.sha(HERE/p) for p in ('model.py','study.py','preflight.py')})
    (HERE/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],maximum_error=max(r['error'] for r in checks),tests=len(checks),pipeline_counts=calls),indent=2));assert not failed
if __name__=='__main__':main()
