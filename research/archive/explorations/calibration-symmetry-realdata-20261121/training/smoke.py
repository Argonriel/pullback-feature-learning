"""Pre-fresh adapter, shared-calibration, and physical deployment smoke."""
import json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import model,study
from adapters import get_adapter
HERE=Path(__file__).resolve().parent
def run():
    data=np.load(HERE/'pilot/1_pb_rate2_lam0.npz');theta=data['states'][int(data['selected_index'])]
    x=model.task.sample_inputs(2099001,256);ties=np.random.default_rng(2099002).random(256)
    a=get_adapter(1.);grid=study.policy.fit_grid(a,theta)
    xc=model.task.sample_inputs(2099003,32768);h=model.task.predict(theta,xc);ht=model.task.predict(theta,model.task.partner(xc))
    records=[]
    for beta in (.2,1.,3.):
        a=get_adapter(beta);scaled=study.scaled_grid(grid,beta)
        folded=study.policy.fit_folded(a,theta,xc,config=(16,16,16),prepared=(beta*h,beta*ht))
        tables={'grid':scaled,'folded':folded}
        for name in study.POLICIES:
            output,mask,calls=study.deploy(a,theta,x,ties,name,tables)
            direct=a.predict(theta,x);partner=a.predict(theta,a.partner(x))
            expected=np.where(mask,.5*(direct+partner),direct)
            assert np.allclose(output,expected,rtol=0,atol=1e-14)
            if name in ('folded','symmetric_jensen'):
                _,flipped,_=study.deploy(a,theta,a.partner(x),ties,name,tables);assert np.array_equal(mask,flipped)
            records.append(dict(beta=beta,policy=name,calls=calls,selected=int(mask.sum()),direct_error=float(np.max(abs(output-expected)))))
        L,C=model.task.network_constants(theta);assert a.difference_bound(theta)==beta*min(1.4+C,1.4*L)
        assert np.max(abs(.5*(a.predict(theta,x)-a.predict(theta,a.partner(x)))))<=a.difference_bound(theta)
    out=dict(status='PASS',fresh_read=False,development_seeds=[2099001,2099002,2099003],cases=records,
      source_hashes={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ('study.py','adapters.py','model.py')})
    (HERE/'SMOKE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
