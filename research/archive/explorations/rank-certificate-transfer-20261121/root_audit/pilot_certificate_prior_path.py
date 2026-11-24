"""Development-only check on three previously completed rings states."""
from pathlib import Path
import json, sys
import numpy as np
from threadpoolctl import threadpool_limits
import certificate as c

def run():
    prior=c.ROOT.parent/'local-balance-20261121';qt=c.kc.q_table();rows=[]
    for cohort,seed in [('rings',1),('rings',8),('rings',16),('transfer',1)]:
        adapter=c.adapters.get_adapter(cohort);task='rings' if cohort=='rings' else 'waveform'
        path=(prior/'validation/policies'/f'{seed}_new.npz') if cohort=='rings' else (c.ROOT/'transfer/pilot_policy'/f'{seed}_pb.npz')
        table=dict(np.load(path))
        bp=c.ROOT/'certificate/pilot'/f'{task}{seed}_v2_sub8_na256_inner64.npz'
        if not bp.exists():continue
        bound=dict(np.load(bp));dist=c.kc.distribution(table,task,131072)
        x=adapter.sample_inputs(9911000+seed,65536);xp=adapter.partner(x);r=adapter.quotient(x)
        h=adapter.predict(table['theta'],x);hp=adapter.predict(table['theta'],xp)
        ql,qh=c.marginal_bounds(x,table,dist,qt,adapter)
        qpl,qph=c.marginal_bounds(xp,table,dist,qt,adapter)
        wl,wh=c.centered_bounds(h,hp,ql,qh,qpl,qph)
        lo,hi=c.local.pointwise_bound(bound,r);B=np.maximum(abs(lo),abs(hi));Z=wl-B/2
        _,C=adapter.rank.network_constants(table['theta']);D=(3 if cohort=='rings' else 1)+C;Bmax=float(bound['uniform_bound'])
        n=len(Z);term=np.log(2/(.05/32));span=3*D/4+Bmax/2
        stat=np.sqrt(2*np.var(Z,ddof=1)*term/n)+7*span*term/(3*(n-1))
        assert np.all(wl<=wh+1e-12)
        assert Z.min()>=-D/4-Bmax/2-1e-12 and Z.max()<=D/2+1e-12
        rows.append(dict(cohort=cohort,seed=seed,n=n,W_lower=float(wl.mean()),q_uncertainty=float((wh-wl).mean()),
            Bmean=float(B.mean()),meanZ=float(Z.mean()),stat=float(stat),gain_lower=float(Z.mean()-stat)))
    (c.HERE/'pilot_checks_v2.json').write_text(json.dumps(dict(development_only=True,rows=rows),indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):run()
