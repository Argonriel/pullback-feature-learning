"""Retain every attempted deterministic curvature-counterexample construction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import curvature as c


def run(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows=[]
    with threadpool_limits(1):
        for bias in (.25,.5,1.,2.):
            theta0=np.array([0.,.5])
            r0=c.predict(theta0,bias=bias)
            aa,_=c.path(theta0,3.,.004,True,bias)
            ff,_=c.path(theta0,3.,.004,False,bias)
            for dm in (-.1,-.2):
                for dp in (.25,1.,3.):
                    for weight in (.1,.25,.5,.75,.9):
                        row=dict(bias=bias,theta0=theta0.tolist(),shifts=[dm,dp],weights=[weight,1-weight])
                        if bias+dm<=0:
                            row.update(status='rejected_invalid_positive_argument',curve=[])
                        else:
                            curve=[]
                            for (r,a),(_,f) in zip(aa[1:],ff[1:]):
                                comp=[float(np.logaddexp(0.,-c.predict(a,s,bias))-
                                            np.logaddexp(0.,-c.predict(f,s,bias))) for s in (dm,dp)]
                                curve.append(dict(training_logit=r,per_input_risk_difference=comp,
                                                  risk_difference=float(np.array([weight,1-weight])@comp)))
                            early=min(curve,key=lambda z:abs(z['training_logit']-(r0+.1)))
                            late=curve[-1]
                            passed=early['risk_difference']*late['risk_difference']<0 and min(abs(early['risk_difference']),abs(late['risk_difference']))>1e-7
                            row.update(status='sign_reversal_candidate' if passed else 'rejected_no_bounded_sign_reversal',
                                       early=early,late=late,curve=curve)
                        rows.append(row)
    result=dict(role='Full deterministic construction search, not inferential experiment',
        step=.004,grid_count=len(rows),candidate_count=sum(r['status']=='sign_reversal_candidate' for r in rows),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        curvature_source_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),rows=rows)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(grid_count=result['grid_count'],candidate_count=result['candidate_count'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    run(p.parse_args().output)
