"""A static distinction between input-map folding and parameter pullback rank.

The witness is an exact one-parameter restriction of the original network.
It makes no optimizer or generalization claim.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from pullback_feature_learning import core


def embed(a):
    p=np.zeros(80)
    p[0]=1.
    p[16]=2*a
    return p


def run(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    x=np.column_stack((np.linspace(-2,2,81),np.zeros(81)))
    rows=[]
    for a in (-.5,-1.,-2.):
        theta=embed(a)
        expected=x.copy()
        expected[:,0]+=a*np.tanh(x[:,0])
        actual=core.representation(theta,x)
        # a is half the original B[0,0] parameter; chain rule contributes 2.
        j=2*core.representation_jacobian(theta,x)[:,16]
        exact=np.zeros(162)
        exact[::2]=np.tanh(x[:,0])
        metric=float(j@j+.1)
        h=1e-5
        plus=x.copy();minus=x.copy()
        plus[:,0]+=h;minus[:,0]-=h
        dx=(core.representation(theta,plus)[:,0]-core.representation(theta,minus)[:,0])/(2*h)
        derivative=1+a*(1-np.tanh(x[:,0])**2)
        row=dict(coefficient=a,metric=metric,
                 minimum_input_determinant=float(derivative.min()),
                 forward_error=float(np.max(np.abs(actual-expected))),
                 parameter_jacobian_error=float(np.max(np.abs(j-exact))),
                 input_derivative_error=float(np.max(np.abs(dx-derivative))))
        assert row['forward_error']<1e-14
        assert row['parameter_jacobian_error']<1e-14
        assert row['input_derivative_error']<1e-9
        rows.append(row)
    assert max(r['metric'] for r in rows)==min(r['metric'] for r in rows)
    critical=float(np.arccosh(np.sqrt(2.)))
    determinant_at_fold=1-2*(1-np.tanh(critical)**2)
    assert abs(determinant_at_fold)<1e-14
    result=dict(status='passed',role='static_algebraic_screen_not_innovation',
        representation='(x1+a*tanh(x1),x2)',active_original_parameter=16,
        restriction='79 original coordinates fixed; a=B1[0,0]/2',
        input_count=81,damping_sum_metric=.1,rows=rows,
        critical_input_coordinates_at_a_minus2=[-critical,critical],
        fold_determinant_error=abs(determinant_at_fold),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_sha256=hashlib.sha256((REPO/'src/pullback_feature_learning/core.py').read_bytes()).hexdigest())
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.output)
