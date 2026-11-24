"""Independent checks of the rank polynomial and centered-gain identities."""
from pathlib import Path
import importlib.util, math, json, sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('rank_policy_audit',ROOT/'selection/policy.py')
rank=importlib.util.module_from_spec(spec);spec.loader.exec_module(rank)

def main():
    probabilities=np.r_[np.linspace(0,1,101),.749,.751,.999999,1e-6]
    reference=[]
    for p in probabilities:
        reference.append(math.fsum(math.comb(255,j)*p**j*(1-p)**(255-j) for j in range(192,256)))
    q=rank.rank_probability(probabilities)
    polynomial_error=float(np.max(abs(q-np.array(reference))))
    assert polynomial_error<3e-12
    nodes,weights=np.polynomial.legendre.leggauss(128)
    integral=float(weights@rank.rank_probability((nodes+1)/2)/2)
    assert abs(integral-.25)<1e-12
    rng=np.random.default_rng(949001)
    a=rng.uniform(-40,40,100000);b=rng.uniform(-40,40,100000)
    q1=rng.uniform(.125,.375,len(a));q2=rng.uniform(.125,.375,len(a))
    p=rng.uniform(0,1,len(a));m=(a+b)/2;d=(a-b)/2
    A=np.logaddexp(0,a)-np.logaddexp(0,m)
    B=np.logaddexp(0,b)-np.logaddexp(0,m)
    W=(q1*(A-d/2)+q2*(B+d/2))/2
    moment=(q1-q2)*d/2
    gain=(q1*(A-p*d)+q2*(B+p*d))/2
    identity_error=float(np.max(abs(gain-(W-(p-.5)*moment))))
    assert identity_error<1e-13
    extreme_margin=0.
    for z1 in (.125,.375):
        for z2 in (.125,.375):
            value=(z1*(A-d/2)+z2*(B+d/2))/2
            scale=abs(d)
            assert np.all(value>=-scale/16-1e-13)
            assert np.all(value<=3*scale/16+1e-13)
            extreme_margin=max(extreme_margin,float(np.max(value-3*scale/16)),float(np.max(-scale/16-value)))
    result=dict(rank_polynomial_max_error=polynomial_error,rank_uniform_mean=integral,
                centered_gain_identity_max_error=identity_error,
                antithetic_range_violations=0,random_pairs=len(a),
                all_four_extreme_q_pairs_checked=True,
                rank_cdf_lipschitz=float(rank.Q_LIPSCHITZ))
    (HERE/'identities.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':main()
