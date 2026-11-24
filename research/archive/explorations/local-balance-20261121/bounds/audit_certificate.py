"""Independent recomputation of the root's frozen finite-sample certificates."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
from threadpoolctl import threadpool_limits
import local
ROOT=Path(__file__).resolve().parent;STUDY=ROOT.parent
sys.path.insert(0,str(STUDY/'reference'));import continuous_policy as field_policy

def main():
    cert=STUDY/'root_audit/certificates';summary=json.loads((cert/'summary.json').read_text());fr=json.loads((STUDY/'root_audit/certificate_freeze.json').read_text())
    for rel,digest in fr['hashes'].items():assert hashlib.sha256((STUDY/rel).read_bytes()).hexdigest()==digest
    with (cert/'per_seed.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    result=[];maximum_B=0.;maximum_W=0.;maximum_lower=0.;forwards=0
    for row in rows:
        seed=int(row['seed']);data=dict(np.load(cert/f'{seed}_observations.npz'));bound=dict(np.load(ROOT/f'fresh/{seed}_local.npz'));field=dict(np.load(STUDY/f'validation/policies/{seed}_continuous.npz'))
        lo,hi=local.pointwise_bound(bound,data['r']);B=np.maximum(abs(lo),abs(hi));err=float(abs(B-data['B']).max());maximum_B=max(maximum_B,err);assert err<1e-12
        Z=data['W']-B/2;assert np.max(abs(Z-data['Z']))<1e-12;n=len(Z);D=3+float(field['C_h']);span=D/4+float(bound['uniform_bound'])/2;delta=.05/16
        deduction=np.sqrt(2*np.var(Z,ddof=1)*np.log(2/delta)/n)+7*span*np.log(2/delta)/(3*(n-1));lower=float(Z.mean()-deduction);maximum_lower=max(maximum_lower,abs(lower-float(row['gain_lower'])));assert abs(lower-float(row['gain_lower']))<1e-13
        W_error=None
        if seed in [1,8,16]:
            rng=np.random.default_rng(941000+seed);r=rng.choice([.5,1.2],n);phi=rng.uniform(0,2*np.pi,n);x=r[:,None]*np.c_[np.cos(phi),np.sin(phi)]+rng.normal(0,.15,(n,2));input_hash=hashlib.sha256(x.tobytes()).hexdigest();assert input_hash==summary['inputs'][seed-1]['x_sha256']
            rr=np.linalg.norm(x,axis=1);a=local.core.representation(field['theta'],x)[:,0];b=local.core.representation(field['theta'],-x)[:,0];m=(a+b)/2
            # Center the softplus before differencing, an independent algebraic form.
            softcenter=lambda z:np.logaddexp(z/2,-z/2)
            U=softcenter(a)-softcenter(m);V=softcenter(b)-softcenter(m);q=.25+field_policy.field_value(field,x,'balanced')/8;qp=.25+field_policy.field_value(field,-x,'balanced')/8
            W=.5*(q*U+qp*V)*(rr<=3);W_error=float(abs(W-data['W']).max());maximum_W=max(maximum_W,W_error);assert W_error<1e-12;forwards+=2*n
        result.append(dict(seed=seed,gain_lower=lower,positive=lower>0,maximum_B_error=err,full_antithetic_W_max_error=W_error))
    audit=dict(status='PASS',models_checked=16,positive_models=sum(r['positive'] for r in result),maximum_bound_value_error=maximum_B,maximum_gain_lower_error=maximum_lower,full_antithetic_models=[1,8,16],maximum_antithetic_W_error=maximum_W,additional_audit_example_forwards=forwards,rows=result)
    (ROOT/'certificate_independent_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='rows'},indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
