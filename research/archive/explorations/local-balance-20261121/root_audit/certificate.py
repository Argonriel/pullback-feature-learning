"""Independent antithetic finite-sample certificate; see CERTIFICATE.md."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, sys
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,str(ROOT/'reference'))
from pullback_feature_learning import core
import continuous_policy as policy

CONFIG=dict(n=262144,seed_base=941000,models=list(range(1,17)),family_delta=.05,
            bounds='local pointwise signed interval, then maximum absolute endpoint',
            range='D/4+Bmax/2',antithetic=True,hidden_prevalence='unrestricted')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def freeze():
    paths=[Path(__file__),HERE/'CERTIFICATE.md',ROOT/'source/pullback_feature_learning/core.py',ROOT/'reference/continuous_policy.py']
    f=HERE/'certificate_freeze.json'
    with f.open('x',encoding='utf8') as out:
        json.dump(dict(timestamp_utc=now(),config=CONFIG,hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},fresh_models_opened=False),out,indent=2)

def sample(seed,n):
    rng=np.random.default_rng(seed)
    latent=rng.choice([.5,1.2],n)
    phi=rng.uniform(0,2*np.pi,n)
    return latent[:,None]*np.column_stack([np.cos(phi),np.sin(phi)])+rng.normal(0,.15,(n,2))

def bound_values(table,r):
    edges=table['r']; j=np.searchsorted(edges,r,side='right')-1
    j=np.clip(j,0,len(edges)-2); width=edges[j+1]-edges[j]
    fraction=np.clip((r-edges[j])/width,0,1)
    linear=(1-fraction)*table['moment'][j]+fraction*table['moment'][j+1]
    correction=width**2*fraction*(1-fraction)/2
    lo=linear-table['angular'][j]-correction*table['brr_hi'][j]
    hi=linear+table['angular'][j]-correction*table['brr_lo'][j]
    B=np.maximum(abs(lo),abs(hi))
    B[(r<edges[0])|(r>edges[-1])|table['exact_even_band'][j]]=0
    assert np.max(B)<=float(table['uniform_bound'])+1e-11
    return B

def run():
    frozen=json.loads((HERE/'certificate_freeze.json').read_text())
    assert frozen['config']==CONFIG
    for rel,digest in frozen['hashes'].items(): assert sha(ROOT/rel)==digest,rel
    dest=HERE/'certificates';dest.mkdir(exist_ok=False)
    rows=[];input_manifest=[]
    with threadpool_limits(1):
        for seed in CONFIG['models']:
            tp=ROOT/'validation/policies'/f'{seed}_continuous.npz'
            bp=ROOT/'bounds/fresh'/f'{seed}_local.npz'
            field=dict(np.load(tp));local=dict(np.load(bp))
            x=sample(CONFIG['seed_base']+seed,CONFIG['n']);r=np.linalg.norm(x,axis=1)
            a=core.representation(field['theta'],x)[:,0]
            partner=core.representation(field['theta'],-x)[:,0]
            m=(a+partner)/2;d=(a-partner)/2
            A=np.logaddexp(0,a)-np.logaddexp(0,m)
            Aprime=np.logaddexp(0,partner)-np.logaddexp(0,m)
            q=.25+policy.field_value(field,x,'balanced')/8
            qp=.25+policy.field_value(field,-x,'balanced')/8
            W=(q*(A-d/2)+qp*(Aprime+d/2))/2
            W*=r<=3
            B=bound_values(local,r)
            Z=W-B/2
            D=3+float(field['C_h']);Bmax=float(local['uniform_bound'])
            low=-D/16-Bmax/2;high=3*D/16;span=high-low
            assert q.min()>=.125-1e-12 and q.max()<=.375+1e-12
            assert Z.min()>=low-1e-12 and Z.max()<=high+1e-12
            n=len(Z);delta=CONFIG['family_delta']/len(CONFIG['models']);logterm=np.log(2/delta)
            var=float(np.var(Z,ddof=1));stat=np.sqrt(2*var*logterm/n)+7*span*logterm/(3*(n-1))
            row=dict(seed=seed,n=n,coordinate_seed=CONFIG['seed_base']+seed,delta=delta,
                     centered_gain_mean=float(W.mean()),local_penalty_mean=float(B.mean()/2),
                     Z_mean=float(Z.mean()),Z_sample_variance=var,range_lower=low,range_upper=high,
                     range_length=span,statistical_deduction=float(stat),gain_lower=float(Z.mean()-stat),
                     positive=bool(Z.mean()-stat>0),uniform_local_bound=Bmax,
                     verification_forwards=2*n,field_sha256=sha(tp),local_sha256=sha(bp))
            rows.append(row)
            np.savez_compressed(dest/f'{seed}_observations.npz',r=r,W=W,B=B,Z=Z)
            input_manifest.append(dict(seed=seed,x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),n=n))
            print(f'certificate {seed}: lower={row["gain_lower"]:.8g}',flush=True)
    with (dest/'per_seed.csv').open('x',newline='',encoding='utf8') as out:
        writer=csv.DictWriter(out,list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(timestamp_utc=now(),family_confidence=.95,positive_models=sum(r['positive'] for r in rows),
                 min_gain_lower=min(r['gain_lower'] for r in rows),mean_gain_lower=float(np.mean([r['gain_lower'] for r in rows])),
                 verification_forwards=sum(r['verification_forwards'] for r in rows),inputs=input_manifest,
                 frozen_config_sha256=sha(HERE/'certificate_freeze.json'),real_arithmetic_only=True)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);arg=parser.parse_args()
    {'freeze':freeze,'run':run}[arg.action]()
