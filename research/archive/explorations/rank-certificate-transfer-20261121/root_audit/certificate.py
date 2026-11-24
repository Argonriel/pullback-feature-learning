"""Independent finite-sample certificates for the unchanged exact-quota rule."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'certificate'))
sys.path.insert(0,str(ROOT/'validation'))
import known_cdf as kc
import local_rank as local
import adapters

CONFIG=dict(n=262144,models=list(range(1,17)),cohorts=['rings','transfer'],
            seed_base={'rings':1011000,'transfer':1051000},family_delta=.05,
            policies=['rank_balanced'],chunk=16384,
            known_events=32,empirical_events=64,
            range='[-D/4-Bmax/2,D/2]',antithetic=True,
            hidden_prevalence='unrestricted',empirical_cdf_is_independent=True)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def marginal_bounds(x,table,dist,qt,adapter,name='rank_balanced'):
    r=adapter.quotient(x);edges=table['radii']
    lo=np.clip(np.searchsorted(edges,r,side='right')-1,0,len(edges)-2)
    t=np.clip((r-edges[lo])/(edges[lo+1]-edges[lo]),0,1)
    a=np.interp(r,edges,table['gain_profile']);outside=(r<edges[0])|(r>edges[-1])
    lower=np.zeros(len(x));upper=np.zeros(len(x))
    for side,aw in ((0,1-t),(1,t)):
        anchor=lo+side;mix=table[name+'_mix'][anchor]
        for component,weight in ((0,mix),(1,1-mix)):
            k=table[name+'_frequency'][anchor,component]
            phase=table[name+'_phase'][anchor,component]
            k=np.where(outside,2,k);phase=np.where(outside,0.,phase)
            angle=adapter.rank.action_angle(x,k)
            score=a+float(table['width'])*((k*angle/(2*np.pi)+phase)%1.)
            qlo,qhi=kc.q_bracket(dist,score,qt)
            lower+=aw*weight*qlo;upper+=aw*weight*qhi
    return np.clip(lower,0,1),np.clip(upper,0,1)

def centered_bounds(h,hp,ql,qh,qpl,qph):
    m=(h+hp)/2
    # log(2cosh(z/2)) is softplus(z)-z/2. This avoids subtraction of d/2.
    center=np.logaddexp(m/2,-m/2)
    e=np.logaddexp(h/2,-h/2)-center
    f=np.logaddexp(hp/2,-hp/2)-center
    lower=(np.where(e>=0,ql,qh)*e+np.where(f>=0,qpl,qph)*f)/2
    upper=(np.where(e>=0,qh,ql)*e+np.where(f>=0,qph,qpl)*f)/2
    return lower,upper

def bound_paths(cohort,seed,mode):
    task='rings' if cohort=='rings' else 'waveform'
    folder=ROOT/'certificate'/('fresh' if mode=='known' else 'fresh_empirical')/task
    return folder/f'{seed}_local.npz',folder/f'{seed}_distribution.npz'

def source_paths():
    return [Path(__file__),HERE/'CERTIFICATE.md',ROOT/'certificate/known_cdf.py',
            ROOT/'certificate/local_rank.py',ROOT/'validation/adapters.py',
            ROOT/'transfer/task.py',ROOT/'transfer/rank_policy.py',
            ROOT/'reference/rank_policy.py',ROOT/'reference/local_jets.py',
            ROOT/'source/pullback_feature_learning/core.py']

def freeze():
    with (HERE/'certificate_freeze.json').open('x',encoding='utf8') as f:
        json.dump(dict(timestamp_utc=now(),config=CONFIG,
            hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths()},
            fresh_models_opened=False,verification_inputs_generated=False),f,indent=2)

def run(mode):
    frozen=json.loads((HERE/'certificate_freeze.json').read_text())
    assert frozen['config']==CONFIG
    for rel,digest in frozen['hashes'].items():assert sha(ROOT/rel)==digest,rel
    destination=HERE/('certificates' if mode=='known' else 'certificates_empirical')
    destination.mkdir(exist_ok=False);qt=kc.q_table();rows=[]
    for cohort in CONFIG['cohorts']:
        adapter=adapters.get_adapter(cohort)
        for seed in CONFIG['models']:
            start=time.perf_counter()
            policy_path=ROOT/'validation/policies'/f'{cohort}_{seed}_pb_rank.npz'
            local_path,dist_path=bound_paths(cohort,seed,mode)
            table=dict(np.load(policy_path));bound=dict(np.load(local_path));dist=dict(np.load(dist_path))
            n=CONFIG['n'];input_seed=CONFIG['seed_base'][cohort]+seed
            x=adapter.sample_inputs(input_seed,n);r=adapter.quotient(x)
            Wlo=np.empty(n);Whi=np.empty(n);rate_lo=np.empty(n);rate_hi=np.empty(n)
            for first in range(0,n,CONFIG['chunk']):
                last=min(n,first+CONFIG['chunk']);xx=x[first:last];xp=adapter.partner(xx)
                h=adapter.predict(table['theta'],xx);hp=adapter.predict(table['theta'],xp)
                ql,qh=marginal_bounds(xx,table,dist,qt,adapter)
                qpl,qph=marginal_bounds(xp,table,dist,qt,adapter)
                wl,wh=centered_bounds(h,hp,ql,qh,qpl,qph)
                inside=(r[first:last]>=table['radii'][0])&(r[first:last]<=table['radii'][-1])
                # Outside the calibrated interval the fallback has exactly even
                # inclusion, so W=q*JensenGap>=0 and b=0. Omitting it is safe.
                Wlo[first:last]=np.where(inside,wl,0.)
                Whi[first:last]=np.where(inside,wh,0.)
                rate_lo[first:last]=(ql+qpl)/2;rate_hi[first:last]=(qh+qph)/2
            blo,bhi=local.pointwise_bound(bound,r);B=np.maximum(abs(blo),abs(bhi))
            Bmax=float(bound['uniform_bound']);assert np.max(B)<=Bmax+1e-9
            _,C=adapter.network_constants(table['theta'])
            D=(float(table['radii'][-1]) if cohort=='rings' else 1.)+C
            lower=-D/4-Bmax/2;upper=D/2;span=upper-lower;Z=Wlo-B/2
            assert Z.min()>=lower-1e-10 and Z.max()<=upper+1e-10
            assert np.all(Wlo<=Whi+1e-12)
            events=CONFIG['known_events'] if mode=='known' else CONFIG['empirical_events']
            delta=CONFIG['family_delta']/events
            if mode=='empirical':assert float(dist['alpha'])<=delta
            term=np.log(2/delta);variance=float(np.var(Z,ddof=1))
            deduction=float(np.sqrt(2*variance*term/n)+7*span*term/(3*(n-1)))
            row=dict(cohort=cohort,seed=seed,mode=mode,n=n,input_seed=input_seed,delta=delta,
                centered_lower_mean=float(Wlo.mean()),centered_upper_mean=float(Whi.mean()),
                centered_cdf_uncertainty=float((Whi-Wlo).mean()),
                mean_bound_penalty=float(B.mean()/2),mean_Z=float(Z.mean()),sample_variance=variance,
                range_lower=lower,range_upper=upper,range_length=span,statistical_deduction=deduction,
                gain_lower=float(Z.mean()-deduction),positive=bool(Z.mean()>deduction),
                mean_rate_lower=float(rate_lo.mean()),mean_rate_upper=float(rate_hi.mean()),
                global_D=D,uniform_bound=Bmax,verification_forwards=2*n,
                seconds=time.perf_counter()-start,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                policy_sha256=sha(policy_path),local_sha256=sha(local_path),distribution_sha256=sha(dist_path))
            rows.append(row)
            np.savez_compressed(destination/f'{cohort}_{seed}_observations.npz',
                r=r,W_lower=Wlo,W_upper=Whi,B=B,Z=Z,rate_lower=rate_lo,rate_upper=rate_hi)
            with (destination/f'{cohort}_{seed}.json').open('x',encoding='utf8') as f:json.dump(row,f,indent=2)
            print(f'{mode} {cohort} {seed}: lower={row["gain_lower"]:.7g}',flush=True)
    with (destination/'per_model.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(timestamp_utc=now(),mode=mode,family_confidence=.95,
        positive_models=sum(r['positive'] for r in rows),models=len(rows),
        min_gain_lower=min(r['gain_lower'] for r in rows),
        mean_gain_lower=float(np.mean([r['gain_lower'] for r in rows])),
        verification_forwards=sum(r['verification_forwards'] for r in rows),
        by_cohort={c:dict(positive=sum(r['positive'] for r in rows if r['cohort']==c),
            mean_gain_lower=float(np.mean([r['gain_lower'] for r in rows if r['cohort']==c])),
            min_gain_lower=min(r['gain_lower'] for r in rows if r['cohort']==c)) for c in CONFIG['cohorts']},
        confidence_scope='32 EB events' if mode=='known' else '32 DKW plus32 EB events',
        real_arithmetic_enclosures=True,formal_float_proof=False,
        freeze_sha256=sha(HERE/'certificate_freeze.json'))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','known','empirical']);a=p.parse_args()
    with threadpool_limits(1):freeze() if a.action=='freeze' else run(a.action)
