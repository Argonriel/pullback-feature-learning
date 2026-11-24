"""All-prior-model development using an explicitly label-free selection criterion."""
from pathlib import Path
import sys,csv,json,hashlib,datetime,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import policy as p
TAUS=(.25,1.,4.);NTEST=16384

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(x):return hashlib.sha256(x).hexdigest()
def write_json(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def csvout(path,rows):
    with path.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def sample(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);phi=rng.uniform(0,2*np.pi,n);return r[:,None]*np.c_[np.cos(phi),np.sin(phi)]+rng.normal(0,.15,(n,2))

def run():
    out=ROOT/'development';out.mkdir(exist_ok=False);(out/'bases').mkdir();(out/'actions').mkdir();tmp=ROOT.parent.parent;records=[];bank={};sourcehash={}
    archive=tmp/'budgeted-rotation-final'/'results'/'models.npz';old=np.load(archive);sourcehash['121_bank']=sha(archive.read_bytes())
    for cohort in (121,611,721,821):
        for seed in range(1,17):
            key=f'{cohort}_{seed}'
            if cohort==121:theta=old[f'{seed}_pb'];path=archive
            else:
                path=(tmp/'predictive-symmetry-final'/'direction2'/'fresh_results'/f'{seed}_training_states.npz' if cohort==611 else tmp/'calibration-quota-20261121'/'idea2a'/'fresh'/'training'/f'{seed}_training_states.npz' if cohort==721 else tmp/'continuous-confirmation-20261121'/'confirmation'/'states'/f'original_{seed}.npz')
                theta=np.load(path)['pb'];sourcehash[key]=sha(path.read_bytes())
            bank[key]=theta
            records.append(dict(key=key,cohort=cohort,pair=seed,state_sha256=sha(theta.tobytes()),source_path=str(path),source_sha256=sha(path.read_bytes())))
    np.savez_compressed(out/'models.npz',**bank);csvout(out/'model_manifest.csv',records);write_json(out/'sources.json',sourcehash)
    scores=[];checks=[]
    for index,record in enumerate(records,1):
        key=record['key'];theta=bank[key];cal_seed=953000+index;test_seed=954000+index;ties=np.random.default_rng(956000+index).random(NTEST);actions=np.random.default_rng(955000+index).random(NTEST);x=sample(test_seed,NTEST)
        h=p.core.representation(theta,x)[:,0];partner=p.core.representation(theta,-x)[:,0];m=(h+partner)/2
        # Centered logistic improvement is input-only; no labels or posterior used.
        benefit=np.logaddexp(h/2,-h/2)-np.logaddexp(m/2,-m/2)
        base=p.prepare(theta,cal_seed);np.savez_compressed(out/'bases'/f'{key}.npz',**base)
        for tau in TAUS:
            table=p.calibrate(theta,cal_seed,tau,base);action_table={k:v for k,v in table.items() if k not in base};np.savez_compressed(out/'actions'/f'{key}_tau{tau:g}.npz',**action_table)
            for name in p.POLICIES:
                mask=p.select(x,h,table,name,ties,actions);gain=float(np.mean(mask*benefit));scores.append(dict(key=key,cohort=record['cohort'],pair=record['pair'],tau=tau,policy=name,n_test=NTEST,centered_gain=gain,cal_seed=cal_seed,test_seed=test_seed,mask_sha256=sha(np.packbits(mask).tobytes()),selected=int(mask.sum()),marginal_rate_min=float(table['rate_profile'].min()),marginal_rate_max=float(table['rate_profile'].max()),mean_calibrated_C=float(np.mean(np.interp(base['cdf_cal_radii'],base['radii'],table[name+'_objective']))),max_anchor_moment=float(abs(table[name+'_moment']).max()),width=float(table['width'])))
            # A direct non-FFT action identity and coefficient feasibility check.
            row=53;k=3;shift=117;phi=2*np.pi*(np.arange(p.NA)+.5)/p.NA;hh=table['h'][row];dd=(hh-np.roll(hh,p.NA//2))/2;AA=np.logaddexp(0,hh)-np.logaddexp(0,(hh+np.roll(hh,p.NA//2))/2);score=table['gain_profile'][row]+float(table['width'])*((k*phi/(2*np.pi)+shift/p.NA)%1);q=p.rank_probability(p.score_cdf(table,score));ix=(k-1)*p.NA+shift
            checks.append(dict(key=key,tau=tau,fft_C_error=abs(float(q@AA/p.NA-table['action_objectives'][row,ix])),fft_b_error=abs(float(q@dd/p.NA-table['action_moments'][row,ix])),max_balanced_anchor_moment=float(abs(table['rank_balanced_moment']).max()),max_even_anchor_moment=float(abs(table['rank_even_moment']).max())))
        print(f'old development {index}/64 {key}',flush=True)
    csvout(out/'scores.csv',scores);csvout(out/'checks.csv',checks)
    means={str(tau):float(np.mean([r['centered_gain'] for r in scores if r['tau']==tau and r['policy']=='rank_balanced'])) for tau in TAUS}
    chosen=max(TAUS,key=lambda tau:(means[str(tau)],-tau))
    write_json(ROOT/'FROZEN_CHOICE.json',dict(timestamp_utc=now(),tau=chosen,criterion='Highest mean held-out OLD input-only centered logistic gain of rank_balanced across64 prior PB models; no labels or posterior.',old_mean_centered_gain=means,old_model_count=64,old_input_count_per_model=NTEST,all_taus_retained=list(TAUS),development_scores_sha256=sha((out/'scores.csv').read_bytes()),new_states_accessed=False,new_outcomes_accessed=False))
    print(json.dumps(means),chosen,flush=True)
if __name__=='__main__':
    with threadpool_limits(1):run()
