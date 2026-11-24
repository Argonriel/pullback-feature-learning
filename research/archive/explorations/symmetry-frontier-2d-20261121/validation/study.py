"""Frozen fresh experiment: training, calibration, measured deployment, analysis."""
from pathlib import Path
import argparse, csv, datetime, gzip, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from threadpoolctl import threadpool_limits
from adapters import get_adapter, nll, worlds
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'method'))
import baseline as policy
COHORTS=('rings','waveform','two_invariants')
BASES=dict(rings=1160000,waveform=1110000,two_invariants=1140000)
N=65536;BATCH=256;QUOTA=64
NAMES=('raw','full','confidence','folded_legacy','folded','symmetric_jensen','grid_nearest','oracle_jensen')
CONFIG=dict(cohorts=COHORTS,bases=BASES,replications=16,n=N,batch=BATCH,quota=QUOTA,
    policies=NAMES,bootstrap_draws=20000,bootstrap_alpha=.05/3,bootstrap_seeds=[1171001,1171002,1171003])
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def jo(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def co(path,rows):
    with Path(path).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def freeze():
    assert not any((HERE/'states').glob('*.npz'))
    paths=[ROOT/'STUDY_SCOPE.md',HERE/'study.py',HERE/'adapters.py',HERE/'PROTOCOL.md',
      ROOT/'method/METHOD_FREEZE.json',ROOT/'method/HISTOGRAM_CHOICE.json',ROOT/'method/baseline.py',
      ROOT/'task2d/TASK_FREEZE.json',ROOT/'task2d/TRAINING_CHOICE.json',ROOT/'task2d/task.py',
      ROOT/'task2d/core4d.py',ROOT/'certification/FREEZE.json',ROOT/'certification/ADDENDUM_FREEZE.json']
    for folder in ('source','legacy','reference'):paths+=list((ROOT/folder).rglob('*.py'))
    jo(HERE/'FREEZE.json',dict(timestamp_utc=now(),config=CONFIG,no_fresh_states=True,
      hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
def verify():
    frozen=json.loads((HERE/'FREEZE.json').read_text())
    assert frozen['config']==json.loads(json.dumps(CONFIG))
    for p,h in frozen['hashes'].items():assert sha(ROOT/p)==h,p
    approval=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
    assert approval['validation_freeze_sha256']==sha(HERE/'FREEZE.json') and approval['fresh_execution_authorized']
def train_one(job):
    cohort,i=job;a=get_adapter(cohort);base=BASES[cohort];start=time.perf_counter()
    with threadpool_limits(limits=1):
        x,y=a.sample_train(base+1000+i);initial=a.initialize(base+2000+i);theta=initial.copy();trace=[]
        assert x.shape==(256,a.dimension) and np.sum(y==1)==128 and np.sum(y==-1)==128
        cfg=a.training
        try:
            for step in range(cfg['steps']):
                theta+=cfg['rate']*a.direction(theta,x,y,method=cfg['method'],damping=cfg['damping'])
                if not np.isfinite(theta).all():raise FloatingPointError((cohort,i,step))
                if (step+1)%100==0:trace.append((step+1,a.loss(theta,x,y)))
                if step==99:checkpoint=theta.copy()
        except Exception as exc:
            failure=HERE/'states'/f'{cohort}_{i}_failure.npz'
            np.savez_compressed(failure,x=x,y=y,initial=initial,theta=theta,loss_trace=np.array(trace))
            jo(failure.with_suffix('.json'),dict(timestamp_utc=now(),cohort=cohort,pair=i,
               step=step+1,exception=repr(exc),state_sha256=sha(failure),seed_replacement=False))
            raise
        path=HERE/'states'/f'{cohort}_{i}.npz'
        np.savez_compressed(path,x=x,y=y,initial=initial,pb100=checkpoint,pb=theta,loss_trace=np.array(trace))
    return dict(cohort=cohort,pair=i,train_seed=base+1000+i,init_seed=base+2000+i,
      steps=cfg['steps'],rate=cfg['rate'],damping=cfg['damping'],parameters=len(theta),
      final_loss=trace[-1][1],seconds=time.perf_counter()-start,sha256=sha(path))
def train():
    verify();(HERE/'states').mkdir(exist_ok=False)
    jo(HERE/'training_start.json',dict(timestamp_utc=now(),freeze_sha256=sha(HERE/'FREEZE.json')))
    jobs=[(c,i) for c in COHORTS for i in range(1,17)];rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for r in pool.map(train_one,jobs):rows.append(r);print('Trained',r['cohort'],r['pair'],flush=True)
    co(HERE/'training.csv',rows)
    jo(HERE/'training_commit.json',dict(timestamp_utc=now(),n_models=len(rows),
       manifest={f"states/{r['cohort']}_{r['pair']}.npz":r['sha256'] for r in rows}))
def calibrate():
    verify();assert (HERE/'training_commit.json').exists();(HERE/'policies').mkdir(exist_ok=False)
    choice=json.loads((ROOT/'method/HISTOGRAM_CHOICE.json').read_text())['choices'];rows=[];manifest={}
    jo(HERE/'calibration_start.json',dict(timestamp_utc=now(),training_commit_sha256=sha(HERE/'training_commit.json')))
    for c in COHORTS:
      for i in range(1,17):
        a=get_adapter(c);theta=np.load(HERE/'states'/f'{c}_{i}.npz')['pb'];original=a.predict;calls=[]
        def counted(th,x):calls.append(len(x));return original(th,x)
        a.predict=counted;start=time.perf_counter();grid=policy.fit_grid(a,theta)
        assert sum(calls)==65536
        grid_seconds=time.perf_counter()-start;grid_calls=calls.copy();calls.clear()
        x=a.sample_inputs(BASES[c]+3000+i,32768);start=time.perf_counter()
        prepared=(a.predict(theta,x),a.predict(theta,a.partner(x)))
        folded=policy.fit_folded(a,theta,x,config=choice[c]['config'],prepared=prepared)
        legacy=policy.fit_folded(a,theta,x,config=policy.histogram_configs(a.quotient_dim)[0],prepared=prepared)
        assert calls==[32768,32768]
        for name,table in [('grid',grid),('folded',folded),('folded_legacy',legacy)]:
            path=HERE/'policies'/f'{c}_{i}_{name}.npz';np.savez_compressed(path,**table);manifest[path.relative_to(HERE).as_posix()]=sha(path)
        rows.append(dict(cohort=c,pair=i,grid_forwards=sum(grid_calls),folded_forwards=sum(calls),
          shared_legacy_extra_forwards=0,grid_invocations=len(grid_calls),folded_invocations=len(calls),
          grid_seconds=grid_seconds,folded_seconds=time.perf_counter()-start,
          config='x'.join(map(str,choice[c]['config'])),input_sha256=ash(x)))
        print('Calibrated',c,i,flush=True)
    co(HERE/'calibration.csv',rows)
    jo(HERE/'policy_commit.json',dict(timestamp_utc=now(),manifest=manifest,
      training_commit_sha256=sha(HERE/'training_commit.json')))
def deploy(a,theta,x,ties,name,tables):
    calls=[]
    def predict(z):calls.append(len(z));return a.predict(theta,z)
    if name in ('symmetric_jensen','grid_nearest','folded','folded_legacy'):
        table=tables['grid'] if name in ('symmetric_jensen','grid_nearest') else tables[name]
        h,mask=policy.deploy(a,theta,x,table,ties,'folded' if name=='folded_legacy' else name,predict)
    else:
        h=predict(x)
        if name=='raw':mask=np.zeros(len(x),bool)
        elif name=='full':mask=np.ones(len(x),bool);h=.5*(h+predict(a.partner(x)))
        elif name=='confidence':
            mask=policy.allocate(-abs(h),ties);h[mask]=.5*(h[mask]+predict(a.partner(x[mask])))
        elif name=='oracle_jensen':
            second=predict(a.partner(x));mask=policy.allocate(policy.jensen(h,second),ties);h[mask]=.5*(h[mask]+second[mask])
        else:raise ValueError(name)
    expected=[256] if name=='raw' else ([256,256] if name in ('full','oracle_jensen') else [256,64])
    assert calls==expected,(name,calls)
    assert int(mask.sum())== (0 if name=='raw' else 256 if name=='full' else 64)
    return h,mask,calls
def score():
    verify();assert (HERE/'policy_commit.json').exists();(HERE/'predictions').mkdir(exist_ok=False)
    jo(HERE/'score_start.json',dict(timestamp_utc=now(),policy_commit_sha256=sha(HERE/'policy_commit.json')))
    rows=[];samples=[];manifests={};total=0;records=0
    with gzip.open(HERE/'batch_ledger.csv.gz','xt',encoding='utf8',newline='') as f:
      w=csv.DictWriter(f,['cohort','pair','policy','batch','inputs','selected','calls','forward_examples']);w.writeheader()
      for c in COHORTS:
       for i in range(1,17):
        a=get_adapter(c);theta=np.load(HERE/'states'/f'{c}_{i}.npz')['pb'];base=BASES[c]
        tables={k:dict(np.load(HERE/'policies'/f'{c}_{i}_{k}.npz')) for k in ('grid','folded','folded_legacy')}
        x=a.sample_inputs(base+4000+i,N);ties=np.random.default_rng(base+5000+i).random(N)
        p=a.posterior(x);label_uniform=np.random.default_rng(base+6000+i).random(N);y=np.where(label_uniform<p,1.,-1.)
        out=np.empty((len(NAMES),N));masks=np.zeros((len(NAMES),N),bool);cost={};alt=worlds(a,x)
        for j,name in enumerate(NAMES):
          start=time.perf_counter();forward=0
          for begin in range(0,N,BATCH):
            h,mask,calls=deploy(a,theta,x[begin:begin+BATCH],ties[begin:begin+BATCH],name,tables)
            out[j,begin:begin+BATCH]=h;masks[j,begin:begin+BATCH]=mask;forward+=sum(calls)
            w.writerow(dict(cohort=c,pair=i,policy=name,batch=begin//BATCH,inputs=256,selected=int(mask.sum()),
               calls='|'.join(map(str,calls)),forward_examples=sum(calls)));records+=1
          total+=forward;cost[name]=(forward,time.perf_counter()-start)
        raw=out[0];partner=2*out[1]-raw;gaps=policy.jensen(raw,partner)
        oracle=(masks[-1]*gaps).reshape(-1,256).mean(1)
        for j,name in enumerate(NAMES):
          h=out[j];gain=(masks[j]*gaps).reshape(-1,256).mean(1)
          if name not in ('raw','full'):assert np.all(gain<=oracle+1e-12)
          r=dict(cohort=c,pair=i,policy=name,n=N,expected_loss=float(nll(h,p).mean()),
            expected_error=float(np.where(h>=0,1-p,p).mean()),sampled_loss=float(np.logaddexp(0,-y*h).mean()),
            sampled_error=float(np.mean(np.where(h>=0,1.,-1.)!=y)),selected_jensen=float(gain.mean()),
            oracle_fraction=float(gain.mean()/oracle.mean()),forward_examples=cost[name][0],seconds=cost[name][1])
          for k,p_alt in alt.items():r[k+'_loss']=float(nll(h,p_alt).mean())
          rows.append(r)
        path=HERE/'predictions'/f'{c}_{i}.npz'
        np.savez_compressed(path,x=x,p=p,y=y,ties=ties,names=np.array(NAMES),outputs=out,masks=np.packbits(masks,axis=1))
        manifests[path.relative_to(HERE).as_posix()]=sha(path)
        samples.append(dict(cohort=c,pair=i,input_seed=base+4000+i,tie_seed=base+5000+i,label_seed=base+6000+i,
          input_sha256=ash(x),tie_sha256=ash(ties),label_sha256=ash(y),posterior_sha256=ash(p)))
        print('Scored',c,i,flush=True)
    assert len(rows)==384 and records==98304 and total==35389440,(len(rows),records,total)
    co(HERE/'per_model.csv',rows);co(HERE/'samples.csv',samples)
    jo(HERE/'score_complete.json',dict(timestamp_utc=now(),n_models=48,n_policy_rows=len(rows),
      n_batch_records=records,actual_forward_examples=total,manifest=manifests))
def analyze():
    verify()
    with (HERE/'per_model.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    summaries=[];contrasts=[]
    numeric=['expected_loss','expected_error','sampled_loss','sampled_error','selected_jensen','oracle_fraction','seconds',
      'half_loss','decreasing_loss','angular_loss']
    for ci,c in enumerate(COHORTS):
      for name in NAMES:
        group=[r for r in rows if r['cohort']==c and r['policy']==name]
        summaries.append(dict(cohort=c,policy=name,n_models=len(group),**{k:float(np.mean([float(r[k]) for r in group])) for k in numeric}))
      def vals(name,key):return np.array([float(next(r for r in rows if r['cohort']==c and int(r['pair'])==i and r['policy']==name)[key]) for i in range(1,17)])
      for other in ('folded','folded_legacy','grid_nearest','confidence','raw','oracle_jensen'):
        for key in ('expected_loss','expected_error','selected_jensen'):
          d=vals('symmetric_jensen',key)-vals(other,key)
          boot=np.random.default_rng(1171001+ci).choice(d,(20000,16)).mean(1)
          alpha=.05/3 if other=='folded' and key=='expected_loss' else .05
          lo,hi=np.quantile(boot,[alpha/2,1-alpha/2])
          contrasts.append(dict(cohort=c,comparison='symmetric_jensen_minus_'+other,metric=key,
             mean=float(d.mean()),lower=float(lo),upper=float(hi),wins=int(np.sum(d>0 if key=='selected_jensen' else d<0)),
             n=16,confidence=1-alpha,primary=(other=='folded' and key=='expected_loss')))
    co(HERE/'summary.csv',summaries);co(HERE/'contrasts.csv',contrasts)
    jo(HERE/'analysis_complete.json',dict(timestamp_utc=now(),summary_sha256=sha(HERE/'summary.csv'),contrasts_sha256=sha(HERE/'contrasts.csv')))
def smoke():
    rows=[]
    for c in COHORTS:
        a=get_adapter(c);theta=a.initialize(1202001);x=a.sample_inputs(1202002,256);ties=np.random.default_rng(1202003).random(256)
        grid=policy.fit_grid(a,theta);fc=policy.fit_folded(a,theta,x)
        tables=dict(grid=grid,folded=fc,folded_legacy=fc)
        for name in NAMES:
            h,mask,calls=deploy(a,theta,x,ties,name,tables)
            assert np.isfinite(h).all();rows.append(dict(cohort=c,policy=name,calls=calls,selected=int(mask.sum())))
    jo(HERE/'smoke.json',dict(timestamp_utc=now(),rows=rows,no_fresh_inputs=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','freeze','train','calibrate','score','analyze']);args=p.parse_args()
    with threadpool_limits(limits=1):globals()[args.stage]()
