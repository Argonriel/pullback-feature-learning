"""Frozen independent iid confirmation supplement for direction 1.

Run only after direction1/new/prediction_commit.json exists. Uses no outcome
table to select cases. Population quadrature outcomes can be compared later.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from pullback_feature_learning import core
CONFIG = dict(n=32768, seed_pairs=list(range(1,17)), stage='trained',
    experiment='single', requested_drop=.02, tasks=['rings','radial','angular'],
    x_seed_base=27755100, label_seed_base=27756100, bootstrap_seed=27757100,
    bootstrap_repeats=10000, outcome='paired change in single-view logistic risk',
    role='Independent iid check of primary deterministic population quadrature',
    selection='All frozen trained-stage .02-drop branches, including failures')

def sha(data): return hashlib.sha256(data).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def read_csv(path):
    with path.open(newline='',encoding='utf8') as f: return list(csv.DictReader(f))
def write_csv(path,rows):
    with path.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def sig(x): return np.exp(-np.logaddexp(0.,-x))

def sample(task,seed):
    rng=np.random.default_rng(CONFIG['x_seed_base']+seed)
    n=CONFIG['n']
    if task=='rings':
        r=rng.choice([.5,1.2],n);a=rng.uniform(0.,2*np.pi,n)
        x=r[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0.,.15,(n,2))
        rr=np.linalg.norm(x,axis=1)
        p=sig((.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(1.2*rr/.15**2))-np.log(np.i0(.5*rr/.15**2)))
    else:
        u=rng.random(n);a=rng.uniform(0.,2*np.pi,n);r=.3+1.2*u
        x=r[:,None]*np.c_[np.cos(a),np.sin(a)]
        p=sig(3*np.cos(2*np.pi*u) if task=='radial' else 3*np.cos(2*a))
    label_rng=np.random.default_rng(CONFIG['label_seed_base']+seed)
    y=(label_rng.random(n)<p).astype(float)
    return x,p,y

def run():
    frozen=json.loads((ROOT/'IID_PROTOCOL.json').read_text())
    assert frozen['script_sha256']==sha(Path(__file__).read_bytes())
    assert frozen['config']==CONFIG
    src=ROOT/'direction1'/'new'
    commit=json.loads((src/'prediction_commit.json').read_text())
    assert sha((src/'predictions.csv').read_bytes())==commit['prediction_sha256']
    assert sha((src/'states.npz').read_bytes())==commit['state_sha256']
    out=ROOT/'iid_results';out.mkdir(exist_ok=False)
    dump(out/'started.json',dict(started_utc=now(),protocol_sha256=sha((ROOT/'IID_PROTOCOL.json').read_bytes()),prediction_commit=commit))
    predictions=read_csv(src/'predictions.csv')
    chosen={}
    for r in predictions:
        if r['stage']==CONFIG['stage'] and r['experiment']==CONFIG['experiment'] and float(r['requested_drop'])==CONFIG['requested_drop']:
            chosen.setdefault(r['row_id'],{})[r['metric']]=r
    state=np.load(src/'states.npz')
    rows=[];pointwise={};cache={}
    for row_id,parts in sorted(chosen.items()):
        r=parts['fit'];task=r['task'];seed=int(r['seed']);key=(task,seed)
        if key not in cache: cache[key]=sample(task,seed)
        x,p,y=cache[key]
        entry=dict(row_id=row_id,task=task,seed=seed,method=r['method'],status=r['status'],n=len(x),
            x_seed=CONFIG['x_seed_base']+seed,label_seed=CONFIG['label_seed_base']+seed,
            x_sha256=sha(x.tobytes()),y_sha256=sha(y.tobytes()),
            predicted_drop_clock=float(parts['fit']['drop_clock'])+float(parts['nuisance']['drop_clock']),
            predicted_old_pooled=float(parts['fit']['old_pooled'])+float(parts['nuisance']['old_pooled']),
            expected_risk_change='',expected_change_se='',label_risk_change='',label_change_se='',
            pre_risk='',post_risk='')
        if r['status']=='ok':
            before=core.representation(state[r['pre_key']],x)[:,0]
            after=core.representation(state[r['end_key']],x)[:,0]
            softdiff=np.logaddexp(0.,after)-np.logaddexp(0.,before)
            expected=softdiff-p*(after-before);labeled=softdiff-y*(after-before)
            entry.update(expected_risk_change=float(expected.mean()),expected_change_se=float(expected.std(ddof=1)/np.sqrt(len(x))),
                label_risk_change=float(labeled.mean()),label_change_se=float(labeled.std(ddof=1)/np.sqrt(len(x))),
                pre_risk=float((np.logaddexp(0.,before)-p*before).mean()),post_risk=float((np.logaddexp(0.,after)-p*after).mean()))
            pointwise[(task,seed,r['method'])]=(expected,labeled)
        rows.append(entry)
    write_csv(out/'by_seed.csv',rows)
    by={(r['task'],r['seed'],r['method']):r for r in rows};contrasts=[]
    for task in CONFIG['tasks']:
        for seed in CONFIG['seed_pairs']:
            a,b=by[task,seed,'gd'],by[task,seed,'pullback']
            if a['status']!='ok' or b['status']!='ok': continue
            e,l=(pointwise[task,seed,'pullback'][i]-pointwise[task,seed,'gd'][i] for i in (0,1))
            contrasts.append(dict(task=task,seed=seed,expected_pb_minus_gd=float(e.mean()),expected_se=float(e.std(ddof=1)/np.sqrt(len(e))),
                labeled_pb_minus_gd=float(l.mean()),labeled_se=float(l.std(ddof=1)/np.sqrt(len(l))),
                clock_pb_minus_gd=b['predicted_drop_clock']-a['predicted_drop_clock'],
                pooled_pb_minus_gd=b['predicted_old_pooled']-a['predicted_old_pooled']))
    write_csv(out/'paired_contrasts.csv',contrasts)
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);summary={}
    for task in CONFIG['tasks']:
        group=[r for r in contrasts if r['task']==task];n=len(group)
        if not n:summary[task]={'n':0};continue
        actual=np.array([r['expected_pb_minus_gd'] for r in group]);lab=np.array([r['labeled_pb_minus_gd'] for r in group]);pr=np.array([r['clock_pb_minus_gd'] for r in group]);old=np.array([r['pooled_pb_minus_gd'] for r in group])
        boot=rng.integers(0,n,(CONFIG['bootstrap_repeats'],n))
        summary[task]=dict(n=n,expected_mean=float(actual.mean()),expected_ci95=np.quantile(actual[boot].mean(1),[.025,.975]).tolist(),
            labeled_mean=float(lab.mean()),labeled_ci95=np.quantile(lab[boot].mean(1),[.025,.975]).tolist(),
            pb_better_count=int((actual<0).sum()),clock_correct_sign_count=int((actual*pr>0).sum()),
            clock_mae=float(np.abs(pr-actual).mean()),pooled_mae=float(np.abs(old-actual).mean()),zero_mae=float(np.abs(actual).mean()))
    dump(out/'summary.json',dict(completed_utc=now(),config=CONFIG,successful_branches=sum(r['status']=='ok' for r in rows),total_branches=len(rows),tasks=summary))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);args=ap.parse_args()
    if args.stage=='freeze':
        path=ROOT/'IID_PROTOCOL.json'
        if path.exists():raise FileExistsError(path)
        dump(path,dict(frozen_utc=now(),config=CONFIG,script_sha256=sha(Path(__file__).read_bytes()),new_outcomes_present=(ROOT/'direction1'/'new'/'outcomes.csv').exists()))
    else:
        with threadpool_limits(1):run()
