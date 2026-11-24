"""Prospective, actual-loss-matched local prediction study. English-only artifacts."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time,datetime,os
from pathlib import Path
from functools import lru_cache
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(WORK/'pullback-feature-learning'/'src'))
LEGACY=next(p for p in (ROOT.parent/'legacy'/'rings-mechanism-20261120',WORK/'tmp'/'rings-mechanism-20261120') if (p/'polar.py').exists())
sys.path.insert(0,str(LEGACY))
import model_core as core
# polar imports the repository package; inject the frozen numerical core if unavailable.
try: import pullback_feature_learning
except ModuleNotFoundError:
    import types
    pkg=types.ModuleType('pullback_feature_learning');pkg.core=core
    sys.modules['pullback_feature_learning']=pkg;sys.modules['pullback_feature_learning.core']=core
import polar
from generator import rings
METRICS=('fit','nuisance','view_value')
TASKS=('rings','radial','angular')
METHODS=('gd','pullback')
CONFIG=dict(old_seeds=list(range(1,7)),new_seeds=list(range(1,17)),damping=.1,
            stages=['initial','trained'],single_drops=[1e-5,.02],path_drop=.005,horizons=[1,2,4,8],
            root_expansion=1.35,root_max_norm=4.,root_residual=2e-12,
            derivative_grid=[48,64],score_grids=[[48,64],[96,128],[192,256]],score_atol=2e-7,
            new_ring_train_base=611000,new_ring_init_base=612000,
            new_synthetic_train_base=27751100,new_synthetic_init_base=27752100,
            old_ring_train_base=71000,old_ring_init_base=73000,
            old_synthetic_train_base=1,old_synthetic_init_base=1001,
            synthetic_n=256,synthetic_gd_steps=300,synthetic_gd_rate=.125,
            rings_pb_steps=100,rings_pb_rate=8.,bootstrap_seed=27753100,bootstrap_repeats=10000)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(v): return hashlib.sha256(v).hexdigest()
def jsonout(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def csvout(p,rows):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def csvin(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def sigmoid(h):return np.exp(-np.logaddexp(0.,-h))
def loss(t,x,y):return core.logistic_loss(core.representation(t,x),y)
@lru_cache(None)
def grid(task,nr,na):
    a=2*np.pi*(np.arange(na)+.5)/na
    if task=='rings':
        r,w0,w1,unit=polar.polar_grid(nr,na);w=w0+w1;p1=w1/w
        p=np.broadcast_to(p1[:,None],(nr,na))
    else:
        nodes,weights=np.polynomial.legendre.leggauss(nr);u=(nodes+1)/2;w=weights/2;r=.3+1.2*u
        unit=np.column_stack([np.cos(a),np.sin(a)])
        b=3*np.cos(2*np.pi*u)[:,None] if task=='radial' else 3*np.cos(2*a)[None,:]
        p=np.broadcast_to(sigmoid(b),(nr,na))
    x=(r[:,None,None]*unit[None,:,:]).reshape(-1,2)
    bayes=np.sum(w[:,None]*(-(p*np.log(np.maximum(p,1e-300))+(1-p)*np.log(np.maximum(1-p,1e-300)))))/na
    return x,w,p,float(bayes)
def components_h(h,w,p,task,derivatives=False):
    nr,na=h.shape
    soft=np.logaddexp(0,h)
    raw=np.sum(w[:,None]*(soft-p*h))/na
    grad=(sigmoid(h)-p)*w[:,None]/na
    if task=='angular':
        bar=(h+np.roll(h,na//2,axis=1))/2
        fitting=np.sum(w[:,None]*(np.logaddexp(0,bar)-p*bar))/na
        gf=(sigmoid(bar)-p)*w[:,None]/na
        pair=fitting;gp=gf
    else:
        bar=h.mean(axis=1)
        fitting=np.sum(w*(np.logaddexp(0,bar)-p[:,0]*bar))
        gf=np.broadcast_to(((sigmoid(bar)-p[:,0])*w/na)[:,None],h.shape)
        pair=0.;gp=np.empty_like(h)
        for i in range(nr):
            av=(h[i,:,None]+h[i,None,:])/2
            pair+=w[i]*np.mean(np.logaddexp(0,av)-p[i,0]*av)
            gp[i]=w[i]*(sigmoid(av)-p[i,0]).mean(axis=1)/na
    vals=np.array([fitting,raw-fitting,raw-pair])
    grads=np.stack([gf,grad-gf,grad-gp]) if derivatives else None
    return vals,grads,float(raw)
def measure(t,task,nr=48,na=64,derivatives=False):
    x,w,p,b=grid(task,nr,na)
    h=core.representation(t,x)[:,0].reshape(nr,na)
    val,gh,raw=components_h(h,w,p,task,derivatives)
    val[0]-=b
    if derivatives:
        jac=core.representation_jacobian(t,x)[0::2]
        g=gh.reshape(3,-1)@jac
    else:g=None
    return val,g,dict(raw=raw,bayes=b,identity_error=abs(val[0]+val[1]-(raw-b)))
def refined(t,task):
    prev=None
    for nr,na in CONFIG['score_grids']:
        v,_,meta=measure(t,task,nr,na)
        if prev is not None:
            err=float(np.max(np.abs(v-prev)))
            if err<=CONFIG['score_atol']:return v,dict(**meta,nr=nr,na=na,grid_error=err,converged=1)
        prev=v
    return v,dict(**meta,nr=nr,na=na,grid_error=err,converged=0)
def dataset(task,seed,cohort):
    prefix='old' if cohort=='old' else 'new'
    if task=='rings':
        ds=CONFIG[prefix+'_ring_train_base']+seed;ini=CONFIG[prefix+'_ring_init_base']+seed
        x,y=rings(ds)
    else:
        ds=CONFIG[prefix+'_synthetic_train_base']+seed;ini=CONFIG[prefix+'_synthetic_init_base']+seed
        rng=np.random.default_rng(ds);u=rng.random(CONFIG['synthetic_n']);a=rng.uniform(0,2*np.pi,len(u));q=rng.random(len(u))
        r=.3+1.2*u;x=np.column_stack([r*np.cos(a),r*np.sin(a)])
        p=sigmoid(3*np.cos(2*np.pi*u) if task=='radial' else 3*np.cos(2*a))
        y=np.where(q<p,1.,-1.)
    return x,y,core.initialize_parameters(ini),ds,ini
def prestates(task,x,y,t):
    ans={'initial':t.copy()}
    if task=='rings':
        for _ in range(CONFIG['rings_pb_steps']):t=t+CONFIG['rings_pb_rate']*core.direction(t,x,y,'pullback',CONFIG['damping'])[0]
    else:
        for _ in range(CONFIG['synthetic_gd_steps']):t=t+CONFIG['synthetic_gd_rate']*core.direction(t,x,y,'gd')[0]
    ans['trained']=t.copy();return ans
def match(t,x,y,d,drop):
    before=loss(t,x,y);slope=-float(core.parameter_gradient(t,x,y)@d)
    common=dict(start_loss=before,requested_drop=drop,train_slope=slope,alpha=0.,actual_drop=0.,residual=drop,attempts=0,displacement_norm=0.,status='')
    if slope<=0 or not np.isfinite(slope):return None,dict(common,status='non_descent')
    if before<=drop:return None,dict(common,status='loss_floor')
    alpha=drop/slope/8;lo=0.;flo=before;target=before-drop
    for count in range(160):
        if np.linalg.norm(alpha*d)>CONFIG['root_max_norm']:return None,dict(common,attempts=count,status='norm_cap')
        value=loss(t+alpha*d,x,y)
        if not np.isfinite(value):return None,dict(common,attempts=count,status='nonfinite')
        if value<=target:
            hi=alpha
            for _ in range(60):
                mid=(lo+hi)/2;lm=loss(t+mid*d,x,y)
                if lm>target:lo=mid
                else:hi=mid
            alpha=(lo+hi)/2;end=t+alpha*d;actual=before-loss(end,x,y)
            return end,dict(common,alpha=alpha,actual_drop=actual,residual=abs(actual-drop),attempts=count+1,displacement_norm=float(np.linalg.norm(alpha*d)),status='ok')
        lo=alpha;flo=value;alpha*=CONFIG['root_expansion']
    return None,dict(common,attempts=160,status='search_cap')
def freeze():
    if (ROOT/'freeze.json').exists():raise FileExistsError('Protocol already frozen')
    protocol='''Prospective actual-loss-drop prediction protocol\n\nExploration: old seed IDs 1..6. Confirmation: wholly new ring train611001..611016/init612001..612016 and synthetic train27751101..27751116/init27752101..27752116. Shared ring states across directions are not independent replications. Same 80-parameter architecture; damping .1. Original rings pre-states initial and PB100(rate8). Synthetic pre-states initial and GD300(rate.125).\n\nMatched transfer: identical X and initialization across radial/angular task, u~Uniform[0,1], phi~Uniform[0,2pi], r=.3+1.2u; logit posterior=3cos(2piu) or 3cos(2phi). These scores have identical arcsine distributions, equal .5 class probability, and identical Bayes risk/accuracy. True radial symmetry is full rotations. True angular symmetry is pi rotations (also reflections); only pi partner used.\n\nFit is Bayes-excess loss of true-symmetry averaged logit; nuisance is exact Jensen loss gap. For radial/rings these are radial fitting/angular nuisance. For angular, fitting retains informative angle dependence; it is NOT radial fitting. View value is one-view risk minus expected two-view averaged-logit risk: independent uniform rotational partner for radial/rings; the pi partner for angular (there it equals nuisance identically).\n\nEvery branch starts at exactly the same theta. GD/PB root searches first detected crossing along direction: geometrically increasing alpha, displacement norm cap4; bisection; failed and unreachable targets retained. Local drops 1e-5 and .02 nats; multistep 8 recomputed directions at .005 nats/drop, horizons1,2,4,8. Prediction at initial state remains frozen over horizon.\n\nPrimary pre-state clock predictor: DQ[d]*delta/(-DL[d]). Diagnostic realized-step tangent: DQ[d]*alpha for one step; DQ[theta_end-theta_start] for path, using actual train-only state displacement; it is outcome-blind but NOT a fully pre-update prediction. Zero and old-seed pooled slopes (same task/stage/method/experiment/horizon/requested_drop) are competitors. Old baseline mean delta per achieved-drop uses only old-seed outcomes, stratified by exact requested drop. If an exact stratum has no reachable old outcome, fallback pools reachable slopes for the same task/stage/method/metric; source and training count are recorded. Training-only endpoint loss/steps do not use population outcomes.\n\nFor each cohort, all endpoint states are generated with training data only, then prediction table is physically saved and SHA256 stamped before ANY updated-state population scores are evaluated. Pre-state metrics/derivatives are allowed. Old cohort is completed first and frozen protocol records old artifacts/code hashes before fresh states. All roots, seed rows, states and failures retained. No fresh-seed winner selection.\n\nReport component MAE/RMSE, sign counts, paired improvement over competitors, aggregated within seed then bootstrap seeds10000 times. Repeated states/targets are not independent samples. Report all strata and combined fixed strata. Identity/finite-difference/grid checks supplied. A first-order prediction can fail at substantial drop or long horizons; preserve these failures.\n'''
    (ROOT/'PROTOCOL.md').write_text(protocol,encoding='utf8')
    manifests={p.name:sha(p.read_bytes()) for p in [ROOT/'study.py',ROOT/'PROTOCOL.md',ROOT/'old'/'outcomes.csv',ROOT/'old'/'predictions.csv']}
    jsonout(ROOT/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,sha256=manifests,core_sha256=sha(Path(core.__file__).read_bytes())))
    print('FROZEN '+json.dumps(manifests),flush=True)
def baseline():
    rows=csvin(ROOT/'old'/'outcomes.csv');b={};fallback={}
    for r in rows:
        if r['status']!='ok':continue
        key=tuple(r[k] for k in ('task','stage','method','experiment','horizon','requested_drop','metric'))
        ratio=float(r['actual_change'])/float(r['actual_drop'])
        b.setdefault(key,[]).append(ratio)
        fk=(r['task'],r['stage'],r['method'],r['metric'])
        fallback.setdefault(fk,[]).append(ratio)
    return ({k:(float(np.mean(v)),len(v)) for k,v in b.items()},{k:(float(np.mean(v)),len(v)) for k,v in fallback.items()})
def generate(cohort):
    out=ROOT/cohort;out.mkdir(exist_ok=False)
    if cohort=='new':
        frozen=json.loads((ROOT/'freeze.json').read_text());assert frozen['sha256']['study.py']==sha(Path(__file__).read_bytes())
    started=now();pred=[];roots=[];bank={};seeds=[];checks=[];base,fallback=baseline() if cohort=='new' else ({},{})
    for seed in CONFIG[cohort+'_seeds']:
        for task in TASKS:
            x,y,t,ds,ini=dataset(task,seed,cohort);bank[f'{task}_{seed}_x']=x;bank[f'{task}_{seed}_y']=y
            seeds.append(dict(seed=seed,task=task,data_seed=ds,initialization_seed=ini,x_sha256=sha(x.tobytes()),y_sha256=sha(y.tobytes()),initial_sha256=sha(t.tobytes())))
            for stage,t in prestates(task,x,y,t).items():
                prekey=f'{task}_{seed}_{stage}';bank[prekey]=t
                pre,gq,meta=measure(t,task,*CONFIG['derivative_grid'],True)
                if seed==CONFIG[cohort+'_seeds'][0]:
                    rng=np.random.default_rng(177);v=rng.normal(size=80);v/=np.linalg.norm(v);eps=1e-5
                    finite=(measure(t+eps*v,task)[0]-measure(t-eps*v,task)[0])/(2*eps)
                    checks.append(dict(task=task,stage=stage,max_derivative_error=float(np.max(abs(finite-gq@v))),identity_error=meta['identity_error']))
                for method in METHODS:
                    d=core.direction(t,x,y,method,CONFIG['damping'])[0];slope=-core.parameter_gradient(t,x,y)@d;deriv=gq@d
                    branches=[]
                    for drop in CONFIG['single_drops']:
                        end,info=match(t,x,y,d,drop);branches.append(('single',1,drop,end,info,[]))
                    curr=t.copy();info=None;path_alpha=[];laststatus='ok'
                    for step in range(1,max(CONFIG['horizons'])+1):
                        if laststatus=='ok':
                            stepd=core.direction(curr,x,y,method,CONFIG['damping'])[0]
                            end,info=match(curr,x,y,stepd,CONFIG['path_drop'])
                            roots.append(dict(seed=seed,task=task,stage=stage,method=method,experiment='path_step',horizon=step,**info))
                            laststatus=info['status'];path_alpha.append(info['alpha'])
                            if end is not None:curr=end
                        if step in CONFIG['horizons']:
                            total=step*CONFIG['path_drop'];pi=dict(info)
                            pi.update(start_loss=loss(t,x,y),requested_drop=total,actual_drop=loss(t,x,y)-loss(curr,x,y),residual=abs(loss(t,x,y)-loss(curr,x,y)-total),status=laststatus,displacement_norm=float(np.linalg.norm(curr-t)))
                            branches.append(('path',step,total,curr.copy() if laststatus=='ok' else None,pi,path_alpha.copy()))
                    for exp,horizon,drop,end,info,alphas in branches:
                        rowid=f'{prekey}_{method}_{exp}_{horizon}_{drop:g}';endkey=rowid+'_end'
                        roots.append(dict(seed=seed,task=task,stage=stage,method=method,experiment=exp,horizon=horizon,**info))
                        if end is not None:bank[endkey]=end
                        for j,m in enumerate(METRICS):
                            key=(task,stage,method,exp,str(horizon),str(drop),m)
                            pooled_slope,pooled_n=base.get(key,fallback.get((task,stage,method,m),(0.,0)))
                            pooled=pooled_slope*drop
                            pooled_source='exact_drop' if key in base else 'pooled_reachable'
                            pred.append(dict(row_id=rowid,seed=seed,task=task,stage=stage,method=method,experiment=exp,horizon=horizon,requested_drop=drop,metric=m,status=info['status'],pre_key=prekey,end_key=endkey,pre_sha256=sha(t.tobytes()),end_sha256=sha(end.tobytes()) if end is not None else '',pre_value=pre[j],train_slope=slope,outcome_derivative=deriv[j],drop_clock=deriv[j]*drop/slope,realized_tangent=float(gq[j]@(end-t)) if end is not None else '',zero=0.,old_pooled=pooled,old_baseline_n=pooled_n,old_baseline_source=pooled_source,actual_drop=info['actual_drop'],alpha=info['alpha'],residual=info['residual']))
        print(f'{cohort} predictions seed {seed} done',flush=True)
    csvout(out/'predictions.csv',pred);csvout(out/'roots.csv',roots);csvout(out/'seeds.csv',seeds);jsonout(out/'derivative_checks.json',checks);np.savez_compressed(out/'states.npz',**bank)
    jsonout(out/'prediction_commit.json',dict(started_utc=started,committed_utc=now(),outcome_evaluation_started=False,prediction_sha256=sha((out/'predictions.csv').read_bytes()),state_sha256=sha((out/'states.npz').read_bytes()),root_sha256=sha((out/'roots.csv').read_bytes()),row_count=len(pred),root_count=len(roots),state_count=len(bank)))
    print(f'{cohort} prediction commit complete',flush=True)
def score(cohort):
    out=ROOT/cohort;commit=json.loads((out/'prediction_commit.json').read_text());assert commit['prediction_sha256']==sha((out/'predictions.csv').read_bytes())
    if (out/'outcomes.csv').exists():raise FileExistsError('Outcomes already scored')
    jsonout(out/'score_start.json',dict(timestamp_utc=now(),prediction_commit=commit))
    rows=csvin(out/'predictions.csv');bank=np.load(out/'states.npz');cache={};outcomes=[];quad=[]
    for r in rows:
        if r['status']=='ok':
            for key in (r['pre_key'],r['end_key']):
                if key not in cache:
                    val,meta=refined(bank[key],r['task']);cache[key]=val;quad.append(dict(key=key,task=r['task'],**meta))
            j=METRICS.index(r['metric']);change=cache[r['end_key']][j]-cache[r['pre_key']][j]
            r.update(actual_pre=cache[r['pre_key']][j],actual_post=cache[r['end_key']][j],actual_change=change)
        else:r.update(actual_pre='',actual_post='',actual_change='')
        outcomes.append(r)
    csvout(out/'outcomes.csv',outcomes);csvout(out/'quadrature.csv',quad)
    jsonout(out/'score_complete.json',dict(timestamp_utc=now(),outcome_sha256=sha((out/'outcomes.csv').read_bytes()),unique_scored_states=len(cache),max_grid_error=max(r['grid_error'] for r in quad),nonconverged=sum(1-r['converged'] for r in quad),max_identity_error=max(r['identity_error'] for r in quad)))
    print(f'{cohort} scoring done {len(cache)} states',flush=True)
def summarize():
    rows=[r for r in csvin(ROOT/'new'/'outcomes.csv') if r['status']=='ok'];rng=np.random.default_rng(CONFIG['bootstrap_seed']);summary=[]
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:
        drop=r['requested_drop'] if r['experiment']=='single' else 'path'
        groups[(r['task'],r['stage'],r['experiment'],r['horizon'],drop,r['metric'])].append(r)
    for key,rr in groups.items():
        for pred in ('drop_clock','realized_tangent','zero','old_pooled'):
            byseed=defaultdict(list);sgn=[]
            for r in rr:
                actual=float(r['actual_change']);pr=float(r[pred]);err=abs(pr-actual);zero=abs(actual);old=abs(float(r['old_pooled'])-actual)
                byseed[int(r['seed'])].append((err,err*err,zero-err,old-err));sgn.append(int(pr*actual>0))
            z=np.array([np.mean(v,axis=0) for _,v in sorted(byseed.items())]);boot=rng.integers(0,len(z),(CONFIG['bootstrap_repeats'],len(z)));ci=np.quantile(z[boot].mean(axis=1),[.025,.975],axis=0)
            summary.append(dict(task=key[0],stage=key[1],experiment=key[2],horizon=key[3],drop=key[4],metric=key[5],predictor=pred,n_rows=len(rr),n_seed=len(z),mae=float(z[:,0].mean()),rmse=float(np.sqrt(z[:,1].mean())),correct_nonzero_sign=sum(sgn),seed_wins_zero=int((z[:,2]>0).sum()),seed_wins_old=int((z[:,3]>0).sum()),mae_advantage_zero=float(z[:,2].mean()),advantage_zero_lo=float(ci[0,2]),advantage_zero_hi=float(ci[1,2]),mae_advantage_old=float(z[:,3].mean()),advantage_old_lo=float(ci[0,3]),advantage_old_hi=float(ci[1,3])))
    csvout(ROOT/'summary.csv',summary)
    contrasts=[]
    index={tuple(r[k] for k in ('seed','task','stage','experiment','horizon','requested_drop','metric','method')):r for r in rows}
    for k,r in index.items():
        if k[-1]!='gd':continue
        q=index.get(k[:-1]+('pullback',))
        if q:
            actual=float(q['actual_change'])-float(r['actual_change'])
            contrasts.append(dict(seed=k[0],task=k[1],stage=k[2],experiment=k[3],horizon=k[4],requested_drop=k[5],metric=k[6],actual_pb_minus_gd=actual,clock_pb_minus_gd=float(q['drop_clock'])-float(r['drop_clock']),pooled_pb_minus_gd=float(q['old_pooled'])-float(r['old_pooled']),tangent_pb_minus_gd=float(q['realized_tangent'])-float(r['realized_tangent'])))
    csvout(ROOT/'method_contrasts.csv',contrasts)
    allrows=csvin(ROOT/'new'/'outcomes.csv');status={}
    for r in allrows:status[r['status']]=status.get(r['status'],0)+1
    jsonout(ROOT/'coverage.json',dict(prediction_rows=len(allrows),successful_rows=len(rows),status_metric_rows=status,unique_cases=len(set(r['row_id'] for r in allrows)),successful_cases=len(set(r['row_id'] for r in rows)),matched_method_contrasts=len(contrasts)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','score','freeze','summarize']);p.add_argument('--cohort',choices=['old','new']);a=p.parse_args()
    with threadpool_limits(1):
        if a.action=='generate':generate(a.cohort)
        elif a.action=='score':score(a.cohort)
        elif a.action=='freeze':freeze()
        else:summarize()



