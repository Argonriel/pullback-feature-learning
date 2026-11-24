"""One bounded PB500 bridge follow-up, with an entirely new confirmation cohort.

Prompted by the PB100 pilot's degenerate all-pi choices. No outcome-based setting
changes are allowed after the six-old-seed exploration. All unreachable .02
loss-drop targets, including loss-floor and displacement-cap failures, remain.
"""
import argparse,csv,json,datetime
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core
import bridge
from experiment import module,OLD,advance,sha,dump,csvout
from experiment import evaluate_one,load_selection

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'late_bridge_source'
RESULTS=HERE/'late_bridge_results'
CFG=dict(old_seeds=list(range(1,7)),new_seeds=list(range(1,17)),
    new_train_base=621000,new_init_base=622000,new_cal_base=625000,old_cal_base=605000,
    pb_updates=500,pb_rate=8.,damping=.1,target_drop=.02,root_norm_cap=4.,
    root_expansion=1.35,root_iterations=160,bisection_iterations=60,
    development='PB100 pilot had no useful global angle changes; this PB500 sensitivity was frozen before a separate third cohort.',
    no_further_optimization=True)


def loss(t,x,y):return core.logistic_loss(core.representation(t,x),y)


def match(t,x,y,d):
    drop=.02;before=loss(t,x,y);slope=-float(core.parameter_gradient(t,x,y)@d)
    common=dict(start_loss=before,requested_drop=drop,train_slope=slope,alpha=0.,actual_drop=0.,residual=drop,attempts=0,displacement_norm=0.,status='')
    if slope<=0 or not np.isfinite(slope):return None,dict(common,status='non_descent')
    if before<=drop:return None,dict(common,status='loss_floor')
    alpha=drop/slope/8;lo=0.;target=before-drop
    for count in range(160):
        if np.linalg.norm(alpha*d)>4:return None,dict(common,attempts=count,status='norm_cap')
        value=loss(t+alpha*d,x,y)
        if not np.isfinite(value):return None,dict(common,attempts=count,status='nonfinite')
        if value<=target:
            hi=alpha
            for _ in range(60):
                mid=(lo+hi)/2
                if loss(t+mid*d,x,y)>target:lo=mid
                else:hi=mid
            alpha=(lo+hi)/2;end=t+alpha*d;actual=before-loss(end,x,y)
            return end,dict(common,alpha=alpha,actual_drop=actual,residual=abs(actual-drop),attempts=count+1,
                displacement_norm=float(np.linalg.norm(alpha*d)),status='ok')
        lo=alpha;alpha*=1.35
    return None,dict(common,attempts=160,status='search_cap')


def generate(cohort):
    out=SOURCE/cohort;out.mkdir(parents=True,exist_ok=False)
    if cohort=='old':old=np.load(HERE.parent/'direction1/old/states.npz')
    rings=module('late_rings',OLD/'generator.py').rings
    bank={};rows=[];records=[]
    for seed in CFG[cohort+'_seeds']:
        if cohort=='old':
            x,y,init=(old[f'rings_{seed}_{name}'] for name in ('x','y','initial'))
        else:
            x,y=rings(621000+seed);init=core.initialize_parameters(622000+seed)
        theta=advance(init,x,y,'pullback',8,500)
        prekey=f'rings_{seed}_trained'
        bank.update({f'rings_{seed}_x':x,f'rings_{seed}_y':y,f'rings_{seed}_initial':init,prekey:theta})
        for method in ('gd','pullback'):
            d,_=core.direction(theta,x,y,method,damping=.1)
            end,info=match(theta,x,y,d)
            endkey=prekey+'_'+method+'_single_1_0.02_end'
            if end is not None:bank[endkey]=end
            rows.append(dict(seed=seed,task='rings',stage='trained',method=method,experiment='single',
                horizon=1,requested_drop=.02,metric='fit',status=info['status'],pre_key=prekey,end_key=endkey,
                alpha=info['alpha'],actual_drop=info['actual_drop'],train_slope=info['train_slope']))
            records.append(dict(seed=seed,method=method,train_hash=sha(x.tobytes()+y.tobytes()),
                init_hash=sha(init.tobytes()),pre_hash=sha(theta.tobytes()),
                end_hash='' if end is None else sha(end.tobytes()),**info))
        print('late_generate',cohort,seed,flush=True)
    np.savez_compressed(out/'states.npz',**bank)
    csvout(out/'predictions.csv',rows);csvout(out/'roots.csv',records)
    dump(out/'training_commit.json',dict(config=CFG,
        state_hash=sha((out/'states.npz').read_bytes()),prediction_hash=sha((out/'predictions.csv').read_bytes()),
        root_hash=sha((out/'roots.csv').read_bytes()),labels_used_only_for_training=True))


def prepare():
    RESULTS.mkdir(exist_ok=True)
    bridge.HERE=RESULTS;bridge.D1=SOURCE
    bridge.CFG=dict(bridge.CFG,calibration_seed_old=605000,calibration_seed_new=625000,
        checkpoint='PB500',reuse='New confirmation cohort621001..621016/622001..622016; independent of main and PB100 bridge.')


def freeze():
    path=HERE/'late_bridge_freeze.json';assert not path.exists()
    dump(path,dict(config=CFG,bridge_config=bridge.CFG,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={p.name:sha(p.read_bytes()) for p in (Path(__file__),HERE/'bridge.py',HERE/'rotation_selectors.py')},
        old_outcomes_sha256=sha((RESULTS/'bridge_old/risks.csv').read_bytes()),
        no_setting_change_after_old_results=True))


def evaluate(cohort):
    out=RESULTS/('bridge_'+cohort)
    commit=json.loads((out/'selector_commit.json').read_text())
    for name,digest in commit['npz_hashes'].items():assert sha((out/name).read_bytes())==digest
    pred=list(csv.DictReader((out/'predictions.csv').open()))
    rows=[];curves=[];audit=[]
    for seed,method in sorted(set((int(r['seed']),r['method']) for r in pred)):
        key=f'{seed}_{method}'
        theta=np.load(out/(key+'_velocity.npz'))['post']
        ref=load_selection(out/(key+'_stale.npz'))
        r,c=evaluate_one(theta,ref,method,seed,selected=['exact_global'],nr=32,na=512)
        rr,cc=evaluate_one(theta,ref,method,seed,selected=['exact_global'],nr=64,na=1024)
        err=max(abs(a['risk']-b['risk']) for a,b in zip(r,rr))
        assert err<2e-7+2e-6*max(a['risk'] for a in rr)
        base=next(a for a in rr if a['policy']=='raw')
        rows.extend(a for a in rr if a['policy']!='exact_global');curves.extend(cc)
        for policy in ('stale','drop_clock_logits','matched_step_logits','first_order_gain','post_recalibrated'):
            cal=load_selection(out/(key+'_'+policy+'.npz'));idx=cal.names.index('exact_global')
            angle=float(cal.angles[idx,0]);gain=float(cc[int(round(angle/(2*np.pi/128)))-1]['gain'])
            rows.append(dict(base,policy=policy,risk=base['risk']-gain,gain=gain,
                fraction=gain/base['angular'],predicted_gain=float(cal.predicted_gains[idx]),alpha_mean=angle))
            audit.append(dict(seed=seed,method=method,policy=policy,grid_error=err))
        print('late_eval',cohort,key,flush=True)
    csvout(out/'risks.csv',rows);csvout(out/'gain_curves.csv',curves);csvout(out/'audits.csv',audit)
    summaries=[];rng=np.random.default_rng(629005)
    for method in ('gd','pullback'):
        seeds=sorted(set(r['pair'] for r in rows if r['model']==method))
        if not seeds:
            summaries.append(dict(method=method,policy='NO_REACHABLE_TARGET',n=0,mean_risk='',
                improvement_over_stale='',ci_low='',ci_high='',wins_over_stale=0,angle_changes=0,
                mean_gain_prediction_error=''));continue
        indices=rng.integers(0,len(seeds),(20000,len(seeds)))
        get=lambda s,p:next(r for r in rows if r['model']==method and r['pair']==s and r['policy']==p)
        for policy in ('pi','stale','drop_clock_logits','matched_step_logits','first_order_gain','post_recalibrated'):
            q=[get(s,policy) for s in seeds];stale=[get(s,'stale') for s in seeds]
            delta=np.array([b['risk']-a['risk'] for a,b in zip(q,stale)])
            aa=np.array([np.pi if policy=='pi' else r['alpha_mean'] for r in q]);bb=np.array([r['alpha_mean'] for r in stale])
            summaries.append(dict(method=method,policy=policy,n=len(seeds),mean_risk=np.mean([r['risk'] for r in q]),
                improvement_over_stale=delta.mean(),ci_low=np.quantile(delta[indices].mean(1),.025),
                ci_high=np.quantile(delta[indices].mean(1),.975),wins_over_stale=int((delta>1e-12).sum()),
                angle_changes=int(np.sum(np.abs(aa-bb)>1e-10)),
                mean_gain_prediction_error=np.mean([r['predicted_gain']-r['gain'] for r in q])))
    csvout(out/'summary.csv',summaries)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate','fit','evaluate','freeze'])
    ap.add_argument('--cohort',choices=['old','new']);args=ap.parse_args()
    with threadpool_limits(limits=1):
        prepare()
        if args.stage=='freeze':freeze()
        else:
            if args.cohort=='new':
                frozen=json.loads((HERE/'late_bridge_freeze.json').read_text())
                for name,digest in frozen['source_sha256'].items():assert sha((HERE/name).read_bytes())==digest
            if args.stage=='generate':generate(args.cohort)
            elif args.stage=='evaluate':evaluate(args.cohort)
            else:bridge.fit(args.cohort)
