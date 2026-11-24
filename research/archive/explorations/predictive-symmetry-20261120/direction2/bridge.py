"""Prospectively forecast a useful post-update second-view angle.

Reuse direction1's matched-loss endpoints, with fresh independent calibration
orbits. Fit stage never evaluates population risk. Primary forecast displacement
is requested_loss_drop*d/(-grad_train dot d); fitted step is a secondary check.
"""
import argparse,csv,json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core
from rotation_selectors import (calibrated_angle_selection, selection_from_logits,
    radii_sample, orbit_logits, phi)
from experiment import sha,dump,csvout,evaluate_one,load_selection

HERE=Path(__file__).resolve().parent
D1=HERE.parent/'direction1'
CFG=dict(calibration_radii=512,calibration_angles=128,radial_bins=8,
    calibration_seed_old=605000,calibration_seed_new=615000,
    requested_drop=.02,task='rings',stage='trained',experiment='single',
    primary='drop_clock_logits',secondary=['matched_step_logits','first_order_gain'],
    evaluation_nr=32,evaluation_na=512,refinement_nr=64,refinement_na=1024,
    reuse='Same direction1 matched-loss cohort, not an independent replication.',
    selection='Complete independent calibration orbits; every selector fixed before population risk.')


def jvp(theta,delta,x):
    """Forward directional derivative, checked against the independent full Jacobian."""
    h=x; v=np.zeros_like(x)
    for (a,b,bias),(da,db,dbias) in zip(core.unpack(theta),core.unpack(delta)):
        t=np.tanh(h@a.T+bias)
        dt=(1-t*t)*(v@a.T+h@da.T+dbias)
        v=v+.5*(dt@b.T+t@db.T)
        h=h+.5*t@b.T
    return v[:,0]


def velocities(theta,delta,r,phase):
    na=CFG['calibration_angles']
    angle=phase[:,None]+2*np.pi*np.arange(na)[None,:]/na
    x=(r[:,None,None]*np.stack((np.cos(angle),np.sin(angle)),axis=-1)).reshape(-1,2)
    v=np.concatenate([jvp(theta,delta,x[i:i+4096]) for i in range(0,len(x),4096)])
    return v.reshape(len(r),na)


def unique_predictions(cohort):
    rr=list(csv.DictReader((D1/cohort/'predictions.csv').open()))
    return [r for r in rr if r['task']=='rings' and r['stage']=='trained'
        and r['experiment']=='single' and float(r['requested_drop'])==.02 and r['metric']=='fit']


def fit(cohort):
    out=HERE/('bridge_'+cohort); out.mkdir(exist_ok=False)
    bank=np.load(D1/cohort/'states.npz')
    rows=[]; rejected=[]
    for pred in unique_predictions(cohort):
        seed=int(pred['seed']);method=pred['method'];key=f'{seed}_{method}'
        if pred['status']!='ok':
            rejected.append(dict(seed=seed,method=method,status=pred['status']));continue
        theta=bank[pred['pre_key']]; post=bank[pred['end_key']]
        x=bank[f'rings_{seed}_x'];y=bank[f'rings_{seed}_y']
        d,_=core.direction(theta,x,y,method,damping=.1)
        grad=core.parameter_gradient(theta,x,y)
        speed=-float(grad@d);eta=.02/speed;alpha=float(pred['alpha'])
        assert speed>0 and abs(speed-float(pred['train_slope']))<1e-10
        delta=eta*d
        check=x[:17]
        j=core.representation_jacobian(theta,check)[0::2]@delta
        assert np.max(abs(j-jvp(theta,delta,check)))<1e-10
        assert np.max(abs(post-theta-alpha*d))<1e-9
        cal_seed=CFG['calibration_seed_'+cohort]+seed
        radii,phases=radii_sample(cal_seed,512)
        start=perf_counter()
        stale=calibrated_angle_selection(theta,radii,phases)
        stale.save(out/(key+'_stale.npz'))
        v=velocities(theta,delta,radii,phases)
        forecast=selection_from_logits(stale.h+v,radii,phases)
        forecast.save(out/(key+'_drop_clock_logits.npz'))
        matched=selection_from_logits(stale.h+v*(alpha/eta),radii,phases)
        matched.save(out/(key+'_matched_step_logits.npz'))
        # First derivative of the Jensen gain itself, without replacing h by h+v.
        sigmoid=lambda z:np.exp(-np.logaddexp(0,-z))
        deriv=[]
        for shift in range(1,65):
            m=.5*(stale.h+np.roll(stale.h,-shift,1))
            mv=.5*(v+np.roll(v,-shift,1))
            deriv.append(np.mean(sigmoid(stale.h)*v-sigmoid(m)*mv))
        first_curve=stale.gain_curves[stale.names.index('exact_global')]+np.array(deriv)
        first_idx=int(first_curve.argmax())
        first=selection_from_logits(stale.h,radii,phases)
        for name in ('exact_global','exact_radial'):
            idx=first.names.index(name)
            first.angles[idx]=np.repeat((first_idx+1)*2*np.pi/128,8)
            first.predicted_gains[idx]=first_curve[first_idx]
            first.gain_curves[idx]=first_curve
        first.save(out/(key+'_first_order_gain.npz'))
        forecast_seconds=perf_counter()-start
        # Strong competitive label-free baseline can observe postupdate logits.
        start=perf_counter()
        recal=calibrated_angle_selection(post,radii,phases)
        recal.save(out/(key+'_post_recalibrated.npz'))
        recal_seconds=perf_counter()-start
        np.savez_compressed(out/(key+'_velocity.npz'),v=v,delta=delta,theta=theta,post=post,
            train_direction=d,train_x=x,train_y=y)
        for policy,cal in [('stale',stale),('drop_clock_logits',forecast),
            ('matched_step_logits',matched),('first_order_gain',first),('post_recalibrated',recal)]:
            idx=cal.names.index('exact_global')
            rows.append(dict(seed=seed,method=method,policy=policy,
                selected_angle=float(cal.angles[idx,0]),predicted_gain=float(cal.predicted_gains[idx]),
                cal_seed=cal_seed,train_slope=speed,loss_clock_scale=eta,actual_matched_scale=alpha,
                target_loss_drop=.02,actual_loss_drop=float(pred['actual_drop']),
                before_hash=sha(theta.tobytes()),endpoint_hash=sha(post.tobytes()),
                cal_hash=sha(radii.tobytes()+phases.tobytes()),forecast_seconds=forecast_seconds,
                recalibration_seconds=recal_seconds,prestate_forward_examples=65536,
                prestate_jvp_examples=65536,post_recalibration_forward_examples=65536))
        print('bridge_fit',cohort,key,flush=True)
    csvout(out/'predictions.csv',rows)
    if rejected:csvout(out/'unreachable.csv',rejected)
    dump(out/'selector_commit.json',dict(config=CFG,script_sha256=sha(Path(__file__).read_bytes()),
        parent_prediction_sha256=sha((D1/cohort/'predictions.csv').read_bytes()),
        parent_state_sha256=sha((D1/cohort/'states.npz').read_bytes()),
        npz_hashes={p.name:sha(p.read_bytes()) for p in out.glob('*.npz')},
        risk_not_evaluated_in_fit_stage=True))


def evaluate(cohort):
    out=HERE/('bridge_'+cohort)
    commit=json.loads((out/'selector_commit.json').read_text())
    for name,digest in commit['npz_hashes'].items():assert sha((out/name).read_bytes())==digest
    pred=list(csv.DictReader((out/'predictions.csv').open()))
    rows=[];curve_rows=[];audit=[]
    for seed,method in sorted(set((int(r['seed']),r['method']) for r in pred)):
        key=f'{seed}_{method}'
        state=np.load(out/(key+'_velocity.npz')); theta=state['post']
        reference=load_selection(out/(key+'_stale.npz'))
        allrows,curve=evaluate_one(theta,reference,method,seed,selected=['exact_global'],nr=32,na=512)
        allrows2,curve2=evaluate_one(theta,reference,method,seed,selected=['exact_global'],nr=64,na=1024)
        err=max(abs(a['risk']-b['risk']) for a,b in zip(allrows,allrows2))
        assert err<2e-7+2e-6*max(r['risk'] for r in allrows2)
        base=next(r for r in allrows2 if r['policy']=='raw')
        rows.extend(r for r in allrows2 if r['policy']!='exact_global')
        curve_rows.extend(curve2)
        for policy in ('stale','drop_clock_logits','matched_step_logits','first_order_gain','post_recalibrated'):
            cal=load_selection(out/(key+'_'+policy+'.npz'))
            idx=cal.names.index('exact_global')
            assert np.array_equal(cal.edges,reference.edges)
            angle=float(cal.angles[idx,0])
            gain=float(curve2[int(round(angle/(2*np.pi/128)))-1]['gain'])
            selected=dict(base,policy=policy,risk=base['risk']-gain,gain=gain,
                fraction=gain/base['angular'],predicted_gain=float(cal.predicted_gains[idx]),alpha_mean=angle)
            rows.append(selected)
            audit.append(dict(seed=seed,method=method,policy=policy,grid_error=err))
        print('bridge_eval',cohort,key,flush=True)
    csvout(out/'risks.csv',rows);csvout(out/'gain_curves.csv',curve_rows);csvout(out/'audits.csv',audit)
    summary=[]
    rng=np.random.default_rng(619005)
    for method in ('gd','pullback'):
        seeds=sorted(set(int(r['pair']) for r in rows if r['model']==method))
        indices=rng.integers(0,len(seeds),(20000,len(seeds)))
        for policy in ('pi','stale','drop_clock_logits','matched_step_logits','first_order_gain','post_recalibrated'):
            q=[next(r for r in rows if r['model']==method and r['pair']==s and r['policy']==policy) for s in seeds]
            stale=[next(r for r in rows if r['model']==method and r['pair']==s and r['policy']=='stale') for s in seeds]
            delta=np.array([b['risk']-a['risk'] for a,b in zip(q,stale)])
            angles=[next(float(r['selected_angle']) for r in pred if r['method']==method and int(r['seed'])==s and r['policy']==policy) for s in seeds] if policy!='pi' else [np.pi]*len(seeds)
            staleangles=[next(float(r['selected_angle']) for r in pred if r['method']==method and int(r['seed'])==s and r['policy']=='stale') for s in seeds]
            summary.append(dict(method=method,policy=policy,n=len(seeds),mean_risk=np.mean([r['risk'] for r in q]),
                improvement_over_stale=delta.mean(),ci_low=np.quantile(delta[indices].mean(1),.025),
                ci_high=np.quantile(delta[indices].mean(1),.975),wins_over_stale=int((delta>1e-12).sum()),
                angle_changes=int(np.sum(np.abs(np.array(angles)-staleangles)>1e-10)),
                mean_gain_prediction_error=np.mean([r['predicted_gain']-r['gain'] for r in q])))
    csvout(out/'summary.csv',summary)


def freeze():
    path=HERE/'bridge_freeze.json'
    assert not path.exists()
    dump(path,dict(config=CFG,script_sha256=sha(Path(__file__).read_bytes()),
        old_results_sha256=sha((HERE/'bridge_old/risks.csv').read_bytes()),
        settings_changed_after_old_exploration=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['fit','evaluate','freeze']);ap.add_argument('--cohort',choices=['old','new'])
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.action=='freeze':freeze()
        else:
            if args.cohort=='new':
                frozen=json.loads((HERE/'bridge_freeze.json').read_text())
                assert frozen['script_sha256']==sha(Path(__file__).read_bytes())
            globals()[args.action](args.cohort)
