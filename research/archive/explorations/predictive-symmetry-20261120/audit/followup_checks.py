"""Independent checks of explicitly labeled, post-discovery follow-ups."""
from collections import Counter,defaultdict
from datetime import datetime
import json
import numpy as np
from audit import core,digest,read_csv,loss,sigmoid


def second_forward(theta,d,x):
    h=x.copy();v=np.zeros_like(h);z=np.zeros_like(h)
    for (a,b,bias),(da,db,dbias) in zip(core.unpack(theta),core.unpack(d)):
        arg=h@a.T+bias
        one=v@a.T+h@da.T+dbias
        two=z@a.T+2*v@da.T
        t=np.tanh(arg);dt=(1-t*t)*one
        ddt=(1-t*t)*(two-2*t*one*one)
        h,v,z=h+.5*t@b.T,v+.5*(dt@b.T+t@db.T),z+.5*(ddt@b.T+2*dt@db.T)
    return h[:,0],v[:,0],z[:,0]


def second_outcome(theta,d,task):
    nodes,weights=np.polynomial.legendre.leggauss(48);u=(nodes+1)/2;r=.3+1.2*u;w=weights/2
    a=2*np.pi*(np.arange(64)+.5)/64;x=(r[:,None,None]*np.stack([np.cos(a),np.sin(a)],axis=-1)).reshape(-1,2)
    p=np.broadcast_to(sigmoid(3*np.cos(2*np.pi*u))[:,None] if task=='radial' else sigmoid(3*np.cos(2*a))[None,:],(48,64))
    h,v,z=[q.reshape(48,64) for q in second_forward(theta,d,x)]
    def curve(hh,vv,zz,pp):
        s=sigmoid(hh);return s*(1-s)*vv*vv+(s-pp)*zz
    raw=np.sum(w[:,None]*curve(h,v,z,p))/64
    if task=='angular':
        hh,vv,zz=[.5*(q+np.roll(q,32,axis=1)) for q in (h,v,z)]
        fit=np.sum(w[:,None]*curve(hh,vv,zz,p))/64;pair=fit
    else:
        fit=np.sum(w*curve(h.mean(1),v.mean(1),z.mean(1),p[:,0]))
        pair=0.
        for i in range(48):
            hh,vv,zz=[.5*(q[i,:,None]+q[i,None,:]) for q in (h,v,z)]
            pair+=w[i]*curve(hh,vv,zz,p[i,0]).mean()
    return np.array([fit,raw-fit,raw-pair])


def audit_second_order(study,balanced=False):
    root=study/('direction1/balanced_control' if balanced else 'direction1/second_order_followup')
    if not (root/'outcomes.csv').exists():return {'status':'pending second order followup'}
    freeze=json.loads((root/'freeze.json').read_text());commit=json.loads((root/'prediction_commit.json').read_text());start=json.loads((root/'score_start.json').read_text())
    provenance=dict(source_hash=digest(root/('balanced.py' if balanced else 'followup.py'))==freeze['script_sha256'],parent_hash=digest(root.parent/'study.py')==freeze['parent_script_sha256'],
        old_outcomes_hash=digest(root.parent/'old/outcomes.csv')==freeze['old_outcomes_sha256'],protocol_hash=digest(root/'PROTOCOL.md')==freeze['protocol_sha256'],
        predictions_hash=digest(root/'predictions.csv')==commit['predictions_sha256'],states_hash=digest(root/'states.npz')==commit['state_sha256'],
        chronological_freeze_commit_score=datetime.fromisoformat(freeze['timestamp_utc'])<datetime.fromisoformat(commit['timestamp_utc'])<datetime.fromisoformat(start['timestamp_utc']))
    assert all(provenance.values())
    rows=read_csv(root/'outcomes.csv');pred=read_csv(root/'predictions.csv');bank=np.load(root/'states.npz')
    assert len(rows)==len(pred) and all(all(p[k]==r[k] for k in p) for p,r in zip(pred,rows))
    unique={r['row_id']:r for r in rows};root_errors=[];formula=[];second_errors=[];corrected_changes=[];trainsecond=[]
    exacts={}
    for seed in range(1,17):
        for task in ('radial','angular'):
            t=bank[f'{task}_{seed}_initial'];x=bank[f'{task}_{seed}_x'];y=bank[f'{task}_{seed}_y']
            assert np.array_equal(x,bank[f'radial_{seed}_x']) and np.array_equal(t,bank[f'radial_{seed}_initial'])
            for method in ('gd','pullback'):
                d=core.direction(t,x,y,method,.1)[0];q=-core.parameter_gradient(t,x,y)@d;w=d/q
                h,v,z=second_forward(t,w,x);s=sigmoid(h);p=(y+1)/2
                exactb=np.mean(s*(1-s)*v*v+(s-p)*z);exactc=second_outcome(t,w,task)
                exacts[str(seed),task,method]=(exactb,exactc)
    for r in rows:
        delta=float(r['requested_drop']);a=float(r['q1']);b=float(r['loss2']);c=float(r['q2'])
        formula.append(abs(delta*a+.5*delta*delta*(c+a*b)-float(r['second_order'])))
        eb,ec=exacts[r['seed'],r['task'],r['method']];j=('fit','nuisance','view_value').index(r['metric'])
        trainsecond.append(abs(b-eb)/max(1.,abs(eb)));second_errors.append(abs(c-ec[j])/max(1.,abs(ec[j])))
        corrected_changes.append(abs(.5*delta*delta*((c-ec[j])+a*(b-eb))))
    for r in unique.values():
        if r['status']!='ok':continue
        t=bank[r['pre_key']];end=bank[r['end_key']];x=bank[f"{r['task']}_{r['seed']}_x"];y=bank[f"{r['task']}_{r['seed']}_y"]
        root_errors.append(abs(loss(t,x,y)-loss(end,x,y)-float(r['requested_drop'])))
    contrasts=read_csv(root/'method_contrasts.csv');summaries=read_csv(root/'contrast_summary.csv');aggregate_errors=[];primary=[]
    for s in summaries:
        rr=[r for r in contrasts if all(r[k]==s[k] for k in ('task','requested_drop','metric'))]
        actual=np.array([float(r['actual']) for r in rr]);pr=np.array([float(r[s['predictor']]) for r in rr]);er=abs(pr-actual)
        first=abs(np.array([float(r['drop_clock']) for r in rr])-actual);old=abs(np.array([float(r['old_pooled']) for r in rr])-actual)
        aggregate_errors.append(max(abs(er.mean()-float(s['mae'])),abs(np.sqrt(np.mean(er*er))-float(s['rmse'])),abs((first-er).mean()-float(s['advantage_first'])),abs((old-er).mean()-float(s['advantage_old']))))
        if s['metric']=='fit' and s['requested_drop']=='0.02' and s['predictor'] in ('drop_clock','second_order'):
            primary.append(dict(task=s['task'],predictor=s['predictor'],complete_pairs=len(rr),mae=float(er.mean()),advantage_over_first=float((first-er).mean()),
                recorded_advantage_ci95=[float(s['advantage_first_lo']),float(s['advantage_first_hi'])]))
    previous=read_csv(root.parent/'new/seeds.csv');newseeds={int(r['data_seed']) for r in rows};oldseeds={int(r['data_seed']) for r in previous}
    assert newseeds.isdisjoint(oldseeds)
    balance_checks=None
    if balanced:
        regenerated=0;complement_error=0.;counts=[]
        for seed in range(1,17):
            rng=np.random.default_rng(27760100+seed);u=rng.random(128);phi=rng.uniform(0,2*np.pi,128);q=rng.random(128)
            uu=np.r_[u,(u+.5)%1];aa=np.r_[phi,(phi+np.pi/2)%(2*np.pi)];qq=np.r_[q,1-q]
            rr=.3+1.2*uu;xx=np.c_[rr*np.cos(aa),rr*np.sin(aa)]
            for task in ('radial','angular'):
                posterior=sigmoid(3*np.cos(2*np.pi*uu) if task=='radial' else 3*np.cos(2*aa));yy=np.where(qq<posterior,1.,-1.)
                assert np.array_equal(xx,bank[f'{task}_{seed}_x']) and np.array_equal(yy,bank[f'{task}_{seed}_y'])
                assert np.array_equal(yy[128:],-yy[:128]);counts.append(int(np.sum(yy==1)));regenerated+=1
                complement_error=max(complement_error,float(np.max(abs(posterior[:128]+posterior[128:]-1))))
        earlier=read_csv(root.parent/'second_order_followup/predictions.csv');assert newseeds.isdisjoint(int(r['data_seed']) for r in earlier)
        assert counts==[128]*32
        balance_checks=dict(regenerated_training_datasets=regenerated,positive_counts=counts,negative_counts=[256-n for n in counts],
            posterior_companion_sum_minus_one_max_error=complement_error,
            marginal_law_argument='Uniform u and angle shifts preserve each input marginal; complement q and posterior give opposite labels with each correct conditional Bernoulli law.',
            independence_unit='128 iid antithetic pairs per training dataset; not 256 iid observations',
            all_old_baseline_rows_flagged_out_of_design=all(r['old_baseline_design']=='iid_out_of_design' for r in rows))
    return dict(status='checked',provenance=provenance,prediction_rows=len(rows),requested_endpoints=len(unique),
        status_counts=dict(Counter(r['status'] for r in unique.values())),max_actual_drop_error=max(root_errors),
        exact_analytic_second_derivative_directions=len(exacts),max_second_order_formula_error=max(formula),
        max_training_curvature_error_scaled_by_max_one_abs_exact=max(trainsecond),
        max_outcome_curvature_error_scaled_by_max_one_abs_exact=max(second_errors),
        max_prediction_error_from_finite_difference_curvature=max(corrected_changes),
        summary_rows_recomputed=len(aggregate_errors),max_summary_mean_error=max(aggregate_errors),
        new_training_cohort_disjoint=True,primary_fit_contrasts=primary,balance_checks=balance_checks)


def audit_bridge(study,late=False):
    d2=study/'direction2'
    root=d2/('late_bridge_results_selected/bridge_new' if late else 'bridge_new')
    parent=d2/'late_bridge_source_selected/new' if late else study/'direction1/new'
    if not (root/'risks.csv').exists():return {'status':'pending bridge results'}
    commit=json.loads((root/'selector_commit.json').read_text());freeze=json.loads((d2/('late_bridge_freeze.json' if late else 'bridge_freeze.json')).read_text())
    hashes={p:digest(root/p)==h for p,h in commit['npz_hashes'].items()}
    frozen=all(digest(d2/p)==h for p,h in freeze['source_sha256'].items()) if late else digest(d2/'bridge.py')==freeze['script_sha256']
    provenance=dict(frozen_source_matches=frozen,all_selector_state_npz_match=all(hashes.values()),
        parent_predictions_match=digest(parent/'predictions.csv')==commit['parent_prediction_sha256'],parent_states_match=digest(parent/'states.npz')==commit['parent_state_sha256'])
    assert all(provenance.values())
    predictions=read_csv(root/'predictions.csv');risks=read_csv(root/'risks.csv');curves=read_csv(root/'gain_curves.csv');summary=read_csv(root/'summary.csv')
    targets={r['seed']+'_'+r['method']:float(r['target_loss_drop']) for r in predictions}
    states=sorted(root.glob('*_velocity.npz'));roots=[];delta_errors=[];formula_errors=[];summary_errors=[];reconstructed=[]
    for path in states:
        d=np.load(path);theta=d['theta'];post=d['post'];x=d['train_x'];y=d['train_y'];v=d['train_direction'];q=-core.parameter_gradient(theta,x,y)@v
        target=targets[path.name.replace('_velocity.npz','')]
        roots.append(abs(loss(theta,x,y)-loss(post,x,y)-target));delta_errors.append(float(np.max(abs(d['delta']-target*v/q))))
    curveindex=defaultdict(list)
    for r in curves:curveindex[r['pair'],r['model']].append(r)
    for r in risks:
        if r['policy'] not in ('stale','drop_clock_logits','matched_step_logits','first_order_gain','post_recalibrated'):continue
        q=curveindex[r['pair'],r['model']];idx=int(round(float(r['alpha_mean'])*128/(2*np.pi)))-1
        formula_errors.append(abs(float(r['gain'])-float(q[idx]['gain'])))
    for s in summary:
        if s['policy']=='NO_REACHABLE_TARGET':continue
        rr=[r for r in risks if r['model']==s['method'] and r['policy']==s['policy']]
        stale={r['pair']:float(r['risk']) for r in risks if r['model']==s['method'] and r['policy']=='stale'}
        change=np.array([stale[r['pair']]-float(r['risk']) for r in rr])
        summary_errors.append(max(abs(np.mean([float(r['risk']) for r in rr])-float(s['mean_risk'])),abs(change.mean()-float(s['improvement_over_stale']))))
    # Independent JVP and selection reconstruction at three representative reachable branches.
    picked=sorted(set([0,len(states)//2,len(states)-1])) if states else []
    max_jvp=0.;max_gain=0.;angle_mismatch=0;population_checks=[];population_coarse_checks=[];population_refinement=[]
    for i in picked:
        path=states[i];key=path.name.replace('_velocity.npz','');state=np.load(path);stale=np.load(root/(key+'_stale.npz'))
        h=stale['h'];v=state['v'];alpha=stale['phases'][:4,None]+2*np.pi*np.arange(128)/128
        xx=(stale['radii'][:4,None,None]*np.stack([np.cos(alpha),np.sin(alpha)],axis=-1)).reshape(-1,2)
        exact=core.representation_jacobian(state['theta'],xx)[::2]@state['delta'];max_jvp=max(max_jvp,float(np.max(abs(exact.reshape(4,128)-v[:4]))))
        pp=[r for r in predictions if r['seed']+'_'+r['method']==key];ratio=float(pp[0]['actual_matched_scale'])/float(pp[0]['loss_clock_scale'])
        firstcurves=[]
        for policy in ('stale','drop_clock_logits','matched_step_logits','first_order_gain','post_recalibrated'):
            cal=np.load(root/(key+'_'+policy+'.npz'));idx=list(cal['names']).index('exact_global')
            hh=h if policy in ('stale','first_order_gain') else h+v if policy=='drop_clock_logits' else h+ratio*v if policy=='matched_step_logits' else cal['h']
            gain=[]
            for j in range(1,65):
                shifted=np.roll(hh,-j,axis=1);mean=(hh+shifted)/2
                value=(.5*(np.logaddexp(0,hh)+np.logaddexp(0,shifted))-np.logaddexp(0,mean)).mean()
                if policy=='first_order_gain':
                    vv=(v+np.roll(v,-j,axis=1))/2
                    value+=(.5*(sigmoid(h)*v+np.roll(sigmoid(h)*v,-j,axis=1))-sigmoid(mean)*vv).mean()
                gain.append(value)
            gain=np.array(gain);choice=(np.argmax(gain)+1)*2*np.pi/128
            angle_mismatch+=int(np.max(abs(cal['angles'][idx]-choice))>1e-12)
            max_gain=max(max_gain,float(np.max(abs(gain-cal['gain_curves'][idx]))))
            reconstructed.append(dict(branch=key,policy=policy,angle=float(choice)))
        if late:
            from experiment_checks import stable_density
            previous={}
            for radial_nodes,angular_nodes in ((96,768),(192,1536)):
                nodes,weights=np.polynomial.legendre.leggauss(radial_nodes);bound=.86419687750821
                r=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in ((0,bound),(bound,3.))]);dr=np.concatenate([(b-a)/2*weights for a,b in ((0,bound),(bound,3.))])
                aa=2*np.pi*(np.arange(angular_nodes)+.231)/angular_nodes;xx=(r[:,None,None]*np.stack([np.cos(aa),np.sin(aa)],axis=-1)).reshape(-1,2)
                hh=core.representation(state['post'],xx)[:,0].reshape(2*radial_nodes,angular_nodes);w0=.5*dr*stable_density(r,.5);w1=.5*dr*stable_density(r,1.2)
                ss,method=key.split('_')
                for policy in ('raw','pi','stale','drop_clock_logits','post_recalibrated'):
                    if policy=='raw':hpolicy=hh
                    else:
                        ang=np.pi if policy=='pi' else next(float(z['selected_angle']) for z in pp if z['policy']==policy)
                        shift=int(round(ang*angular_nodes/(2*np.pi)));hpolicy=.5*(hh+np.roll(hh,-shift,axis=1))
                    risk=w0@np.logaddexp(0,hpolicy).mean(1)+w1@np.logaddexp(0,-hpolicy).mean(1)
                    record=next(z for z in risks if z['pair']==ss and z['model']==method and z['policy']==policy)
                    if previous.get(policy) is not None:
                        population_checks.append(abs(risk-float(record['risk'])));population_refinement.append(abs(risk-previous[policy]))
                    else:population_coarse_checks.append(abs(risk-float(record['risk'])))
                    previous[policy]=risk
    unreached=read_csv(root/'unreachable.csv') if (root/'unreachable.csv').exists() else []
    assert angle_mismatch==0
    late_checks=None
    if late:
        bank=np.load(parent/'states.npz');regenerated=0;retrained=[]
        for seed in range(1,17):
            rng=np.random.default_rng(621000+seed);y=np.repeat([-1.,1.],128);angle=rng.uniform(0,2*np.pi,256);radius=np.where(y<0,.5,1.2)
            x=np.c_[radius*np.cos(angle),radius*np.sin(angle)]+rng.normal(0,.15,(256,2));order=rng.permutation(256);x=x[order];y=y[order]
            init=core.initialize_parameters(622000+seed)
            assert np.array_equal(x,bank[f'rings_{seed}_x']) and np.array_equal(y,bank[f'rings_{seed}_y']) and np.array_equal(init,bank[f'rings_{seed}_initial'])
            regenerated+=1
            if seed in (1,16):
                t=init
                for _ in range(500):t=t+8*core.direction(t,x,y,'pullback',.1)[0]
                retrained.append(float(np.max(abs(t-bank[f'rings_{seed}_trained']))))
        choice=json.loads((d2/'late_feasibility_choice.json').read_text());feas=read_csv(d2/'late_feasibility_all_roots.csv')
        assert digest(d2/'late_feasibility_all_roots.csv')==choice['root_table_hash']
        counts={}
        for drop in (.005,.001):
            rr={(r['seed'],r['method']):r['status'] for r in feas if float(r['requested_drop'])==drop}
            counts[str(drop)]=sum(rr[str(s),'gd']=='ok' and rr[str(s),'pullback']=='ok' for s in range(1,7))
        assert counts==choice['paired_success_counts']
        late_checks=dict(regenerated_new_train_and_init_datasets=regenerated,retrained_pb500_states=len(retrained),max_retrained_parameter_error=max(retrained),
            verified_old_feasibility_complete_pairs=counts,selected_target=float(choice['selected_drop']),old_pilot_02_preserved=(d2/'late_bridge_pilot02.py').exists() and (d2/'late_bridge_source/old/roots.csv').exists(),
            independent_alternative_population_risks=len(population_checks),max_independent_population_risk_error=max(population_checks),
            preserved_max_coarse_grid_risk_discrepancy=max(population_coarse_checks),max_independent_grid_refinement_change=max(population_refinement))
    primary=[{k:s[k] for k in ('method','policy','n','mean_risk','improvement_over_stale','ci_low','ci_high','wins_over_stale','angle_changes')} for s in summary if s['policy'] in ('stale','drop_clock_logits','first_order_gain','post_recalibrated')]
    return dict(status='checked',cohort='independent PB500 followup' if late else 'shared main PB100 bridge',provenance=provenance,
        reachable_branches=len(states),unreachable_branches=len(unreached),requested_drop_values=sorted(set(targets.values())),failure_statuses=dict(Counter(r['status'] for r in unreached)),
        max_actual_drop_error=max(roots) if roots else None,max_loss_clock_displacement_error=max(delta_errors) if delta_errors else None,
        risk_choice_gain_mapping_error=max(formula_errors) if formula_errors else None,
        summary_rows_recomputed=len(summary_errors),max_summary_mean_error=max(summary_errors) if summary_errors else None,
        representative_reconstructed_selection_policies=len(reconstructed),max_independent_jvp_error=max_jvp,
        max_gain_forecast_curve_error=max_gain,angle_selection_mismatches=angle_mismatch,
        primary_results=primary,all_selected_angles_pi=all(abs(float(r['selected_angle'])-np.pi)<1e-12 for r in predictions),late_cohort_checks=late_checks)
