"""Read-only checks of frozen investigator data using independent arithmetic."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
from audit import core, digest, read_csv, sigmoid, loss, import_file


def i0e(z):
    return np.i0(z)*np.exp(-np.abs(z))


def xlogy(x,y):
    return x*np.log(np.maximum(y,1e-300))


def stable_density(radius, center):
    z=radius*center/.15**2
    return radius/.15**2*np.exp(-.5*((radius-center)/.15)**2)*i0e(z)


def independent_measure(theta,task,nr=48,na=64,v=None):
    phi=2*np.pi*(np.arange(na)+.5)/na
    nodes,weights=np.polynomial.legendre.leggauss(nr//2 if task=='rings' else nr)
    if task=='rings':
        bound=.86419687750821
        r=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in ((0,bound),(bound,3.))])
        dr=np.concatenate([(b-a)/2*weights for a,b in ((0,bound),(bound,3.))])
        w0=.5*dr*stable_density(r,.5);w1=.5*dr*stable_density(r,1.2)
        weight=w0+w1;eta=np.broadcast_to((w1/weight)[:,None],(nr,na))
    else:
        u=(nodes+1)/2;r=.3+1.2*u;weight=weights/2
        eta=np.broadcast_to(sigmoid(3*np.cos(2*np.pi*u))[:,None] if task=='radial' else sigmoid(3*np.cos(2*phi))[None,:],(nr,na))
    x=(r[:,None,None]*np.stack([np.cos(phi),np.sin(phi)],axis=-1)).reshape(-1,2)
    h=core.representation(theta,x)[:,0].reshape(nr,na)
    entropy=(-xlogy(eta,eta)-xlogy(1-eta,1-eta))
    bayes=np.sum(weight[:,None]*entropy)/na
    risk=np.sum(weight[:,None]*(np.logaddexp(0,h)-eta*h))/na
    if task=='angular':
        projected=(h+np.roll(h,na//2,axis=1))/2
        fit=np.sum(weight[:,None]*(np.logaddexp(0,projected)-eta*projected))/na
        two=fit
    else:
        projected=h.mean(1)
        fit=np.sum(weight*(np.logaddexp(0,projected)-eta[:,0]*projected))
        two=sum(weight[i]*(np.logaddexp(0,(h[i,:,None]+h[i,None,:])/2)-eta[i,0]*(h[i,:,None]+h[i,None,:])/2).mean() for i in range(nr))
    q=np.array([fit-bayes,risk-fit,risk-two])
    if v is None:return q
    vel=(core.representation_jacobian(theta,x)[::2]@v).reshape(nr,na)
    drisk=np.sum(weight[:,None]*(sigmoid(h)-eta)*vel)/na
    if task=='angular':
        dv=(vel+np.roll(vel,na//2,axis=1))/2
        dfit=np.sum(weight[:,None]*(sigmoid(projected)-eta)*dv)/na
        dtwo=dfit
    else:
        dfit=np.sum(weight*(sigmoid(projected)-eta[:,0])*vel.mean(1))
        dtwo=sum(weight[i]*((sigmoid((h[i,:,None]+h[i,None,:])/2)-eta[i,0])*(vel[i,:,None]+vel[i,None,:])/2).mean() for i in range(nr))
    return q,np.array([dfit,drisk-dfit,drisk-dtwo])


def audit_direction1(study):
    root=study/'direction1';out=root/'new'
    if not (out/'outcomes.csv').exists():return {'status':'pending new outcomes'}
    preds=read_csv(out/'predictions.csv');rows=read_csv(out/'outcomes.csv');roots=read_csv(out/'roots.csv');seeds=read_csv(out/'seeds.csv')
    bank=np.load(out/'states.npz');commit=json.loads((out/'prediction_commit.json').read_text());freeze=json.loads((root/'freeze.json').read_text())
    start=json.loads((out/'score_start.json').read_text())
    checks={
        'source_matches_freeze':digest(root/'study.py')==freeze['sha256']['study.py'],
        'predictions_match_preoutcome_commit':digest(out/'predictions.csv')==commit['prediction_sha256'],
        'states_match_preoutcome_commit':digest(out/'states.npz')==commit['state_sha256'],
        'roots_match_preoutcome_commit':digest(out/'roots.csv')==commit['root_sha256'],
        'commit_precedes_scoring':datetime.fromisoformat(commit['committed_utc'])<datetime.fromisoformat(start['timestamp_utc']),
        'outcomes_preserve_all_prediction_rows':len(preds)==len(rows) and all(all(p[k]==r[k] for k in p) for p,r in zip(preds,rows)),
        'frozen_core_matches_repository':digest(root/'model_core.py')==digest(Path(core.__file__)),
    }
    assert all(checks.values()),checks
    unique={r['row_id']:r for r in rows}
    root_errors=[];hash_errors=0;ray_errors=[];endpoint_checks=0
    for r in unique.values():
        t=bank[r['pre_key']];x=bank[f"{r['task']}_{r['seed']}_x"];y=bank[f"{r['task']}_{r['seed']}_y"]
        import hashlib
        hash_errors+=hashlib.sha256(t.tobytes()).hexdigest()!=r['pre_sha256']
        if r['status']!='ok':continue
        end=bank[r['end_key']];actual=loss(t,x,y)-loss(end,x,y)
        hash_errors+=hashlib.sha256(end.tobytes()).hexdigest()!=r['end_sha256']
        root_errors.append(abs(actual-float(r['requested_drop'])))
        endpoint_checks+=1
        if r['experiment']=='single':
            v=core.direction(t,x,y,r['method'],.1)[0]
            ray_errors.append(np.max(abs((t+float(r['alpha'])*v)-end)))
    assert hash_errors==0
    pair_checks=[]
    for seed in range(1,17):
        pair_checks.append(np.array_equal(bank[f'radial_{seed}_x'],bank[f'angular_{seed}_x']))
        pair_checks.append(np.array_equal(bank[f'radial_{seed}_initial'],bank[f'angular_{seed}_initial']))
    assert all(pair_checks)
    oldseeds=read_csv(root/'old'/'seeds.csv')
    disjoint=set(int(r['data_seed']) for r in oldseeds).isdisjoint(int(r['data_seed']) for r in seeds)
    assert disjoint
    # Distribution-level Bayes parity independently on a dense common uniform grid.
    u=(np.arange(131072)+.5)/131072
    pr=sigmoid(3*np.cos(2*np.pi*u));pa=sigmoid(3*np.cos(4*np.pi*u))
    bayes=dict(radial_class_fraction=float(pr.mean()),angular_class_fraction=float(pa.mean()),
        radial_bayes_log_loss=float((-xlogy(pr,pr)-xlogy(1-pr,1-pr)).mean()),
        angular_bayes_log_loss=float((-xlogy(pa,pa)-xlogy(1-pa,1-pa)).mean()),
        radial_bayes_error=float(np.minimum(pr,1-pr).mean()),angular_bayes_error=float(np.minimum(pa,1-pa).mean()),
        angular_posterior_pi_invariance_max_error=float(np.max(abs(pa-sigmoid(3*np.cos(4*np.pi*u+2*np.pi))))),
        angular_posterior_quarter_turn_max_violation=float(np.max(abs(pa-sigmoid(3*np.cos(4*np.pi*u+np.pi))))))
    derivative=[];fine=[]
    index={(r['seed'],r['task'],r['stage'],r['method'],r['experiment'],r['requested_drop'],r['metric']):r for r in rows if r['experiment']=='single'}
    for seed in (1,16):
        for task in ('rings','radial','angular'):
            x=bank[f'{task}_{seed}_x'];y=bank[f'{task}_{seed}_y']
            for stage in ('initial','trained'):
                t=bank[f'{task}_{seed}_{stage}']
                for method in ('gd','pullback'):
                    v=core.direction(t,x,y,method,.1)[0]
                    q,deriv=independent_measure(t,task,v=v)
                    eps=1e-5/np.linalg.norm(v)
                    fd=(independent_measure(t+eps*v,task)-independent_measure(t-eps*v,task))/(2*eps)
                    rr=[index[str(seed),task,stage,method,'single','1e-05',m] for m in ('fit','nuisance','view_value')]
                    saved=np.array([float(r['outcome_derivative']) for r in rr]);clock=np.array([float(r['drop_clock']) for r in rr])
                    sl=-core.parameter_gradient(t,x,y)@v
                    derivative.append(dict(seed=seed,task=task,stage=stage,method=method,
                        saved_derivative_error=float(np.max(abs(saved-deriv))),
                        normalized_finite_difference_error=float(np.max(abs(fd-deriv))/np.linalg.norm(v)),
                        clock_formula_error=float(np.max(abs(clock-1e-5*deriv/sl)))))
                    if stage=='trained' and rr[0]['status']=='ok':
                        end=bank[rr[0]['end_key']]
                        exact=independent_measure(end,task,192,256)-independent_measure(t,task,192,256)
                        coarse=independent_measure(end,task,48,64)-independent_measure(t,task,48,64)
                        reported=np.array([float(r['actual_change']) for r in rr])
                        fine.append(dict(seed=seed,task=task,method=method,max_change_error=float(np.max(abs(exact-reported))),
                            coarse_to_fine_paired_change_error=float(np.max(abs(exact-coarse))),
                            sign_disagreements=int(np.sum(np.sign(exact)!=np.sign(reported)))))
    aggregate=[]
    if (root/'summary.csv').exists():
        summaries=read_csv(root/'summary.csv')
        for s in summaries:
            subset=[r for r in rows if r['status']=='ok' and all(r[k]==s[k] for k in ('task','stage','experiment','horizon','metric')) and (s['drop']=='path' or r['requested_drop']==s['drop'])]
            byseed=defaultdict(list)
            for r in subset:
                a=float(r['actual_change']);err=abs(a-float(r[s['predictor']]))
                byseed[r['seed']].append([err,err*err,abs(a)-err,abs(a-float(r['old_pooled']))-err])
            z=np.array([np.mean(v,axis=0) for v in byseed.values()])
            target=np.array([z[:,0].mean(),np.sqrt(z[:,1].mean()),z[:,2].mean(),z[:,3].mean()])
            saved=np.array([float(s[k]) for k in ('mae','rmse','mae_advantage_zero','mae_advantage_old')])
            aggregate.append(float(np.max(abs(target-saved))))
    return dict(status='checked',provenance=checks,source_sha256=digest(root/'study.py'),
        prediction_rows=len(rows),unique_requested_cases=len(unique),successful_endpoint_loss_recomputations=endpoint_checks,
        failure_cases=dict(Counter(r['status'] for r in unique.values() if r['status']!='ok')),
        preserved_failed_attempt_directories=[p.name for p in root.glob('*attempt*') if p.is_dir()],
        max_actual_drop_error=float(max(root_errors)),max_saved_single_ray_endpoint_error=float(max(ray_errors)),
        saved_state_hash_errors=hash_errors,paired_synthetic_x_and_initialization_checks=len(pair_checks),
        old_new_data_seeds_disjoint=disjoint,bayes_and_symmetry=bayes,derivative_checks=derivative,
        max_saved_derivative_error=max(r['saved_derivative_error'] for r in derivative),
        max_normalized_finite_difference_error=max(r['normalized_finite_difference_error'] for r in derivative),
        max_clock_formula_error=max(r['clock_formula_error'] for r in derivative),
        common_fine_grid_tiny_drop_checks=fine,max_tiny_drop_fine_grid_change_error=max(r['max_change_error'] for r in fine),
        max_tiny_drop_coarse_to_fine_paired_change_error=max(r['coarse_to_fine_paired_change_error'] for r in fine),
        tiny_drop_sign_disagreements=sum(r['sign_disagreements'] for r in fine),
        aggregate_rows_recomputed=len(aggregate),max_aggregate_error=max(aggregate) if aggregate else None)


def audit_direction2(study):
    root=study/'direction2';out=root/'fresh_results'
    if not (out/'risks.csv').exists():return {'status':'pending fresh risks'}
    cfg=json.loads((root/'protocol.json').read_text());freeze=json.loads((root/'freeze.json').read_text());frozen=json.loads((out/'selectors_frozen.json').read_text())
    sourcepaths={p:root/p for p in cfg['scripts']}
    sourcepaths.update({'generator.py':study/'legacy/budgeted-rotation-final/generator.py',
        'polar.py':study/'legacy/rings-mechanism-20261120/polar.py','core.py':Path(core.__file__)})
    source={p:digest(sourcepaths[p])==h for p,h in cfg['scripts'].items()}
    checks=dict(source_matches_freeze=all(source.values()),protocol_matches_freeze=digest(root/'protocol.json')==freeze['protocol_sha256'],
        model_bank_matches_freeze=digest(out/'models.npz')==frozen['state_bank_hash'],
        all_selectors_match_freeze=all(digest(out/p)==h for p,h in frozen['selector_hashes'].items()))
    assert all(checks.values()),checks
    bank=np.load(out/'models.npz');rows=read_csv(out/'risks.csv');metadata=read_csv(out/'training_and_calibration.csv')
    maxgain=0.;maxpower=0.;selection_mismatches=0;maxpred=0.;count=0
    calibrations={}
    for path in sorted(out.glob('*_selector.npz')):
        d=np.load(path);h=d['h'];r=d['radii'];edges=d['edges'];names=list(d['names']);na=h.shape[1]
        angle=2*np.pi*np.arange(1,na//2+1)/na;membership=np.searchsorted(edges[1:-1],r,side='right');counts=np.bincount(membership,minlength=8)
        # Independent full FFT rather than investigator rFFT code.
        ft=np.fft.fft(h,axis=1)/na;power=np.abs(ft[:,1:na//2+1])**2+np.abs(ft[:,np.arange(na-1,na//2-1,-1)])**2;power[:,-1]/=2
        maxpower=max(maxpower,float(np.max(abs(power-d['power']))))
        curves=np.array([(.5*(np.logaddexp(0,h)+np.logaddexp(0,np.roll(h,-j,axis=1)))-np.logaddexp(0,.5*(h+np.roll(h,-j,axis=1)))).mean(1) for j in range(1,na//2+1)]).T
        curv=sigmoid(h.mean(1))*(1-sigmoid(h.mean(1)));removal=np.sin(np.arange(1,na//2+1)[:,None]*angle[None,:]/2)**2
        for idx,name in enumerate(names):
            stem,policy=name.rsplit('_',1)
            if stem=='exact':score=curves
            else:
                k=int(stem.split('K')[-1]);score=power[:,:k]@removal[:k]
                if stem.startswith('fourier'):score=.5*curv[:,None]*score
            globalcurve=score.mean(0);radialcurve=np.array([score[membership==b].mean(0) for b in range(8)])
            choice=np.full(8,np.argmax(globalcurve)) if policy=='global' else np.argmax(radialcurve,axis=1)
            selection_mismatches+=int(np.sum(angle[choice]!=d['angles'][idx]))
            maxgain=max(maxgain,float(np.max(abs(globalcurve-d['gain_curves'][idx]))),float(np.max(abs(radialcurve-d['radial_gain_curves'][idx]))))
            pred=(counts*radialcurve[np.arange(8),choice]).sum()/len(r)
            maxpred=max(maxpred,float(abs(pred-d['predicted_gains'][idx])))
        calibrations[path.name.replace('_selector.npz','')]=d;count+=1
    # Verify independent calibration draws and orbit logits on six representative models.
    logit_errors=[];sym_errors=[];risk_errors=[];iid_errors=[];budget=[]
    row_index={(r['pair'],r['model'],r['policy']):r for r in rows}
    iid=read_csv(out/'independent_iid_test.csv') if (out/'independent_iid_test.csv').exists() else []
    iid_index={(r['pair'],r['model'],r['policy']):r for r in iid}
    sys.path.insert(0,str(root));selector=import_file(root/'rotation_selectors.py','audit_rotation_selectors')
    for pair in (1,16):
        for model in ('gd','pb','hybrid'):
            key=f'{pair}_{model}';d=calibrations[key];theta=bank[key]
            angles=d['phases'][:5,None]+2*np.pi*np.arange(128)/128
            x=(d['radii'][:5,None,None]*np.stack([np.cos(angles),np.sin(angles)],axis=-1)).reshape(-1,2)
            logit_errors.append(float(np.max(abs(core.representation(theta,x)[:,0].reshape(5,128)-d['h'][:5]))))
            # A differently phased, finer radial/angle grid than reported calculation.
            bounds=np.unique(np.r_[0.,d['edges'][1:-1],.86419687750821,3.]);nodes,weights=np.polynomial.legendre.leggauss(80)
            r=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in zip(bounds[:-1],bounds[1:])]);dr=np.concatenate([(b-a)/2*weights for a,b in zip(bounds[:-1],bounds[1:])])
            phi=2*np.pi*(np.arange(768)+.173)/768;x=(r[:,None,None]*np.stack([np.cos(phi),np.sin(phi)],axis=-1)).reshape(-1,2)
            h=core.representation(theta,x)[:,0].reshape(len(r),768);w0=.5*dr*stable_density(r,.5);w1=.5*dr*stable_density(r,1.2)
            for name in ('raw','pi','exact_global','exact_radial','fourier_K8_global'):
                if name=='raw':hh=h
                else:
                    alpha=np.full(len(r),np.pi) if name=='pi' else d['angles'][list(d['names']).index(name),np.searchsorted(d['edges'][1:-1],r,side='right')]
                    shift=np.rint(alpha*768/(2*np.pi)).astype(int)
                    hh=.5*(h+np.take_along_axis(h,(np.arange(768)[None,:]+shift[:,None])%768,axis=1))
                    sym_errors.append(float(np.max(abs(hh.mean(1)-h.mean(1)))))
                risk=w0@np.logaddexp(0,hh).mean(1)+w1@np.logaddexp(0,-hh).mean(1)
                risk_errors.append(dict(pair=pair,model=model,policy=name,error=float(abs(risk-float(row_index[str(pair),model,name]['risk'])))))
            rng=np.random.default_rng(614000+pair);n=32768;y=rng.integers(0,2,n);rr=np.where(y==0,.5,1.2);aa=rng.uniform(0,2*np.pi,n);xx=rr[:,None]*np.c_[np.cos(aa),np.sin(aa)]+rng.normal(0,.15,(n,2));rad=np.linalg.norm(xx,axis=1)
            eta=stable_density(rad,1.2)/(stable_density(rad,.5)+stable_density(rad,1.2));hh=core.representation(theta,xx)[:,0]
            for name in ('raw','pi','exact_global','exact_radial'):
                if name=='raw':pred=hh
                else:
                    alpha=np.full(n,np.pi) if name=='pi' else d['angles'][list(d['names']).index(name),np.searchsorted(d['edges'][1:-1],rad,side='right')]
                    c,s=np.cos(alpha),np.sin(alpha);rot=np.c_[c*xx[:,0]-s*xx[:,1],s*xx[:,0]+c*xx[:,1]];pred=.5*(hh+core.representation(theta,rot)[:,0])
                if iid:
                    rr=iid_index[str(pair),model,name]
                    iid_errors.append(max(abs((np.logaddexp(0,pred)-eta*pred).mean()-float(rr['expected_risk'])),abs((np.logaddexp(0,pred)-y*pred).mean()-float(rr['sampled_label_risk']))))
            # Execute deployment with an observation-only wrapper to count forwards.
            obj=selector.RotationSelection(list(d['names']),d['edges'],d['angles'],d['predicted_gains'],d['gain_curves'],d['radial_gain_curves'],d['radii'],d['phases'],d['h'],d['power'],float(d['calibration_seconds']),int(d['forward_examples']),int(d['velocity_examples']))
            calls=[];original=selector.core.representation
            def counted(t,z):calls.append(len(z));return original(t,z)
            selector.core.representation=counted
            try:selector.deploy(theta,xx[:257],obj,'exact_radial')
            finally:selector.core.representation=original
            budget.append(calls)
    aggregates=[]
    for model in ('gd','pb','hybrid'):
        for policy in ('pi','random_alpha','fourier_K8_global','exact_global','exact_radial'):
            rr=[r for r in rows if r['model']==model and r['policy']==policy]
            gain=np.array([float(r['gain']) for r in rr]);p=np.array([float(row_index[r['pair'],model,'pi']['gain']) for r in rr]);diff=gain-p
            rng=np.random.default_rng(946001);draws=rng.integers(0,len(diff),(10000,len(diff)));ci=np.quantile(diff[draws].mean(1),[.025,.975])
            aggregates.append(dict(model=model,policy=policy,n=len(rr),risk=float(np.mean([float(r['risk']) for r in rr])),gain=float(gain.mean()),gain_over_pi=float(diff.mean()),gain_over_pi_ci95=ci.tolist(),wins_over_pi=int((diff>0).sum())))
    seeds={role:{cfg[f'{role}_seed_base']+p for p in range(1,17)} for role in ('train','init','cal','test')}
    disjoint=all(not seeds[a]&seeds[b] for i,a in enumerate(seeds) for b in list(seeds)[i+1:])
    assert selection_mismatches==0 and disjoint and all(c==[257,257] for c in budget)
    return dict(status='checked',provenance=checks,script_sha256={p:digest(sourcepaths[p]) for p in cfg['scripts']},
        models=48,calibrations_recomputed=count,policies_per_calibration=len(next(iter(calibrations.values()))['names']),
        max_full_fft_power_error=maxpower,max_selector_curve_error=maxgain,max_selector_prediction_error=maxpred,
        selected_bin_angle_mismatches=selection_mismatches,max_recomputed_calibration_logit_error=max(logit_errors),
        representative_population_risk_checks=risk_errors,max_independent_population_risk_error=max(r['error'] for r in risk_errors),
        max_orbit_mean_preservation_error=max(sym_errors),representative_iid_risks_rebuilt=len(iid_errors),max_iid_risk_error=max(iid_errors) if iid_errors else None,
        actual_deployment_call_sizes=budget,calibration_forwards_per_model=[int(r['calibration_forward_examples']) for r in metadata],
        training_initialization_calibration_test_seed_ranges_disjoint=disjoint,
        per_seed_primary_aggregates=aggregates)


def audit_iid_supplement(study):
    out=study/'iid_results'
    if not (out/'by_seed.csv').exists():return {'status':'pending iid supplement'}
    protocol=json.loads((study/'IID_PROTOCOL.json').read_text());cfg=protocol['config']
    checks={'source_matches_frozen':digest(study/'iid_validation.py')==protocol['script_sha256'],
        'protocol_records_no_new_outcomes_at_freeze':not protocol['new_outcomes_present']}
    assert all(checks.values())
    rows=read_csv(out/'by_seed.csv');quad=read_csv(study/'direction1/new/outcomes.csv');bank=np.load(study/'direction1/new/states.npz')
    parts=defaultdict(dict)
    for r in quad:parts[r['row_id']][r['metric']]=r
    pointwise={};errors=[];hash_checks=[];comparison=[];status=Counter();pairedx={}
    import hashlib
    for row in rows:
        seed=int(row['seed']);task=row['task'];n=int(row['n']);rng=np.random.default_rng(cfg['x_seed_base']+seed)
        if task=='rings':
            radii=rng.choice([.5,1.2],n);a=rng.uniform(0,2*np.pi,n)
            x=radii[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0,.15,(n,2));r=np.linalg.norm(x,axis=1)
            eta=stable_density(r,1.2)/(stable_density(r,.5)+stable_density(r,1.2))
        else:
            u=rng.random(n);a=rng.uniform(0,2*np.pi,n);r=.3+1.2*u;x=r[:,None]*np.c_[np.cos(a),np.sin(a)]
            eta=sigmoid(3*np.cos(2*np.pi*u) if task=='radial' else 3*np.cos(2*a))
            pairedx[task,seed]=hashlib.sha256(x.tobytes()).hexdigest()
        y=(np.random.default_rng(cfg['label_seed_base']+seed).random(n)<eta).astype(float)
        hash_checks.extend([hashlib.sha256(x.tobytes()).hexdigest()==row['x_sha256'],hashlib.sha256(y.tobytes()).hexdigest()==row['y_sha256']])
        p=parts[row['row_id']];pre=p['fit'];assert pre['status']==row['status'];status[row['status']]+=1
        if row['status']!='ok':continue
        before=core.representation(bank[pre['pre_key']],x)[:,0];after=core.representation(bank[pre['end_key']],x)[:,0]
        sf=np.logaddexp(0,after)-np.logaddexp(0,before);expected=sf-eta*(after-before);labeled=sf-y*(after-before)
        values=np.array([expected.mean(),expected.std(ddof=1)/np.sqrt(n),labeled.mean(),labeled.std(ddof=1)/np.sqrt(n)])
        saved=np.array([float(row[k]) for k in ('expected_risk_change','expected_change_se','label_risk_change','label_change_se')])
        errors.append(float(np.max(abs(values-saved))))
        target=float(p['fit']['actual_change'])+float(p['nuisance']['actual_change'])
        comparison.append(dict(row_id=row['row_id'],task=task,seed=seed,method=row['method'],quadrature_delta=target,iid_delta=float(expected.mean()),
            difference=float(expected.mean()-target),difference_standard_errors=float((expected.mean()-target)/(expected.std(ddof=1)/np.sqrt(n)))))
        pointwise[task,seed,row['method']]=(expected,labeled)
    contrast_rows=read_csv(out/'paired_contrasts.csv');contrast_error=[];summaries=[]
    for row in contrast_rows:
        task=row['task'];seed=int(row['seed']);a=pointwise[task,seed,'pullback'];b=pointwise[task,seed,'gd']
        e=a[0]-b[0];l=a[1]-b[1]
        contrast_error.append(max(abs(e.mean()-float(row['expected_pb_minus_gd'])),abs(l.mean()-float(row['labeled_pb_minus_gd'])),abs(e.std(ddof=1)/np.sqrt(len(e))-float(row['expected_se']))))
    for task in cfg['tasks']:
        rr=[r for r in contrast_rows if r['task']==task]
        if not rr:summaries.append(dict(task=task,complete_pairs=0));continue
        value=np.array([float(r['expected_pb_minus_gd']) for r in rr]);pred=np.array([float(r['clock_pb_minus_gd']) for r in rr]);old=np.array([float(r['pooled_pb_minus_gd']) for r in rr])
        summaries.append(dict(task=task,complete_pairs=len(rr),expected_mean=float(value.mean()),clock_mae=float(abs(pred-value).mean()),
            old_pooled_mae=float(abs(old-value).mean()),zero_mae=float(abs(value).mean()),clock_correct_signs=int((value*pred>0).sum()),pb_better_count=int((value<0).sum())))
    assert all(hash_checks) and all(pairedx['radial',s]==pairedx['angular',s] for s in range(1,17))
    return dict(status='checked',provenance=checks,requested_branches=len(rows),coverage=dict(status),
        input_and_label_hash_checks=len(hash_checks),max_independent_iid_recalculation_error=max(errors),
        complete_pairs_recomputed=len(contrast_rows),max_paired_contrast_recalculation_error=max(contrast_error),
        matched_synthetic_evaluation_x_checks=16,paired_summary=summaries,
        quadrature_vs_iid=comparison,max_absolute_standardized_iid_quadrature_difference=max(abs(r['difference_standard_errors']) for r in comparison),
        iid_quadrature_differences_within_1_96_se=sum(abs(r['difference_standard_errors'])<=1.96 for r in comparison))
