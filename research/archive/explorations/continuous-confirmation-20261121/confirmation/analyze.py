"""Prespecified paired risk and end-to-end calibration-amortization reporting."""
import math,json
from pathlib import Path
import numpy as np
from study import HERE,CONFIG,POLICIES,csvin,csvout,dump,sha,now


def summarize():
    out=HERE/'results';rows=csvin(out/'per_seed.csv');timing=csvin(out/'latency.csv');cal=csvin(HERE/'calibration/costs.csv')
    index={(r['geometry'],r['model'],int(r['budget']),r['policy'],int(r['pair'])):r for r in rows}
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);draw=rng.integers(0,16,(20000,16))
    summaries=[];contrasts=[];latency=[];costs=[];cost_summary=[]
    comparisons=[('confidence_radial','confidence_pi','primary_angle_addition'),
        ('confidence_radial','confidence_global','radius_specific_vs_global'),
        ('confidence_global','confidence_pi','global_angle_addition'),
        ('confidence_radial','legacy_iid_folded_pi','strong_legacy_gate'),
        ('confidence_radial','joint','confidence_vs_joint'),
        ('confidence_radial','raw','improvement_over_raw'),
        ('confidence_radial','full_pi','different_budget_full_pi'),
        ('confidence_radial','full_radial','different_budget_full_radial'),
        ('joint','confidence_pi','joint_vs_confidence_halfturn')]
    for geometry in CONFIG['geometries']:
        for model in CONFIG['models']:
            for budget in CONFIG['budgets']:
                for policy in POLICIES:
                    for metric in ('expected_risk','sampled_risk'):
                        values=np.array([float(index[geometry,model,budget,policy,p][metric]) for p in range(1,17)])
                        lo,hi=np.quantile(values[draw].mean(1),[.025,.975])
                        summaries.append(dict(geometry=geometry,model=model,budget=budget,policy=policy,metric=metric,
                            mean=values.mean(),ci_low=lo,ci_high=hi,n_pairs=16))
                    seed_times=[]
                    for pair in range(1,17):
                        t=[float(r['ms_per_batch256']) for r in timing if r['geometry']==geometry and r['model']==model and int(r['budget'])==budget and r['policy']==policy and int(r['pair'])==pair]
                        seed_times.append(float(np.median(t)))
                    latency.append(dict(geometry=geometry,model=model,budget=budget,policy=policy,
                        median_ms_per_batch256=float(np.median(seed_times)),
                        p10_seed_ms=float(np.quantile(seed_times,.1)),p90_seed_ms=float(np.quantile(seed_times,.9))))
                for policy,reference,kind in comparisons:
                    for metric in ('expected_risk','sampled_risk'):
                        values=np.array([float(index[geometry,model,budget,policy,p][metric])-float(index[geometry,model,budget,reference,p][metric]) for p in range(1,17)])
                        boot=values[draw].mean(1);lo,hi=np.quantile(boot,[.025,.975]);flo,fhi=np.quantile(boot,[.0125,.9875])
                        contrasts.append(dict(geometry=geometry,model=model,budget=budget,policy=policy,reference=reference,
                            comparison=kind,metric=metric,mean_difference=values.mean(),ci95_low=lo,ci95_high=hi,
                            ci97_5_low=flo,ci97_5_high=fhi,lower_risk_pairs=int((values < -1e-12).sum()),
                            higher_risk_pairs=int((values>1e-12).sum()),ties=int((abs(values)<=1e-12).sum()),n_pairs=16,
                            primary=int(model=='pb' and budget==8192 and kind=='primary_angle_addition' and metric=='expected_risk')))
                for pair in range(1,17):
                    med={policy:float(np.median([float(r['ms_per_batch256']) for r in timing if r['geometry']==geometry and r['model']==model and int(r['budget'])==budget and r['policy']==policy and int(r['pair'])==pair])) for policy in POLICIES}
                    orbit=float(next(r['calibration_seconds'] for r in cal if r['geometry']==geometry and r['model']==model and int(r['budget'])==budget and int(r['pair'])==pair and r['kind']=='orbit'))*1000
                    iid=float(next(r['calibration_seconds'] for r in cal if r['geometry']==geometry and r['model']==model and int(r['budget'])==budget and int(r['pair'])==pair and r['kind']=='legacy_iid'))*1000
                    for reference in ('confidence_pi','full_pi','legacy_iid_folded_pi','full_radial'):
                        refcal=0 if reference in ('confidence_pi','full_pi') else (iid if reference=='legacy_iid_folded_pi' else orbit)
                        setup=orbit-refcal;delta=med[reference]-med['confidence_radial']
                        if setup<=0 and delta>=0:latency_batches=0
                        elif delta>0:latency_batches=math.ceil(max(setup,0)/delta)
                        else:latency_batches=None
                        # Calibration cost and deployment savings in model-example evaluations.
                        setup_calls=budget-(0 if reference in ('confidence_pi','full_pi') else budget)
                        save_calls=192 if reference.startswith('full_') else 0
                        count_batches=math.ceil(setup_calls/save_calls) if save_calls>0 else (0 if setup_calls<=0 else None)
                        costs.append(dict(geometry=geometry,model=model,pair=pair,budget=budget,
                            policy='confidence_radial',reference=reference,calibration_ms=orbit,reference_calibration_ms=refcal,
                            policy_ms_per_batch256=med['confidence_radial'],reference_ms_per_batch256=med[reference],
                            deployment_ms_saved_per_batch=delta,calibration_ms_difference=setup,
                            latency_break_even_batches='' if latency_batches is None else latency_batches,
                            latency_break_even_inputs='' if latency_batches is None else 256*latency_batches,
                            latency_break_even_finite=int(latency_batches is not None),
                            forward_break_even_batches='' if count_batches is None else count_batches,
                            forward_break_even_inputs='' if count_batches is None else 256*count_batches,
                            calibration_forward_cost=budget,deploy_forward_calls_per_input=1.25,
                            calibration_runtime_10pct_inputs=256*math.ceil(orbit/(.1*med['confidence_radial'])),
                            calibration_runtime_1pct_inputs=256*math.ceil(orbit/(.01*med['confidence_radial'])),
                            calibration_forwards_10pct_inputs=256*math.ceil(budget/(.1*320)),
                            calibration_forwards_1pct_inputs=256*math.ceil(budget/(.01*320))))
    for geometry in CONFIG['geometries']:
        for model in CONFIG['models']:
            for budget in CONFIG['budgets']:
                for reference in ('confidence_pi','full_pi','legacy_iid_folded_pi','full_radial'):
                    q=[r for r in costs if r['geometry']==geometry and r['model']==model and r['budget']==budget and r['reference']==reference]
                    finite=[r['latency_break_even_inputs'] for r in q if r['latency_break_even_finite']]
                    cost_summary.append(dict(geometry=geometry,model=model,budget=budget,reference=reference,
                        median_calibration_ms=float(np.median([r['calibration_ms'] for r in q])),
                        median_calibration_ms_difference=float(np.median([r['calibration_ms_difference'] for r in q])),
                        median_policy_ms_per_batch256=float(np.median([r['policy_ms_per_batch256'] for r in q])),
                        median_reference_ms_per_batch256=float(np.median([r['reference_ms_per_batch256'] for r in q])),
                        finite_latency_break_even_models=len(finite),no_finite_latency_break_even_models=16-len(finite),
                        median_latency_break_even_inputs_conditional_on_finite=float(np.median(finite)) if finite else '',
                        median_runtime_10pct_inputs=float(np.median([r['calibration_runtime_10pct_inputs'] for r in q])),
                        median_runtime_1pct_inputs=float(np.median([r['calibration_runtime_1pct_inputs'] for r in q])),
                        forward_break_even_inputs=q[0]['forward_break_even_inputs'],
                        forwards_10pct_inputs=q[0]['calibration_forwards_10pct_inputs'],
                        forwards_1pct_inputs=q[0]['calibration_forwards_1pct_inputs']))
    csvout(out/'summary.csv',summaries);csvout(out/'paired_contrasts.csv',contrasts)
    csvout(out/'latency_summary.csv',latency);csvout(out/'amortization_by_seed.csv',costs);csvout(out/'amortization_summary.csv',cost_summary)
    return summaries,contrasts,latency


def figures(summary,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.8),sharey=True)
    names=['confidence_pi','confidence_global','confidence_radial','legacy_iid_folded_pi','joint']
    labels=['Confidence + 180°','Confidence + global','Confidence + radius','Legacy folded + 180°','Joint radius policy']
    for ax,geometry,title in zip(axes,CONFIG['geometries'],('Original rings','Known ellipse symmetry')):
        rows=[next(r for r in summary if r['geometry']==geometry and r['model']=='pb' and r['budget']==8192 and r['policy']==p and r['metric']=='expected_risk') for p in names]
        val=[r['mean'] for r in rows]
        ax.bar(range(len(names)),val,color=['#879caf','#4d88a7','#185c7d','#bd936b','#b8c1c8'])
        ax.set_xticks(range(len(names)),labels,rotation=32,ha='right');ax.set_title(title)
    axes[0].set_ylabel('Mean held-out log loss (lower is better)')
    fig.suptitle('Independent confirmation: 16 pullback models per geometry, 64 extra views per batch',y=1.03)
    fig.tight_layout();fig.savefig(HERE/'results/confirmation_risks.png',dpi=200,bbox_inches='tight');fig.savefig(HERE/'results/confirmation_risks.svg',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.8,4))
    x=[];labels=[];val=[];err=[]
    for geometry in CONFIG['geometries']:
        for model in CONFIG['models']:
            r=next(r for r in contrasts if r['geometry']==geometry and r['model']==model and r['budget']==8192 and r['comparison']=='primary_angle_addition' and r['metric']=='expected_risk')
            x.append(len(x));labels.append(geometry+' / '+model);val.append(-r['mean_difference']*1000)
            err.append([(r['ci95_high']-r['mean_difference'])*1000,(r['mean_difference']-r['ci95_low'])*1000])
    ax.errorbar(x,val,yerr=np.array(err).T,fmt='o',color='#185c7d',capsize=4)
    ax.axhline(0,color='gray',lw=.8);ax.set_xticks(x,labels,rotation=25,ha='right')
    ax.set_ylabel('Improvement over confidence + 180°\n(milli-nats per input)');ax.set_title('Calibrated radius angles with the same confidence mask')
    fig.tight_layout();fig.savefig(HERE/'results/confirmation_contrasts.png',dpi=200);fig.savefig(HERE/'results/confirmation_contrasts.svg');plt.close(fig)


if __name__=='__main__':
    s,c,l=summarize();figures(s,c)
    print(json.dumps([r for r in c if r['primary']],indent=2))
