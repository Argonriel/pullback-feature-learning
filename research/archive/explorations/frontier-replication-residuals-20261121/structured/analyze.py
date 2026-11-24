"""Output-only summaries, cost/statistics ledgers, and readable report figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,ci,co,jo,now,sha
from math_utils import LAWS,EPS,LOOKS,FAMILY,MOMENT_EVENTS

def main():
    curves=ci(HERE/'certificate_curves.csv');truth=ci(HERE/'stress_results.csv');moments=ci(HERE/'moment_bounds.csv')
    evidence=json.loads((HERE/'EVIDENCE_COMMIT.json').read_text());summary=[]
    groups=sorted({(int(r['n_batches']),int(r['law_index']),r['kind'],r['scenario'],float(r['epsilon']),float(r['true_defect'])) for r in curves})
    for n,j,kind,scenario,eps,defect in groups:
        rows=[r for r in curves if int(r['n_batches'])==n and int(r['law_index'])==j and r['scenario']==scenario]
        result=dict(n_batches=n,law=LAWS[j]['name'],kind=kind,scenario=scenario,epsilon=eps,true_defect=defect,assumptions_covered=LAWS[j]['covered'],lines=len(rows))
        for key in ['known_lower','estimated_lower','old_global_lower']:
            values=np.array([float(r[key]) for r in rows]);result[key+'_mean']=float(values.mean());result[key+'_min']=float(values.min());result[key+'_max']=float(values.max())
            result[key+'_positive_values']=int((values>0).sum());result[key+'_certified_positive']=int((values>0).sum()) if LAWS[j]['covered'] else 0
        summary.append(result)
    co(HERE/'certificate_summary.csv',summary)
    truth_summary=[]
    for law in LAWS:
        for world,eps in sorted({(r['world'],float(r['epsilon'])) for r in truth if r['law']==law['name']}):
            rows=[r for r in truth if r['law']==law['name'] and r['world']==world and float(r['epsilon'])==eps and r['policy'] in ['symmetric_jensen','folded']]
            result=dict(law=law['name'],assumptions_covered=law['covered'],world=world,epsilon=eps,lines=len(rows))
            for key in ['known_density_gain','direct_gain','expected_nll','expected_error','known_density_batch_se']:
                values=np.array([float(r[key]) for r in rows]);result[key+'_mean']=float(values.mean());result[key+'_min']=float(values.min());result[key+'_max']=float(values.max())
                if key.endswith('gain'):result[key+'_negative']=int((values<0).sum())
            truth_summary.append(result)
    co(HERE/'stress_summary.csv',truth_summary)
    comparisons=[]
    for row in curves:
        if int(row['n_batches'])!=2048:continue
        worlds=['cell'] if row['kind']=='labeled' else ['cosine','adversarial','orbit_worst']
        for world in worlds:
            t=next(t for t in truth if t['law']==row['law'] and t['policy']==row['policy'] and t['pair']==row['pair'] and t['world']==world and float(t['epsilon'])==float(row['true_defect']))
            comparisons.append(dict(pair=row['pair'],law=row['law'],policy=row['policy'],kind=row['kind'],scenario=row['scenario'],world=world,
                assumptions_covered=row['assumptions_covered'],assumed_epsilon=float(row['epsilon']),true_posterior_defect=float(row['true_defect']),
                known_lower=float(row['known_lower']),estimated_lower=float(row['estimated_lower']),truth_estimate=float(t['known_density_gain']),
                truth_batch_se=float(t['known_density_batch_se']),truth_minus_known=float(t['known_density_gain'])-float(row['known_lower']),
                truth_minus_estimated=float(t['known_density_gain'])-float(row['estimated_lower'])))
    co(HERE/'certificate_vs_truth.csv',comparisons)
    c=ci(HERE/'certificate_costs.csv');t=ci(HERE/'stress_costs.csv');e=ci(HERE/'evidence_costs.csv');ledger=ci(HERE/'certificate_ledger.csv')
    costs=dict(parameter_inputs=sum(int(r['inputs']) for r in e if r['kind']=='parameter'),
        labeled_audit_inputs=sum(int(r['inputs']) for r in e if r['kind']=='labels'),actual_new_audit_labels=sum(int(r['sampled_labels']) for r in e),
        evidence_model_forwards=0,verification_inputs=sum(int(r['coordinate_inputs']) for r in c),verification_forwards=sum(int(r['forward_examples']) for r in c),
        verification_512_forwards=sum(int(r['forward_examples']) for r in ledger if int(r['batch'])<512),
        stress_inputs=sum(int(r['coordinate_inputs']) for r in t),stress_forwards=sum(int(r['forward_examples']) for r in t),
        deployment_replay_forwards=sum(int(r['extra_audit_forwards']) for r in t),
        coordinate_proposals=sum(int(r['proposals']) for data in [c,t,e] for r in data),
        verification_seconds_descriptive=sum(float(r['seconds']) for r in c),stress_seconds_descriptive=sum(float(r['seconds']) for r in t),
        new_fitting_labels=0,new_model_or_selector_fits=0,historical_selector_forwards=8*2*65536,
        deployment_forwards_per_256_inputs=320,union_selected_min=min(int(r['union_selected']) for r in ledger),union_selected_max=max(int(r['union_selected']) for r in ledger),
        max_identity_error=max(float(r['max_identity_error']) for r in t),max_extremal_error=max(float(r['max_extremal_error']) for r in t))
    recovery=json.loads((HERE/'STRESS_RECOVERY_FREEZE.json').read_text())
    costs['original_failed_attempt_truth_inputs']=recovery['failed_attempt']['coordinate_inputs']
    costs['original_failed_attempt_forwards']=recovery['failed_attempt']['total_forwards']
    costs['recovery_endpoint_roundoff']=json.loads((HERE/'STRESS_RECOVERY_COMMIT.json').read_text())['probability_roundoff']
    valid=[r for r in comparisons if r['assumptions_covered']=='True']
    primary=[r for r in summary if r['n_batches']==2048 and r['law']=='covered_third' and ((r['kind']=='grid' and r['epsilon']==0) or (r['kind']=='labeled' and r['true_defect']==0))]
    results=dict(utc=now(),source_freeze_sha256=sha(HERE/'FREEZE.json'),family_events=FAMILY,joint_coverage=.95,primary=primary,
        label_audits=evidence['labels'],all_parameter_intervals_cover=all(r['coverage'] for r in evidence['parameters']),
        all_label_intervals_cover=all(r['cell_interval_coverage'] for r in evidence['labels']),costs=costs,
        valid_min_truth_minus_known=min(r['truth_minus_known'] for r in valid),valid_min_truth_minus_estimated=min(r['truth_minus_estimated'] for r in valid),
        comparisons_are_mc_diagnostics_not_coverage_proofs=True,invalid_sensitivity_excluded_from_coverage=True)
    jo(HERE/'RESULTS.json',results)
    jo(HERE/'STATISTICS_LEDGER.json',dict(utc=now(),raw_batch_moment_series=len(moments)//2,stored_batch_moment_values=(len(moments)//2)*2048,
        moment_look_reductions=len(moments),moment_concentration_calculations=2*len(moments),allocated_moment_events=MOMENT_EVENTS,
        parameter_intervals=14,cell_intervals=32,total_probability_events=FAMILY,curve_rows=len(curves),truth_rows=len(truth),
        extra_network_calls_for_saved_logit_statistics=0,algebra_random_records=49152,quadrature_angular_points=1048576))
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':8})
    base=[r for r in curves if int(r['n_batches'])==2048 and r['law']=='covered_third' and r['scenario']=='e0']
    base.sort(key=lambda r:(r['policy']!='symmetric_jensen',int(r['pair'])))
    labeled={(r['pair'],r['policy']):r for r in curves if int(r['n_batches'])==2048 and r['law']=='covered_third' and r['scenario']=='label0'}
    fig,ax=plt.subplots(figsize=(7.1,3.0));xx=np.arange(16)
    for offset,color,marker,label,values in [(-.18,'#087c80','o','Known coefficients; assumed epsilon=0',[float(r['known_lower']) for r in base]),
        (0,'#cb8418','^','Estimated coefficients; assumed epsilon=0',[float(r['estimated_lower']) for r in base]),
        (.18,'#765395','s','Estimated coefficients + independent labels',[float(labeled[r['pair'],r['policy']]['estimated_lower']) for r in base])]:
        ax.scatter(xx+offset,values,s=20,color=color,marker=marker,label=label,zorder=3)
    ax.axhline(0,color='gray',lw=.8);ax.axvline(7.5,color='#d1d5d8',lw=.6)
    ax.set_xticks(xx,[('I' if r['policy']=='symmetric_jensen' else 'F')+r['pair'] for r in base]);ax.set(xlabel='Fixed model-policy pair (I: interpolation; F: folded)',ylabel='Simultaneous gain lower bound',title='A covered third harmonic, with coefficient and label uncertainty')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False,ncol=1)
    low,high=ax.get_ylim();ax.set_ylim(low-.003,high+(high-low)*.4)
    fig.subplots_adjust(left=.12,right=.99,bottom=.2,top=.86);fig.savefig(HERE/'residual_report.png',dpi=220);fig.savefig(HERE/'residual_report.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,3.2));names=[r['name'] for r in LAWS];xx=np.arange(len(names))
    ss=[next(r for r in summary if r['n_batches']==2048 and r['law']==law and r['scenario']=='e0') for law in names]
    tt=[next(r for r in truth_summary if r['law']==law and r['world']=='orbit_worst' and r['epsilon']==0) for law in names]
    ax.plot(xx,[r['estimated_lower_mean'] for r in ss],'o-',color='#087c80',label='Estimated-envelope value')
    ax.plot(xx,[r['known_density_gain_mean'] for r in tt],'s-',color='#b7484c',label='Independent worst-posterior gain estimate')
    ax.axvspan(5.5,6.45,color='#f8e5d8',alpha=.7);ax.text(6,.98,'INVALID\nrho=0',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8,color='#954b12')
    ax.axhline(0,color='gray',lw=.8);ax.set_xticks(xx,['Uniform','First .01','Third .10','Mixed','Covered\nresidual','Third .60','Omitted\nfifth .60'])
    ax.set(ylabel='Mean loss gain / lower-bound value',title='A supplied residual envelope still matters');ax.legend(frameon=False,loc='lower left',fontsize=8)
    fig.subplots_adjust(left=.12,right=.99,bottom=.21,top=.88);fig.savefig(HERE/'residual_stress.png',dpi=220);fig.savefig(HERE/'residual_stress.svg');plt.close(fig)
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
