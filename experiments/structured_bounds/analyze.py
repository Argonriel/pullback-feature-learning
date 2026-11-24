"""Summarize fixed outputs; never changes frozen samples or decisions."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,ci,co,jo,now,sha

def main():
    curves=ci(HERE/'certificate_curves.csv');stress=ci(HERE/'stress_results.csv');moments=ci(HERE/'moment_bounds.csv')
    summary=[]
    for n in [512,2048]:
        for a in [0.,.01,.1,.6]:
            for eps in [0.,.025,.05,.1,.2,.4,.8,1.]:
                for policy in ['symmetric_jensen','folded']:
                    subset=[r for r in curves if int(r['n_batches'])==n and float(r['a'])==a and float(r['epsilon'])==eps and r['policy']==policy]
                    row=dict(n_batches=n,a=a,epsilon=eps,policy=policy,models=len(subset))
                    for key in ['known_lower','estimated_lower','old_lower','known_shift_penalty','estimated_shift_penalty','old_shift_penalty']:
                        values=np.array([float(r[key]) for r in subset]);row[key+'_mean']=float(values.mean());row[key+'_min']=float(values.min());row[key+'_max']=float(values.max())
                        if key.endswith('lower'):row[key+'_positive']=int((values>0).sum())
                    summary.append(row)
    co(HERE/'certificate_summary.csv',summary)
    truth_summary=[]
    for a in [0.,.01,.1,.6]:
        for eps in [0.,.025,.05,.1,.2,.4,.8,1.]:
            for world in ['cosine','adversarial','orbit_worst']:
                for policy in ['raw','full','symmetric_jensen','folded']:
                    subset=[r for r in stress if float(r['a'])==a and float(r['epsilon'])==eps and r['world']==world and r['policy']==policy]
                    row=dict(a=a,epsilon=eps,world=world,policy=policy,models=len(subset))
                    for key in ['known_density_gain','direct_gain','expected_nll','expected_error','known_density_batch_se','direct_batch_se']:
                        values=np.array([float(r[key]) for r in subset]);row[key+'_mean']=float(values.mean());row[key+'_min']=float(values.min());row[key+'_max']=float(values.max())
                        if key.endswith('gain'):row[key+'_negative']=int((values<0).sum())
                    truth_summary.append(row)
    co(HERE/'stress_summary.csv',truth_summary)
    indexed={(int(r['pair']),float(r['a']),float(r['epsilon']),r['policy']):r for r in curves if int(r['n_batches'])==2048}
    comparisons=[]
    for row in stress:
        if row['policy'] not in ['symmetric_jensen','folded']:continue
        bound=indexed[int(row['pair']),float(row['a']),float(row['epsilon']),row['policy']]
        comparisons.append(dict(pair=int(row['pair']),a=float(row['a']),epsilon=float(row['epsilon']),policy=row['policy'],world=row['world'],
            known_lower=float(bound['known_lower']),estimated_lower=float(bound['estimated_lower']),old_lower=float(bound['old_lower']),
            truth_estimate=float(row['known_density_gain']),truth_batch_se=float(row['known_density_batch_se']),
            truth_minus_known=float(row['known_density_gain'])-float(bound['known_lower']),
            truth_minus_estimated=float(row['known_density_gain'])-float(bound['estimated_lower'])))
    co(HERE/'certificate_vs_truth.csv',comparisons)
    c=ci(HERE/'certificate_costs.csv');s=ci(HERE/'stress_costs.csv');ledger=ci(HERE/'certificate_ledger.csv');params=ci(HERE/'parameter_intervals.csv')
    costs=dict(certificate_coordinate_inputs=sum(int(r['coordinate_inputs']) for r in c),
        certificate_model_forwards=sum(int(r['forward_examples']) for r in c),
        certificate_512_model_forwards=sum(int(r['forward_examples']) for r in ledger if int(r['batch'])<512),
        certificate_invocations=sum(int(r['invocations']) for r in c),
        certificate_seconds_descriptive=sum(float(r['seconds']) for r in c),
        parameter_coordinate_inputs=sum(int(r['coordinate_inputs']) for r in params),parameter_model_forwards=0,
        stress_coordinate_inputs=sum(int(r['coordinate_inputs']) for r in s),stress_model_forwards=sum(int(r['shared_forward_examples']) for r in s),
        deployment_replay_forwards=sum(int(r['extra_audit_forwards']) for r in s),
        stress_seconds_descriptive=sum(float(r['seconds']) for r in s),
        new_fitting_labels=0,new_fitting_forwards=0,historical_selector_forwards=8*2*65536,
        deployment_inputs_per_batch=256,deployment_model_forwards_per_batch=320,
        union_selected_min=min(int(r['union_selected']) for r in ledger),union_selected_max=max(int(r['union_selected']) for r in ledger),
        coordinate_sampler_proposals=sum(int(r['proposals']) for rows in [c,s,params] for r in rows),
        max_identity_error=max(float(r['max_identity_error']) for r in s),
        max_lower_violation=max(float(r['max_lower_violation']) for r in s),
        max_extremal_error=max(float(r['max_extremal_error']) for r in s),
        max_partner_error=max(float(r['max_partner_error']) for r in s))
    primary=[r for r in summary if r['n_batches']==2048 and r['a']==.01 and r['epsilon']==0]
    results=dict(timestamp_utc=now(),primary=primary,costs=costs,parameter_intervals=params,
        family_events=1284,joint_coverage=.95,models_are_fixed=True,
        min_truth_minus_known=min(r['truth_minus_known'] for r in comparisons),
        min_truth_minus_estimated=min(r['truth_minus_estimated'] for r in comparisons),
        comparison_is_mc_diagnostic_not_coverage_proof=True,
        source_freeze_sha256=sha(HERE/'FREEZE.json'))
    jo(HERE/'RESULTS.json',results)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,4.8));colors={'known_lower':'#087c80','estimated_lower':'#cb8418','old_lower':'#b7484c'}
    labels={'known_lower':'Known family parameter','estimated_lower':'Estimated parameter + CI','old_lower':'Old global-TV bound'}
    rows=[r for r in curves if int(r['n_batches'])==2048 and float(r['a'])==.01 and float(r['epsilon'])==0]
    rows.sort(key=lambda r:(r['policy']!='symmetric_jensen',int(r['pair'])))
    x=np.arange(16)
    for offset,key in [(-.18,'known_lower'),(0,'estimated_lower'),(.18,'old_lower')]:
        axes[0].scatter(x+offset,[float(r[key]) for r in rows],s=24,label=labels[key],color=colors[key])
    axes[0].axhline(0,color='gray',lw=.8);axes[0].set_xticks(x,[f"{'I' if r['policy']=='symmetric_jensen' else 'F'}{r['pair']}" for r in rows],rotation=60)
    axes[0].set(title='Small input shift: a=.01, epsilon=0',ylabel='Simultaneous loss-gain lower bound',xlabel='Eight fixed models per policy');axes[0].legend(fontsize=8,frameon=False)
    for key in ['known_lower','estimated_lower']:
        rr=[]
        for eps in [0,.025,.05,.1,.2,.4]:
            v=np.array([float(r[key]) for r in curves if int(r['n_batches'])==2048 and float(r['a'])==.01 and float(r['epsilon'])==eps])
            rr.append((eps,v.mean(),v.min(),v.max()))
        rr=np.array(rr);axes[1].plot(rr[:,0],rr[:,1],color=colors[key],label=labels[key]);axes[1].fill_between(rr[:,0],rr[:,2],rr[:,3],color=colors[key],alpha=.13)
    axes[1].axhline(0,color='gray',lw=.8);axes[1].set(title='Posterior assumptions still matter',xlabel='Assumed posterior TV upper bound',ylabel='Lower bound at a=.01');axes[1].legend(fontsize=8,frameon=False)
    for world,color,label in [('cosine','#087c80','Cosine posterior, epsilon=0'),('orbit_worst','#b7484c','Worst posterior, epsilon=0'),('adversarial','#765395','Adversarial posterior, epsilon=.4')]:
        eps=.4 if world=='adversarial' else 0
        xx=[];yy=[];lo=[];hi=[]
        for a in [0,.01,.1,.6]:
            values=np.array([float(r['known_density_gain']) for r in stress if float(r['a'])==a and float(r['epsilon'])==eps and r['world']==world and r['policy'] in ['symmetric_jensen','folded']])
            xx.append(a);yy.append(values.mean());lo.append(values.min());hi.append(values.max())
        axes[2].plot(xx,yy,'o-',color=color,label=label);axes[2].fill_between(xx,lo,hi,color=color,alpha=.10)
    axes[2].axhline(0,color='gray',lw=.8);axes[2].set_xscale('symlog',linthresh=.01);axes[2].set_xticks([0,.01,.1,.6],['0','.01','.1','.6'])
    axes[2].set(title='Independent truth stress',xlabel='Orientation asymmetry a',ylabel='Known-density expected gain');axes[2].legend(fontsize=8,frameon=False)
    fig.suptitle('Structured assumptions recover useful guarantees; larger violations can remain harmful',fontsize=13)
    fig.text(.5,.015,'Shading: range across fixed model-policy pairs, not a confidence interval. I: interpolation; F: folded histogram. Truth stress is descriptive.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(HERE/'structured_frontier.png',dpi=180);fig.savefig(HERE/'structured_frontier.svg');plt.close(fig)
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
