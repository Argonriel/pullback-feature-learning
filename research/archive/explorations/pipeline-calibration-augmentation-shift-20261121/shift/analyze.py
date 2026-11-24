"""Descriptive aggregation of the frozen experiment; no selector fitting."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,ci,co,jo,now

def main():
    bounds=ci(HERE/'certificate_bounds.csv');curves=ci(HERE/'certificate_curves.csv');stress=ci(HERE/'stress_results.csv')
    summary=[]
    for n in [512,2048]:
        for a in [0.,.01,.1,.6]:
            for policy in ['symmetric_jensen','folded']:
                rows=[r for r in bounds if int(r['n_batches'])==n and float(r['a'])==a and r['policy']==policy]
                out=dict(n_batches=n,a=a,eta=2*a/np.pi,policy=policy,models=len(rows))
                for key in ['c','D','mean_A','mean_B','A_lower','B_upper','input_tv_penalty','lower_epsilon0','eta_tolerance_epsilon0','epsilon_tolerance']:
                    v=np.array([float(r[key]) for r in rows]);out[key+'_mean']=float(v.mean());out[key+'_min']=float(v.min());out[key+'_max']=float(v.max())
                out['positive_epsilon0']=sum(float(r['lower_epsilon0'])>0 for r in rows);summary.append(out)
    co(HERE/'certificate_summary.csv',summary)
    curve_summary=[]
    for n in [512,2048]:
        for a in [0.,.01,.1,.6]:
            for policy in ['symmetric_jensen','folded']:
                for eps in [0.,.025,.05,.1,.2,.4,.8,1.]:
                    v=np.array([float(r['lower']) for r in curves if int(r['n_batches'])==n and float(r['a'])==a and r['policy']==policy and float(r['epsilon'])==eps])
                    curve_summary.append(dict(n_batches=n,a=a,epsilon=eps,policy=policy,mean=float(v.mean()),minimum=float(v.min()),maximum=float(v.max()),positive_models=int((v>0).sum())))
    co(HERE/'certificate_curve_summary.csv',curve_summary)
    ssummary=[]
    for a in [0.,.01,.1,.6]:
        for world in ['cosine','adversarial']:
            for eps in [0.,.025,.05,.1,.2,.4,.8,1.]:
                for policy in ['raw','full','symmetric_jensen','folded','confidence']:
                    r=[v for v in stress if float(v['a'])==a and v['world']==world and float(v['epsilon'])==eps and v['policy']==policy]
                    out=dict(a=a,world=world,epsilon=eps,policy=policy)
                    for k in ['expected_nll','expected_error','direct_gain','known_density_gain','density_correction','unweighted_paired_gain']:
                        v=np.array([float(x[k]) for x in r if x[k]!=''])
                        out[k+'_mean']=float(v.mean()) if len(v) else ''
                        if k in ['direct_gain','known_density_gain']:
                            out[k+'_negative_models']=int((v<0).sum()) if len(v) else ''
                    ssummary.append(out)
    co(HERE/'stress_summary.csv',ssummary)
    c=ci(HERE/'certificate_costs.csv');s=ci(HERE/'stress_costs.csv');clip=json.loads((HERE/'CLIP_COMMIT.json').read_text())['records']
    costs=dict(certificate_forward_examples=sum(int(r['forward_examples']) for r in c),
        certificate_coordinate_inputs=sum(int(r['coordinate_inputs']) for r in c),
        certificate_seconds=sum(float(r['seconds']) for r in c),
        stress_shared_forwards=sum(int(r['shared_forward_examples']) for r in s),
        stress_extra_audit_forwards=sum(int(r['extra_deployment_audit_forwards']) for r in s),
        stress_coordinate_inputs=sum(int(r['inputs']) for r in s),stress_seconds=sum(float(r['seconds']) for r in s),
        sampler_coordinate_only_inputs=262144,new_fitting_forward_examples=0,
        historical_standalone_selector_forwards=8*2*65536,
        deployment_forward_examples_per_256=320,
        maximum_binary_identity_error=max(float(r['maximum_binary_identity_error']) for r in s))
    ledger=ci(HERE/'certificate_ledger.csv')
    costs['certificate_512_forward_examples']=sum(int(r['forward_examples']) for r in ledger if int(r['batch'])<512)
    costs['certificate_union_per_batch_min']=min(int(r['union_selected']) for r in ledger)
    costs['certificate_union_per_batch_max']=max(int(r['union_selected']) for r in ledger)
    jo(HERE/'RESULTS.json',dict(timestamp_utc=now(),costs=costs,model_constants=clip,
        primary_summary=[r for r in summary if r['n_batches']==2048]))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15.4,4.8))
    colors={'symmetric_jensen':'#166b8c','folded':'#d37b1f'}
    for policy,label in [('symmetric_jensen','Interpolated Jensen'),('folded','Folded histogram')]:
        r=[v for v in summary if v['n_batches']==2048 and v['policy']==policy]
        x=np.array([v['a'] for v in r]);y=np.array([v['lower_epsilon0_mean'] for v in r])
        lo=np.array([v['lower_epsilon0_min'] for v in r]);hi=np.array([v['lower_epsilon0_max'] for v in r])
        axes[0].plot(x,y,'o-',color=colors[policy],label=label);axes[0].fill_between(x,lo,hi,alpha=.12,color=colors[policy])
    axes[0].axhline(0,color='gray',lw=.8);axes[0].set(xlabel='Orientation asymmetry a',ylabel='Certified NLL gain at posterior defect 0',title='Global TV penalty dominates')
    axes[0].set_xscale('symlog',linthresh=.01);axes[0].set_xticks([0,.01,.1,.6],['0','.01','.1','.6']);axes[0].legend(frameon=False,fontsize=8)
    for policy,label in [('symmetric_jensen','Interpolation'),('folded','Folded')]:
        for a,ls in [(0.,'-'),(.01,'--')]:
            r=[v for v in curve_summary if v['n_batches']==2048 and v['a']==a and v['policy']==policy]
            axes[1].plot([v['epsilon'] for v in r],[v['mean'] for v in r],ls,color=colors[policy],label=f'{label}, a={a:g}')
    axes[1].axhline(0,color='gray',lw=.8);axes[1].set(xlabel='Posterior TV bound epsilon',ylabel='Mean simultaneous lower bound',title='Both defects consume the margin');axes[1].legend(frameon=False,fontsize=8)
    for policy,label in [('symmetric_jensen','Interpolation'),('folded','Folded')]:
        for eps,ls in [(0.,'-'),(.4,'--')]:
            r=[v for v in ssummary if v['policy']==policy and v['world']=='adversarial' and v['epsilon']==eps]
            axes[2].plot([v['a'] for v in r],[v['known_density_gain_mean'] for v in r],'o'+ls,color=colors[policy],label=f'{label}, epsilon={eps:g}')
    axes[2].axhline(0,color='gray',lw=.8);axes[2].set(xlabel='Orientation asymmetry a',ylabel='Known-density expected NLL gain',title='Actual gains need not track the bound');axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Eight fixed historical models; two unchanged exact-quota policies',fontsize=13)
    fig.text(.5,.01,'Left shading: range across eight fixed models, not a confidence interval. Right: independent evaluation, not a certificate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(HERE/'shift_frontier.png',dpi=180);fig.savefig(HERE/'shift_frontier.svg');plt.close(fig)
    print(json.dumps(costs,indent=2))
if __name__=='__main__':main()
