"""Post-score summaries only; does not alter any frozen policy or raw result."""
from pathlib import Path
import csv, json, hashlib, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import study

ROOT=Path(__file__).resolve().parent
METHODS=['base_independent','base_joint','uniform_independent','uniform_joint','adaptive_joint','pointwise_oracle']
LABELS=['Original interval bound','Joint objective, same grid','Uniform refinement, old objective','Uniform refinement, joint','Adaptive refinement, joint','Pointwise oracle (2 calls)']
EPS=[0.,.1,.25,.5]

def read(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(path,rows):
    with Path(path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
def stats(v,ix):
    v=np.asarray(v,float);q=np.quantile(v[ix].mean(1),[.025,.975])
    return dict(mean=float(v.mean()),ci_low=float(q[0]),ci_high=float(q[1]),minimum=float(v.min()),maximum=float(v.max()))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    rng=np.random.default_rng(study.CONFIG['bootstrap_seed']);ix=rng.integers(0,16,(10000,16))
    aggregated=[];contrasts=[];costs=[];allsummary={};numerics=[];checks=[];base_latency=[]
    freeze=json.loads((ROOT/'freeze.json').read_text())
    for p,h in freeze['source_hashes'].items():assert sha(ROOT/p)==h
    for cohort in ['old','fresh']:
        cal=ROOT/cohort/'calibration';out=ROOT/cohort/'results'
        commit=json.loads((cal/'selector_commit.json').read_text());start=json.loads((out/'score_start.json').read_text());complete=json.loads((out/'score_complete.json').read_text())
        assert freeze['timestamp_utc']<commit['timestamp_utc']<start['timestamp_utc']<complete['timestamp_utc']
        assert start['selector_commit_sha256']==sha(cal/'selector_commit.json')
        for p,h in commit['npz_sha256'].items():assert sha(cal/p)==h
        for p,h in complete['csv_sha256'].items():assert sha(out/p)==h
        rows=read(out/'by_model.csv');lookup={(float(r['epsilon']),r['method'],int(r['pair'])):r for r in rows}
        assert len(lookup)==4*6*16
        def vec(e,m,k):return np.array([float(lookup[e,m,i][k]) for i in range(1,17)])
        for e in EPS:
            for m in METHODS:
                result=dict(cohort=cohort,epsilon=e,method=m,n_models=16,n_test_per_model=32768)
                for k in ['mean_gain','mean_risk','selected_fraction','calls_per_input','mean_lower_bound']:
                    result.update({k+'_'+name:value for name,value in stats(vec(e,m,k),ix).items()})
                result['harmful_selected']=int(sum(vec(e,m,'harmful_selected')))
                result['maximum_bound_violation']=float(vec(e,m,'max_bound_violation').max())
                aggregated.append(result)
            for a,b in [('adaptive_joint','base_independent'),('adaptive_joint','uniform_joint'),('base_joint','base_independent'),('uniform_joint','uniform_independent')]:
                v=vec(e,a,'mean_gain')-vec(e,b,'mean_gain')
                contrasts.append(dict(cohort=cohort,epsilon=e,method=a,reference=b,**stats(v,ix),positive_models=int((v>1e-14).sum()),negative_models=int((v< -1e-14).sum()),equal_models=int((abs(v)<=1e-14).sum())))
        cov=vec(.1,'adaptive_joint','selected_fraction');gain=vec(.1,'adaptive_joint','mean_gain');oracle=vec(.1,'pointwise_oracle','mean_gain');difference=gain-vec(.1,'base_independent','mean_gain')
        ratio=float(gain.mean()/oracle.mean());bratio=gain[ix].mean(1)/oracle[ix].mean(1)
        criteria=dict(coverage_at_least_2_percent=bool(cov.mean()>=.02),gain_at_least_half_oracle=bool(ratio>=.5),paired_improvement_ci_positive=bool(stats(difference,ix)['ci_low']>0))
        allsummary[cohort]=dict(primary_epsilon=.1,coverage=stats(cov,ix),gain=stats(gain,ix),oracle_gain=stats(oracle,ix),oracle_gain_fraction=ratio,oracle_gain_fraction_ci=np.quantile(bratio,[.025,.975]).tolist(),gain_difference_vs_original=stats(difference,ix),gain_difference_vs_equal_cost_uniform=stats(gain-vec(.1,'uniform_joint','mean_gain'),ix),criteria=criteria,all_criteria_pass=all(criteria.values()),models_meeting_coverage_target=int((cov>=.02).sum()),models_with_positive_certified_gain=int((gain>0).sum()),fresh_cohort=cohort=='fresh',chronology=dict(freeze=freeze['timestamp_utc'],selector_commit=commit['timestamp_utc'],score_start=start['timestamp_utc'],score_complete=complete['timestamp_utc']))
        for grid in ['base','uniform','adaptive']:
            cr=[r for r in read(cal/'costs.csv') if r['grid']==grid]
            costs.append(dict(cohort=cohort,grid=grid,n_models=len(cr),interval_propagations_per_model=int(cr[0]['interval_propagations']),total_interval_propagations=sum(int(r['interval_propagations']) for r in cr),final_leaves_per_model=int(cr[0]['leaves']),nodes_per_model=int(cr[0]['nodes']),mean_setup_seconds=float(np.mean([float(r['total_setup_seconds']) for r in cr])),median_setup_seconds=float(np.median([float(r['total_setup_seconds']) for r in cr])),coordinate_only_inputs_per_model=int(cr[0]['coordinate_only_calibration_inputs']),npz_bytes_total=sum((cal/f'{i}_{grid}.npz').stat().st_size for i in range(1,17))))
        nr=read(cal/'root_diagnostics.csv')
        numerics.append(dict(cohort=cohort,cubic_solves=sum(int(r['cubic_solves']) for r in nr),fallback_cells=sum(int(r['fallback_cells']) for r in nr),max_stationary_residual=max(float(r['max_stationary_residual']) for r in nr)))
        audit=json.loads((out/'deployment_checks.json').read_text());br=read(out/'by_batch.csv');en=read(out/'enclosure_checks.csv')
        checks.append(dict(cohort=cohort,deployment_checks=len(audit),max_extra_calls=max(a['selected'] for a in audit),max_batch_selected=max(int(r['selected']) for r in br),max_output_error=max(a['output_error'] for a in audit),max_logit_enclosure_violation=max(float(r['max_logit_enclosure_violation']) for r in en),outside_grid=sum(int(r['outside_grid']) for r in en),harmful_selected=sum(int(r['harmful_selected']) for r in rows),source_hashes_match=True,calibration_hashes_match=True,result_hashes_match=True,chronology_pass=True))
        if cohort=='fresh':
            # An outcome-neutral timing supplement uses exactly the fixed test batches.
            for i in range(1,17):
                theta=study.model(cohort,i);x=study.sample(study.CONFIG['fresh_test_seed']+i,32768)[:2048];ties=np.random.default_rng(study.CONFIG['fresh_tie_seed']+i).random(len(x))
                for name in ['raw','pointwise_oracle']:
                    for rep in range(3):
                        t=time.perf_counter_ns()
                        for k in range(8):
                            z=x[k*256:(k+1)*256];a=study.core.representation(theta,z)[:,0]
                            if name=='pointwise_oracle':
                                b=study.core.representation(theta,-z)[:,0];mask=study.allocate(study.objective(a,b,.1),ties[k*256:(k+1)*256]);result=np.where(mask,.5*(a+b),a)
                        if rep:base_latency.append(dict(pair=i,epsilon=.1,method=name,repeat=rep,ms_per_batch=(time.perf_counter_ns()-t)/8/1e6))
    write(ROOT/'aggregate.csv',aggregated);write(ROOT/'paired_contrasts.csv',contrasts);write(ROOT/'cost_summary.csv',costs);write(ROOT/'baseline_latency.csv',base_latency)
    timing=read(ROOT/'fresh/results/latency.csv')+base_latency;lat=[]
    for method in ['raw']+METHODS:
        vals=[float(r['ms_per_batch']) for r in timing if r['method']==method and float(r['epsilon'])==.1]
        lat.append(dict(method=method,n=len(vals),median_ms_per_256=float(np.median(vals)),mean_ms_per_256=float(np.mean(vals))))
    write(ROOT/'latency_summary.csv',lat)
    allsummary.update(numerical_checks=numerics,execution_checks=checks,bootstrap=dict(seed=study.CONFIG['bootstrap_seed'],resamples=10000,unit='complete model pair',interval='pointwise percentile 95%; not simultaneous'),physical_study_interval_propagations=sum(c['total_interval_propagations'] for c in costs),latency=lat,disclosure='All setup times include four epsilon objectives and shared fixed-grid objective variants. Adaptive partition timing includes priority roots; equal interval counts do not imply equal time. Baseline timings are a post-score, outcome-neutral supplement. Historical models do not add fresh training replications.')
    dump(ROOT/'summary.json',allsummary)
    # Scientific primary comparison, preserving the oracle's different cost.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(13.3,5.4),sharey=True)
    records={r['method']:r for r in aggregated if r['cohort']=='fresh' and r['epsilon']==.1}
    colors=['#9ca3af','#7199b4','#b4bfc7','#3f7ea6','#087f72','#d5a64d']
    for ax,key,mult,title in zip(axes,['selected_fraction','mean_gain'],[100,1],['Inputs receiving a second view (%)','Expected risk reduction (nats / input)']):
        means=np.array([records[m][key+'_mean'] for m in METHODS])*mult;lo=np.array([records[m][key+'_ci_low'] for m in METHODS])*mult;hi=np.array([records[m][key+'_ci_high'] for m in METHODS])*mult
        bars=ax.barh(np.arange(6),means,color=colors,height=.65,xerr=np.array([means-lo,hi-means]),error_kw=dict(capsize=3,elinewidth=1))
        bars[-1].set_hatch('//');ax.set_title(title,pad=14);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.set_yticks(np.arange(6),LABELS)
        if mult==100:ax.axvline(2,color='#555',ls='--',lw=1,label='2% target');ax.legend(loc='lower right',frameon=False)
        else:ax.axvline(.5*records['pointwise_oracle']['mean_gain_mean'],color='#555',ls='--',lw=1,label='Half oracle gain');ax.legend(loc='lower right',frameon=False)
    axes[0].set_ylim(5.6,-.6)
    fig.suptitle('Tighter adaptive certificates meet the frozen target at epsilon = 0.1',fontsize=14,y=.98)
    fig.text(.015,.015,'16 fresh PB500 models; 32,768 independent test inputs per model. Error bars: pointwise 95% model bootstrap.\nDeployable gates use at most 64 extra calls per 256 inputs; the diagnostic oracle uses 2 calls per input.',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(ROOT/'certificate_primary.png',dpi=180);fig.savefig(ROOT/'certificate_primary.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for m,c in zip(METHODS,colors):
        rr=[next(r for r in aggregated if r['cohort']=='fresh' and r['epsilon']==e and r['method']==m) for e in EPS]
        axes[0].plot(EPS,[100*r['selected_fraction_mean'] for r in rr],marker='o',color=c,label=LABELS[METHODS.index(m)])
        axes[1].plot(EPS,[r['mean_gain_mean'] for r in rr],marker='o',color=c)
    axes[0].set_ylabel('Selected inputs (%)');axes[1].set_ylabel('Expected risk reduction (nats / input)')
    for ax in axes:ax.set_xlabel('Externally supplied epsilon');ax.set_xticks(EPS);ax.grid(alpha=.2)
    fig.suptitle('Adaptive partition tuned to epsilon = 0.1: benefit does not transfer uniformly')
    fig.legend(loc='lower center',bbox_to_anchor=(.5,0),ncol=2,frameon=False,fontsize=9);fig.tight_layout(rect=(0,.16,1,.94));fig.savefig(ROOT/'certificate_sensitivity.png',dpi=180);fig.savefig(ROOT/'certificate_sensitivity.pdf');plt.close(fig)
    print(json.dumps(allsummary['fresh'],indent=2));print('Physical interval propagations',allsummary['physical_study_interval_propagations']);print('Numerics',numerics)

if __name__=='__main__':
    with threadpool_limits(1):main()
