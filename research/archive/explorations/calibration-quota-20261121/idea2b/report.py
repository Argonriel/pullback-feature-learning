"""Nonselective reporting of frozen quota experiments; no fitting or new evaluation."""
import sys,csv,json,re
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e

def report_csv(path,rows):
    if path.exists():
        old=e.ci(path)
        assert len(old)==len(rows)
        assert all(dict(a)=={k:str(v) for k,v in b.items()} for a,b in zip(old,rows)), f'Report table changed: {path}'
    else:e.co(path,rows)

def main():
    rows=e.ci(ROOT/'fresh'/'per_seed.csv');ix={(r['method'],int(r['budget']),r['policy'],int(r['seed'])):r for r in rows}
    rng=np.random.default_rng(742000);draw=rng.integers(0,16,(10000,16));extra=[]
    def record(method,budget,policy,ref,kind,metric,reference_budget=None):
        b0=budget if reference_budget is None else reference_budget
        z=np.array([float(ix[method,budget,policy,k][metric])-float(ix[method,b0,ref,k][metric]) for k in range(1,17)]);ci=np.quantile(z[draw].mean(1),[.025,.975]);extra.append(dict(method=method,budget=budget,reference_budget=b0,policy=policy,reference=ref,comparison=kind,metric=metric,mean_difference=z.mean(),ci_lo=ci[0],ci_hi=ci[1],lower_risk_seeds=int((z<0).sum()),n_seed=16))
    for method in e.CONFIG['models']:
        for metric in ('expected_risk','sampled_label_risk'):
            for budget in e.CONFIG['budgets']:
                for angle in ('pi','default','global','radial'):record(method,budget,'confidence_'+angle,'radius_pi_'+angle,'remaining_fixed_angle_pairwise',metric)
                for policy,ref in [('joint','confidence_radial'),('legacy_iid_folded_pi','confidence_pi'),('confidence_radial','full_radial'),('confidence_pi','full_pi')]:record(method,budget,policy,ref,'descriptive_reference',metric)
            for policy in e.POLICIES:record(method,65536,policy,policy,'prespecified_budget_sensitivity',metric,8192)
    report_csv(ROOT/'supplementary_contrasts.csv',extra)
    summaries=e.ci(ROOT/'summary.csv');contrasts=e.ci(ROOT/'contrasts.csv');latency=e.ci(ROOT/'latency_summary.csv')
    mean={(r['method'],int(r['budget']),r['policy'],r['metric']):float(r['mean']) for r in summaries}
    lat={(r['method'],int(r['budget']),r['policy']):float(r['median_ms']) for r in latency}
    cal=e.ci(ROOT/'fresh'/'calibration.csv');cost=[]
    for budget in e.CONFIG['budgets']:
        for design in ('orbit','legacy_iid_antipodes'):
            z=[float(r['calibration_seconds']) for r in cal if int(r['budget'])==budget and r['calibration_design']==design]
            cost.append(dict(budget=budget,calibration_design=design,n_models=len(z),median_seconds=float(np.median(z)),independent_inputs_or_radii=budget//(128 if design=='orbit' else 2),forward_examples_per_policy=budget))
    report_csv(ROOT/'calibration_cost.csv',cost)
    # Equal-cap policy accounting, distinguishing0-cost fixed/confidence rules.
    counts=[]
    for budget in e.CONFIG['budgets']:
        for policy in e.POLICIES:
            if policy=='raw' or policy=='full_pi' or policy in ('random_pi','random_default','confidence_pi','confidence_default'):
                calcost=0;pool='none; historical default uses frozen old archive when applicable'
            elif policy.startswith('legacy_iid_'):calcost=budget;pool='IID inputs plus antipodes'
            else:calcost=budget;pool='shared angular orbit'
            counts.append(dict(budget_cap=budget,policy=policy,per_model_calibration_forwards=calcost,calibration_pool=pool,deployment_base=256,deployment_extra=0 if policy=='raw' else (256 if policy.startswith('full_') else 64),deployment_per_input=1 if policy=='raw' else (2 if policy.startswith('full_') else 1.25)))
    report_csv(ROOT/'policy_costs.csv',counts)
    # Use the compact archived curves first; original external files are optional.
    compact=ROOT/'old_pilot'/'historical_gain_curves.npz'
    history=json.loads((ROOT/'old_pilot'/'historical_defaults.json').read_text())
    if compact.exists():
        curves=dict(np.load(compact))
    else:
        prior=ROOT.parent.parent/'predictive-symmetry-final'/'direction2'/'exploration';curves={}
        expected={r['file']:r['sha256'] for r in history['source_hashes']}
        for method in e.CONFIG['models']:
            records=[]
            for seed in range(1,17):
                p=prior/f'{seed}_{method}.npz';assert e.sha(p.read_bytes())==expected[p.name];z=np.load(p);records.append(z['gain_curves'][z['names'].tolist().index('exact_global')])
            curves[method]=np.array(records)
        np.savez_compressed(compact,**curves)
    for method in e.CONFIG['models']:
        np.testing.assert_allclose(curves[method].mean(axis=0),history['mean_gain_curves'][method],rtol=0,atol=1e-15)
        assert int(curves[method].mean(axis=0).argmax()+1)==e.CONFIG['default_index'][method]
    lines=['# Idea2B: fixed quota and calibrated second-view angles','',
    'The primary joint radius policy did not beat the strong first-view confidence baseline. On16 fresh pullback models, at the8192-forward calibration cap, joint risk exceeded confidence-plus-half-turn by0.00165668 nats (95% paired seed-bootstrap interval[0.00035641,0.00291192]); joint was worse on14/16 seeds. Against the original IID-calibrated folded half-turn gate, joint was better by0.00066659 nats on average, but the interval[-0.00178090,0.00052721] included zero. Independently sampled labels gave the same conclusions.','',
    'There was a useful prespecified factorial result: keeping the confidence mask exactly fixed while replacing the half-turn with a learned radius angle reduced risk by0.00258288 nats[-0.00364939,-0.00157176], winning15/16 seeds. Changing the reference radius mask after choosing radius angles added only0.00001930 nats mean improvement, with an interval crossing zero. The evidence supports angle calibration as an addition to a strong allocator; it does not support replacing confidence with the tested joint radius surrogate. This interpretation does not select or refit a new policy on these results: all arms were frozen beforehand.','',
    '## Primary comparison at both frozen calibration caps','',
    '|Model|Calibration forwards|Joint minus confidence+pi|95%CI|Joint minus legacy IID folded+pi|95%CI|','|---|---:|---:|---|---:|---|']
    for method in e.CONFIG['models']:
        for budget in e.CONFIG['budgets']:
            a=next(r for r in contrasts if r['method']==method and int(r['budget'])==budget and r['policy']=='joint' and r['reference']=='confidence_pi' and r['metric']=='expected_risk')
            b=next(r for r in contrasts if r['method']==method and int(r['budget'])==budget and r['policy']=='joint' and r['reference']=='legacy_iid_folded_pi' and r['metric']=='expected_risk')
            lines.append(f"|{method}|{budget}|{float(a['mean_difference']):+.7f}|[{float(a['ci_lo']):+.7f},{float(a['ci_hi']):+.7f}]|{float(b['mean_difference']):+.7f}|[{float(b['ci_lo']):+.7f},{float(b['ci_hi']):+.7f}]|")
    lines+=['','The pullback8192 contrasts are primary. Other methods, the65536 cap, sampled-label risks and all factorial contrasts are reported as prespecified secondary/sensitivity results. Every interval resamples sixteen complete seed records, not batches. Intervals are pointwise; correlated policy/endpoint tests are not independent evidence. Calibration has one independent pool per model/seed here, so this study does not isolate variability over repeated calibration pools; Idea2A supplies that separate analysis.','',
    '## All pullback policies, primary calibration cap','',
    '|Policy|Expected risk|Sampled-label risk|New-model calibration forwards|Deployment calls/input|Median ms/batch256|','|---|---:|---:|---:|---:|---:|']
    costs={(r['budget_cap'],r['policy']):r for r in counts}
    for name in e.POLICIES:
        c=costs[8192,name];lines.append(f"|{name}|{mean['pb',8192,name,'expected_risk']:.6f}|{mean['pb',8192,name,'sampled_label_risk']:.6f}|{c['per_model_calibration_forwards']}|{c['deployment_per_input']}|{lat['pb',8192,name]:.4f}|")
    lines+=['','Policy naming: random, confidence and radius_pi prefixes fix the selected mask across pi/default/global/radial angle rules. `default` is frozen from old models: GD180degrees, pullback143.4375degrees, hybrid163.125degrees. `radius_pi` always ranks the estimated half-turn gain. `folded_pi` uses the orbit-derived folded table. The two `legacy_iid_*` rules use the unchanged prior Gate.fit algorithm on IID inputs and antipodes. `radius_angle_*` updates a radius priority for that same angle; `joint` uses the maximizing radius angle and corresponding gain priority.','',
    'The full-two-view rows are distinct-budget references, NOT universal risk ceilings. A confidence-dependent mask can skip orientation-specific harmful second views. Indeed pullback confidence+radius-angle risk0.100349 is below the full-radius reference0.102995 despite using1.25 versus2 calls/input. No universal dominance claim follows from forward counts or Jensen symmetry. The folded half-turn guarantee applies only to that pair and is not extended to arbitrary angles. See THEORY.md.','',
    '## Samples, calibration and deployment accounting','',
    'The48 fixed models come from16 new paired training draws shared with Idea2A: train721001..721016, init722001..722016; GD500, PB500 and PB100/GD400. This shared model cohort is not an independent replication of Idea2A. Each seed uses65536 independent test inputs, arranged into256 batches256, shared across all models, budgets and policies. Input seeds736000+i, independent ties737000+i and independent Bernoulli-label draws738000+i are retained by hashes. Conditional expected risk uses the known posterior only in the evaluator.','',
    'Primary calibration uses64 independent radii and128-point angular orbits, exactly8192 forwards; secondary uses512 radii and65536 forwards, with the primary radii as a prefix. Radius seeds735000+i and separate phase seeds745000+i are independent of test data and Idea2A calibration. All angle/radius/folded orbit tables reuse these forwards. Sixteen radius bins contain at least4 independent radii; angular samples are correlated quadrature nodes, never counted as independent calibration draws.','',
    'The stronger unchanged legacy gate uses4096 IID inputs and their antipodes at the primary cap, or32768 at the secondary cap, with seed741000+i and shared prefixes. Its original empty-cell fallback is preserved. Each individual gate respects the same forward cap, but its sample design is different. At8192, IID folding improves over orbit folding by0.00023461 nats on pullback, with interval[-0.00052762,-0.00003028], showing why retaining this control mattered.','',
    'Confidence/random with fixed pi or historical default require zero new-model calibration. Learned policies incur their stated budget. Historical defaults reuse old calibration work and are not free to discover. The two budgets and every losing policy remain reported. Across the final confirmation, actual shared calibration work is48*(65536+8192+65536)=6684672 model-example forwards: one nested orbit pool and separately executed legacy fits at both caps. This research total is distinct from any one deployed policy\'s cost. The preserved pre-test phase attempt incurred additional replay work of the same size and is not hidden.','']
    for r in cost:lines.append(f"- {r['calibration_design']} at{r['budget']} forwards: median{r['median_seconds']:.4f}seconds per model, {r['independent_inputs_or_radii']} independent radii/inputs.")
    lines+=['','All18 limited policies actually select64 entries per batch after their first logits and coordinates are available. Only those entries receive second-view evaluations; no candidate second views are probed during routing. Every model×budget×policy has an independently counted first-batch deployment audit with calls[256,64], including random allocation. Raw calls[256] and full references[256,256] are separate. Whole-test evaluation vectorizes these identical batch allocations; per-batch risks and actual mask hashes are saved. Timings randomize policy order, include routing/rotation overhead and repeat16 real batches. Local CPU latency does not imply accelerator-service throughput.','',
    '## Frozen boundaries and retained attempts','',
    'DESIGN.md/design_freeze.json and experiment_v1_frozen.py/freeze.json preserve the initial protocol/code. Before fresh calibration/testing, ADDENDUM.md and freeze_addendum.json added unchanged IID strong baselines for calibration sampling fairness. Before any test data or outcomes, PHASE_ADDENDUM.md and freeze_phase_addendum.json corrected a restarted RNG stream to an independent phase stream. The entire earlier calibration remains in fresh_v2_shared_phase with its own commit and experiment_v2_frozen.py; it was never evaluated for choosing a result. The final fresh/policy_commit.json precedes fresh/score_start.json, and all policy hashes remain immutable through testing. No failed scientific policy was dropped or retuned.','',
    '## Reproduction and files','',
    'Python dependencies: NumPy and threadpoolctl for experiments; Matplotlib for figures. The numerical model and unchanged gate are imported from the surrounding archived source/ and legacy/ folders. Fresh model states are supplied by ../idea2a/fresh/training/. No network, repository checkout or account access is needed for confirmation. The48 historical gain curves used for old-only defaults are copied into old_pilot/historical_gain_curves.npz with source hashes and means in historical_defaults.json. Use `python replay_old_pilot.py` to reproduce the original12 old-pilot rows and historical choices entirely from the archived compact curves and old model bank. That portable replay passed. The frozen original oldpilot command remains preserved for provenance and is not required for reproduction.','',
    'In a fresh archive-layout copy, preserve the supplied freezes, source and input model states, omit fresh/outputs, then run: `python experiment.py calibrate`, `python experiment.py score`, `python experiment.py summarize`. Existing calibration/results are refused. Reporting is `python report.py`; it reads and verifies the archived compact historical curves and needs no original external exploration path. Existing report tables are verified unchanged before regenerating prose/figures. No scientific fitting depends on report generation.','',
    'Primary raw files: fresh/per_seed.csv, fresh/per_batch.csv.gz, fresh/samples.csv, fresh/deployment_audit.csv, fresh/calibration.csv, fresh/latency.csv, frozen policies and full calibration orbit arrays. summary.csv/contrasts.csv contain every prespecified result. supplementary_contrasts.csv completes pairwise fixed-angle mask comparisons and every budget sensitivity; descriptive comparisons against full references are labeled. policy_costs.csv, calibration_cost.csv and latency_summary.csv distinguish calibration, calls and measured time.','']
    text='\n'.join(lines)
    text=re.sub(r'\b(On|on|the|by|winning|only|Idea|per|at|in|from|The|Each|uses|into|batches|seeds|ties|draws|is|least|median|after|all|every|model|exactly|and|or|with|of|for|risk|reference|versus|using)(?=\d)',r'\1 ',text)
    text=re.sub(r'(?<=\d)(degrees|seconds|times)\b',r' \1',text)
    text=text.replace('95%CI','95% CI').replace('interval[','interval [').replace('nats[','nats [')
    (ROOT/'README.md').write_text(text,encoding='utf8')
    # Three-panel standalone scientific figure; same seed paired contrasts.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(17,5.2),constrained_layout=True)
    selected=[('joint','confidence_pi'),('joint','legacy_iid_folded_pi'),('confidence_radial','confidence_pi'),('radius_pi_radial','radius_pi_pi'),('joint','radius_pi_radial')]
    labels=['Joint − confidence + pi','Joint − legacy folded + pi','Confidence: radial angle − pi','Fixed radius mask: radial − pi','Joint − fixed radius mask']
    for j,((a,b),lab) in enumerate(zip(selected,labels)):
        rr=next(r for r in contrasts if r['method']=='pb' and int(r['budget'])==8192 and r['policy']==a and r['reference']==b and r['metric']=='expected_risk')
        mu=float(rr['mean_difference']);lo=float(rr['ci_lo']);hi=float(rr['ci_hi']);axes[0].errorbar(mu,j,xerr=[[mu-lo],[hi-mu]],fmt='o',color='#007f8b' if mu<0 else '#c65e29',capsize=3)
    from matplotlib.ticker import MaxNLocator
    axes[0].xaxis.set_major_locator(MaxNLocator(4))
    axes[0].axvline(0,color='gray',ls='--');axes[0].set_yticks(range(5),labels);axes[0].invert_yaxis();axes[0].set_xlabel('Expected risk contrast (nats)');axes[0].set_title('Pullback contrasts, calibration cap 8192')
    names=['random','confidence','radius_pi'];angles=['pi','default','global','radial'];matrix=np.array([[mean['pb',8192,f'{m}_{a}','expected_risk'] for a in angles] for m in names]);im=axes[1].imshow(matrix,aspect='auto',cmap='YlGnBu');axes[1].set_xticks(range(4),['Pi','Historical','Global','Radius']);axes[1].set_yticks(range(3),['Random','Confidence','Radius pi-gain']);axes[1].set_title('Same selected mask across each row')
    for i in range(3):
        for j in range(4):axes[1].text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',color='white' if matrix[i,j]>.109 else 'black')
    for name,color in [('joint','#c65e29'),('confidence_pi','#777777'),('confidence_radial','#007f8b'),('legacy_iid_folded_pi','#8b6bb1')]:
        axes[2].plot([8192,65536],[mean['pb',b,name,'expected_risk'] for b in (8192,65536)],'o-',label={'joint':'Joint radius','confidence_pi':'Confidence + pi','confidence_radial':'Confidence + radius angle','legacy_iid_folded_pi':'Legacy folded + pi'}[name],color=color)
    axes[2].set_xscale('log',base=2);axes[2].set_xticks([8192,65536],['8192','65536']);axes[2].set_ylabel('Expected risk (nats)');axes[2].set_xlabel('Offline calibration forward cap');axes[2].set_title('Larger calibration does not reverse ranking');axes[2].legend(fontsize=8,loc='center',bbox_to_anchor=(.52,.35))
    fig.suptitle('Pullback models: 64 extra views per 256 inputs\nAngle calibration helps; joint radius allocation loses to confidence',fontsize=13)
    fig.savefig(ROOT/'quota_factorial.png',dpi=180);fig.savefig(ROOT/'quota_factorial.pdf');plt.close(fig)
if __name__=='__main__':main()
