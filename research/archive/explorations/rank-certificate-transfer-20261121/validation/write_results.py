"""Render completed validation tables and figures; never changes frozen methods."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def read(name):
    with (HERE/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def main():
    summary=read('summary.csv');contrast=read('contrasts.csv');gains=read('world_gains.csv')
    costs=read('calibration_costs.csv');individual=read('per_model.csv')
    frozen=json.loads((HERE/'freeze.json').read_text());audit=json.loads((HERE/'analysis_audit.json').read_text())
    index={(r['cohort'],r['method'],r['policy'],r['outcome']):r for r in summary}
    get=lambda cohort,method,policy,outcome:float(index[cohort,method,policy,outcome]['mean'])
    labels={'raw':'Raw','full':'Full half-turn','confidence':'Confidence + half-turn',
        'folded':'Folded symmetric gate','rank_balanced':'Rank-balanced','rank_even':'Rank-even',
        'rank_unbalanced':'Rank-unbalanced','confidence_radial':'Confidence + quotient angle',
        'full_radial':'Full quotient angle'}
    lines=['# Independent validation: empirical quality and invariant-posterior sensitivity','',
        'The confidence comparison changes on the new harmonic task. Rank-balanced loses to confidence on every fresh rings model, but beats it on 14 of 16 harmonic models. It has a small mean loss advantage over rank-even in both cohorts. The strong folded symmetric gate remains essential: the rings comparison is unresolved, and folded wins on all 16 harmonic models. No selector settings were changed after the freezes.','',
        'The primary sample contains 16 independently trained PB models in each of two cohorts, with 65,536 independent IID test inputs per pair. The 16 transfer GD states are underfitted optimizer context, not a third selector cohort. Training data, architecture, observation law and Bayes difficulty differ between the two tasks, so their absolute losses are not an equal-difficulty comparison. The four-dimensional transfer action is R(alpha) on the first pair and R(2alpha) on the second; its exact half-turn flips only the first pair. Its quotient remains scalar.','',
        '## Prespecified primary family','',
        'Differences are rank-balanced minus the reference; negative favors rank-balanced. Intervals are 98.75% paired-model percentile bootstrap intervals for the four prespecified comparisons. Ordinary 95% intervals are also retained. These model-cluster bootstrap intervals are empirical uncertainty summaries, distinct from the separate finite-sample certificate.','',
        '| Cohort | Reference | Mean loss difference | 98.75% interval | Balanced wins |','|---|---|---:|---|---:|']
    for r in contrast:
        if r['primary']=='True':lines.append(f"| {r['cohort']} | {labels[r['reference']]} | {float(r['delta_risk']):+.7f} | [{float(r['low98_75']):+.7f}, {float(r['high98_75']):+.7f}] | {r['wins']}/16 |")
    lines+=['','The folded comparisons are descriptive:','']
    for cohort in ('rings','transfer'):
        r=next(r for r in contrast if r['cohort']==cohort and r['method']=='pb' and r['policy']=='rank_balanced' and r['reference']=='folded' and r['outcome']=='expected_risk')
        lines.append(f"- {cohort}: balanced minus folded = {float(r['delta_risk']):+.7f}, 95% interval [{float(r['low95']):+.7f}, {float(r['high95']):+.7f}], {r['wins']}/16 balanced wins.")
    lines+=['','## Loss and classification metrics','',
        'Each entry is a mean across 16 trained models. Expected loss uses the evaluator-only analytic posterior. The sampled-label loss and error use a separate Bernoulli-label stream. Logistic loss and classification error are different outcomes; a loss advantage is not automatically an accuracy advantage.']
    for cohort,title in [('rings','Original rings: primary PB'),('transfer','Two-harmonic task: primary PB')]:
        lines+=['',f'### {title}','',
            '| Policy | Expected logistic loss | Sampled-label loss | Expected error | Sampled error |',
            '|---|---:|---:|---:|---:|']
        for name in frozen['policies']:
            lines.append(f"| {labels[name]} | {get(cohort,'pb',name,'expected_risk'):.6f} | {get(cohort,'pb',name,'sampled_label_risk'):.6f} | {100*get(cohort,'pb',name,'expected_error'):.3f}% | {100*get(cohort,'pb',name,'sampled_error'):.3f}% |")
    lines+=['','On the harmonic task, the loss advantage over confidence is explicitly **not an accuracy advantage**: rank-balanced has 14.766% expected classification error versus confidence at 13.803%. The folded gate also has lower logistic loss but higher error than confidence. Confidence plus quotient angles has the best mean classification error among the limited policies in this cohort. These are different quality objectives; the primary claims concern logistic loss.']
    lines+=['','### Transfer GD: optimizer context only','',
        '| Policy | Expected logistic loss | Sampled-label loss | Expected error | Sampled error |',
        '|---|---:|---:|---:|---:|']
    for name in ('raw','full'):
        lines.append(f"| {labels[name]} | {get('transfer','gd',name,'expected_risk'):.6f} | {get('transfer','gd',name,'sampled_label_risk'):.6f} | {100*get('transfer','gd',name,'expected_error'):.3f}% | {100*get('transfer','gd',name,'sampled_error'):.3f}% |")
    lines+=['','Both optimizers use 500 updates, but their work per update is different. Rates were selected from the bounded four-seed supervised pilot. These GD/PB results are not a matched-compute optimizer comparison. The task Bayes classification error is 10.9375%, and these GD states remain visibly underfitted.','',
        '## Four invariant-posterior sensitivity scenarios','',
        'Entries below are mean raw-minus-policy logistic loss; positive means an improvement. The model, coordinates, policy, quota and selected views are held fixed while only the hypothetical invariant posterior changes. Half uses p=1/2; rising uses normalized quotient u; falling uses 1-u; threshold uses 1[u>=1/2]. Rings use u=clip((r-.25)/1.25,0,1); transfer uses u=r-.4. These are four prespecified scenarios, not sharp adversarial extrema or a proof over every invariant posterior. Per-model signs and 95% model-bootstrap intervals are in world_gains.csv.']
    world_names=[('half_risk','Half'),('increasing_risk','Rising'),('decreasing_risk','Falling'),('threshold_risk','Threshold')]
    shown=['confidence','rank_balanced','rank_even','folded','full']
    for cohort in ('rings','transfer'):
        lines+=['',f'### {cohort}','',
            '| World | Confidence | Rank-balanced | Rank-even | Folded gate | Full half-turn |',
            '|---|---:|---:|---:|---:|---:|']
        for world,label in world_names:
            numbers=[float(next(r['mean_gain'] for r in gains if r['cohort']==cohort and r['method']=='pb' and r['policy']==name and r['world']==world)) for name in shown]
            lines.append('| '+label+' | '+' | '.join(f'{v:+.6f}' for v in numbers)+' |')
    lines+=['','Confidence has a negative mean gain in the half and falling worlds in both tasks. In falling worlds it harms all 16 models in each cohort. Rank-balanced and folded improve all 16 models in every displayed scenario. The folded gate has an exact half-turn symmetry property; near-flat finite-sample gains are not themselves a proof of exact identification. The new rank certificate is evaluated separately on independent probes.','',
        '## Calibration and actual prediction work','',
        '| Cohort | Shared fit | Forwards/model | Additional coordinate-only quotient draws/model | Mean fit seconds |',
        '|---|---|---:|---:|---:|']
    for r in costs:lines.append(f"| {r['cohort']} | {r['kind']} | {int(r['forwards_per_model']):,} | {int(r['coordinate_only_per_model']):,} | {float(r['mean_calibration_seconds']):.4f} |")
    forward_total=sum(int(r['first_forwards'])+int(r['second_forwards']) for r in individual)
    lines+=['',
        f"Across 32 primary models, these independent fits total {audit['total_calibration_forwards']:,} network forwards plus {audit['total_coordinate_only_draws']:,} coordinate-only quotient draws. The rank fit is shared by balanced/even/unbalanced policies; folded scores and angles are separate fits. The angle fit has 64 independent quotient clusters, not 8,192 independent observations. Confidence has zero model-specific calibration cost; GD context has no policy fit.",
        '',
        f"All 81,920 scored batches pass actual call and quota checks. Raw uses [256], limited policies [256,64], and full references [256,256]. The 320 model-policy scoring rows used {forward_total:,} forward examples as research evaluation work. There are 2,097,152 unique primary test inputs; transfer GD reuses its PB cohort's inputs. Certificate, model-training and local-bound costs are separate. No new latency or GPU claim is made in this round.",
        '',
        '## Independent analysis audit and chronology','',
        f"The post-run audit reconstructs all 320 outcome rows from saved predictions using an algebraically distinct logistic-loss expression, then recomputes all 136 contrasts with bootstrap multiplicity weights rather than indexed row averages. Maximum outcome discrepancy is {audit['maximum_outcome_error']:.3g}; maximum bootstrap discrepancy is {audit['maximum_bootstrap_error']:.3g}; all win counts match. All 160 mean/interval summaries also pass. The audit source and outputs are outside the frozen scientific files.",
        '',
        'Root independently reconstructed all 320 risks, checked all 81,920 ledgers, reproduced nine complete training trajectories exactly, and replayed 54 full-input deployments. The frozen design and preserved raw files make these checks reviewable. No failed training or evaluation was silently excluded.','']
    for label,file in [('Validation freeze','freeze.json'),('Root approval','ROOT_APPROVAL.json'),('Training start','training_start.json'),('Training commit','training_commit.json'),('Policy commit','policy_commit.json'),('Scoring start','score_start.json'),('Scoring complete','score_complete.json')]:
        obj=json.loads((HERE/file).read_text());lines.append(f"- {label}: {obj.get('timestamp_utc','recorded in '+file)}.")
    lines+=['','## Files and reproduction','',
        'Frozen scientific sources are study.py, adapters.py and PROTOCOL.md plus the dependencies listed in freeze.json. States, policy tables, predictions and manifests use relative paths. The results report and audit code are separate from method selection. The complete metrics are per_model.csv, summary.csv, contrasts.csv and world_gains.csv; analysis_audit.json and its CSVs document the independent arithmetic audit.',
        '',
        'For a fresh output copy with the same relative source/reference/legacy/transfer dependencies, run the stages below. Fresh training requires an approval record linked to its new freeze hash; stages refuse to overwrite completed outputs. The archived approval documents this completed run.',
        '',
        '```text','python validation/study.py freeze','python validation/study.py train',
        'python validation/study.py calibrate','python validation/study.py score',
        'python validation/study.py analyze','python validation/audit_analysis.py',
        'python validation/write_results.py','```','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    # Primary/descriptive paired contrasts: cohorts use different scales deliberately.
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.8))
    for ax,cohort,title in zip(axes,('rings','transfer'),('Original rings','Two-harmonic transfer')):
        for y,reference in enumerate(('confidence','rank_even','folded')):
            r=next(r for r in contrast if r['cohort']==cohort and r['method']=='pb' and r['policy']=='rank_balanced' and r['reference']==reference and r['outcome']=='expected_risk')
            primary=reference!='folded';lo=float(r['low98_75'] if primary else r['low95'])*1000;hi=float(r['high98_75'] if primary else r['high95'])*1000;mean=float(r['delta_risk'])*1000
            ax.errorbar(mean,2-y,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o' if primary else 's',color='#007F82' if primary else '#657487',capsize=4,lw=1.5)
        ax.axvline(0,color='#888888',ls='--',lw=1);ax.set_yticks([2,1,0],['Versus confidence','Versus rank-even','Versus folded'])
        ax.set_ylim(-.6,2.6);ax.set_xlabel('Balanced minus reference loss × 1,000\nNegative favors balanced');ax.set_title(title);ax.grid(axis='x',alpha=.18)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.suptitle('Empirical ranking changes with the task; the folded control remains strong',fontsize=11)
    fig.text(.5,.012,'16 fresh PB models per cohort. Primary intervals: 98.75%; folded comparison: descriptive 95%. Cohort axes have different scales.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.91));fig.savefig(HERE/'primary_contrasts.png',dpi=200);fig.savefig(HERE/'primary_contrasts.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.8))
    for ax,cohort,title in zip(axes,('rings','transfer'),('Original rings','Two-harmonic transfer')):
        for name,label,color in [('confidence','Confidence','#C77720'),('rank_balanced','Rank-balanced','#007F82'),('folded','Folded gate','#7654A7')]:
            r=[next(r for r in gains if r['cohort']==cohort and r['method']=='pb' and r['policy']==name and r['world']==world) for world,_ in world_names]
            mean=np.array([float(v['mean_gain']) for v in r]);lo=np.array([float(v['low95']) for v in r]);hi=np.array([float(v['high95']) for v in r])
            ax.errorbar(np.arange(4),mean,yerr=np.vstack((mean-lo,hi-mean)),fmt='o-',label=label,color=color,capsize=3,lw=1.4)
        ax.axhline(0,color='#888888',ls='--',lw=1);ax.set_xticks(np.arange(4),[label for _,label in world_names]);ax.set_title(title)
        ax.set_ylabel('Raw minus policy logistic loss');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.suptitle('Four fixed posterior worlds expose the limits of confidence routing',fontsize=11)
    fig.text(.5,.012,'Positive gain is better. Means and 95% model-bootstrap intervals; these finite scenarios are not a uniform-posterior guarantee.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.91));fig.savefig(HERE/'world_sensitivity.png',dpi=200);fig.savefig(HERE/'world_sensitivity.svg');plt.close(fig)

if __name__=='__main__':main()
