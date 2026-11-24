"""Render descriptive study artifacts from immutable primary outputs."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import runner

OUT=runner.OUT
def rows(name):return list(csv.DictReader((OUT/name).open(newline='')))
def mean(rs,key):return float(np.mean([float(r[key]) for r in rs]))
def main():
    runner.verify();runner.verify_commit('SCORE_COMPLETE.json')
    summary=rows('summary.csv');contrast=rows('contrasts.csv');training=rows('training.csv');actual=rows('actual_training_work.csv');temps=rows('temperature_fits.csv');per=rows('per_model.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,arm in enumerate(('raw_T','full_T')):
        for regime,names,style in [('500 updates',[f'fixed_{n}' for n in runner.NS],'-o'),('Matched nominal arithmetic',['flop_256','flop_768','flop_1280','fixed_2304'],'--s')]:
            chosen=[next(r for r in summary if r['procedure']==name and r['arm']==arm) for name in names]
            for row,metric in enumerate(('expected_nll','expected_error')):
                axes[row,col].plot(runner.NS,[float(r[metric]) for r in chosen],style,label=regime)
        for row,metric in enumerate(('expected_nll','expected_error')):
            ex=next(r for r in summary if r['procedure']=='examples_256' and r['arm']==arm)
            va=next(r for r in summary if r['procedure']=='validation_1280' and r['arm']==arm)
            axes[row,col].scatter([256],[float(ex[metric])],marker='D',s=55,label='Exact example work',color='#9b59b6')
            axes[row,col].scatter([1280],[float(va[metric])],marker='*',s=120,label='Validation allocation',color='#2e8b57')
            axes[row,col].set_xlabel('Training labels (all calibrated procedures: 4352 total labels)')
            axes[row,col].set_ylabel('Expected log loss' if row==0 else 'Expected classification error')
            axes[row,col].grid(alpha=.2);axes[row,col].set_xticks(runner.NS)
        axes[0,col].set_title('Raw output, own temperature' if col==0 else 'Full mean output, own temperature')
    axes[0,0].legend(fontsize=8);fig.suptitle('Allocation outcomes across 16 paired model seeds')
    for ext in ('png','pdf'):fig.savefig(OUT/f'allocation_outcomes.{ext}',dpi=180)
    plt.close(fig)
    primary=[r for r in contrast if r['primary']=='True']
    fig,ax=plt.subplots(figsize=(11,5.2),constrained_layout=True)
    labels=[]
    for i,r in enumerate(primary):
        center=float(r['mean_difference']);lo=float(r['lower']);hi=float(r['upper']);ax.errorbar(center,i,xerr=[[center-lo],[hi-center]],fmt='o',capsize=4,color='#215b8b')
        labels.append(r['contrast'].replace('_',' '))
    ax.set_yticks(range(len(primary)),labels);ax.invert_yaxis();ax.axvline(0,color='black',lw=1);ax.set_xlabel('Expected NLL difference (first procedure minus comparator)')
    ax.set_title('Eight primary contrasts: multiplicity-adjusted paired bootstrap intervals');ax.grid(axis='x',alpha=.2)
    for ext in ('png','pdf'):fig.savefig(OUT/f'primary_contrasts.{ext}',dpi=180)
    plt.close(fig)
    base_raw=next(r for r in summary if r['procedure']=='fixed_2304' and r['arm']=='raw_T')
    base_full=next(r for r in summary if r['procedure']=='fixed_2304' and r['arm']=='full_T')
    selected=[int(r['selected_step']) for r in training if r['procedure']=='validation_1280']
    selection_counts={step:selected.count(step) for step in (100,250,500,896)}
    fit_status={status:sum(r['status']==status for r in temps) for status in sorted({r['status'] for r in temps})}
    text=['# Label allocation and training-work results','',
      'All nine frozen procedures completed on all 16 fresh paired seeds. Every calibrated procedure used 4,352 fitting labels. The fixed 2,304-training/2,048-calibration baseline is retained. The prior completed study was not modified.','',
      '## Main results','',
      f"The baseline achieved expected NLL {float(base_raw['expected_nll']):.6f} with calibrated raw output and {float(base_full['expected_nll']):.6f} with calibrated full mean output. The table reports means over paired model seeds; lower is better.",'',
      '| Procedure | Raw NLL | Full NLL | Raw error | Full error |','|---|---:|---:|---:|---:|']
    for p in runner.PROCEDURES:
        a=next(r for r in summary if r['procedure']==p['name'] and r['arm']=='raw_T');b=next(r for r in summary if r['procedure']==p['name'] and r['arm']=='full_T')
        text.append(f"| {p['name']} | {float(a['expected_nll']):.6f} | {float(b['expected_nll']):.6f} | {float(a['expected_error']):.6f} | {float(b['expected_error']):.6f} |")
    text+=['','## Prespecified inference','','Eight NLL contrasts use 20,000 paired resamples of 16 seed clusters and alpha 0.05/8 each. Intervals are approximate bootstrap uncertainty, not a theorem or proof of equivalence. Negative favors the first named procedure.','','| Contrast | Difference | Adjusted interval | Wins / 16 |','|---|---:|---:|---:|']
    for r in primary:text.append(f"| {r['contrast']} | {float(r['mean_difference']):+.6f} | [{float(r['lower']):+.6f}, {float(r['upper']):+.6f}] | {r['wins']} |")
    text+=['','## Cost interpretation','',
      'Fixed steps do not equalize training work. The exact-example endpoint has 1,152,000 example-updates for both n=256/4500 updates and n=2304/500 updates, with 4,500 versus 500 LU solves. The nominal-arithmetic arms include all counted Jacobian/Gram/solve arithmetic. They use 4,332/1,485/896 updates at n=256/768/1280; maximum budget shortfall is 0.05996%. Elementary-function internals, memory, pivot comparisons and elapsed time are separate. No whole-system compute or time equality is claimed.','',
      'Each update has n standalone representation forwards plus n forwards within the analytic Jacobian, 4n Jacobian rows, and one 144-dimensional solve. The Jacobian contraction forms the loss gradient; there is no independent reverse-mode backward call. Validation consumes 4,096 extra forward examples for the selected arm. Full inference costs two forwards per item; raw inference costs one.','',
      f"Actual shared trajectory work: {sum(int(r['training_example_updates']) for r in actual):,} example-updates; {sum(int(r['lu_solves_144']) for r in actual):,} solves; {sum(int(r['nominal_arithmetic_flops']) for r in actual):,} nominal arithmetic operations. Diagnostic training-loss monitoring: {sum(int(r['training_diagnostic_forward_examples']) for r in actual):,} additional forward examples. Calibration actual shared forward examples: {sum(int(r['shared_actual_forward_examples']) for r in temps if r['pipeline']=='raw'):,}. Test forwards: {sum(int(r['inference_forwards']) for r in per):,}.",'',
      'Per-procedure standalone costs are in `training.csv`; actual amortized experiment costs are in `actual_training_work.csv`. The label union is 7,424 fitting labels per paired seed, with 4,352 visible to each calibrated procedure. Synthetic rejection proposal counts are recorded separately. Training elapsed times collected during concurrent work are descriptive only.','',
      '## Calibration and limitations','',
      'Validation selected-update counts: '+', '.join(f'{step}: {count}/16' for step,count in selection_counts.items())+'. Temperature optimum statuses across all 288 fits: '+', '.join(f'{status}: {count}' for status,count in fit_status.items())+'. The validation arm consumes all 896 scheduled training updates even when an earlier checkpoint is deployed.','',
      'Both raw and full deployed outputs were calibrated against sampled iid labels with independently fitted positive scalar temperatures. Full output with raw temperature is retained as a transfer diagnostic, not selected as the winning calibration. Calibration preserves each fixed pipeline\'s prediction signs. Known posterior probabilities are used only in generation and final evaluation, never in fitting. Uncalibrated diagnostics use fewer labels and are not included as equal-budget competitors.','',
      'The modest prespecified grid cannot identify a global optimal allocation. The bootstrap has only 16 model clusters; an interval crossing zero is inconclusive rather than evidence of equivalence. The structured synthetic data law, fixed small model and full-batch PB optimizer limit external validity. Matched nominal arithmetic is a transparent accounting model, not measured hardware FLOPs or equal time.','',
      '## Reproduction and audit','',
      'The complete procedure definitions and work derivation are in `PROTOCOL.md`. The source freeze, passing preflight, internal approval, model commit, fit commit and scoring commit establish chronological separation. The source runner is `pullback-feature-learning/experiments/budget_allocation/runner.py`, with the workspace `.venv/Scripts/python.exe`. Run stages train, calibrate, score, analyze into a fresh branch after its own protocol/approval. Retained `states/`, `splits/`, `temperatures/`, and `predictions/` reproduce every per-model risk. The five output arms reconstruct from committed raw/full per-item logits and temperatures.','',
      'Figure files: `allocation_outcomes.png/.pdf`, `primary_contrasts.png/.pdf`. Primary and secondary contrast tables are `contrasts.csv` and `paired_contrasts.csv`; all 720 procedure/output/seed rows are in `per_model.csv`.','',
      'Temperature scaling is prior art ([Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)). Compute allocation is established broadly ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)); this small PB experiment does not extend their scaling law.']
    if (OUT/'TIMING_AUDIT.json').exists():
        timing=json.loads((OUT/'TIMING_AUDIT.json').read_text());text+=['','## Isolated timing audit','',timing['summary']]
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps(dict(primary=primary,source_freeze=runner.sha(OUT/'SOURCE_FREEZE.json')),indent=2))

if __name__=='__main__':main()
