"""Derived accounting, figures and English report; never changes primary fits."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import runner

def account():
    runner.verify();runner.verify_commit('ANALYSIS_COMPLETE.json');rows=[];totals={}
    for phase in ('development','confirmation'):
        runner.verify_commit(f'{phase}/MODEL_COMMIT.json');runner.verify_commit(f'{phase}/FIT_COMMIT.json');runner.verify_commit(f'{phase}/SCORE_COMPLETE.json')
        train=runner.read_csv(runner.OUT/phase/'training.csv');fits=runner.read_csv(runner.OUT/phase/'temperature_fits.csv');deploy=runner.read_csv(runner.OUT/phase/'deployment_ledger.csv');count,_=runner.phase_config(phase)
        sampling=[json.loads((runner.OUT/phase/'splits'/f'{p}_sampling.json').read_text()) for p in range(1,count+1)]
        for tr in train:
            for pipeline in ('raw','full'):
                fit=next(r for r in fits if r['pair']==tr['pair'] and r['procedure']==tr['procedure'] and r['pipeline']==pipeline)
                rows.append(dict(phase=phase,pair=tr['pair'],procedure=tr['procedure'],pipeline=pipeline,unique_fit_labels=4352,n_train=int(tr['n_train']),n_validation=0,n_calibration=int(tr['n_calibration']),
                  training_updates=int(tr['updates_executed']),training_example_updates=int(tr['training_example_updates']),standalone_representation_forward_examples=int(tr['standalone_representation_forward_examples']),jacobian_forward_examples=int(tr['jacobian_forward_examples']),
                  jacobian_rows=int(tr['jacobian_rows']),jacobian_scalar_entries=int(tr['jacobian_scalar_entries']),loss_derivative_examples=int(tr['loss_derivative_examples']),separate_reverse_mode_backward_calls=0,
                  lu_144_solves=int(tr['lu_144_solves']),nominal_training_arithmetic=int(tr['nominal_arithmetic_flops']),tanh_evaluations=int(tr['tanh_scalar_evaluations']),exp_evaluations=int(tr['exp_scalar_evaluations']),logaddexp_evaluations=int(tr['logaddexp_scalar_evaluations']),
                  training_diagnostic_forward_examples=int(tr['training_diagnostic_forward_examples']),validation_forwards=0,standalone_calibration_forwards=int(fit['standalone_calibration_forwards']),
                  scalar_temperature_parameters=1,temperature_derivative_evaluations=int(fit['derivative_evaluations']),temperature_objective_evaluations=2,temperature_derivative_label_evaluations=int(fit['derivative_label_evaluations']),temperature_objective_label_evaluations=int(fit['objective_label_evaluations']),
                  deployment_forward_examples_per_item=1 if pipeline=='raw' else 2))
        totals[phase]=dict(models=len(train),temperature_fits=len(fits),accepted_training_label_union=sum(r['accepted_training_union'] for r in sampling),calibration_label_union=count*2048,validation_label_union=0,
          fit_label_union=sum(r['unique_fitting_label_union'] for r in sampling),evaluation_label_union=count*65536,synthetic_training_rejection_proposals=sum(m['proposals'] for r in sampling for m in r['metadata']),
          training_example_updates=sum(int(r['training_example_updates']) for r in train),nominal_training_arithmetic=sum(int(r['nominal_arithmetic_flops']) for r in train),lu_144_solves=sum(int(r['lu_144_solves']) for r in train),
          diagnostic_training_forwards=sum(int(r['training_diagnostic_forward_examples']) for r in train),validation_forwards=0,actual_shared_calibration_forwards=sum(int(r['actual_shared_calibration_forwards']) for r in fits if r['pipeline']=='raw'),
          temperature_derivative_label_evaluations=sum(int(r['derivative_label_evaluations']) for r in fits),temperature_objective_label_evaluations=sum(int(r['objective_label_evaluations']) for r in fits),
          actual_evaluation_forwards=sum(int(r['inference_forward_examples']) for r in deploy),actual_evaluation_invocations=sum(int(r['inference_invocations']) for r in deploy))
    timing_totals=None
    if (runner.OUT/'TIMING_AUDIT.json').exists():
        timing_rows=runner.read_csv(runner.OUT/'complete_fit_timing.csv');winner=runner.selection()['selected_procedure'];paired=[]
        for block in range(4):
            base=next(r for r in timing_rows if int(r['block'])==block and r['procedure']==runner.BASELINE)
            selected=next(r for r in timing_rows if int(r['block'])==block and r['procedure']==winner)
            paired.append(dict(block=block,role=base['role'],selected_procedure=winner,selected_seconds=float(selected['total_fit_seconds']),baseline_seconds=float(base['total_fit_seconds']),ratio=float(selected['total_fit_seconds'])/float(base['total_fit_seconds']),identical_procedure=winner==runner.BASELINE))
        runner.cw(runner.OUT/'timing_paired_ratios.csv',paired)
        calibration_reference=runner.read_csv(runner.OUT/'development/temperature_fits.csv')
        timing_totals={role:dict(complete_fits=sum(r['role']==role for r in timing_rows),nominal_training_arithmetic=sum(int(r['nominal_arithmetic_flops']) for r in timing_rows if r['role']==role),
          training_example_updates=sum(int(r['training_example_updates']) for r in timing_rows if r['role']==role),lu_solves=sum(int(r['lu_144_solves']) for r in timing_rows if r['role']==role),
          diagnostic_forwards=sum(int(r['training_diagnostic_forwards']) for r in timing_rows if r['role']==role),calibration_forwards=sum(int(r['calibration_forwards']) for r in timing_rows if r['role']==role),
          temperature_derivative_label_evaluations=sum(sum(int(f['derivative_label_evaluations']) for f in calibration_reference if f['pair']=='1' and f['procedure']==r['procedure']) for r in timing_rows if r['role']==role),
          temperature_objective_label_evaluations=sum(sum(int(f['objective_label_evaluations']) for f in calibration_reference if f['pair']=='1' and f['procedure']==r['procedure']) for r in timing_rows if r['role']==role),new_unique_labels=0) for role in ('warmup','measured')}
    runner.cw(runner.OUT/'standalone_cost_ledger.csv',rows)
    runner.jo(runner.OUT/'COST_TOTALS.json',dict(utc=runner.now(),phases=totals,per_fitted_calibrated_procedure_labels=4352,
      development_recipe_search_labels=totals['development']['fit_label_union']+totals['development']['evaluation_label_union'],
      all_accepted_fit_plus_evaluation_label_union=sum(r['fit_label_union']+r['evaluation_label_union'] for r in totals.values()),timing_audit_work=timing_totals,
      note='The 4,352 budget is per fitted procedure, not end-to-end recipe discovery. All development grid fits and 524,288 selection labels are charged separately. Timing reuses existing development labels and is additional computational work; see TIMING_AUDIT.json. Synthetic rejection proposals are separately disclosed, and expected confirmation metrics use known posterior access.'))

def report():
    runner.verify();runner.verify_commit('ANALYSIS_COMPLETE.json');selection=runner.selection();winner=selection['selected_procedure'];dev=runner.read_csv(runner.OUT/'development/summary.csv');conf=runner.read_csv(runner.OUT/'confirmation/summary.csv');contrasts=runner.read_csv(runner.OUT/'confirmation/contrasts.csv')
    costs=json.loads((runner.OUT/'COST_TOTALS.json').read_text());timing=json.loads((runner.OUT/'TIMING_AUDIT.json').read_text()) if (runner.OUT/'TIMING_AUDIT.json').exists() else None
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for arm,label,style in [('raw_T','Raw, own temperature: selection criterion','-o'),('full_T','Full mean, own temperature: descriptive only','--s')]:
        rs=[next(r for r in dev if r['procedure']==p['name'] and r['arm']==arm) for p in runner.PROCEDURES]
        ax.plot(runner.NS,[float(r['sampled_nll']) for r in rs],style,label=label)
    selected=next(r for r in dev if r['procedure']==winner and r['arm']=='raw_T');ax.scatter([int(selected['n_train'])],[float(selected['sampled_nll'])],marker='*',s=220,color='#c43c39',zorder=4,label='Selected on development only')
    ax.set_xticks(runner.NS);ax.set_xlabel('Training labels (4352 total fitting labels per procedure)');ax.set_ylabel('Sampled-label NLL, mean over 8 paired seeds');ax.set_title('Development frontier: selected minimum is not confirmation evidence');ax.grid(alpha=.2);ax.legend(fontsize=9)
    for ext in ('png','pdf'):fig.savefig(runner.OUT/f'development_frontier.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    for i,arm in enumerate(('raw_T','full_T')):
        rows=runner.read_csv(runner.OUT/'confirmation/per_model.csv');base=sorted([r for r in rows if r['procedure']==runner.BASELINE and r['arm']==arm],key=lambda r:int(r['pair']));chosen=sorted([r for r in rows if r['procedure']==winner and r['arm']==arm],key=lambda r:int(r['pair']))
        if winner!=runner.BASELINE:
            for a,b in zip(base,chosen):axes[i].plot([0,1],[float(a['expected_nll']),float(b['expected_nll'])],'-o',color='#b5c7d5',alpha=.7,markersize=3)
            axes[i].plot([0,1],[np.mean([float(r['expected_nll']) for r in base]),np.mean([float(r['expected_nll']) for r in chosen])],'-o',color='#174a70',lw=3,markersize=8,label='16-seed mean')
            axes[i].set_xticks([0,1],['Baseline 2304',winner.replace('train_','Selected ')])
        else:
            axes[i].scatter([int(r['pair']) for r in base],[float(r['expected_nll']) for r in base]);axes[i].set_xlabel('Fresh confirmation seed')
        title='Raw calibrated output' if arm=='raw_T' else 'Full calibrated output'
        if winner==runner.BASELINE:title+='\nBaseline retained; absolute performance only'
        axes[i].set_ylabel('Expected NLL');axes[i].set_title(title,fontsize=10);axes[i].grid(alpha=.2)
    fig.suptitle('Independent confirmation: 16 new paired seeds' if winner!=runner.BASELINE else 'Independent confirmation: 16 fresh baseline fits')
    for ext in ('png','pdf'):fig.savefig(runner.OUT/f'confirmation_pairs.{ext}',dpi=180)
    plt.close(fig)
    primary=[r for r in contrasts if r['primary']=='True'];lookup=lambda records,name,arm:next(r for r in records if r['procedure']==name and r['arm']==arm)
    raw=lookup(conf,winner,'raw_T');base=lookup(conf,runner.BASELINE,'raw_T');full=lookup(conf,winner,'full_T');basefull=lookup(conf,runner.BASELINE,'full_T')
    text=['# Allocation frontier: development selection and independent confirmation','',
      f"The eight-seed development criterion selected `{winner}` from the four frozen allocations. The selected recipe was committed before any of the 16 confirmation seed streams opened. All planned fits, calibrations and evaluations completed; no replacement seeds or outcome-driven schedule changes were used.",'',
      '## Development selection and full grid','',
      'The selection criterion was mean sampled-label NLL of calibrated raw output. Full output, expected posterior risk, error and timing did not enter selection. Development metrics below are descriptive; the selected minimum is subject to winner\'s bias and is not independent evidence of improvement.','',
      '| Training | Calibration | Updates | Raw selection NLL | Full NLL (descriptive) | Selected |','|---:|---:|---:|---:|---:|---|']
    for p in runner.PROCEDURES:
        a=lookup(dev,p['name'],'raw_T');b=lookup(dev,p['name'],'full_T');text.append(f"| {p['n']} | {p['calibration']} | {p['steps']} | {float(a['sampled_nll']):.6f} | {float(b['sampled_nll']):.6f} | {'yes' if p['name']==winner else ''} |")
    descriptive_full=min([r for r in dev if r['arm']=='full_T'],key=lambda r:float(r['sampled_nll']))
    if descriptive_full['procedure']!=winner:text += ['',f"The descriptive full-output development minimum occurred at `{descriptive_full['procedure']}` ({float(descriptive_full['sampled_nll']):.6f}). Full-output loss was not the selection criterion, so this observation did not promote that allocation. It has no independent candidate-versus-baseline confirmation in this round."]
    text += ['',f"Tie tolerance was 1e-12, with the smallest training allocation preferred inside tolerance. The development selection used 524,288 labeled evaluation items beyond 47,104 unique fitting labels across its eight paired datasets. The resulting 571,392-label recipe-discovery cost is separate from 4,352 labels per fitted procedure.",'',
      '## Independent confirmation','',
      f"Calibrated raw expected NLL: selected {float(raw['expected_nll']):.6f}, baseline {float(base['expected_nll']):.6f}. Calibrated full expected NLL: selected {float(full['expected_nll']):.6f}, baseline {float(basefull['expected_nll']):.6f}.",'',
      '| Output | Selected minus baseline NLL | Adjusted interval | Status |','|---|---:|---|---|']
    for r in primary:
        interval=f"[{float(r['lower']):+.6f}, {float(r['upper']):+.6f}]" if r['lower'] else 'not applicable'
        text.append(f"| {r['arm']} | {float(r['mean_difference']):+.6f} | {interval} | {r['inference_status']} |")
    if winner==runner.BASELINE:text += ['','The baseline won selection. It was trained once per confirmation seed; selected-minus-baseline is zero by identity. No runner-up was promoted, and no improvement claim or test is made.']
    else:
        findings=[]
        for r in primary:
            direction='favored the selected allocation' if float(r['upper'])<0 else 'favored the baseline' if float(r['lower'])>0 else 'was inconclusive'
            findings.append(f"{r['arm']} {direction}")
        text += ['','The primary adjusted intervals '+ '; '.join(findings)+'. These are approximate paired bootstrap intervals from 20,000 resamples of 16 seed clusters, with alpha 0.05/2 for each of the two expected-NLL contrasts.']
    text += ['',f"The chosen raw development selection NLL was {selection['selected_development_sampled_nll']:.6f}; its fresh confirmation sampled-label NLL was {float(raw['sampled_nll']):.6f}. These are distinct datasets; the latter was never used to reconsider selection.",'',
      '## Resource accounting','',
      'All calibrated procedures spend 4,352 accepted fitting labels. The nominal arithmetic reference is 207,260,412,000 operations with C(n)=179045n+2001144 per update. Update counts 500/375/321/300 have maximum floor shortfall 0.30552%. These counts include Jacobian, Gram and general LU arithmetic; special functions, memory and control overhead are separate. One update uses two representation-forward streams, 4n Jacobian rows, and one 144-dimensional solve, with no separate reverse-mode backward call.','',
      'Validation labels and forwards are zero by prospective choice. Four saved diagnostic checkpoints per trajectory incur additional training forwards and do not select stopping. Raw inference costs one model forward per item and full mean costs two. Both final outputs have their own label-fitted temperature; scalar calibration and shared raw/partner costs are explicit. Uncalibrated diagnostics have smaller actual label counts and are not equal-budget competitors.','',
      '| Phase | Models | Fit-label union | Evaluation-label union | Example-updates | Nominal arithmetic | Calibration forwards | Evaluation forwards |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for phase,c in costs['phases'].items():text.append(f"| {phase} | {c['models']} | {c['fit_label_union']:,} | {c['evaluation_label_union']:,} | {c['training_example_updates']:,} | {c['nominal_training_arithmetic']:,} | {c['actual_shared_calibration_forwards']:,} | {c['actual_evaluation_forwards']:,} |")
    text += ['',f"Total accepted fitting plus evaluation label union across both phases: {costs['all_accepted_fit_plus_evaluation_label_union']:,}. Synthetic rejection proposals are separately counted in COST_TOTALS.json. The temperature derivative/objective label evaluations and all per-procedure forward/Jacobian/solve counts are in standalone_cost_ledger.csv. Concurrent training times are descriptive only."]
    if timing:
        text += ['','## Isolated complete-procedure latency','','An exclusive CPU window timed one warmup block and three measured paired blocks on development pair 1. Each fit includes all training updates, diagnostic checkpoint work, both calibration forward streams, and both scalar temperature fits. Input generation/loading, output I/O, held-out evaluation and verification are excluded. Parameters and temperatures matched the committed development fits within 1e-12; reused labels added zero new unique observations.','','| Procedure | Median total seconds | Observed range | Median training | Median calibration |','|---|---:|---|---:|---:|']
        for t in timing['summaries']:text.append(f"| {t['procedure']} | {t['median_total_seconds']:.4f} | [{t['minimum_total_seconds']:.4f}, {t['maximum_total_seconds']:.4f}] | {t['median_training_seconds']:.4f} | {t['median_calibration_seconds']:.4f} |")
        ratios=[float(r['ratio']) for r in runner.read_csv(runner.OUT/'timing_paired_ratios.csv') if r['role']=='measured']
        text += ['', 'Measured paired-block selected/baseline time ratios: '+', '.join(f'{r:.4f}' for r in ratios)+f'; median {float(np.median(ratios)):.4f}.']
        if winner==runner.BASELINE:text += ['','Because the selected procedure is the baseline itself, these ratios are one by identity. Only baseline latency was measured; no comparative speed claim is made.']
        text += ['','These are actual complete-procedure times for three repeats of one fixed dataset, not projections or independent-dataset latency uncertainty. Timing did not select the allocation. Nominal arithmetic equality does not imply equal time.']
    text += ['','## Limits, prior art and auditability','',
      'The allocation changes training size, calibration size and update count together; this experiment does not separately identify calibration-estimation variance and representation learning effects. It explores four prespecified points rather than proving a global optimum. Eight development and 16 confirmation seeds limit precision. Posterior expected metrics rely on the known synthetic law and support no real-data guarantee. A confidence interval crossing zero is inconclusive. All negative or inconclusive findings are retained.','',
      'Temperature scaling is prior art ([Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)); compute allocation is established broadly ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)); finite-sample model-selection bias is established ([Cawley and Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). The present contribution is a controlled frontier selection and independent confirmation under the existing small PB model.','',
      'PROTOCOL.md and SOURCE_FREEZE.json define the experiment before fresh outcomes. Each phase has MODEL_COMMIT, FIT_COMMIT and SCORE_COMPLETE artifacts. SELECTION_COMMIT binds the selected recipe before confirmation starts. Raw per-item predictions, labels, model states and checkpoints allow independent reconstruction; posterior values are retained only for confirmation evaluation. Frozen portable source and commands are in source/ and REPRODUCE.md. No earlier archive or source file was changed.']
    (runner.OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['account','report']);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.out:runner.OUT=args.out.resolve()
    globals()[args.stage]()
