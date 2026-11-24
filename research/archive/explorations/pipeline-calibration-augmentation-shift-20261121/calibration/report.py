"""Readable report and figure from completed frozen outcomes, without fitting."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def read(name):
    with (HERE/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def run():
    assert (HERE/'ANALYSIS_COMPLETE.json').exists()
    summary=read('summary.csv');per=read('per_model.csv');contrasts=read('contrasts.csv');fits=read('temperature_fits.csv');training=read('training.csv')
    idx={(r['family'],r['arm']):r for r in summary};families=['legacy','selected','extra_train']
    labels={'legacy':'Legacy training','selected':'Selected training','extra_train':'More training labels'}
    colors={'legacy':'#9e5a35','selected':'#166d94','extra_train':'#348253'}
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.1),layout='constrained')
    arms=['raw_rawT','full_finalT','interp_finalT']
    for family in families:
        axs[0].plot(range(3),[float(idx[(family,a)]['expected_nll']) for a in arms],'-o',color=colors[family],label=labels[family])
    axs[0].axhline(.4929065112,ls=':',color='0.4',label='Bayes')
    axs[0].set(xticks=range(3),xticklabels=['Raw (1)','Full (2)','Interp. (1.25)'],ylabel='Expected log loss',title='Final temperatures: equal 4,352 labels');axs[0].tick_params(axis='x',rotation=25);axs[0].legend(fontsize=7)
    for j,label in enumerate(['selected_full_final_vs_rawT','selected_interp_final_vs_rawT']):
        r=next(r for r in contrasts if r['contrast']==label and r['metric']=='expected_nll')
        mean=float(r['mean_difference']);lo=float(r['lower']);hi=float(r['upper'])
        axs[1].errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color='#166d94',label='Mean and 98.75% interval' if j==0 else None)
    axs[1].axhline(0,ls='--',color='0.5',lw=1)
    axs[1].set(xticks=range(2),xticklabels=['Full (2 calls)','Interp. (1.25 calls)'],ylabel='Loss: final-fitted T minus raw-fitted T',title='Selected model: calibration target');axs[1].legend(fontsize=7)
    for family in families:
        axs[2].plot(range(3),[float(idx[(family,a)]['expected_error']) for a in arms],'-o',color=colors[family],label=labels[family])
    axs[2].axhline(.226562651,ls=':',color='0.4',label='Bayes');axs[2].set(xticks=range(3),xticklabels=['Raw','Full','Interp.'],ylabel='Expected classification error',title='Error: temperature preserves signs');axs[2].legend(fontsize=7)
    for ax in axs:ax.grid(axis='y',alpha=.15)
    fig.savefig(HERE/'final_pipeline_results.png',dpi=180);fig.savefig(HERE/'final_pipeline_results.pdf');plt.close(fig)
    text='''# Final-pipeline calibration at an equal fitting-label budget

The prospective confirmation is complete: 16 independent seed pairs, three fixed-capacity model families, three fixed prediction pipelines, 144 scalar-temperature fits, and 384 performance rows. No temperature, field, checkpoint rule, or label allocation was selected from performance outcomes. The earlier two-amplitude task, posterior law, and 144-parameter architecture are unchanged.

**Final-pipeline calibration helps, and allocating more of the fixed label budget to training helps more on this task.** For the selected model, fitting temperature to the final output lowers expected log loss by 0.013098 for full averaging and 0.006157 for interpolation, with improvement on all 16 seeds. Reallocating 2,048 labels from calibration to training lowers raw calibrated loss by 0.094195 versus the legacy allocation and improves raw classification error by 9.78 percentage points. However, the previously selected validation-based training allocation is worse than the legacy allocation in the primary raw-calibrated comparison (+0.005094 loss). That unfavorable primary result is retained; the study does not support every more elaborate procedure.

## Equal resources and fixed inference rules

| Procedure | Training labels | Validation labels | Calibration labels | Total unique fitting labels |
|---|---:|---:|---:|---:|
| Legacy PB4/500 | 256 | 0 | 4,096 | 4,352 |
| Previously selected PB2 + checkpoint selection | 256 | 2,048 | 2,048 | 4,352 |
| Extra labels assigned to PB training | 2,304 | 0 | 2,048 | 4,352 |

Each extra-training model uses the identical base 256 examples and initialization as its paired controls, plus an independent balanced 2,048-example sample. It remains exactly class balanced. The experimental union of all fitting splits has 8,448 unique labels per seed, because the alternative procedures share some, but not all, samples. The uncalibrated outputs are diagnostic arms with smaller actual label use; the equal-budget comparisons concern calibrated procedures.

The PB metric is a Jacobian sum while the loss gradient is a mean. For 2,304 training examples the fixed rate/damping are therefore 36/0.9, equivalent to 4/0.1 at 256 examples after normalizing the metric by sample count. Independent finite-difference and duplicate-data checks passed before release. Giving the model more data did not silently slow each update by a factor of nine. All extra-training and legacy models execute 500 updates; selected models evaluate the same four earlier declared checkpoints. Runtime and update budgets are reported separately from label counts.

The interpolation field is fitted once from 65,536 unlabelled model forwards per base model and committed before any temperature is fitted. Its exact 64-of-256 mask is identical across uncalibrated, raw-temperature, and final-temperature arms. Each temperature is fitted from sampled labels to the outputs of the relevant fixed pipeline. Calibration-batch ties and performance ties are independent. Positive scaling preserves every pipeline's classification signs, a property checked on every performance input.

## Expected risks on the independent performance inputs

All entries below use exactly 4,352 fitting labels. Raw inference costs one forward per input, full averaging costs two, and fixed interpolation costs 1.25. Inference cost is not equalized across these separate pipeline references.

| Procedure | Raw: raw-fitted T | Full: raw-fitted T | Full: final-fitted T | Interp.: raw-fitted T | Interp.: final-fitted T |
|---|---:|---:|---:|---:|---:|
'''
    for family in families:
        values=[float(idx[(family,a)]['expected_nll']) for a in ('raw_rawT','full_rawT','full_finalT','interp_rawT','interp_finalT')]
        text+='| '+labels[family]+' | '+' | '.join(f'{v:.6f}' for v in values)+' |\n'
    text+='''
| Primary paired log-loss contrast | Mean difference | 98.75% interval | Seed wins |
|---|---:|---:|---:|
'''
    primary=[r for r in contrasts if r['primary']=='True']
    for r in primary:text+=f"| {r['contrast']} | {float(r['mean_difference']):+.6f} | [{float(r['lower']):+.6f}, {float(r['upper']):+.6f}] | {r['wins']}/16 |\n"
    text+='''
Negative differences favor the first named arm. The four prespecified primary comparisons share a nominal 95% bootstrap family through alpha=0.05/4 per interval. Bootstrap draws resample 16 paired model seeds, not the individual performance batches. These are approximate finite-model uncertainty intervals, not distribution-free certificates. All other metrics and comparisons remain secondary.

Classification must be interpreted separately. A positive scalar temperature changes neither the fixed allocation mask nor the sign of its final logits, so raw-fitted and final-fitted temperatures have exactly equal classification errors within each model/pipeline. Any error differences between model families arise from training or the prediction pipeline, not scalar calibration.

| Procedure | Raw error | Full error | Interpolation error |
|---|---:|---:|---:|
'''
    for family in families:text+='| '+labels[family]+' | '+' | '.join(f"{float(idx[(family,a)]['expected_error']):.6f}" for a in ('raw_rawT','full_finalT','interp_finalT'))+' |\n'
    text+='''
The Bayes values for this synthetic task are approximately 0.492907 log loss and 0.226563 classification error. The study tests allocation of a fixed number of fitting labels and calibration targets; it does not claim the task is solved or that a scalar temperature can repair classification errors. Every uncalibrated reference, failed attempt (if any), and unfavorable result remains in the raw files.

## Calibration and computational accounting

The same calibration labels may support several alternative scalar fits; this is not counted as new unique labels for each temperature. The code shares paid full-view calibration logits across the three fits. Actual temperature-calibration forwards total 262,144. Standalone per-model fitting costs are n for raw, 2n for full, and 1.25n for interpolation; their sum across all alternative fits is 557,056 forwards. Field fitting adds 3,145,728 forwards. Validation checkpoint selection adds 131,072 forwards across the 16 selected models, while the fixed-step families use no validation selection.

The complete physical performance replay executes 36,962,304 model-example forwards and checks 98,304 batch records. It does not claim unexecuted caching savings. Training additionally computes gradients and full representation Jacobians, with elapsed seconds retained in `training.csv`. Equal labels and capacity do not imply equal wall-clock cost. Field calibration is unlabeled but is not free model work.

## Artifacts and reproduction

`PROTOCOL.md`, `PREFLIGHT.json`, `SOURCE_FREEZE.json`, and `ROOT_APPROVAL.json` document design, numerical checks, immutable sources, and release chronology. `states/` retains all chosen models and selected-model checkpoints; `splits/` retains labeled validation/calibration draws; `MODEL_COMMIT.json`, `FIELD_COMMIT.json`, and `TEMPERATURE_COMMIT.json` commit each stage before its dependent work. The fields are frozen before scalar fitting. `temperature_fits.csv` and `temperatures/` retain all fit outputs, derivative/boundary diagnostics and exact masks. `predictions/`, `deployment_ledger.csv.gz`, `per_model.csv`, and `reliability_bins.csv` retain performance outputs and costs; `contrasts.csv` and `paired_contrasts.csv` retain every prespecified model-seed comparison.

Dependencies are Python, NumPy, matplotlib, and threadpoolctl. All numerical source dependencies are included under this folder; prior studies are untouched. To reproduce a new run in an isolated copy with separate output directories:

```text
python preflight.py
python study.py freeze
python study.py train
python study.py fields
python study.py temperatures
python study.py score
python study.py analyze
python report.py
```

Fresh execution requires the matching internal approval record. Existing freezes and output directories are protected against overwrite. Reporting alone reads completed tables and fits no model or calibration parameter. The previous checkpoint/temperature machinery and selector are included as frozen reference copies. There is no new hyperparameter search in this confirmation.
'''
    (HERE/'README.md').write_text(text,encoding='utf-8')
    details={}
    for family in families:
        details[family]={}
        for pipeline in ('raw','full','interp'):
            rs=[r for r in fits if r['family']==family and r['pipeline']==pipeline]
            details[family][pipeline]=dict(mean_beta=float(np.mean([float(r['beta']) for r in rs])),mean_temperature=float(np.mean([float(r['temperature']) for r in rs])),
               statuses={status:sum(r['fit_status']==status for r in rs) for status in sorted(set(r['fit_status'] for r in rs))})
    (HERE/'fit_summary.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    print('Report complete')
if __name__=='__main__':run()
