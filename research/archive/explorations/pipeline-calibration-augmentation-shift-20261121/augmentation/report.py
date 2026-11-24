"""Post-outcome descriptive report; does not train, fit, select, or score models."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def read(name):
    with (HERE/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def table(rows,columns):
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(str(r[k]) for k in columns)+' |' for r in rows)
def main():
    if not (HERE/'serving_summary.csv').exists():
        latency=read('latency.csv');speed=[]
        for pair in [1,2,3]:
            for arm in ['noaug','aug']:
                for mode in ['cpu','cuda_resident','cuda_transfer']:
                    med={m:float(np.median([float(r['ms_per_batch']) for r in latency if int(r['pair'])==pair and r['arm']==arm and r['mode']==mode and r['method']==m])) for m in ['raw','full_fused','full_sequential']}
                    best=min(['full_fused','full_sequential'],key=lambda m:med[m])
                    speed.append(dict(pair=pair,arm=arm,mode=mode,raw_ms=med['raw'],full_fused_ms=med['full_fused'],full_sequential_ms=med['full_sequential'],
                        strongest_full=best,strongest_full_ms=med[best],full_over_raw_latency=med[best]/med['raw']))
        with (HERE/'serving_summary.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,list(speed[0]));w.writeheader();w.writerows(speed)
    q=read('summary.csv');contrasts=read('contrasts.csv');train=read('training.csv');serv=read('serving_summary.csv')
    diag=read('diagnostics.csv');classes=read('classwise.csv')
    get=lambda arm,pipeline,metric:float(next(r[metric] for r in q if r['dataset']=='holdout' and r['arm']==arm and r['pipeline']==pipeline))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    colors={'noaug':'#c66a45','aug':'#277f97'};xx=np.arange(2);width=.34
    for j,arm in enumerate(['noaug','aug']):
        for ax,metric,scale in [(axes[0],'ce',1),(axes[1],'error',100)]:
            ax.bar(xx+(j-.5)*width,[get(arm,p,metric)*scale for p in ['raw_Traw','full_Tfull']],width,
                   color=colors[arm],label='No training flips' if arm=='noaug' else '50% training flips')
            ax.set_xticks(xx,['Raw, own calibrated','Full, own calibrated'])
        rr=[r for r in diag if r['dataset']=='holdout' and r['arm']==arm]
        original=get(arm,'raw_Traw','error')*100;reflected=np.mean([float(r['reflected_error']) for r in rr])*100
        axes[2].bar(xx+(j-.5)*width,[original,reflected],width,color=colors[arm])
    axes[0].set_ylabel('Cross entropy (nats)');axes[1].set_ylabel('Classification error (%)')
    axes[2].set_ylabel('Classification error (%)');axes[2].set_xticks(xx,['Original input','Reflected input'])
    axes[0].legend(frameon=False,fontsize=8);axes[0].set_title('Pipeline-matched calibration');axes[1].set_title('Temperature preserves argmax')
    axes[2].set_title('Transformation reliability')
    fig.suptitle('Within-round holdout: three paired CNN initializations, 10,000 images',fontsize=13)
    fig.savefig(HERE/'augmentation_quality.png');fig.savefig(HERE/'augmentation_quality.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,mode,title in zip(axes,['cpu','cuda_resident','cuda_transfer'],['One-thread CPU','GPU, resident','GPU, transfers included']):
        for j,arm in enumerate(['noaug','aug']):
            rr=[r for r in serv if r['mode']==mode and r['arm']==arm]
            ax.bar(xx+(j-.5)*width,[np.median([float(r[k]) for r in rr]) for k in ['raw_ms','strongest_full_ms']],width,color=colors[arm],label=arm)
        ax.set_xticks(xx,['Calibrated raw','Calibrated full']);ax.set_ylabel('ms per 256 images');ax.set_title(title)
    axes[0].legend(frameon=False);fig.suptitle('Serving latency: fastest measured full implementation per model',fontsize=13)
    fig.savefig(HERE/'augmentation_serving.png');fig.savefig(HERE/'augmentation_serving.svg');plt.close(fig)
    simple=[]
    for r in q:
        if r['pipeline'] in ['raw_Traw','full_Traw','full_Tfull']:
            simple.append(dict(dataset=r['dataset'],training=r['arm'],pipeline=r['pipeline'],CE=f"{float(r['ce']):.6f}",error_percent=f"{100*float(r['error']):.3f}"))
    primary=[]
    for r in contrasts:
        if r['primary']=='True':primary.append(dict(contrast=r['contrast'],metric=r['metric'],difference=f"{float(r['mean']):+.7f}",interval=f"[{float(r['lower']):+.7f}, {float(r['upper']):+.7f}]"))
    training_seconds=sum(float(r['train_seconds']) for r in train);cal_seconds=sum(float(r['calibration_seconds']) for r in train)
    costs=json.loads((HERE/'TIMING_COMPLETE.json').read_text())
    tt=[]
    for mode in ['cpu','cuda_resident','cuda_transfer']:
        rr=[r for r in serv if r['mode']==mode]
        tt.append(dict(mode=mode,raw_ms=f"{np.median([float(r['raw_ms']) for r in rr]):.4f}",full_ms=f"{np.median([float(r['strongest_full_ms']) for r in rr]):.4f}",full_over_raw=f"{np.median([float(r['full_over_raw_latency']) for r in rr]):.3f}"))
    cc=[]
    for arm in ['noaug','aug']:
        rr=[r for r in classes if r['dataset']=='holdout' and r['arm']==arm and int(r['class_id'])==9]
        cc.append(dict(training=arm,original_error_percent=f"{100*np.mean([float(r['original_error']) for r in rr]):.3f}",reflected_error_percent=f"{100*np.mean([float(r['reflected_error']) for r in rr]):.3f}",full_error_percent=f"{100*np.mean([float(r['full_error']) for r in rr]):.3f}"))
    text='''# Learning reflection and calibrating the final CNN pipeline

Six CNNs were trained from scratch: three paired initialization/order streams,
with and without probability-0.5 horizontal flips. Every arm uses 50,000 unique
labels (40,000 training, 5,000 checkpoint selection, 5,000 calibration).
Two separate positive scalar fits reuse the same 5,000 calibration labels:
one fit targets raw logits and the other the final two-view mean logits.

**Data boundary:** the primary 10,000-image holdout is excluded from every fit
in this round, but those images appeared in the previous study's training set.
No previous weights are reused. The previously inspected official test is
secondary descriptive evidence. Neither set is a pristine external replication.

## Prediction quality

The primary result is conditional: training reflection makes two-view averaging
useful for these models, but does not clearly improve raw inference. The
adjusted intervals for the raw augmentation contrast include zero. Within the
augmented arm, averaging reduces both cross entropy and classification error
in all three models, with the four-comparison adjusted mean intervals excluding
zero. Its mean error improvement is 0.28 percentage points, not a large accuracy
jump. Own-output calibration gives no detectable additional holdout benefit
over the raw-fitted scalar for the augmented full pipeline. For unaugmented
models it reduces full-pipeline loss, but cannot repair the harmful class
predictions caused by reflecting inputs the model has not learned to handle.

The following are means over three paired model initializations. Temperature
scaling changes cross entropy but does not change predicted classes.

'''+table(simple,['dataset','training','pipeline','CE','error_percent'])+'''

Primary differences (first minus second; negative favors the first) have four
Bonferroni-adjusted approximate image-bootstrap intervals, nominal 95% family
coverage. They share image resamples across all six fixed models. These are
not finite-sample certificates or estimates of between-training-dataset variation.

'''+table(primary,['contrast','metric','difference','interval'])+'''

The prior failure on ankle boots is checked directly using class 9 on the new
within-round holdout. These errors use actual reflected forward passes, not
reconstructed partner predictions:

'''+table(cc,['training','original_error_percent','reflected_error_percent','full_error_percent'])+'''

All ten classes, all per-seed results, reflected-input cross entropy, and flip
agreement are retained in `classwise.csv`, `quality.csv`, and `diagnostics.csv`.
The artificial equally weighted original/reflected pool is reported separately;
it does not establish symmetry of the original image distribution.

## Computation and serving

'''+table(tt,['mode','raw_ms','full_ms','full_over_raw'])+f'''

Serving values are medians over six models of each model's seven-block median.
The ratio column is the median of model-specific ratios, not a ratio of table
medians. Each block contains four 256-image validation batches. Raw uses one
forward per image; full uses two. Both fused and sequential full implementations
are timed and the faster is the per-model reference. Timers include flipping,
merging, calibration multiplication, and transfers where named. Input decoding,
normalization, and model loading are excluded. Measurements use one CPU thread
or the available RTX 5070 Ti with synchronized CUDA execution. These measurements
do not compare against a many-thread CPU implementation.
Decoding, normalization, and model loading were not individually timed, so this
study does not estimate complete cold-start latency. The separately measured
predeployment components are training/checkpoint validation and calibration.

Training and checkpoint validation took {training_seconds:.3f} GPU seconds in
total; direct two-view calibration plus both scalar fits took {cal_seconds:.3f}
seconds. These are elapsed process measurements, not isolated serving benchmarks.
There were 2,880,000 training forward/backward examples, 120,000 validation
forward examples, 60,000 calibration forward examples, and 240,000 shared quality
evaluation forward examples. The five quality pipelines share two actual
queries per image; their hypothetical standalone costs must not be added to
that actual shared ledger. Timing used {costs['recorded_forward_examples']:,}
recorded and {costs['warmup_forward_examples']:,} warm-up forward examples.
The parent's independent complete holdout replay added 120,000 audit forward
examples, outside those scientific and timing ledgers. Its independent CNN
definition reproduced all saved holdout logits exactly and reconstructed all
quality rows, classwise rows, scalar fits, primary and secondary intervals,
split membership, checkpoint choices, and serving summaries.

`cpu_cuda_equivalence.csv` records logit and class agreement on the last 256-image
validation batch in the first recorded block for every model and method.
Evaluation uses directly queried views with float64 combination; deployed fused
and sequential arithmetic is FP32 and may differ slightly. No method was selected
or revised from the holdout, official test, or timing results.

## Reproduction and source integrity

The official Fashion-MNIST files and their MIT license are copied unchanged from
the earlier checksum-verified download. Source: [official repository](https://github.com/zalandoresearch/fashion-mnist);
dataset paper: [Xiao, Rasul and Vollgraf, 2017](https://arxiv.org/abs/1708.07747).
`SOURCE_FREEZE.json`, `SPLIT_AUDIT.json`, the parent `ROOT_APPROVAL.json`,
`MODEL_COMMIT.json`, and `PAIRING_AUDIT.json` retain protocol ordering and hashes.
Initial weights and all 12 epoch permutations are checked identical within each
pair. All checkpoints, selected models, fits, direct predictions, and results
are retained. There is no replacement of a failed or unfavorable seed.

From a fresh copy of this branch, retain its data and `prior_splits.npz`, then
run the following stages with the workspace `.venv-gpu/Scripts/python.exe`:
`study.py freeze`, obtain the parent's scientific `ROOT_APPROVAL.json`, then
`study.py train`, `study.py score`, `study.py benchmark`, `study.py analyze`.
The benchmark requires a quiet interval without other numerical jobs.
`report.py` renders this report and the two figures using NumPy and matplotlib.
Existing runs deliberately refuse to overwrite committed artifacts.

This experiment examines transformation reliability and pipeline-specific
calibration. It does not evaluate a quota selector, establish a natural-image
symmetry certificate, or identify generalization beyond this dataset and model.
'''
    (HERE/'README.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
