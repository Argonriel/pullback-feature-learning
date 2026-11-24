"""Descriptive report from immutable outcomes; no new policy selection."""
from pathlib import Path
import csv,json
import numpy as np
HERE=Path(__file__).resolve().parent
def read(name):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read('summary.csv');cost=read('cost_quality.csv');quality=read('quality.csv');train=read('training.csv');cal=read('selector_costs.csv')
    contrasts=read('contrasts.csv');equiv=read('cpu_cuda_equivalence.csv');calls=read('deployment_ledger.csv')
    latency=read('latency.csv');warm=read('timing_calls.csv')
    timed_forwards=sum(int(r['forward_examples']) for r in latency)
    warmup_forwards=3*sum(int(r['forward_examples']) for r in warm)
    names=['raw','confidence','ridge','histogram','knn','full_fused','oracle_jensen']
    labels=['Raw','Confidence','Ridge','Histogram','kNN','Full flip','J oracle']
    colors=['#434343','#bd7e23','#137f7d','#8270a2','#4982b7','#8a8a8a','#b34c4c']
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,metric,title in zip(axes,['original_loss','paired_loss'],['Original official test images','Deliberately flip-paired evaluation']):
        values=[float(next(r[metric] for r in summary if r['device']=='cuda' and r['policy']==n)) for n in names]
        ax.bar(np.arange(len(names)),values,color=colors)
        ax.set_xticks(np.arange(len(names)),labels,rotation=35,ha='right');ax.set_ylabel('Cross-entropy (nats)');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same frozen models: symmetry construction changes the conclusion\nThree initializations; 10,000 original test images per model',fontsize=12)
    fig.savefig(HERE/'quality_boundary.png',dpi=180);fig.savefig(HERE/'quality_boundary.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),constrained_layout=True)
    for ax,mode,title in zip(axes,['cpu','cuda_resident','cuda_transfer'],['CPU, one thread','GPU, resident inputs','GPU, transfers included']):
        for j,name in enumerate(['raw','confidence','ridge','histogram','knn']):
            values=[float(r['speedup_vs_full']) for r in cost if r['mode']==mode and r['policy']==name]
            ax.scatter(np.full(3,j),values,s=26,color=colors[names.index(name)])
            ax.plot([j-.22,j+.22],[np.mean(values)]*2,color='black',lw=2)
        ax.axhline(1,color='grey',ls='--',lw=1);ax.axhline(1.1,color='grey',ls=':',lw=1)
        ax.set_xticks(range(5),['Raw','Confidence','Ridge','Histogram','kNN'],rotation=35,ha='right');ax.set_ylabel('Speedup versus fastest full reference');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Forward-count savings and end-to-end serving time are separate\nPoints are three models; full reference is the faster fused/sequential median',fontsize=12)
    fig.savefig(HERE/'serving_cost.png',dpi=180);fig.savefig(HERE/'serving_cost.svg');plt.close(fig)
    s={r['policy']:r for r in summary if r['device']=='cuda'}
    table='\n'.join(f"|{label}|{float(s[n]['original_loss']):.6f}|{100*float(s[n]['original_error']):.3f}%|{float(s[n]['paired_loss']):.6f}|{float(s[n]['oracle_fraction']):.4f}|" for n,label in zip(names,labels))
    speed=[]
    for mode in ['cpu','cuda_resident','cuda_transfer']:
        for n in ['confidence','ridge','histogram','knn']:
            group=[r for r in cost if r['mode']==mode and r['policy']==n]
            speeds=np.array([float(r['speedup_vs_full']) for r in group])
            passes=sum(r['practical_pass']=='True' for r in group)
            speed.append(f"|{mode}|{n}|{np.median(speeds):.3f}x|{speeds.min():.3f}–{speeds.max():.3f}x|{passes}/3|")
    ct='\n'.join(f"|{r['world']}|{r['contrast']}|{float(r['mean']):.6f}|[{float(r['lower']):.6f}, {float(r['upper']):.6f}]|" for r in contrasts if r['device']=='cuda')
    train_seconds=sum(float(r['training_seconds']) for r in train)
    cal_seconds=sum(float(r['forward_seconds'])+float(r['fitting_seconds'])+float(r['hyperparameter_selection_seconds']) for r in cal)
    max_logits=max(float(r['max_logit_difference']) for r in equiv);maskdiff=sum(int(r['changed_masks']) for r in equiv)
    rawloss=float(s['raw']['original_loss']);ridge_loss=float(s['ridge']['original_loss'])
    body=f'''# Fashion-MNIST: a negative original-distribution transfer result

The frozen real-data study is complete. Original Fashion-MNIST performance
does not support applying horizontal-flip averaging to these models. The
candidate ridge selector's original-test cross-entropy is {ridge_loss:.6f},
versus {rawloss:.6f} for raw prediction. The separately constructed flip-paired
evaluation improves, illustrating the distributional boundary of the theorem.
No model, selector, temperature, or threshold was retuned after test access.

## All primary CUDA quality results

|Policy|Original loss|Original error|Pair-closed loss|Paired-J oracle fraction|
|---|---:|---:|---:|---:|
{table}

The selected kNN is the stronger validation control for every model. The
table reports mean metrics over three initialization/order replications on
one shared training split. The original and paired columns refer to different
evaluation laws. The full and oracle references cost 2 forward examples per
input; deployable limited policies cost 1.25, and raw costs 1. Oracle fractions
refer to the exact-quota Jensen objective, not original-test loss optimality.

![Original versus explicitly paired quality](quality_boundary.png)

## Fixed paired cluster-bootstrap contrasts

Negative loss differences favor ridge. These 95% intervals resample the 40
original-image batches, retaining all 10,000 images including the final 16-image
batch and keeping both orientations and all three fixed models together.
They are descriptive approximations, not three-dataset replication or a
distribution-free real-data guarantee.

|Evaluation law|Contrast|Mean|Interval|
|---|---|---:|---|
{ct}

## Serving measurements

The latency comparator is the faster median full_fused/full_sequential path
for each model and mode. Values below summarize the three per-model medians.
The frozen practical target is at least 1.10x speedup and original loss within
0.01 nats of full averaging. It is kept unchanged even though full averaging
itself is harmful here. A pass against that reference is therefore not a
claim of useful predictive deployment. Raw prediction is both cheaper in
model examples and substantially more accurate on the original test law.
This raw-baseline interpretation is an explicit post-result assessment,
not a changed prespecified success threshold.

|Mode|Policy|Median speedup|Model range|Frozen full-relative target passes|
|---|---|---:|---:|---:|
{chr(10).join(speed)}

![Actual serving cost](serving_cost.png)

All feature extraction, kNN distance work, scoring, sorting, selection,
transformations, model calls, and output merging are timed. GPU-transfer
mode additionally includes image/needed-key transfer and output return.
Both modes begin with normalized FP32 tensors. Download, decoding,
normalization, model loading, training, and selector calibration are separate
setup costs. This is not a disk-to-prediction benchmark. Models and tables
are resident; full references never transfer unused tie keys. Three warmups
and seven randomized blocks of four batches are retained for every path.

## Costs and integrity

- Training used 1,440,000 forward/backward image examples in total, with 60,000
  validation forward examples and 15,000 temperature-fitting examples.
  The fixed training protocol took {train_seconds:.3f} GPU wall seconds summed
  across three sequential models, including validation/checkpoint overhead.
- Selector fitting and selection used 60,000 shared CNN forward examples:
  16,000 fit plus 4,000 selection-validation per model. Shared selector
  preparation took {cal_seconds:.3f} seconds, with component timings retained.
- Executed original-test quality paths used {sum(int(r['forward_examples']) for r in calls):,}
  CNN forward examples across three models, CPU/CUDA, and eight policies.
  Paired diagnostics reuse those paid all-view outputs and do not give free
  partner access to a deployable policy.
- CPU quality was separately evaluated on every test image. Maximum saved
  CPU/CUDA logit difference was {max_logits:.6g}; total mask differences were
  {maskdiff}. Per-policy/seed risk and class differences are preserved.
- All official compressed files match the published MD5 checksums. The MIT
  license is included. All five training-data splits are disjoint; the test
  arrays were decoded only after source freeze, model/policy commit, and
  root approval. Every development candidate and all checkpoints are kept.
- The benchmark used {timed_forwards:,} measured and {warmup_forwards:,}
  warmup forward examples, or {timed_forwards+warmup_forwards:,} in total.
  Additional post-fit temperature diagnostics use 15,000 validation forwards
  and reuse the stored temperature-fitting logits. They change no choice and
  are explicitly descriptive because validation selected the checkpoint.

`LITERATURE_AND_THEORY.md` states the provenance and multiclass identity.
`PRETEST_CLARIFICATION.md` fixes the exact 1.10x threshold and FP32 pairing
qualification. Source hashes and read-only access boundaries are in the
training/source freezes, model/policy commits, and test approval. The primary
sources were not edited after test access. Independent true-partner and
classwise audits are maintained by the root audit branch.

## Interpretation and next action

This is a valid negative result for the proposed real-data application.
Semantic plausibility of a reflected garment label does not make the
original input distribution reflection-invariant. The manufactured paired
law meets a different condition and must not be used to advertise an
original-test guarantee. A later study could separately validate augmentation
training or transformation choice on new development data, then freeze and
test anew. None was added to rescue this result. The current evidence favors
raw prediction on this official test distribution.

The 98-feature ridge/kNN/histogram adaptation is not the previous interpolation
algorithm. Three CNN initializations, a small grayscale dataset, and a single
reflection do not establish general real-data or GPU utility. Forward-count
savings are reported separately from latency, and favorable timing against
a harmful full reference is not a sufficient reason to deploy augmentation.

## Reproduction

Use the installed GPU environment or a compatible PyTorch/NumPy runtime.
Run in a new copy with generated outputs absent; write-once files prevent
silently overwriting results. The original commands are:

```text
python data.py
python train.py freeze
python train.py train
python study.py freeze
python study.py calibrate
# After independent review, write TEST_APPROVAL.json binding POLICY_COMMIT.json.
python study.py score
python study.py benchmark
python study.py analyze
python report.py
```

Keep protocol/source freezes and actual model/policy hashes in the reproduced
approval. Do not bypass the test lock or reuse the historical approval hash
with newly generated artifacts. Downloaded bytes, split indices, checkpoints,
calibration targets, all test predictions, ledgers, raw latency blocks, and
summary tables are included for read-only reconstruction.
'''
    (HERE/'README.md').write_text(body,encoding='utf-8')
    print(json.dumps(dict(original_raw_loss=rawloss,original_ridge_loss=ridge_loss,max_cpu_cuda_logit=max_logits,mask_differences=maskdiff),indent=2))

if __name__=='__main__':main()
