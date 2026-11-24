# Learning reflection and calibrating the final CNN pipeline

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

| dataset | training | pipeline | CE | error_percent |
| --- | --- | --- | --- | --- |
| holdout | noaug | raw_Traw | 0.256317 | 9.410 |
| holdout | noaug | full_Traw | 0.554614 | 18.280 |
| holdout | noaug | full_Tfull | 0.516069 | 18.280 |
| holdout | aug | raw_Traw | 0.257226 | 9.437 |
| holdout | aug | full_Traw | 0.250729 | 9.157 |
| holdout | aug | full_Tfull | 0.250742 | 9.157 |
| official_test | noaug | raw_Traw | 0.274545 | 10.087 |
| official_test | noaug | full_Traw | 0.576744 | 19.023 |
| official_test | noaug | full_Tfull | 0.534300 | 19.023 |
| official_test | aug | raw_Traw | 0.281153 | 10.233 |
| official_test | aug | full_Traw | 0.273295 | 10.003 |
| official_test | aug | full_Tfull | 0.273555 | 10.003 |

Primary differences (first minus second; negative favors the first) have four
Bonferroni-adjusted approximate image-bootstrap intervals, nominal 95% family
coverage. They share image resamples across all six fixed models. These are
not finite-sample certificates or estimates of between-training-dataset variation.

| contrast | metric | difference | interval |
| --- | --- | --- | --- |
| augmentation_raw | ce | +0.0009092 | [-0.0035058, +0.0051483] |
| augmentation_raw | error | +0.0002667 | [-0.0026333, +0.0032333] |
| augmented_full_vs_raw | ce | -0.0064843 | [-0.0080703, -0.0048510] |
| augmented_full_vs_raw | error | -0.0028000 | [-0.0047000, -0.0009333] |

The prior failure on ankle boots is checked directly using class 9 on the new
within-round holdout. These errors use actual reflected forward passes, not
reconstructed partner predictions:

| training | original_error_percent | reflected_error_percent | full_error_percent |
| --- | --- | --- | --- |
| noaug | 2.700 | 100.000 | 72.533 |
| aug | 3.333 | 3.033 | 2.867 |

All ten classes, all per-seed results, reflected-input cross entropy, and flip
agreement are retained in `classwise.csv`, `quality.csv`, and `diagnostics.csv`.
The artificial equally weighted original/reflected pool is reported separately;
it does not establish symmetry of the original image distribution.

## Computation and serving

| mode | raw_ms | full_ms | full_over_raw |
| --- | --- | --- | --- |
| cpu | 28.7614 | 57.5303 | 2.014 |
| cuda_resident | 0.3670 | 0.6554 | 1.780 |
| cuda_transfer | 0.6299 | 0.8071 | 1.327 |

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

Training and checkpoint validation took 45.989 GPU seconds in
total; direct two-view calibration plus both scalar fits took 0.615
seconds. These are elapsed process measurements, not isolated serving benchmarks.
There were 2,880,000 training forward/backward examples, 120,000 validation
forward examples, 60,000 calibration forward examples, and 240,000 shared quality
evaluation forward examples. The five quality pipelines share two actual
queries per image; their hypothetical standalone costs must not be added to
that actual shared ledger. Timing used 645,120
recorded and 276,480 warm-up forward examples.
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
