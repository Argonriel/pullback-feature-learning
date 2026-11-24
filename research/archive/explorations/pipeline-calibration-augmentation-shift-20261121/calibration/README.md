# Final-pipeline calibration at an equal fitting-label budget

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
| Legacy training | 0.641332 | 0.623911 | 0.615151 | 0.628423 | 0.623959 |
| Selected training | 0.646426 | 0.627230 | 0.614132 | 0.632169 | 0.626012 |
| More training labels | 0.547136 | 0.531482 | 0.528741 | 0.535428 | 0.533790 |

| Primary paired log-loss contrast | Mean difference | 98.75% interval | Seed wins |
|---|---:|---:|---:|
| selected_full_final_vs_rawT | -0.013098 | [-0.016366, -0.010031] | 16/16 |
| selected_interp_final_vs_rawT | -0.006157 | [-0.008040, -0.004516] | 16/16 |
| extra_training_vs_legacy_rawT | -0.094195 | [-0.102746, -0.084668] | 16/16 |
| selected_vs_legacy_rawT | +0.005094 | [+0.001644, +0.007989] | 3/16 |

Negative differences favor the first named arm. The four prespecified primary comparisons share a nominal 95% bootstrap family through alpha=0.05/4 per interval. Bootstrap draws resample 16 paired model seeds, not the individual performance batches. These are approximate finite-model uncertainty intervals, not distribution-free certificates. All other metrics and comparisons remain secondary.

Classification must be interpreted separately. A positive scalar temperature changes neither the fixed allocation mask nor the sign of its final logits, so raw-fitted and final-fitted temperatures have exactly equal classification errors within each model/pipeline. Any error differences between model families arise from training or the prediction pipeline, not scalar calibration.

| Procedure | Raw error | Full error | Interpolation error |
|---|---:|---:|---:|
| Legacy training | 0.367731 | 0.339134 | 0.347697 |
| Selected training | 0.374092 | 0.341344 | 0.353112 |
| More training labels | 0.269886 | 0.252151 | 0.257324 |

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
