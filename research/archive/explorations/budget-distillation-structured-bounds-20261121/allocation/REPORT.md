# Label allocation and training-work results

All nine frozen procedures completed on all 16 fresh paired seeds. Every calibrated procedure used 4,352 fitting labels. The fixed 2,304-training/2,048-calibration baseline is retained. The prior completed study was not modified.

## Main results

The baseline achieved expected NLL 0.553049 with calibrated raw output and 0.531434 with calibrated full mean output. The table reports means over paired model seeds; lower is better.

| Procedure | Raw NLL | Full NLL | Raw error | Full error |
|---|---:|---:|---:|---:|
| fixed_256 | 0.644989 | 0.616589 | 0.370830 | 0.339047 |
| fixed_768 | 0.590185 | 0.561397 | 0.313018 | 0.284060 |
| fixed_1280 | 0.568220 | 0.543541 | 0.289317 | 0.265729 |
| fixed_2304 | 0.553049 | 0.531434 | 0.274712 | 0.253999 |
| flop_256 | 0.662787 | 0.641127 | 0.389600 | 0.362465 |
| flop_768 | 0.588779 | 0.559949 | 0.306759 | 0.280004 |
| flop_1280 | 0.562381 | 0.540497 | 0.282387 | 0.262481 |
| examples_256 | 0.663056 | 0.641551 | 0.390006 | 0.362968 |
| validation_1280 | 0.562589 | 0.540475 | 0.282601 | 0.262542 |

## Prespecified inference

Eight NLL contrasts use 20,000 paired resamples of 16 seed clusters and alpha 0.05/8 each. Intervals are approximate bootstrap uncertainty, not a theorem or proof of equivalence. Negative favors the first named procedure.

| Contrast | Difference | Adjusted interval | Wins / 16 |
|---|---:|---:|---:|
| flop_256_vs_baseline_raw_T | +0.109738 | [+0.100439, +0.120509] | 0 |
| flop_768_vs_baseline_raw_T | +0.035730 | [+0.025968, +0.045837] | 0 |
| flop_1280_vs_baseline_raw_T | +0.009332 | [+0.000781, +0.017029] | 4 |
| validation_vs_flop1280_raw_T | +0.000209 | [-0.000101, +0.000806] | 5 |
| flop_256_vs_baseline_full_T | +0.109693 | [+0.100480, +0.120544] | 0 |
| flop_768_vs_baseline_full_T | +0.028515 | [+0.020235, +0.037058] | 0 |
| flop_1280_vs_baseline_full_T | +0.009063 | [+0.003129, +0.014887] | 3 |
| validation_vs_flop1280_full_T | -0.000021 | [-0.000274, +0.000207] | 10 |

## Cost interpretation

Fixed steps do not equalize training work. The exact-example endpoint has 1,152,000 example-updates for both n=256/4500 updates and n=2304/500 updates, with 4,500 versus 500 LU solves. The nominal-arithmetic arms include all counted Jacobian/Gram/solve arithmetic. They use 4,332/1,485/896 updates at n=256/768/1280; maximum budget shortfall is 0.05996%. Elementary-function internals, memory, pivot comparisons and elapsed time are separate. No whole-system compute or time equality is claimed.

Each update has n standalone representation forwards plus n forwards within the analytic Jacobian, 4n Jacobian rows, and one 144-dimensional solve. The Jacobian contraction forms the loss gradient; there is no independent reverse-mode backward call. Validation consumes 4,096 extra forward examples for the selected arm. Full inference costs two forwards per item; raw inference costs one.

Actual shared trajectory work: 73,461,760 example-updates; 118,096 solves; 13,389,287,921,024 nominal arithmetic operations. Diagnostic training-loss monitoring: 155,648 additional forward examples. Calibration actual shared forward examples: 950,272. Test forwards: 75,497,472.

Per-procedure standalone costs are in `training.csv`; actual amortized experiment costs are in `actual_training_work.csv`. The label union is 7,424 fitting labels per paired seed, with 4,352 visible to each calibrated procedure. Synthetic rejection proposal counts are recorded separately. Training elapsed times collected during concurrent work are descriptive only.

## Calibration and limitations

Validation selected-update counts: 100: 0/16, 250: 0/16, 500: 1/16, 896: 15/16. Temperature optimum statuses across all 288 fits: interior: 288. The validation arm consumes all 896 scheduled training updates even when an earlier checkpoint is deployed.

Both raw and full deployed outputs were calibrated against sampled iid labels with independently fitted positive scalar temperatures. Full output with raw temperature is retained as a transfer diagnostic, not selected as the winning calibration. Calibration preserves each fixed pipeline's prediction signs. Known posterior probabilities are used only in generation and final evaluation, never in fitting. Uncalibrated diagnostics use fewer labels and are not included as equal-budget competitors.

The modest prespecified grid cannot identify a global optimal allocation. The bootstrap has only 16 model clusters; an interval crossing zero is inconclusive rather than evidence of equivalence. The structured synthetic data law, fixed small model and full-batch PB optimizer limit external validity. Matched nominal arithmetic is a transparent accounting model, not measured hardware FLOPs or equal time.

## Reproduction and audit

The complete procedure definitions and work derivation are in `PROTOCOL.md`. The source freeze, passing preflight, internal approval, model commit, fit commit and scoring commit establish chronological separation. The source runner is `pullback-feature-learning/experiments/budget_allocation/runner.py`, with the workspace `.venv/Scripts/python.exe`. Run stages train, calibrate, score, analyze into a fresh branch after its own protocol/approval. Retained `states/`, `splits/`, `temperatures/`, and `predictions/` reproduce every per-model risk. The five output arms reconstruct from committed raw/full per-item logits and temperatures.

Figure files: `allocation_outcomes.png/.pdf`, `primary_contrasts.png/.pdf`. Primary and secondary contrast tables are `contrasts.csv` and `paired_contrasts.csv`; all 720 procedure/output/seed rows are in `per_model.csv`.

Temperature scaling is prior art ([Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)). Compute allocation is established broadly ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)); this small PB experiment does not extend their scaling law.

## Isolated timing audit

After all predictive results were committed, an exclusive CPU window measured seven 30-update blocks per n with one BLAS thread, following five warm-up updates. Each block resets to the same old development state and duplicates the same 256 training examples. Median block times project to fixed_2304 12.954 s (1.000 times baseline), flop_256 11.275 s (0.870 times baseline), flop_768 11.312 s (0.873 times baseline), flop_1280 12.569 s (0.970 times baseline), examples_256 11.713 s (0.904 times baseline). These are development-workload projections, not timed complete prospective fits; they exclude diagnostics, calibration, validation, data generation and I/O, and do not alter any schedule.
