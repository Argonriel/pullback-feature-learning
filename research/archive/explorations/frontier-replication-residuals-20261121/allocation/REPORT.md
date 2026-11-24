# Allocation frontier: development selection and independent confirmation

The eight-seed development criterion selected `train_2304` from the four frozen allocations. The selected recipe was committed before any of the 16 confirmation seed streams opened. All planned fits, calibrations and evaluations completed; no replacement seeds or outcome-driven schedule changes were used.

## Development selection and full grid

The selection criterion was mean sampled-label NLL of calibrated raw output. Full output, expected posterior risk, error and timing did not enter selection. Development metrics below are descriptive; the selected minimum is subject to winner's bias and is not independent evidence of improvement.

| Training | Calibration | Updates | Raw selection NLL | Full NLL (descriptive) | Selected |
|---:|---:|---:|---:|---:|---|
| 2304 | 2048 | 500 | 0.553170 | 0.530731 | yes |
| 3072 | 1280 | 375 | 0.558800 | 0.530289 |  |
| 3584 | 768 | 321 | 0.570441 | 0.532742 |  |
| 3840 | 512 | 300 | 0.576344 | 0.533609 |  |

The descriptive full-output development minimum occurred at `train_3072` (0.530289). Full-output loss was not the selection criterion, so this observation did not promote that allocation. It has no independent candidate-versus-baseline confirmation in this round.

Tie tolerance was 1e-12, with the smallest training allocation preferred inside tolerance. The development selection used 524,288 labeled evaluation items beyond 47,104 unique fitting labels across its eight paired datasets. The resulting 571,392-label recipe-discovery cost is separate from 4,352 labels per fitted procedure.

## Independent confirmation

Calibrated raw expected NLL: selected 0.552949, baseline 0.552949. Calibrated full expected NLL: selected 0.532814, baseline 0.532814.

| Output | Selected minus baseline NLL | Adjusted interval | Status |
|---|---:|---|---|
| raw_T | +0.000000 | not applicable | identical procedure; no improvement test |
| full_T | +0.000000 | not applicable | identical procedure; no improvement test |

The baseline won selection. It was trained once per confirmation seed; selected-minus-baseline is zero by identity. No runner-up was promoted, and no improvement claim or test is made.

The chosen raw development selection NLL was 0.553170; its fresh confirmation sampled-label NLL was 0.552729. These are distinct datasets; the latter was never used to reconsider selection.

## Resource accounting

All calibrated procedures spend 4,352 accepted fitting labels. The nominal arithmetic reference is 207,260,412,000 operations with C(n)=179045n+2001144 per update. Update counts 500/375/321/300 have maximum floor shortfall 0.30552%. These counts include Jacobian, Gram and general LU arithmetic; special functions, memory and control overhead are separate. One update uses two representation-forward streams, 4n Jacobian rows, and one 144-dimensional solve, with no separate reverse-mode backward call.

Validation labels and forwards are zero by prospective choice. Four saved diagnostic checkpoints per trajectory incur additional training forwards and do not select stopping. Raw inference costs one model forward per item and full mean costs two. Both final outputs have their own label-fitted temperature; scalar calibration and shared raw/partner costs are explicit. Uncalibrated diagnostics have smaller actual label counts and are not equal-budget competitors.

| Phase | Models | Fit-label union | Evaluation-label union | Example-updates | Nominal arithmetic | Calibration forwards | Evaluation forwards |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 32 | 47,104 | 524,288 | 36,851,712 | 6,622,064,466,432 | 73,728 | 16,777,216 |
| confirmation | 16 | 69,632 | 1,048,576 | 18,432,000 | 3,316,166,592,000 | 65,536 | 8,388,608 |

Total accepted fitting plus evaluation label union across both phases: 1,689,600. Synthetic rejection proposals are separately counted in COST_TOTALS.json. The temperature derivative/objective label evaluations and all per-procedure forward/Jacobian/solve counts are in standalone_cost_ledger.csv. Concurrent training times are descriptive only.

## Isolated complete-procedure latency

An exclusive CPU window timed one warmup block and three measured paired blocks on development pair 1. Each fit includes all training updates, diagnostic checkpoint work, both calibration forward streams, and both scalar temperature fits. Input generation/loading, output I/O, held-out evaluation and verification are excluded. Parameters and temperatures matched the committed development fits within 1e-12; reused labels added zero new unique observations.

| Procedure | Median total seconds | Observed range | Median training | Median calibration |
|---|---:|---|---:|---:|
| train_2304 | 11.2746 | [11.1380, 11.3194] | 11.2680 | 0.0066 |

Measured paired-block selected/baseline time ratios: 1.0000, 1.0000, 1.0000; median 1.0000.

Because the selected procedure is the baseline itself, these ratios are one by identity. Only baseline latency was measured; no comparative speed claim is made.

These are actual complete-procedure times for three repeats of one fixed dataset, not projections or independent-dataset latency uncertainty. Timing did not select the allocation. Nominal arithmetic equality does not imply equal time.

## Limits, prior art and auditability

The allocation changes training size, calibration size and update count together; this experiment does not separately identify calibration-estimation variance and representation learning effects. It explores four prespecified points rather than proving a global optimum. Eight development and 16 confirmation seeds limit precision. Posterior expected metrics rely on the known synthetic law and support no real-data guarantee. A confidence interval crossing zero is inconclusive. All negative or inconclusive findings are retained.

Temperature scaling is prior art ([Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)); compute allocation is established broadly ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)); finite-sample model-selection bias is established ([Cawley and Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). The present contribution is a controlled frontier selection and independent confirmation under the existing small PB model.

PROTOCOL.md and SOURCE_FREEZE.json define the experiment before fresh outcomes. Each phase has MODEL_COMMIT, FIT_COMMIT and SCORE_COMPLETE artifacts. SELECTION_COMMIT binds the selected recipe before confirmation starts. Raw per-item predictions, labels, model states and checkpoints allow independent reconstruction; posterior values are retained only for confirmation evaluation. Frozen portable source and commands are in source/ and REPRODUCE.md. No earlier archive or source file was changed.
