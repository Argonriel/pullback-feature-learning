# Independent GPU and CPU review

The final optimized timing and quality summaries pass independent recomputation. No model was retrained, policy retuned, or timing benchmark repeated for this review. The subsequent CPU quality replay uses the unchanged weights, tables and complete saved test inputs.

## Raw-data audit

`gpu_checks.py` verifies the source-freeze chain, trained weights, calibration tables, original deployment records, saved predictions and final timing summaries. It recomputes all 63 expected-risk rows, with maximum discrepancy 1.39e-17. All 4,032 original deployment call records have the specified example counts and exact quotas. The 126 recorded original/optimized equivalence rows cover 8,064 policy-device batches and have identical selected sets and zero logit difference. Eighteen sequential/fused checks also pass.

The audit independently groups all 2,646 retained timing records into 378 seven-repeat medians, recomputes the strongest equivalent full baseline, every speed ratio and break-even field, and all 126 summary rows. The recomputation differences are exactly zero. The full comparator is the minimum of sequential and fused full inference across both serving implementations, within each model and timing condition. The final v2 transfer code does not transfer unused tie keys for raw/full inference.

All 144 limited-policy GPU comparisons are slower than that full comparator: nine models, two timing conditions, two implementations and four limited policies. None has a GPU wall-time break-even. This does not mean that GPU network inference is slow; it means the selective paths fail to beat the strong full-GPU reference in this eager serving implementation.

The optimized folded-fused path is faster on one CPU thread at widths 256 and 1024, in all three seeds at each width. Mean per-model speed ratios are 1.192677 and 1.504954 respectively. Their calibration-only break-evens are 31/12/10 and 2/2/2 batches. Width 64 has mean ratio 0.630373 and is slower. Calibration-only amortization excludes rank preparation, model/table loading and supplied random-key generation; CPU serving uses GPU-fitted tables. The separate forward-example break-even is 43 batches.

## Direct CPU quality replay

The original cost-quality files attach CUDA-scored risks to CPU timing rows. To check the CPU tradeoff directly, `cpu_quality_checks.py` executes optimized full-fused, folded-fused and confidence-plus-radius-angle inference on all 16,384 saved inputs for every one of the nine models. This produces 27 quality records and 1,728 batch-call records. The actual verification cost is 2,304 network invocations and 663,552 forward examples; this is not a deployment or calibration cost.

All CPU/CUDA selected sets and classification decisions agree. The maximum absolute logit difference is 7.62939453125e-6, and the maximum absolute expected-risk difference is 1.760191423172408e-9. Thus the CPU speed-quality interpretation is directly supported for these three policies at the reported precision. Cross-device bitwise equality is neither claimed nor needed. CPU predictions, masks and input/output hashes are retained alongside the replay-start source commit. The frozen experiment sources remain unchanged.

| Width | Direct CPU full risk | Direct CPU folded risk | Direct CPU radius-angle risk |
|---|---|---|---|
| 64 | 0.03079114627 | 0.03083094030 | 0.03043498412 |
| 256 | 0.03643279444 | 0.03645858725 | 0.03621969722 |
| 1024 | 0.03459324615 | 0.03463286534 | 0.03409882071 |

## Interpretation limits

There are three model replications per width. Training and test data are paired across widths, so the sweep has three independent training-data draws, not nine. Seven timing repeats do not increase the model sample size. The widths, batch size, FP32 arithmetic and serving family were fixed before testing; all nine trained models are retained.

These are ordinary two-hidden-layer MLPs trained with 1,000 Adam updates on 1,024 examples. They differ from the main 80-parameter residual PB500 cohort in architecture, optimizer, data size and precision. This experiment supports an implementation-specific inference-cost conclusion. It is not a controlled training-algorithm comparison, a pullback-scaling demonstration, a deployment of the new rank-balanced selector on GPU, or an all-hidden-posterior guarantee for the larger models.

The combined REPORT.md and REPRODUCE.md were reviewed against these records and the completed local-bound and rank-diagnostic checks. No substantive interpretation blocker was found. They correctly keep the useful old-field certificate separate from the approximate new-rank diagnostics and distinguish model-specific numerical endpoints from a common worst-case labeling law.

## Reproduction

Run `python root_audit/gpu_checks.py` in a study copy to recheck saved files without inference. The main NumPy environment suffices.

Run `python root_audit/cpu_quality_checks.py` using the separate recorded PyTorch environment in a new output copy. It intentionally refuses to overwrite its replay commit, CSVs and prediction directory. It does not call training, calibration, test generation or timing stages.

Audit outputs are `gpu_checks.json`, `cpu_quality_checks.csv`, `cpu_quality_checks.json`, `cpu_quality_calls.csv`, `cpu_quality_replay_start.json` and `cpu_quality_predictions/`. The direct CPU replay supplements the earlier GPU raw-data audit; the original CUDA risk columns remain unchanged.
