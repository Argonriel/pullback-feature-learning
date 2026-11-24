# Fresh two-invariant task diagnostics

All 16 prespecified fresh models are included. This is a descriptive post-confirmation analysis of saved predictions. The parent had already released aggregate log-loss and classification-error results when these reliability diagnostics were specified. No model, temperature, threshold, policy, or calibration parameter was fit, and no new model calls were made. The original model and policy freezes are untouched.

| Policy | Calls/input | Expected log loss | Expected classification error | Expected Brier loss | Mean confidence | Expected accuracy | Confidence minus accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.700872 | 0.371238 | 0.241276 | 0.736100 | 0.628762 | 0.107338 |
| folded | 1.25 | 0.653311 | 0.352905 | 0.225828 | 0.712476 | 0.647095 | 0.065381 |
| symmetric_jensen | 1.25 | 0.647930 | 0.350019 | 0.223884 | 0.709229 | 0.649981 | 0.059248 |
| full | 2 | 0.629271 | 0.340438 | 0.217515 | 0.704148 | 0.659562 | 0.044586 |

The raw model learns real class signal: its mean expected accuracy is 62.88%, compared with 50% for the constant classifier. Its mean log loss is 0.700872, compared with log(2)=0.693147; learning useful decisions is therefore not evidence of well-calibrated probabilities. Its mean top-label confidence is 73.61%, exceeding expected accuracy by 10.73 percentage points. The fixed-bin reliability curves show the direction of probability calibration errors rather than inferring calibration solely from log loss.

Interpolation reduces mean expected log loss to 0.647930 and classification error to 0.350019. The full half-turn reference reaches 0.629271 and 0.340438, at twice the raw call count. These policies still remain far from the task's Bayes log loss 0.4929065 and Bayes classification error 0.2265627. The Brier excess, equal to mean squared difference between predicted and true probabilities, is 0.081271 for raw, 0.063879 for interpolation, and 0.057510 for full averaging. That excess mixes probability calibration and lost predictive information; it is not a pure calibration statistic.

The fixed-bin posterior ECE averages are 0.108135, 0.064030, and 0.051954 respectively. This is the weighted absolute binwise difference between predicted probability and the known posterior, computed separately per model in ten fixed equal-width bins. It removes sampled-label noise but remains a coarse binned diagnostic, not an exact conditional-calibration norm. The plot pools all equal-size test samples for readability; seed-specific bins and counts are retained. No confidence interval or new superiority claim is inferred from these descriptive summaries.

A grounded next direction is to improve training regularization or probability calibration on a separate development/validation split, then freeze any change and repeat the same independent comparison. The current pilot already showed that lower training loss at larger rate or more steps increased population log loss. That observation and the fresh overconfidence diagnostics support investigating calibration and generalization, rather than claiming the task is solved or applying a correction fitted on this test set. A positive temperature rescaling preserves the logit's sign, so it would leave classification error at threshold 0.5 unchanged; addressing the substantial Bayes classification gap requires more than that correction.

The data are synthetic, bounded, and phase-coupled; the quotient genuinely has two coordinates on this support. This is not unrestricted ambient-four-dimensional or real-data validation. The fitted model sees raw coordinates. Bayes probabilities enter these diagnostics only as evaluator information, never as deployment inputs.

Artifacts are in `root_audit/fresh_task_diagnostics/`: `PROTOCOL.json`, all 64 model/policy metric rows, 640 reliability-bin rows including empty bins, 1,024 two-dimensional quotient-cell rows, pooled curves, summary table, and PNG/PDF figures. All 16 saved prediction files were verified against the immutable main scoring manifest. Reproduce into a new output folder with `python root_audit/fresh_task_diagnostics.py`; the script refuses to overwrite its existing directory. It performs no model evaluations.
