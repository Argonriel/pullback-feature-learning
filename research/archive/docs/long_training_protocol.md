# Bounded long-training diagnostic

This supplementary experiment asks whether the existing fixed methods still improve when training is extended to 100,000 updates. It is planned after the formal results have been inspected and is therefore an exploratory convergence diagnostic, not a replacement for the formal comparison or a newly independent test of method superiority. The authoritative settings are in [configs/long_training.json](../configs/long_training.json). Commit this protocol, configuration, and runner before executing this batch.

The existing development validation sets were used to select the optimizer settings at 2,000 updates. Reusing them can help describe learning behavior, but their scores are not unbiased estimates of performance on unseen tasks or data. This experiment generates and evaluates no test split and no formal-stage data. The earlier tuning and formal artifacts remain unchanged.

## Fixed methods and paired data

The two-block, 80-parameter residual architecture, fixed readout, mean logistic loss, summed Euclidean representation metric, and raw-input convention remain unchanged. Use CPU, NumPy, float64, and one BLAS thread. Each seed has 256 training examples and 256 validation examples, with coordinate noise standard deviation 0.15.

| Method | Learning rate | Damping | Measured updates per seed |
| --- | ---: | ---: | ---: |
| GD | 0.125 | Not applicable | 100,000 |
| Damped representation pullback | 8 | 0.1 | 100,000 |

These settings are inherited from the 2,000-update development selection. They are not retuned for the longer horizon. In particular, damping 0.1 remains the upper boundary of the original finite grid; this diagnostic does not resolve that limitation.

| Pair | Training seed | Validation seed | Initialization seed | Within-pair order |
| --- | ---: | ---: | ---: | --- |
| dev1 | 1701 | 1702 | 23 | GD, pullback |
| dev2 | 1801 | 1802 | 24 | Pullback, GD |
| dev3 | 1901 | 1902 | 25 | GD, pullback |

Execute these six trajectories sequentially in the listed order. The two methods receive identical data and initial parameters within each pair. Each run performs five disposable warmup updates and then resets to the original initialization. The measured trajectory restarts at step zero; it does not resume from a selected checkpoint.

## Budgets and recorded states

Each run ends after exactly 100,000 measured updates unless a numerical failure prevents completion. There is no validation-based early stopping, learning-rate schedule, adaptive damping, fallback optimizer, or best-checkpoint selection. The allocation is 300,000 updates per method and 600,000 updates overall. Equal update counts do not imply equal computation cost.

Record initialization and every 1,000 updates. Also force records at steps 100 and 500, so the saved states include every explicitly declared checkpoint: 0, 100, 500, 1,000, 2,000, 5,000, 10,000, 20,000, 50,000, and 100,000. At every recorded state preserve parameters, update count, training and validation logistic loss and accuracy, and actual cumulative core training time. Use the same fixed-readout classification rule as the earlier experiments.

All updates use training inputs and labels only. Validation scores are diagnostics and cannot change the trajectory or stopping point. Do not use the lowest observed validation loss to replace the declared 100,000-update endpoint or any prior formal endpoint. Plot the full sampled training and validation curves; present the predeclared intermediate checkpoints alongside the terminal result so that late deterioration remains visible.

Core update timing includes each method's required forward and gradient work, pullback Jacobian and Gram construction and linear solve, and the parameter update. GD retains its independent reverse-mode gradient. Separately report warmup, evaluation and recording diagnostics, and end-to-end batch time. Curves against measured core time describe the cost of these fixed settings; they are not an independently time-tuned benchmark. Do not extrapolate beyond the observed trajectories or claim a matched-time endpoint that was not saved.

## Aggregation and the question of further training

Report all three seeds, method means and sample standard deviations (`ddof = 1`), and paired differences defined as pullback minus GD. Accuracy variability is descriptive, not a confidence interval or significance claim. Only form a complete paired aggregate when all six runs reach the declared endpoint successfully; no survivor-only comparison is allowed.

Assess whether to propose a separate extension beyond 100,000 updates using the following rule, fixed before this batch:

1. For a method, compare its arithmetic mean training loss at 50,000 and 100,000 updates. The relative decrease must be at least 5%.
2. Its arithmetic mean validation loss must also decrease by at least 5% over the same interval.
3. Validation loss must decrease strictly in at least two of its three paired development seeds over that interval.

For a positive starting mean loss, the relative decrease is `(loss_at_50000 - loss_at_100000) / loss_at_50000`. If a required starting mean is zero, nonfinite, or unavailable, the rule is not satisfied. At least one method must satisfy all three conditions before proposing an extension on this basis. The batch does not automatically run beyond 100,000 updates, regardless of the outcome. If a later extension is justified, specify its new endpoint and equal treatment of both methods before running it.

This 5% threshold is a practical screening convention, not a theorem, power calculation, or proof of convergence. Failing the rule does not prove that no later improvement is possible; passing it does not imply that pullback will outperform GD or improve on a fresh test set. Lower training loss without a corresponding validation improvement is evidence against continuing solely to improve held-out performance under these settings. No test result is used to make this continuation assessment.

## Numerical failures, preservation, and provenance

A failed warmup, failed solve, numerical exception, or nonfinite state or recorded metric makes the run unsuccessful. Preserve its failure reason, the latest finite parameter state, completed update count, available curves, and measured cost. A failed attempt consumes its scheduled slot and is not replaced. Attempt the remaining scheduled runs and report the batch as incomplete if any trajectory fails. Do not manufacture missing endpoints or silently change settings. Any software correction and rerun must retain the original evidence and be documented separately.

Use a fresh output directory and never overwrite the earlier runs. Archive the configuration, deterministic schedule, protocol/source Git revision, source and configuration hashes, software and thread environment, data and initialization fingerprints, saved parameter states, per-seed metrics, summaries, plots, and timing. Check that the step-2,000 development states and metrics agree with the previously saved selected-candidate records within the documented numerical tolerance; a discrepancy requires investigation before drawing conclusions about the longer horizon. Independent metric recomputation from saved parameters verifies that the plotted and tabulated results correspond to the archived states.

This diagnostic addresses whether longer optimization helps the existing settings. The proposal's separate frozen-feature linear-probe and spectral interpretation work remains necessary to assess representation quality and explain differences between the methods.
