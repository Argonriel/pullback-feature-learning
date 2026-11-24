# Formal paired experiment protocol

This document specifies the execution details for stage 5 before the formal test results are inspected. It implements the settings and endpoints already frozen in [configs/formal.json](../configs/formal.json) under the [stage-4 protocol](tuning_protocol.md). It does not change the chosen hyperparameters, data seeds, or comparison budgets.

The purpose is to evaluate the two fixed methods on fresh paired data and initializations, with separate iteration and computation-time comparisons. Frozen-feature linear probes and a full interpretation of representation quality remain stage 6.

## Reproduce this stage

Use the installed project environment described in the [README](../README.md). From the repository root, run the tests and numerical verification before starting formal evaluation:

```sh
python -m unittest discover -s tests -v
python -m pullback_feature_learning.verify --output runs/formal-verification.json
```

Continue only if both checks succeed. Then run the frozen formal configuration:

```sh
python -m pullback_feature_learning.formal --config configs/formal.json --output runs/formal --frozen-config-commit 8e24690f4f2af0e2c20a28701a2c0ea591cef4e7
```

The command compares the supplied JSON configuration with `configs/formal.json` in the specified stage-4 Git commit before training. It requires that commit to be available in the local repository history. A fresh output directory is required; existing evidence is never overwritten. Use new output paths for an explicitly documented repeat. A failed batch also retains its output directory and records.

## Fixed methods and data

The model remains the two-block, 80-parameter residual network with a fixed readout, mean logistic loss, and summed Euclidean representation metric. Data generation and initialization follow [the original protocol](protocol.md). All computations use NumPy, CPU, float64, and one BLAS thread.

| Method | Learning rate | Damping |
| --- | ---: | ---: |
| GD | 0.125 | Not applicable |
| Damped representation pullback | 8 | 0.1 |

No optimizer setting is changed in response to formal training, validation, or test performance. There is no line search, adaptive damping, validation-based early stopping, or test-selected checkpoint.

Each formal seed has 256 training examples, 256 validation examples, and 2,000 test examples. All coordinate noise has standard deviation 0.15. Data and initial parameters are identical within a method pair.

| Pair | Training seed | Validation seed | Test seed | Initialization seed |
| --- | ---: | ---: | ---: | ---: |
| formal1 | 2701 | 2702 | 2703 | 123 |
| formal2 | 2801 | 2802 | 2803 | 124 |
| formal3 | 2901 | 2902 | 2903 | 125 |

These formal data and initialization seeds were declared before the stage-4 tuning results. They differ from the development seeds. The two methods receive copies of the same arrays and parameter vector for each pair; fingerprints establish this pairing in the saved record.

## Predetermined execution order

The six runs execute sequentially in this order:

1. formal1: GD, then pullback.
2. formal2: pullback, then GD.
3. formal3: GD, then pullback.

Alternating the within-pair order reduces a consistent first-method timing advantage. Three pairs cannot perfectly balance two possible orders, and this schedule is not a substitute for a dedicated hardware benchmark. Runs are not reordered after inspecting results.

Each run performs five disposable warmup updates and then resets exactly to its saved initial parameters. Warmup does not count toward either formal budget. Its elapsed time is recorded separately.

## Two distinct endpoint states

Let `t_k` be the cumulative measured core training time immediately after update `k`, with `t_0 = 0`.

- **Iteration endpoint:** the parameter state after exactly 2,000 updates.
- **Time endpoint:** the parameter state at `k_time = max {k : t_k <= 2.0 seconds}`.

Continue training until the run has completed at least 2,000 updates and has **strictly crossed** the 2.0-second core-training threshold. Preserve the states on both sides of the time threshold: the time endpoint and the first following update with `t_k > 2.0`. If an update ends at exactly 2.0 seconds, it is eligible as the time endpoint and one additional update establishes the upper bracket. If the first measured update already exceeds the budget, the time endpoint is the initial state at step 0.

The iteration endpoint and time endpoint can occur in either order. A run that reaches 2,000 updates quickly continues to the time threshold; a run that crosses the threshold early continues to update 2,000. Preserve both requested states even if either is not the terminal state. The eventual termination state is not itself a universally matched-iteration or matched-time comparison point.

Report the actual update count and measured time at every endpoint, as well as the duration of the crossing update. Whole updates are indivisible. Do not interpolate model parameters to invent a state at exactly 2.0 seconds or extrapolate performance beyond the observed trajectory.

The settings were tuned for the 2,000-iteration validation objective. The time comparison is therefore a cost diagnostic for those same iteration-tuned settings, not a separate benchmark with time-objective hyperparameter tuning.

## Training, diagnostics, and timing boundaries

The online training procedure uses training inputs and labels only. It receives no validation or test objective for update selection or stopping decisions.

Core update timing includes the forward and gradient computations required by the method, full Jacobian and Gram construction for pullback, the damped linear solve, and the parameter update. GD uses its independent reverse-mode gradient and is not charged for an unnecessary full Jacobian. The same timer and timing convention apply to both methods.

Parameter copies, timing-bookkeeping work outside the update, saved-state collection, extra representation evaluations, and cumulative feature-displacement calculations are diagnostic or recording work. They are excluded from the 2.0-second core budget. Explicit representation and snapshot diagnostics are timed separately; loop wall time also contains residual bookkeeping overhead. Warmup, offline evaluation, SVD diagnostics, serialization, and plotting are also excluded from core training. Record loop and batch wall time so that the core budget is not mistaken for end-to-end elapsed time.

For every measured update, accumulate the RMS per-example representation displacement on the fixed training inputs:

$$
D_k=D_{k-1}+\left(\frac1n\sum_{i=1}^{n}\|z_i(\theta_k)-z_i(\theta_{k-1})\|_2^2\right)^{1/2}.
$$

This quantity is a diagnostic. It cannot affect an update, its learning rate, damping, or stopping rule. It is the sum of successive displacements, not simply the initial-to-final distance.

The time budget measures this implementation on the recorded machine. Operating-system activity and timer variation can change the number of updates completed within two seconds. Record software versions, thread settings, and actual endpoint counts; do not claim that a repeated wall-clock run must end at the same update number.

## Saved states and offline evaluation

Save snapshots at initialization and every 100 updates. Force additional records at update 1,000, update 2,000, the lower and upper time-threshold states, and the eventual termination state. Multiple roles may refer to the same state; their identities must remain explicit. Each snapshot contains at least the step, actual cumulative core time, cumulative feature displacement, and parameters.

Only after all six fixed training trajectories complete successfully, generate the validation and test arrays and evaluate the saved snapshots on the training, validation, and test splits. This separation keeps test evaluation outside training and budget decisions. If any scheduled training attempt fails, no validation or test arrays are generated and no offline evaluation or aggregate comparison is performed. Test scores from a successful batch are reported for the predeclared endpoint states and sampled learning curves; they are not used to choose a new checkpoint or revise the method.

Use the same fixed-readout logistic loss and classification rule for both methods. Prediction at a zero logit follows the existing code's deterministic positive-class convention. These scores are distinct from later linear-probe scores.

Compute Jacobian singular values and gradient alignment offline using the training representation map at the initial, intermediate, and endpoint checkpoints. Preserve the full two-coordinate representation Jacobian and the documented numerical-rank convention. These are explanatory diagnostics, not extra selection criteria. Full linear-probe evaluation and any distribution-shift extension are outside this stage.

Training loss and test accuracy are plotted against both iteration count and measured training time. Show the common iteration range through update 2,000 and the common core-time range through two seconds. Display per-seed trajectories or clearly state how sampled curves are aligned for a mean. A display interpolation must not be treated as an observed parameter state or replace an endpoint measurement. Do not extend a run beyond its observed range in a comparison plot.

## Aggregation and interpretation

For each endpoint type separately, report every seed's result and the method mean and sample standard deviation across the three seeds (`ddof = 1`). Do not pool all test examples across the three datasets and treat them as independent training runs.

Also report within-pair differences, defined as **pullback minus GD**, and their mean and sample standard deviation. For loss, a negative difference favors pullback; for accuracy, a positive difference favors pullback. Include actual update counts, endpoint times, and core/diagnostic costs so that the performance and cost comparisons remain interpretable.

Three pairs provide a small controlled comparison, not strong evidence of statistical significance or broad generalization across architectures and datasets. The sample standard deviation is descriptive and is not a confidence interval. Retain the previously documented limitations of the finite tuning grid, including the selected pullback damping at its upper boundary.

## Failure handling and provenance

If warmup, an update, or recorded diagnostics produce a nonfinite state or a numerical failure, preserve the latest finite prefix, recorded update count, timing, and failure reason. Save that finite prefix even when it falls between the regular 100-update snapshots; the initial state is retained if warmup fails. Do not replace the run with a new seed, substitute a different optimizer, or silently change the configuration. Attempt and retain all six scheduled training runs, including those after an earlier failure.

If any of the six training attempts fails, mark the entire batch incomplete and report the completed and scheduled trajectory counts. Stop before validation/test generation and before all offline evaluation. Write the retained trajectories, environment, timing, and failure summary, but produce no endpoint aggregates or survivor-only mean. A failed run cannot supply a missing endpoint. If a software defect is found, preserve the original attempt, document the correction and rerun policy, and distinguish any corrected run from the original record.

Archive the frozen configuration, schedule, source and data fingerprints, initial states, saved snapshots, endpoint roles, measured costs, offline metrics, spectra, plots, and summary tables. The iteration endpoint can be checked by deterministic replay under the same numerical environment; the time endpoint is audited against its saved state and measured bracket because timing can vary across repetitions.

Stage 5 is complete when all scheduled runs are accounted for, the two endpoint comparisons are reported under the fixed rules, raw evidence is preserved, and independent checks confirm pairing, endpoint selection, and metric recomputation. No parameter retuning follows inspection of these test results.
