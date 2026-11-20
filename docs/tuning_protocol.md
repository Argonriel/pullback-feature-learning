# Predeclared tuning protocol

This document specifies stage 4 before the tuning batch is executed. The authoritative machine-readable settings are [configs/tuning.json](../configs/tuning.json). The protocol and configuration are committed to GitHub before inspecting this batch's outcomes. The earlier pilot informed the search ranges; it is development evidence, not an independent evaluation.

Stage 4 chooses fixed optimizer hyperparameters and freezes the subsequent formal-run settings. It does not evaluate a test set, fit a linear probe, or establish an independent performance ranking.

## Fixed experimental setting

The model, initialization distribution, mean logistic objective, summed Euclidean representation metric, and raw-input convention are unchanged from [the experimental protocol](protocol.md). All runs use NumPy, CPU, float64, one BLAS thread, 256 training examples, 256 validation examples, and two-moons coordinate noise with standard deviation 0.15.

Each candidate receives exactly 2,000 update opportunities on each of three development seeds. Each method has 12 candidates, hence 36 scheduled runs and at most 72,000 updates per method. There are 72 scheduled runs in total. This equalizes candidate-run count and per-run update budget, **not computation time**. Actual training, diagnostic, and batch costs are reported separately.

| Development seed | Training seed | Validation seed | Initialization seed |
| --- | ---: | ---: | ---: |
| dev1 | 1701 | 1702 | 23 |
| dev2 | 1801 | 1802 | 24 |
| dev3 | 1901 | 1902 | 25 |

The first development seed was used in the earlier pilot. It is explicitly treated as development data. For every candidate, the two methods receive the same data and initial parameter vector for a given seed. No test split is generated or evaluated during tuning.

## Candidate grids

GD uses the following 12 learning rates:

```text
0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5,
1, 2, 4, 8, 16, 32
```

Damped pullback uses the Cartesian product of four learning rates and three damping values:

```text
learning_rate: 2, 8, 32, 128
damping:       0.001, 0.01, 0.1
```

The damping value stays fixed within a run. These ranges include the earlier pilot settings. Before any tuning runs, the provisional pullback range was changed from `[8, 32, 128, 512]` to `[2, 8, 32, 128]`: the 2,000-step horizon is ten times the pilot horizon, so the final grid includes a smaller update scale while retaining the pilot learning rate of 32. This is a recorded pre-run design choice, not an adjustment to observed tuning results.

Candidate IDs are assigned deterministically from the ordered grids. The 72-run schedule is shuffled with seed `20261119` and saved before execution. The shuffle reduces systematic run-order effects on timing while preserving exact reproducibility of the schedule.

## Execution and failure handling

Each run begins with five disposable warmup updates, then resets to the original parameters before its measured 2,000 updates. Warmup cannot alter the measured initial state. Warmup failure still counts as a failed scheduled run.

Training time includes the required forward/gradient work, Jacobian and Gram construction for pullback, the linear solve, and the update. Additional evaluation, cumulative feature-displacement measurements, and artifact collection are diagnostic work. The timing convention matches the pilot. Timing is not a model-selection criterion in this stage.

Record loss and accuracy curves at initialization and every 100 updates, including update 2,000. Accumulate RMS per-example feature displacement after every update on the fixed training inputs. Save candidate settings, seed identities, source and data fingerprints, final parameters, timings, and failure information. Validation is evaluated for logging and terminal selection only; no validation-based early stopping is used.

A run fails if warmup or measured training encounters a numerical exception, a failed solve, or a nonfinite parameter, representation, loss, or recorded displacement. Preserve its failure reason, completed update count, and available records. A failed run consumes its scheduled trial slot and is not replaced. Other scheduled runs remain part of the recorded batch.

A candidate is eligible for selection only if all three development runs complete successfully with finite terminal metrics. No fallback optimizer, line search, adaptive damping, or post-hoc candidate replacement is introduced. If a method has no eligible candidate, report that failure instead of fabricating a selected setting or silently widening its grid.

## Selection rule

For each eligible candidate, compute the arithmetic mean of validation logistic loss at **exactly update 2,000**, with equal weight for the three development seeds. Choose the candidate with the smallest mean. The best intermediate checkpoint is not eligible to replace this terminal value.

Use the following deterministic lexicographic ranking:

1. Smaller mean terminal validation loss.
2. If that mean is exactly tied, larger mean terminal validation accuracy.
3. If both are exactly tied, smaller candidate ID in the predefined ordering.

Do not introduce a post-hoc near-tie tolerance. Report all candidate scores, per-seed outcomes, eligibility, and actual cost, not only the winners. Mark whether a selected learning rate or damping lies at an edge of the declared grid. Selection establishes the best eligible candidate within this finite grid; it does not establish a globally optimal hyperparameter value.

The selected validation scores are development results. Optimizing a selection criterion on finite data can itself overfit and bias a subsequent performance comparison. This motivates separating hyperparameter selection from the fresh formal evaluation; it is not a claim that three development seeds eliminate selection bias. See Gavin C. Cawley and Nicola L. C. Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11(70):2079-2107, 2010 ([primary source](https://jmlr.org/papers/v11/cawley10a.html)).

## Freeze the subsequent formal study

After applying the selection rule once, save one fixed learning rate for GD and one fixed learning-rate/damping pair for pullback. Freeze the selected settings, candidate-grid identity, and formal protocol. Stage 4 does not run the formal study.

The formal study uses three fresh paired seeds:

| Formal seed | Training seed | Validation seed | Test seed | Initialization seed |
| --- | ---: | ---: | ---: | ---: |
| formal1 | 2701 | 2702 | 2703 | 123 |
| formal2 | 2801 | 2802 | 2803 | 124 |
| formal3 | 2901 | 2902 | 2903 | 125 |

Each split retains the planned sample sizes: 256 training, 256 validation, and 2,000 test examples. The selected hyperparameters remain unchanged across formal seeds. Each pair shares data and initialization across methods. Test results cannot change the settings, stopping rule, or endpoints.

Two formal endpoints are declared separately:

- **Iteration endpoint:** the state immediately after update 2,000.
- **Time endpoint:** the state after the last completed update whose cumulative core training time is at or before 2.0 seconds.

Continue each trajectory until both the 2,000-update budget and the 2.0-second core-training budget have been reached. Preserve the state immediately before crossing the time threshold, as well as the crossing update and exact iteration endpoint. The eventual termination state is not itself a universally matched-iteration or matched-time comparison point.

Save regular curves every 100 updates and force records at the declared endpoints even when they do not coincide with the regular logging interval. A whole update is indivisible; report the actual elapsed time of the retained time-endpoint state. Do not extrapolate beyond an observed trajectory or interpolate model parameters to invent a state at exactly 2.0 seconds.

The hyperparameters are selected for the 2,000-iteration objective. The time comparison therefore serves as a **cost diagnostic for those same iteration-tuned settings**, not as a separate benchmark in which each method has been tuned for a time-budget objective. Report the two endpoint comparisons distinctly, with no early stopping or test-driven changes.

## Stage 4 completion criteria

Stage 4 is complete when the predeclared batch has been recorded, all failures are accounted for, eligible candidates are ranked by the stated rule, selected settings and the formal protocol are saved, and the results can be regenerated from the archived configuration and code. Any later change to the search or evaluation rules must be a separately documented experiment rather than a replacement for this record.
