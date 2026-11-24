# Stage 4: selected settings and interpretation

Stage 4 is complete. The [protocol and grid](tuning_protocol.md) were committed before execution in [26e278d](https://github.com/Argonriel/pullback-feature-learning/commit/26e278dbd708b7eab6b687f1e398e0741c225164). The batch was executed once, with no candidate replacement, grid expansion, early stopping, or post-hoc change of selection objective.

## What was run

Each method received 12 candidate configurations, each evaluated on three paired development seeds for 2,000 updates. All 72 scheduled runs completed with finite terminal results. A shuffled schedule was fixed in advance. The candidate and update budgets were equal; actual compute budgets were not.

The entire batch, including setup, diagnostics, serialization, and plotting, took **66.35 seconds** on the recorded CPU environment. GD used 6.01 seconds of core training and 3.78 seconds of diagnostics across its 36 runs. Pullback used 49.07 seconds of core training and 5.06 seconds of diagnostics across its 36 runs. Warmup and other overhead explain the remainder. These are measurements of this implementation and machine, not universal speed ratios.

## Settings selected by the declared rule

| Method | Learning rate | Damping | Mean terminal validation loss | Across-seed sample SD | Mean validation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| GD | 0.125 | Not applicable | 0.049446 | 0.021092 | 98.31% |
| Damped pullback | 8 | 0.1 | 0.072279 | 0.014322 | 98.18% |

Selection used the mean validation **logistic loss** at update 2,000, not accuracy or the best earlier checkpoint. Logistic loss also reflects prediction confidence, so the lowest-loss configuration need not have the highest classification accuracy. The standard deviation summarizes these three development runs; it is not a confidence interval.

The ranking figure uses a logarithmic vertical axis for GD because its largest learning rate produces a high-loss outlier; the pullback panel uses a linear vertical axis. Compare the labeled loss values, not point heights across panels.

Both configurations were valid on all three development seeds. GD's selected learning rate is inside the grid. Pullback's learning rate is inside its grid, but damping **0.1 is the upper boundary**. It is therefore the best eligible setting within this declared grid, not evidence that 0.1 is the globally best damping. Any broader search would be a separate documented experiment with a stated comparison budget.

## What changed in our understanding

The 200-step pilot used illustrative settings and showed lower loss for pullback at that iteration count. After giving both methods a declared parameter search and a longer fixed horizon, the selected GD configuration has lower mean development validation loss, while the two selected accuracies are close. The settings, horizon, and seed set differ from the pilot, so the reversal cannot be attributed to one cause in isolation.

This is a reason to retain tuned baselines and distinguish iteration efficiency from computation cost. It is not a final finding that GD generalizes better. The same development data helped select these configurations, the search spaces differ in dimensionality, there are only three development seeds, and the pullback damping optimum may lie outside the declared range. Fresh formal evaluation and frozen-feature linear probes are still needed.

## Frozen next-stage configuration

[configs/formal.json](../configs/formal.json) contains the selected settings and the already declared fresh seed triplets. Both methods will use the same training/validation/test data and initialization within each formal pair. None of these formal arrays has been generated or evaluated in stage 4.

The planned formal study has two separate comparison endpoints:

1. The exact state after 2,000 updates.
2. The last completed update at or before 2.0 seconds of cumulative core training time.

Each trajectory continues until both budgets have been reached, while preserving the two endpoint states and the time-threshold bracket. The eventual termination state must not be treated as a common-budget endpoint. The time comparison is a cost diagnostic for the same iteration-tuned settings, not a separate time-objective tuning exercise.

Stage 5 has not been executed. No test scores, test-selected stopping rule, or linear-probe result is reported here.

## Reproducibility records

The [archived batch](../reports/tuning-v1/README.md) includes the grid, shuffled schedule, complete per-seed outcomes, sampled learning curves, terminal parameters, development arrays, source/data fingerprints, actual timings, candidate rankings, and the selected formal specification. The [test log](../reports/tuning_test_results.txt) records all 26 passing checks, including numerical correctness, complete-seed selection, failure retention, warmup reset, and development-only data generation. The [independent artifact audit](../reports/tuning_audit.json) checks the saved results against the archived data and checkpoints.
