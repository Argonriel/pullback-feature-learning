# Whole-update displacement refinement protocol

The first archived-state analysis exposed a measurement-resolution limitation: at a 25% displacement ceiling, GD's first saved 100-update checkpoint already exceeds the budget, so the saved-state rule retains initialization. This refinement resolves that gap by deterministically replaying the original six trajectories and locating each crossing to a single whole update. It preserves the previous analysis, original trajectories, iteration/time endpoints, and optimizer settings.

This protocol is written after inspecting the sparse-checkpoint results and is therefore a documented diagnostic refinement. Commit this protocol and runner before executing it. The budget fractions and their original training-based base values remain unchanged; no outcome is used to choose a different ceiling, hyperparameter, or checkpoint rule.

## Replay and exact verification

Use `reports/formal-v1/config.json`, `data.json`, and `trajectories.json`. For each of the three formal seeds, replay GD and pullback from the exact archived initial parameters on the exact archived training arrays through exactly 2,000 updates. The frozen settings remain GD learning rate 0.125, pullback learning rate 8 and damping 0.1, the same 80-parameter model, and CPU float64 with one BLAS thread. No validation or test data participate in updates. Disposable warmup is unnecessary for this deterministic verification because replay timings do not replace the original timing experiment.

Accumulate displacement after every update using exactly the original formula:

```text
D_next = D_previous + sqrt(mean(sum((Z_next - Z_previous)^2, axis=1))).
```

At every archived checkpoint up to and including step 2,000, require exact equality of the replayed parameter array and cumulative displacement with the archived values. Record the maximum absolute parameter error and displacement error, both of which must be zero. This intentionally strong requirement is appropriate to the recorded local environment; results on another platform may require a separately documented numerical-tolerance protocol instead.

A failed verification, failed solve, numerical exception, or nonfinite state aborts this replay batch before any new test evaluation. Preserve the attempted replay, current finite state, failure reason, checks performed, and measured diagnostic cost. Do not silently loosen the equality rule, alter the optimizer, or overwrite failed evidence.

## Fixed ceilings and whole-update brackets

For each paired seed, the base ceiling remains the smaller of the two cumulative displacements at the original exact 2,000-update endpoints. The four targets are 25%, 50%, 75%, and 100% of that archived value.

For each method and target, retain the last whole update whose cumulative displacement is at or below the target. Preserve the immediately following whole update strictly above the target. Their step indices must differ by exactly one. Exact equality belongs to the retained lower state. If multiple updates have equal displacement, use the last eligible one before a strict crossing. For the special case where the target equals the step-2,000 displacement, the lower state is step 2,000 and no upper state exists inside the fixed replay horizon.

Preserve both parameter vectors, step indices, cumulative distances, target values, and residual gaps. Do not interpolate parameters or metrics. Realized lower-state distances can still differ between methods because the stopping points are whole updates, but the residual shortfall is bounded by one observed update whenever an upper crossing exists.

## Evaluation and aggregation

Finish all six replays, verify every original shared state, and select all 24 lower/upper brackets before reconstructing any test features or computing any new test metric. Save successful replay records before test evaluation starts.

Then evaluate the original training and formal test sets at every available lower and upper state, using the existing fixed readout and stable logistic loss. No classifier or probe is fitted here. This test split was already evaluated in the original formal study, so these scores remain a secondary descriptive analysis rather than fresh independent test evidence. Metric values do not influence checkpoint selection.

For each ceiling fraction, report all three pairs and summarize the lower states with mean and sample SD (`ddof = 1`) for training/test loss, accuracy, update count, realized cumulative displacement, and unused budget. Also report the largest unused budget per method and the paired pullback-minus-GD differences for training loss, test loss, and test accuracy. Upper-state metrics remain visible in the per-state table to show crossing behavior. No survivor-only aggregate is permitted after a failed replay.

The per-seed base ceilings differ, so equal fractions mean the same paired budget convention, not identical absolute motion across seeds. This provides a comparison under tightly bracketed common displacement budgets; it does not assert identical continuous-time feature paths or prove a general ordering of the optimizers.

Plot the retained lower states at their actual cumulative displacement, with separate panels for training loss and test accuracy. Method colors and seed line styles identify every paired trajectory; markers denote observed states, and connecting lines are visual guides only. They do not create interpolated metric estimates or select additional checkpoints.

## Artifacts and timing

Use a fresh output directory. Archive input/source hashes, software and thread details, the committed protocol, base budgets, all replay verification checks, selected lower/upper weights, their metrics, descriptive summaries, and actual replay/evaluation wall time. Record timestamps showing that all replay verification finished before test evaluation began.

Replay cost is an additional diagnostic cost. It must not substitute for the original core-training times or change the original two-second endpoints. Retain the sparse archived analysis as evidence of why this refinement was necessary. Interpret the refined comparison alongside the original fixed-iteration, fixed-time, probe, and local-spectrum results without selecting a favorable endpoint after seeing test scores.

If offline evaluation fails after successful replay, preserve the replay records, evaluation-start timestamp, failure reason, incomplete summary, and elapsed diagnostic cost. Do not report a complete aggregate or silently replace failed metrics.
