# Adaptive approximate-symmetry certificates

The frozen target was met on 16 fresh original-ring PB500 models at epsilon 0.1. An adaptive, offline interval table selected 2.7571% of inputs for a second prediction and reduced expected logistic risk by 0.00754331 nats per input. This recovered 89.60% of the gain of a pointwise diagnostic oracle that already uses two model calls per input. The deployed certificate uses only 1.02757 calls per input on average, with a hard maximum of 64 extra calls per batch of 256.

This is a call-budget and risk result. It is not a wall-clock speed improvement on the small, 80-parameter network: the adaptive policy takes about 0.14034 ms per batch with resident tables, versus 0.06176 ms for full half-turn averaging. It also does not establish a uniformly useful certificate for larger mismatch bounds. The partition was optimized for epsilon 0.1, performs worse than uniform refinement at epsilon 0.25, and essentially abstains at epsilon 0.5.

## Design and mathematical result

For two antipodal logits a and b, the exact conditional pair-averaged gain from averaging logits is

    J(a,b) - (p-q)(a-b)/4,

where J is the logistic Jensen gap. An externally supplied bound |p-q| <= epsilon gives the lower bound F = J - epsilon |a-b|/4. The earlier certificate separately minimized J and maximized the separation within a rectangle of possible logits. This study minimizes F jointly over that rectangle. Every minimum is attained at a corner, a diagonal candidate, or a stationary point on an edge. The edge candidates are the valid positive roots of a cubic. There are no smooth, non-diagonal interior minima. See THEORY.md for the derivation.

One wording clarification to the frozen derivation: the stated interior stationary pair is unique **up to interchanging a and b**. Both signs of the anti-diagonal pair occur. This does not change the boundary characterization, implementation, or any result.

Conservative interval propagation encloses the residual tanh network over each polar cell and its antipode. The adaptive partition starts with 64 by 128 cells and makes 12,288 binary splits in 24 batches. Its priority uses independent unlabeled input mass and a conservative possible-gain bound at epsilon 0.1. It never uses labels or the evaluation posterior. All parent work is charged: 65,536 interval propagations produce 32,768 evaluated tree nodes and 20,480 final leaves. The equal-cost uniform control has 128 by 256 cells and the same propagation count.

The table score and fixed tie-breaking variables are invariant under flipping any one input. Selecting the top positive scores, up to 64 per batch, therefore gives a nonnegative expected pair/orbit risk improvement under the antipodally symmetric input law and the external epsilon assumption. This is an expectation over the orbit, not a guarantee for every labeled observation. Unused budget is allowed. Epsilon is not estimated or certified from unlabeled predictions.

The mathematical enclosure and minimization statements are exact. The implementation uses ordinary floating point, conservative padding, residual checks, and fallback logic; it is numerically validated rather than a formally verified directed-rounding proof system.

## Locked confirmation

The algorithm, source hashes, seeds, epsilon grid, budgets, and success thresholds were frozen at 2026-11-21 08:56:41 UTC. Fresh calibration tables were committed at 08:58:07 UTC; fresh evaluation began at 08:59:09 UTC. The 16 fresh original-ring model states were not inspected by this investigator before the algorithm freeze. Training and initialization seeds are 821001–821016 and 822001–822016. Each model has 32,768 independent test inputs, shared across its policy comparisons. Models, not batches, are the independent replication units.

Each adaptive model uses 16,384 coordinate-only calibration inputs with seeds 8650001–8650016. Test inputs use 8651001–8651016 and ties use 8652001–8652016. The evaluation posterior is the fixed mixture

    p_epsilon(r,phi) = (1-epsilon) p_ring(r) + epsilon (1+cos(phi))/2.

It has antipodal difference epsilon cos(phi). This posterior is used only to score risk. It neither changes the input distribution nor enters the gate. Historical 16-model checks use separate inputs and seeds; they are not counted as fresh training replications.

### Primary comparison: epsilon 0.1

| Policy | Interval propagations per model | Selected inputs | Mean risk reduction | Calls per input |
| --- | ---: | ---: | ---: | ---: |
| Original 128 by 128 independent bound | 32,768 | 0.3979% | 0.00199521 | 1.00398 |
| Joint objective on the original grid | 32,768 | 0.9392% | 0.00390410 | 1.00939 |
| Uniform refinement, independent bound | 65,536 | 0.6884% | 0.00319194 | 1.00688 |
| Uniform refinement, joint bound | 65,536 | 1.3725% | 0.00513821 | 1.01373 |
| Adaptive refinement, joint bound | 65,536 | 2.7571% | 0.00754331 | 1.02757 |
| Pointwise diagnostic oracle | No interval table | 4.2873% | 0.00841891 | 2.00000 |

Risk reductions are nats per input, computed from the known conditional posterior and paired orientations. The oracle obtains both logits before deciding where to average, so it cannot be treated as a 1.25-call deployment policy. It is an oracle for the pointwise lower-bound score, not an upper bound on every possible risk-optimal allocation.

All three prespecified criteria passed:

- Coverage was 2.7571%, above the 2% cohort-mean target. Its pointwise 95% model-bootstrap interval is [2.0184%, 3.5311%]. Ten of 16 individual models met the 2% target.
- The ratio of adaptive to oracle mean gain was 89.60%, above the 50% target; its bootstrap interval is [85.59%, 92.25%].
- The paired improvement over the original certificate was 0.00554809 nats, with a 95% interval [0.00388245, 0.00728136].

The paired improvement over the mandatory, equal-propagation uniform joint control was 0.00240509 [0.00170483, 0.00322997]. All 16 models had greater gain under the adaptive policy than under either of these controls. Confidence intervals use 10,000 paired model bootstrap resamples, seed 8653000; they are pointwise, not simultaneous. The study does not infer 16 times 128 independent trained-model replications from its batches.

### Sensitivity and negative findings

| Epsilon | Original gain | Uniform joint gain | Adaptive gain | Oracle gain | Adaptive coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00659809 | 0.00808345 | 0.00951123 | 0.00995739 | 13.0621% |
| 0.1 | 0.00199521 | 0.00513821 | 0.00754331 | 0.00841891 | 2.7571% |
| 0.25 | 0.00045112 | 0.00269415 | 0.00095325 | 0.00633264 | 0.1825% |
| 0.5 | 0 | 0.0000000952 | 0.0000000952 | 0.00026922 | 0.0001907% |

The epsilon 0.25 result is a real failure of transfer for the fixed epsilon 0.1 partition. No alternative partition or threshold was selected after seeing it. At epsilon 0.5, the joint gates each selected only one of 524,288 total fresh test inputs. That is essentially abstention, not evidence of useful coverage. The historical cohort showed a similar primary improvement: adaptive coverage 2.7319%, gain 0.00674372, and 90.48% of its oracle gain. These historical models are a development check only.

## Cost and execution checks

Fresh mean setup times, including all four epsilon objectives and shared objective variants, were 0.5255 seconds per original table, 1.0441 seconds per uniform table, and 1.3919 seconds per adaptive table. Adaptive partition time includes priority-root work. These are measured study setup times, not minimal timings for the independent-only baseline. Equal interval propagation counts do not mean equal arithmetic or elapsed time. The adaptive table also uses 16,384 coordinate-only calibration draws; the uniform table does not need these draws.

Across old and fresh cohorts, the physical experiment performed 5,242,880 interval propagations. Each cohort performed 35,651,584 cubic solves, counting partition priorities and final objective work; there were no numerical fallbacks. Maximum stationary residual was 5.42e-16. Stored adaptive arrays occupy about 5.41 MB per model in memory, including all objectives, parent nodes, intervals, and calibration radii. This is the implemented representation, not a claimed minimal serving format.

The original frozen evaluator used archive-backed NumPy tables and repeatedly decompressed arrays during lookup. Its latency values are preserved under old/results and fresh/results and labeled archive-backed. A separate outcome-neutral benchmark loaded the same arrays once, verified identical masks, predictions, and calls, and randomized method order after warmup. No scientific policy changed. The final benchmark includes raw and full-half-turn executed-prediction references:

| Method | Median of 16 model-median times, ms per 256 inputs |
| --- | ---: |
| Raw prediction | 0.030125 |
| Full half-turn averaging, two calls | 0.061756 |
| Original interval gate | 0.082081 |
| Uniform joint gate | 0.087700 |
| Adaptive joint gate | 0.140341 |
| Pointwise oracle gate, two calls | 0.100947 |

The adaptive policy is about 2.27 times slower than full averaging in this implementation because routing dominates the tiny model's computation. Its query-count savings are therefore not a measured time saving. Timings use one BLAS thread, eight batches per measurement, two warmup blocks, and eight measured blocks per model. The original archive-backed, first resident-table, and final resident-table benchmarks are all retained.

All 640 explicit first-batch deployment checks across both cohorts passed. Every saved batch respected the maximum of 64 extra predictions. Eighty additional archive-versus-memory checks passed exactly. No selected pair had negative expected gain within tolerance. All held-out logits were enclosed, no tested point lay outside radius 3, and source, table, result hashes, and freeze/commit/score ordering matched. Independent root audits additionally checked 1,500 random rectangles and all fresh gain rows and adaptive split records. No scientific setting or failed numerical attempt was removed.

## Interpretation and next step

This completes a bounded positive result: joint rectangle minimization plus an input-mass-guided partition makes the approximate-symmetry certificate substantially less conservative at a prespecified small mismatch bound, and the benefit replicates on fresh trained models. The success is conditional on the known input symmetry, a credible external epsilon bound, the current architecture, and a partition targeted at epsilon 0.1.

The most useful next step for this component is to reduce routing and table overhead while preserving the existing invariant mask and guarantee, then test on a model whose second prediction is genuinely expensive. A separately frozen epsilon-aware partition would be needed before claiming performance at epsilon 0.25. Neither claim is established here. No further experiment is required to interpret this study's positive primary result and its recorded limitations.

## Files

- `THEORY.md`, `PROTOCOL.md`, and `freeze.json`: frozen mathematical and experimental specification.
- `bounds.py`, `study.py`: frozen policy and scoring code.
- `old/calibration`, `fresh/calibration`: every evaluated cell, model, interval, objective, adaptive trace, setup cost, and selector commitment.
- `old/results`, `fresh/results`: every model/epsilon/method outcome, batch count, enclosure check, deployment check, and original archive-backed timing.
- `aggregate.csv`, `paired_contrasts.csv`, `cost_summary.csv`, `summary.json`: auditable post-score summaries.
- `serving_with_full`: final randomized, resident-table serving benchmark; `serving` retains the first benchmark.
- `certificate_primary.png` and `.pdf`, `certificate_sensitivity.png` and `.pdf`: standalone scientific figures, visually inspected.
- `REPRODUCE.md`, `reproduce.py`: non-destructive reproduction into a new directory.
