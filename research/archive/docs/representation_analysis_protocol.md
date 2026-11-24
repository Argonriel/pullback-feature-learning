# Archived representation analysis protocol

This stage completes the proposal's feature-geometry, local-spectrum, and cumulative-displacement diagnostics using the already frozen formal study. The analysis is specified after the formal fixed-readout and supplementary long-training results have been seen. It is descriptive interpretation, not a new independent test of optimizer superiority. Commit this protocol and its runner before executing the analysis.

The source is `reports/formal-v1`. No network parameters, learning rates, damping values, seeds, or original endpoints are changed. This command does not fit linear probes; the separately recorded probe protocol governs that analysis. Test predictions are not recomputed here. Any reported test scores are copied from the original formal metrics, and do not select a new checkpoint or replace the formal endpoints.

## Fixed archived states and feature clouds

Analyze all three formal pairs and both methods at four states:

1. Initial parameters, step zero.
2. The pre-existing intermediate checkpoint at step 1,000.
3. The exact iteration endpoint at step 2,000.
4. The saved last-whole-update endpoint at or before two core-training seconds.

The last category has different update counts between methods and seeds. Label every panel with the actual count. Preserve the initial state for each method even though the paired initial parameters and features coincide; this makes the common origin explicit.

Plot only training representations, in their raw two-dimensional coordinates. Each seed receives a two-row, four-column panel grid, with method rows and state columns. Use common coordinate limits across all eight panels within that seed, equal coordinate aspect ratios, class colors, and the vertical line at coordinate one equal to zero, which is the fixed classification boundary. Axis limits can differ across seeds and must not imply a common global scale. No point-cloud normalization, rotation, label-based rearrangement, or selected-example filtering is performed.

Archive every plotted feature coordinate and its label. A visually wider separation or larger feature movement does not on its own establish useful features or better generalization. The independent frozen-feature probe assessment supplies the predictive diagnostic.

## Spectra and local gradient alignment

Reuse the singular values and squared gradient projections saved in `spectra.csv` and the rank metadata in `spectral_metadata.json`. The original rank threshold is

```text
tau = max(J.shape) * machine_epsilon(float64) * largest_singular_value.
```

Treat a singular direction as numerically active exactly when its singular value exceeds this recorded threshold. Record the numerical rank, largest singular value, smallest active singular value, and their active condition-number ratio. Full column rank here is at most 80, while the representation vector has 512 coordinates. It does not imply that every representation-space gradient direction is reachable.

Let `a_i` denote the saved fraction `(u_i.T @ g)^2 / ||g||^2`, where `g` is the representation loss gradient. Report its sum over numerically active directions and the original unreachable fraction. For each archived state, evaluate both hypothetical local spectral weightings:

```text
q_i(GD)       = sigma_i^2
q_i(pullback) = sigma_i^2 / (sigma_i^2 + lambda),   lambda = 0.1.
```

Report the following quantities, summing over the numerically active directions:

```text
gradient_weighted_gain = sum_i a_i q_i
velocity_norm / gradient_norm = sqrt(sum_i a_i q_i^2)
cosine(-g, J u) = (sum_i a_i q_i) / sqrt(sum_i a_i q_i^2)
predicted_step_RMS_motion = learning_rate * ||g|| * sqrt(sum_i a_i q_i^2) / sqrt(n).
```

The predicted motion includes each method's frozen learning rate: 0.125 for GD and 8 for pullback. Recompute only `g` and its norm from the archived training state; the full Jacobian or optimizer trajectory is not retrained or recomputed. If the velocity or gradient is zero, its directional cosine is undefined and must be represented as missing rather than an invented zero-angle value.

Both candidate directions in this table are evaluated at the same archived state. Comparing a GD-trajectory state with a pullback-trajectory state additionally changes the Jacobian and gradient. A local cosine or gain is not an actual finite-step observation, a generalization score, or a causal explanation of a whole trajectory. The earlier numerical verification establishes the local first-order identities, not exactness for the finite learning rates used throughout training.

Plot singular values in descending order and their gradient-alignment fractions for all four states, retaining every archived direction. Individual singular-vector alignment can be unstable within repeated or nearly repeated singular-value subspaces; do not assign a semantic interpretation to a particular index or infer task relevance from the spectrum alone.

## Secondary comparison at common displacement ceilings

Use the original cumulative RMS feature displacement, which was accumulated from every training update on the fixed training inputs. It is a path length rather than the distance between initial and final feature clouds.

For each paired seed, take the smaller of the two displacements at exactly 2,000 updates as a common base budget. Define four ceilings at 25%, 50%, 75%, and 100% of that value. These choices depend only on already archived training displacement, never on validation or test performance.

Only saved checkpoints at or before step 2,000 are eligible. For each method and ceiling:

1. Retain the last saved checkpoint whose cumulative displacement is at or below the ceiling. Exact equality is eligible; if a plateau contains multiple saved states, retain the last one.
2. Preserve the next saved checkpoint strictly above the ceiling, if one exists, and record the number of updates between the two saved states. This is a saved-checkpoint bracket, not a bracket between consecutive updates.
3. Record the target, realized displacement, unused budget, upper gap, update count, measured core time, and original training/test metrics at the retained state.
4. If the target equals a trajectory's final saved displacement, there need not be an upper checkpoint. Targets outside the observed range are errors.

There is no parameter, displacement, loss, accuracy, or time interpolation. Because regular logging is every 100 updates, realized feature motion may differ substantially between methods even under the same ceiling. This analysis must be described as a comparison under common displacement budgets using saved states, not as a claim of exactly matched realized motion. Plot the sampled training loss against cumulative displacement through step 2,000, mark the retained states at their actual displacements, and show the target ceilings.

Report all 24 per-seed/method/budget records, with both bracketing states. Descriptive aggregates use all three pairs: means, sample standard deviations (`ddof = 1`), and pullback-minus-GD paired differences. Per-pair base budgets differ, so a fraction denotes the same within-pair budget convention, not a common absolute displacement across seeds. Missing pairs or incomplete formal trajectories prevent an aggregate; no survivor-only comparison is allowed.

## Provenance and interpretation limits

Refuse to overwrite an existing output directory. Archive input artifact hashes, source hashes, the protocol/source commit, software and thread environment, figure coordinates, local spectral summaries, displacement brackets, descriptive summaries, and analysis wall time. Use CPU float64 and one BLAS thread for training-feature reconstruction.

The conclusions are limited to one small network, one noisy two-moons distribution, and three paired seeds. Geometry and alignment are measured associations. No claim of a new optimizer, Fisher-natural-gradient equivalence, Wasserstein geometry, universal optimizer ranking, or distribution-shift robustness follows. Keep earlier formal results intact, and use the separate frozen-feature probe results to distinguish representation quality from the original fixed readout.
