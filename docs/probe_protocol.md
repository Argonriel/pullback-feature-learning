# Frozen-feature probe protocol

This is the feature-quality component planned in the [original proposal](proposal.tex), implemented after the formal fixed-readout results and supplementary long-training diagnostic were inspected. The probe protocol, configuration, and implementation are recorded before computing these probe test results. This is a planned representation diagnostic on the existing formal splits, not a newly independent confirmation study or a reproduction of another paper's experiment.

## Research question and frozen inputs

Do the learned representations support a better affine linear classifier than their initial representations, and how do GD and damped representation pullback compare under the same probe protocol?

Use the six archived trajectories in `reports/formal-v1/trajectories.json`, their exact saved train/validation/test arrays in `data.json`, and the unchanged network implementation. Do not train the network, select another network checkpoint, retune its learning rate or damping, or alter the old formal results. These network settings were selected independently on development data before the formal runs. The supplementary development trajectories at 100,000 updates are not substituted for formal checkpoints.

The three paired formal seeds share input arrays and initialization across methods. For each method and seed, evaluate the following six states:

| Label | Saved state |
| --- | --- |
| initial | Exactly 0 updates |
| early100 | Exactly 100 updates |
| early500 | Exactly 500 updates |
| middle1000 | Exactly 1,000 updates |
| iteration2000 | Exactly 2,000 updates |
| time | Original saved last whole update at or before 2 seconds |

This produces 36 probe tasks. Initial GD and pullback features within a pair must be identical, so their fitted probes and predictions must agree. Time endpoints have different update counts across methods; they remain distinct from the iteration endpoint. Report every state without promoting a favorable intermediate state to the primary formal endpoint.

## Probe model, normalization, and fitting

The frozen network produces two-dimensional features `z`. Fit each coordinate's mean and population standard deviation (`ddof=0`) on the training features only. Use that same mean and scale for validation and test transformations. Floor each standard deviation at `1e-12`; record the fitted values. Each checkpoint has its own training-fitted standardization under this shared rule.

For standardized features `s` and labels `y` in `{-1, +1}`, fit an affine score `a.T @ s + b` by minimizing

```text
mean(log(1 + exp(-y * (a.T @ s + b)))) + alpha / 2 * ||a||^2.
```

Penalize the two coefficients, not the intercept. This uses the mean logistic loss, and `alpha` is a probe penalty distinct from the network's pullback damping. Classify nonnegative scores as `+1`. Report unpenalized logistic loss and accuracy on every split.

Use NumPy float64 on CPU with one BLAS thread. Fit each candidate independently from zero coefficients and intercept using Newton steps with backtracking. The convergence criterion is infinity norm of the regularized objective gradient at most `1e-9`, with at most 100 Newton updates. Use Armijo constant `1e-4` and at most 40 backtracking attempts. Preserve fitted coefficients, normalization, final objective, gradient norm, iteration count, status, and costs. The source implementation and actual recorded solver outcomes are part of the reproducibility evidence.

The common candidate grid is `alpha = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]`: eight candidates for every task, or 288 candidate fits overall. Select the smallest unpenalized validation loss for each task; an exact tie favors the larger alpha. Do not use test metrics for fitting, normalization, regularization selection, or choosing a network state. Do not refit on training plus validation after selection.

## Evaluation sequence and failure handling

Complete all training/validation fits and all 36 regularization selections first. Save the full candidate records and selected models before computing any test representation or probe test prediction. Only then evaluate all selected probes on the saved test arrays. The test arrays already exist in the source JSON; loading that archive is not a new independent test draw.

A nonfinite input/state/metric, failed linear solve, exhausted line search, or unconverged candidate is an analysis failure. Retain its evidence. Do not silently omit candidates or use a partial successful grid to select a probe. If any task lacks a fully successful candidate grid, stop before probe test evaluation and report an incomplete batch. Any numerical correction must be documented and must preserve the failed batch. No test-score-dependent solver or grid changes are permitted.

## Comparisons and interpretation

For each state, report individual seeds, arithmetic means, sample standard deviations (`ddof=1`), and paired differences defined as pullback minus GD. Also compare each method's learned-feature probe against its paired initial-feature probe. Compare probe scores to the original fixed-readout score at the exact same network state; these classifiers have different fitting rules and should not be conflated.

The initial-feature comparison controls for what is already linearly accessible before training. A good probe measures information accessible to this affine classifier on the evaluated distribution. It does not establish broad representation quality, causal mechanisms, transfer performance, or distribution-shift robustness. Improvements over the fixed readout can reflect refitting, scale/calibration, a changed decision boundary, or use of both coordinates. Without additional controls, do not attribute them specifically to the second feature coordinate or label them the same hidden-learning phenomenon as another paper.

These are descriptive comparisons across three seeds, not significance tests. Probe test data overlap the original formal evaluation, whose results have already been seen. The original proposal included probes, but the detailed grid and solver were specified at this later stage. Clearly preserve that chronology.

The companion [representation-analysis protocol](representation_analysis_protocol.md) addresses training-feature point clouds, checkpoint spectra and gradient alignment, and saved-state comparisons at cumulative feature-displacement budgets. Associations between these diagnostics and probe outcomes do not prove causality.

## Reproducibility and next stage

Preserve source artifact hashes, source/configuration fingerprints, the protocol Git revision, environment and thread settings, every candidate record, selected probe parameters, per-seed metrics, summaries, figures, and measured evaluation costs. Use a fresh output directory and never overwrite prior runs. Verify probe metrics independently from saved weights and normalization, initial-pair equality, selection rules, and aggregate calculations.

After these analyses, the remaining core task is to assemble the final report: mathematical derivation, local spectral predictions, optimization/cost results, representation diagnostics, and clearly delimited findings. The optional distribution-shift experiment is not required to complete this core scope.
