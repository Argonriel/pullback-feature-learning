# Frozen-feature probe findings

The representation-quality analysis specified in the [original proposal](proposal.tex) is complete for the six archived formal trajectories. The [probe protocol](probe_protocol.md), configuration, implementation, and 79 passing tests were committed in [6cb553d](https://github.com/Argonriel/pullback-feature-learning/commit/6cb553d09784d0769f0f03138d76e8accce2d9f8) before this probe evaluation. All 288 regularization candidates converged; all 36 validation selections were saved before any probe test features were evaluated.

The result distinguishes early feature learning from terminal performance. Pullback features support much better linear classification at 100 and 500 updates. GD retains better probe performance at both original formal endpoints. Refitting a classifier does not reveal a hidden terminal advantage for pullback under this protocol.

## What was fitted

The network parameters stayed fixed. Each two-dimensional feature set received an affine logistic classifier with an unpenalized intercept, training-fitted coordinate standardization, and the same eight positive regularization candidates. Selection minimized unpenalized validation loss; test scores did not select a regularization value or network checkpoint. Initial, early, intermediate, iteration-endpoint, and time-endpoint probes all used this same procedure.

This is further analysis of the same formal experiment, whose fixed-readout test results were already known. It is not a new independent evaluation set. The detailed probe choices were recorded at this later stage, although linear probing itself was in the original proposal. Network hyperparameters, formal endpoints, and raw formal evidence remain unchanged.

## Results across three paired seeds

Entries below are mean test accuracy plus/minus sample standard deviation in percentage points. Sample SD is descriptive, not a confidence interval.

| Frozen network state | GD probe accuracy | Pullback probe accuracy | Paired difference, pullback minus GD |
| --- | ---: | ---: | ---: |
| Initial | 87.50% +/- 0.39 pp | 87.50% +/- 0.39 pp | 0.00 pp |
| 100 updates | 87.62% +/- 0.42 pp | 98.50% +/- 0.48 pp | +10.88 pp |
| 500 updates | 92.62% +/- 1.99 pp | 98.53% +/- 0.55 pp | +5.92 pp |
| 1,000 updates | 98.43% +/- 0.20 pp | 98.20% +/- 0.26 pp | -0.23 pp |
| 2,000 updates | 98.68% +/- 0.06 pp | 98.18% +/- 0.33 pp | -0.50 pp |
| Original two-second endpoint | 98.72% +/- 0.19 pp | 98.20% +/- 0.28 pp | -0.52 pp |

The paired initial probes agree exactly. At 100 and 500 updates, pullback has higher probe accuracy and lower probe loss in every pair. At 2,000 updates and at the two-second endpoint, GD has higher probe accuracy and lower probe loss in every pair. The 1,000-update accuracy comparison is mixed across seeds. These statements describe the measured pairs and do not establish population-wide superiority or statistical significance.

Both optimizers learn features that are substantially more linearly useful than the initial features: at 2,000 updates, probe accuracy rises by 11.18 percentage points for GD and 10.68 points for pullback relative to their paired initial baseline. The strong early pullback advantage is per update. The original two-second comparison does not show a terminal compute-budget advantage, and this probe analysis does not infer a ranking at every possible elapsed time.

## Fixed readout versus fitted probe

| Original endpoint | Method | Fixed-readout test accuracy | Probe test accuracy | Fixed-readout test loss | Probe test loss |
| --- | --- | ---: | ---: | ---: | ---: |
| 2,000 updates | GD | 98.87% | 98.68% | 0.03479 | 0.04157 |
| 2,000 updates | Pullback | 98.22% | 98.18% | 0.07985 | 0.07859 |
| Two seconds | GD | 98.82% | 98.72% | 0.06222 | 0.04099 |
| Two seconds | Pullback | 98.22% | 98.20% | 0.08264 | 0.07980 |

Refitting a regularized classifier does not guarantee higher test accuracy than the fixed readout. The probe optimizes a penalized training objective and selects its penalty by validation loss, not test accuracy. For example, the GD time-endpoint probe reduces mean test loss but has slightly lower mean accuracy. Logistic loss and accuracy measure different properties, and the test set does not determine the fitting rule.

The initial probe improves on the initial fixed readout (87.50% versus 72.08%) without any representation learning. This illustrates why a probe-versus-fixed-readout gain alone cannot be attributed to newly learned features. The learned-versus-initial probe comparison and the paired optimizer comparison are the more direct feature diagnostics here. No claim is made that the second coordinate alone explains a gain, or that the hidden-learning phenomenon in Arbel et al. (2023) has been reproduced.

## Numerical evidence and limits

All candidate gradient infinity norms meet the declared `1e-9` tolerance; the maximum is approximately `9.43e-10`, and the largest iteration count is 17. Five of the 36 selected probes choose the lowest regularization candidate, `1e-6`: the two identical formal1 initial probes and formal1 GD at 100, 500, and 1,000 updates. No original terminal probe selects a grid boundary. The finite grid is preserved without post-test expansion; its selections are not guaranteed global optima over all positive penalties.

The batch took 3.29 seconds including fitting, recording, evaluation, and plotting. Candidate fitting consumed 0.139 seconds. These are diagnostic costs and are separate from the original network-training timing.

The [independent stage-6 audit](../reports/stage6_audit.json) passed 1,814 checks across probes and the initial representation analysis. It independently reconstructed feature normalization, all candidate objective gradients and scores, selections, test predictions, pairing, summaries, and provenance. Prediction metrics and summaries matched exactly; the maximum independently computed objective/gradient discrepancy was `6.11e-16`.

All saved models, candidate scores, full-precision metrics, and figures are in the [probe archive](../reports/probes-v1/README.md). The [representation findings](representation_findings.md) relate the results to local spectra, point clouds, and displacement diagnostics. The final interpretation remains limited to this small architecture, this data distribution, this affine probe family, these frozen optimizer settings, and three paired seeds.
