# Project plan and proposal alignment

Status recorded on November 20, 2026. The [original proposal](proposal.tex) is preserved as submitted. This document records execution progress and subsequent additions without rewriting the original commitments.

## Original research question

The proposal asks:

> Does a representation-space optimization rule produce more task-effective feature motion than ordinary parameter gradient descent?

It specifies one small residual network, noisy two-moons data, a Euclidean representation pullback with positive damping, and a GD comparison. The intended explanation connects the Jacobian spectrum to learning behavior and includes the cost of constructing Jacobians and solving linear systems.

Its outcome statement is explicit:

> The core deliverables are the metric-to-update derivation, local spectral analysis, and a reproducible numerical comparison. Both improvements and limitations, such as iteration savings offset by computational cost, will be informative.

The hypothesis motivates the comparison; a positive performance result for pullback is not a completion requirement. Likewise, related papers motivate diagnostic questions but do not guarantee that this model reproduces their findings.

## Commitments and current status

| Stage or component | Proposal commitment | Current status and evidence |
| --- | --- | --- |
| 1. Specification and repository | One controlled model, metric, dataset, and objective | Complete. The 80-parameter residual model and fixed readout are recorded in the [protocol](protocol.md) and [mathematics](mathematics.md). |
| 2. Derivation and related work | Derive the pullback update and local spectral/descent properties | Complete. The mathematical note distinguishes the representation metric from Fisher/GGN and Wasserstein constructions. The [literature review](literature.md) now covers eleven primary papers. |
| 3. Numerical checks and pilot | Verify small-step feature motion, then establish feasibility | Complete. Independent gradient/Jacobian and linear-system checks precede the [paired CPU pilot](pilot_findings.md). |
| 4. Validation-only tuning | Separate learning rates, positive fixed damping, equal tuning budgets | Complete. Twelve candidates per method and three development seeds produced 72 runs under the [tuning protocol](tuning_protocol.md). |
| 5. Formal comparison | At least three paired seeds; performance by updates and time | Complete. The [formal study](formal_findings.md) preserves separate 2,000-update and two-second endpoints, per-seed records, uncertainty descriptions, and compute costs. |
| Supplementary long training | Added after the formal results to examine the insufficient-training hypothesis | Complete. Six fixed-parameter development runs reached 100,000 updates. The [diagnostic](long_training_findings.md) preserves the original formal results and supports no further automatic extension. |
| 6a. Frozen-feature probes | Compare initial and learned features using common normalization and regularization-selection procedures | In progress. The [probe protocol](probe_protocol.md) specifies fitting on training features and selecting regularization on validation data before test evaluation. |
| 6b. Feature geometry and spectra | Show point clouds and relate singular values and gradient alignment to learning | In progress. The [representation-analysis protocol](representation_analysis_protocol.md) uses archived states and training-feature visualizations. Spectra were recorded during the formal evaluation. |
| 6c. Cumulative displacement | Compare progress at matched cumulative RMS feature displacement as a secondary diagnostic | In progress. Training-derived budgets will be bracketed using archived states. Realized displacement gaps must be reported; coarse checkpoints do not imply exact matching. |
| 7. Final report | Combine derivation, comparison, costs, and representation analysis | Pending completion and review of stage 6. The final narrative must distinguish established measurements, local mathematical predictions, and plausible mechanisms. |
| Optional distribution shift | Add a nuisance feature only after core work is complete | Not started. This is optional and requires a separate protocol and evaluation data. |

## Additions and preserved boundaries

The original proposal did not specify a 100,000-update endpoint. That experiment was added to examine whether the selected optimizers simply needed longer training. It used the original development splits, retained the settings selected at 2,000 updates, and did not generate test or formal-stage data. Its validation results are diagnostic evidence, not a replacement for the frozen formal comparison.

Frozen-feature evaluation was already a core proposal commitment. The implementation now makes its details concrete: archived initial, early, intermediate, fixed-iteration, and fixed-time states; training-only feature normalization; a common regularization grid; and validation-only selection. All network weights stay fixed. The additional early checkpoints describe the learning trajectory and are not candidates for replacing the formal endpoint with a favorable test score.

The formal fixed-readout test results have already been examined. The probe protocol must be recorded before probe test evaluation, and the output should be described as further analysis of the archived formal study, not an independent replication on unseen data. A change to network architecture, optimizer settings, or model-selection rules would require a separately documented experiment; renewed final-performance claims would need an appropriate unused evaluation set.

## Immediate sequence

1. Record and verify the probe and representation-analysis procedures before running the new analyses.
2. Fit all probe candidates on training features, select regularization on validation features, and save those choices before evaluating test features.
3. Generate the feature clouds, spectral summaries, and displacement-budget diagnostics from the archived experiment.
4. Independently recompute reported metrics, inspect the figures, and interpret fixed-readout and probe results together.
5. Write the final report around the measured findings and their limitations. Consider distribution shift only after these core deliverables are complete.

The next stage can reveal useful features that the fixed readout did not exploit, or it can confirm that the current pullback settings offer no feature-quality advantage. Both outcomes answer the original research question within this experiment's limited scope.
