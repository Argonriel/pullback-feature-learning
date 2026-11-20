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
| 6a. Frozen-feature probes | Compare initial and learned features using common normalization and regularization-selection procedures | Complete. All 288 candidates converged and 36 validation selections preceded test evaluation. The [probe findings](probe_findings.md) distinguish early feature gains from the terminal comparison. |
| 6b. Feature geometry and spectra | Show point clouds and relate singular values and gradient alignment to learning | Complete. The [representation findings](representation_findings.md) cover 24 training-feature states and local spectral summaries, with explicit limits on causal interpretation. |
| 6c. Cumulative displacement | Compare progress at matched cumulative RMS feature displacement as a secondary diagnostic | Complete with refinement. Sparse checkpoints were insufficient at small budgets; [exact replay](displacement_refinement_protocol.md) now brackets the same ceilings by consecutive updates and records all residual gaps. |
| 7. Final report | Combine derivation, comparison, costs, and representation analysis | Ready to write. The experiments and core diagnostic analyses are complete; the final narrative must distinguish established measurements, local mathematical predictions, and plausible mechanisms. |
| Optional distribution shift | Add a nuisance feature only after core work is complete | Not started. This is optional and requires a separate protocol and evaluation data. |

## Additions and preserved boundaries

The original proposal did not specify a 100,000-update endpoint. That experiment was added to examine whether the selected optimizers simply needed longer training. It used the original development splits, retained the settings selected at 2,000 updates, and did not generate test or formal-stage data. Its validation results are diagnostic evidence, not a replacement for the frozen formal comparison.

Frozen-feature evaluation was already a core proposal commitment. The implementation now makes its details concrete: archived initial, early, intermediate, fixed-iteration, and fixed-time states; training-only feature normalization; a common regularization grid; and validation-only selection. All network weights stay fixed. The additional early checkpoints describe the learning trajectory and are not candidates for replacing the formal endpoint with a favorable test score.

The formal fixed-readout test results had already been examined when stage 6 was specified. The probe protocol was recorded before probe test evaluation. Its output is further analysis of the archived formal study, not an independent replication on unseen data. A change to network architecture, optimizer settings, or model-selection rules would require a separately documented experiment; renewed final-performance claims would need an appropriate unused evaluation set.

The first displacement analysis exposed an additional recording limitation: checkpoints spaced 100 updates apart could leave very different realized movement under a shared ceiling. A separate refinement protocol was then committed before replay and new endpoint evaluation. It retains the same training-derived ceilings and unchanged optimizer trajectories, verifies all 126 shared archived parameter states and cumulative distances exactly, and uses consecutive-update crossings. This improves measurement resolution rather than selecting a new model or changing an optimizer.

## Immediate sequence

1. Assemble the final report using the derivation, frozen formal comparison, supplementary training diagnostic, probes, local spectra, and refined displacement results.
2. Present the measured distinction: pullback learns useful features earlier per update and uses feature movement more effectively under the chosen budgets, while GD retains better terminal probe performance and lower required update cost in this implementation.
3. Check figure captions, definitions, uncertainty descriptions, bibliography, and reproducibility links against the archived evidence. Keep local geometric explanations separate from untested causal claims.
4. Consider distribution shift only as an optional subsequent study; it is not needed to finish the present core project.

Stage 6 did not reveal a hidden terminal probe advantage for pullback. It did identify an advantage under closely matched feature-displacement budgets, alongside the early per-update probe advantage. These qualified findings directly answer the original research question while preserving the earlier limitations.
