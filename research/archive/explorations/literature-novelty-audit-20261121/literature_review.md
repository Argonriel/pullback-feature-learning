# Prior work and candidate novelty in selective symmetry averaging

Review date: 21 November 2026. Status: literature and novelty assessment; no new experiment or research-goal commitment.

## Assessment

The broad proposal of using selective test-time augmentation under a compute budget, with a statistical decision about whether to deploy it, is already substantially covered by prior work. Our completed experiments and conditional bounds are useful preliminary results, but they do not yet establish a substantial new method or theory contribution.

The strongest candidate worth examining is a **sharp, resource-aware certifiability boundary for selective averaging with imperfect input and posterior symmetry**. This remains a proposed contribution. It would need a result that is stronger than applying existing risk-control tools to paired losses, together with a precise comparison against the closest symmetry theorems. An intersection of several existing topics is not sufficient evidence of novelty.

## Closest prior work

The implications in the last column are our assessment of overlap, rather than claims made by the cited authors. TTA denotes test-time augmentation unless adaptation is explicitly stated.

| Work and primary source | What is already established | Consequence for our claim |
|---|---|---|
| Chen, Dobriban and Lee, **A Group-Theoretic Framework for Data Augmentation**, JMLR 2020. [Paper](https://jmlr.org/papers/v21/20-163.html) | Group-orbit averaging, variance reduction, and analysis extending beyond exact invariance. | Group averaging, Jensen arguments, and allowing approximate invariance are not new in themselves. |
| Lyle et al., **On the Benefits of Invariance in Neural Networks**, 2020. [Paper](https://arxiv.org/abs/2005.00178) | Distinguishes data augmentation from feature averaging and develops risk, gradient, and PAC-Bayes analyses. | A general claim that averaging exploits invariance or improves generalization is occupied. |
| Elesedy and Zaidi, **Provably Strict Generalisation Benefit for Equivariant Models**, ICML 2021. [Paper](https://proceedings.mlr.press/v139/elesedy21a.html) | Quantifies a strict generalization benefit in invariant/equivariant linear models under invariant distributions, using averaging-operator structure. | Positive benefit under exact symmetry is an established baseline, not the proposed discovery. |
| Kim, Kim and Kim, **Learning Loss for Test-Time Augmentation**, NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/2ba596643cbbbc20318224181fa46b28-Paper.pdf) | Predicts transformation-specific losses from an original input and selects favorable transformations before aggregating predictions. | Selecting views by predicted benefit, including identity among candidates, already has direct precedent. |
| Mocerino et al., **Adaptive Test-Time Augmentation for Low-Power CPU**, 2021 author preprint; published as AdapTTA. [Paper](https://arxiv.org/abs/2105.06183) | Sequentially averages predictions and stops using the margin between the largest two probabilities; includes embedded-CPU measurements. | Adaptive pass counts, uncertainty-based stopping, and measured serving costs are established. |
| Son and Kang, **Efficient improvement of classification accuracy via selective test-time augmentation**, Information Sciences 2023. [Publisher](https://doi.org/10.1016/j.ins.2023.119148), [institutional record](https://pure.skku.edu/en/publications/efficient-improvement-of-classification-accuracy-via-selective-te/) | The author abstract explicitly applies augmentation only to uncertain queries and reports a configuration augmenting 10 percent of samples. | Selecting only a fraction of inputs is a direct precedent. Only the primary abstract was accessible, so its exact ranking and quota mechanics are not asserted here. |
| Ozturk, Prabhushankar and AlRegib, **Intelligent Multi-View Test Time Augmentation**, ICIP 2024. [Paper](https://arxiv.org/html/2406.08593v1) | Uses uncertainty to decide whether to augment and the predicted class to choose an extra view, then averages the original and selected predictions. | Both routing decisions and two-view aggregation already exist. A stricter prospective evaluation would improve evidence without inventing the routing concept. |
| Chen et al., **When Do Histology Foundation Models Need Geometric Test-Time Augmentation?**, 16 November 2026 preprint. [Paper](https://arxiv.org/html/2611.21565v1) | Section 4.7 gates D4 group averaging using entropy from the original prediction. | Selective symmetry-group averaging is directly anticipated. The abstract and body disagree about the augmented fraction, so no efficiency percentage from this preprint is used as evidence here. |
| Shanmugam et al., **Better Aggregation in Test-Time Augmentation**, ICCV 2021. [Paper](https://arxiv.org/html/2011.11156v2) | Studies corrected and corrupted predictions and learns augmentation-dependent and class-dependent aggregation weights. | Learning weights or acknowledging harmful augmentations is not sufficient novelty. |
| Conde et al., **Approaching Test Time Augmentation in the Context of Uncertainty Calibration for Deep Neural Networks**, 2023 preprint, revised 2024. [Paper](https://arxiv.org/html/2304.05104v2) | M/V-ATTA adjusts the mixture with the original prediction to preserve its top-1 decision while improving calibration empirically. | Accuracy preservation is already possible by construction. This differs from certifying a beneficial policy that may change decisions. |
| Shanmugam et al., **Test-time Augmentation Improves Efficiency in Conformal Prediction**, CVPR 2025. [Paper](https://arxiv.org/html/2505.22764v1) | Learns augmentation weights on a separate split before conformal calibration. Its expanded policy includes transformations that need not preserve labels. | TTA plus independent calibration plus potentially label-changing transformations is already present. The guarantee concerns marginal prediction-set coverage under exchangeability, not improvement in mean classification loss. |
| Lawrence et al., **To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking**, ICLR 2026. [Paper](https://arxiv.org/html/2510.01349v2), [proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/106509d2c9263224610fe4145fbed3e8-Abstract-Conference.html) | Distinguishes input-distribution symmetry from target symmetry; gives regression benefit/harm results and a counterexample with arbitrarily harmful test-time symmetrization. Appendix D.2.4 uses a learned classifier for selective augmentation. | Our basic asymmetry warning and a detect-asymmetry-then-select proposal are already anticipated. The closest remaining difference is a quantitative, same-distribution certificate for a fixed nonlinear classifier with approximate posterior symmetry. |
| Beck and Ochs, **Symmetries in PAC-Bayesian Learning**, ICML 2026. [Paper](https://proceedings.mlr.press/v306/beck26a.html) | Extends symmetry analysis to non-invariant distributions. Its orbit average uses the conditional distribution of transformations given the orbit representative; risk contraction assumes exact target equivariance and a suitable convex loss. | Density-aware averaging itself is not new. Our unweighted two-view rule has a different target and needs to justify its advantage over this alternative. |
| Angelopoulos et al., **Learn then Test**, 2021 preprint, revised 2022. [Paper](https://arxiv.org/html/2110.01052) | Uses valid hypothesis tests and multiple-testing control to select configurations with finite-sample risk guarantees. | A held-out accept-or-decline test for a fixed family of TTA policies can be built from existing machinery. |
| Laufer-Goldshtein et al., **Efficiently Controlling Multiple Risks with Pareto Testing**, ICLR 2023. [Paper](https://arxiv.org/html/2210.07913) | Combines multi-objective search and risk-controlled configuration selection, including adaptive computation, accuracy constraints, and inference cost. | Combining compute allocation, abstention from unsupported configurations, and risk control is already established. |
| Einbinder, Ringel and Romano, **Semi-Supervised Risk Control via Prediction-Powered Inference**, 2024 preprint, revised July 2025. [Paper](https://arxiv.org/html/2412.11174v2) | Uses unlabeled data and labeled corrections to improve calibration efficiency when the surrogate is informative, retaining validity with suitable finite-sample upper bounds. | Claiming that unlabeled data reduce the labeled cost of certification is too broad. This is an essential comparison for any proposed label-efficient method. |
| Jhawar and Wang, **When Is Test-Time Adaptation Identifiable from Unlabeled Evidence?**, September 2026 preprint. [Paper](https://arxiv.org/html/2609.11235v1) | Studies whether unlabeled evidence identifies the best adaptation action, with indistinguishable-world limits and a finite-batch Gaussian boundary. | Generic identifiability, impossibility, or help-versus-harm boundary framing is also occupied. Its stateful adaptation setting differs from our frozen-model averaging problem. |
| Tahmasebi and Weber, **Test-Time Symmetrization: Scaling Laws for Model-Agnostic Learning under Structure**, 18 November 2026 preprint. [Paper](https://arxiv.org/html/2611.24032v1) | Gives uniform sparse approximation to a finite-group averaging operator, with logarithmic group-size dependence and a matching lower-bound construction. Its risk corollary compares with full symmetrization and averages over uniform starting representations. | Frozen-model symmetrization, compute scaling, and information lower bounds are active prior art. Its result does not establish that full averaging improves the raw predictor under an arbitrary nonuniform deployment distribution. |

The September and November 2026 preprints above are treated as prior disclosures rather than established peer-reviewed conclusions. Publication status is not itself a reason to ignore overlap.

## What our completed work currently adds

The latest local theory concerns a fixed nonlinear classifier, multiclass log loss, a two-element transformation group, and an exactly budgeted batch selector. It separates the gain from averaging from penalties for unequal input frequencies and unequal conditional label distributions. It propagates independent coefficient uncertainty and a supplied residual envelope into a conditional certificate.

The most specific existing technical result is the joint worst-case posterior penalty

    h_epsilon(r) = [abs(r) + (1-abs(r))*epsilon] * S/2 - r*t,

and its interval envelope

    H_epsilon([l,u]) = max(h_epsilon(l), h_epsilon(u)).

Here r measures the signed density imbalance within an input pair, epsilon is a posterior-total-variation bound in [0,1], S is the range of the antisymmetric logits, and t is the centered odd log-partition difference. The formula retains their interaction instead of independently maximizing separate penalties. Coordinate-wise invariance of the selection indicator permits the orbit identity to remain valid with dependent quota selection.

These are potentially useful specializations. Their novelty is not settled by this review. Pointwise sharpness of a posterior bound, and endpoint maximization over an interval, do not establish a globally sharp uncertainty set when coefficients, moments, or residual restrictions couple different inputs. They also do not establish a minimax sample-complexity theorem or an optimal allocation policy.

The current positive certificates depend on explicit synthetic density structure, a supplied residual bound, and, for the labeled extension, a finite-cell posterior model. Independently maximizing posterior choices at each input can conflict with this within-cell restriction. The independent verification and labeling burden has not been shown to beat a direct paired-loss audit. The natural-image distillation result failed external confirmation. Those limitations prevent treating the completed package as an already validated general solution.

Local evidence: [latest archived theory](C:/Users/liwei/Documents/Research/pullback-feature-learning/explorations/frontier-replication-residuals-20261121/structured/THEORY.md), [scope and reproduction record](C:/Users/liwei/Documents/Research/pullback-feature-learning/explorations/frontier-replication-residuals-20261121/README.md).

## The proposed innovation to examine

**Candidate claim:** For a declared class of imperfect symmetries and frozen nonlinear predictors, characterize how input asymmetry, posterior mismatch, selection budget, and finite evidence jointly determine whether a positive deployment gain can be certified. Establish a tight boundary or a demonstrable statistical advantage over existing generic risk audits under the same declared information and resource conditions.

The central contribution would have to be the additional structural result. A theorem merely saying that an independent confidence bound is valid is insufficient. Two substantive forms are worth considering:

1. **A sharp structural boundary.** Derive a useful bound for the actual deployment distribution, permit both kinds of symmetry error, and give compatible distributions or a lower bound showing when the result cannot be improved within the stated class. A local worst-case calculation alone does not meet this requirement.
2. **A quantified evidence advantage.** Show precisely when symmetry information reduces the labeled evidence or model queries needed to certify a specified gain, compared with direct paired evaluation and semi-supervised risk control. Any structural assumptions and their acquisition costs must be shared, charged, or explicitly distinguished in the comparison. Improvement in a constant or an example must not be advertised as a universal rate improvement.

The desired scientific distinction is between approximating an orbit average and establishing that this average, selectively used, is better than the original predictor on the actual population. The recent scaling-law work treats the former and retains the full-average risk as an unresolved model/task term. Our candidate would address the latter under explicit imperfect-symmetry assumptions. This is a precise difference in target; it is not yet proof of an unoccupied or important research gap.

## Comparisons that determine whether the claim survives

A generic paired audit directly measures the loss difference between a frozen candidate and the original predictor on the same labeled examples. It needs no semantic label-preservation assumption to evaluate those two predictions against the original label. With bounded losses or an independently justified range, it fits existing risk-testing frameworks. Their sampling assumptions still apply, typically independent calibration data from the deployment population; guarantees do not automatically survive a new target shift. Log loss is unbounded in general, so its range or tail assumptions must be specified. For quota policies, independent batches can be the audit units; examples sharing a batch must not automatically be treated as independent policy outcomes.

If the proposed certificate can be reproduced by that audit, by Learn then Test/Pareto Testing, or by semi-supervised risk control with a symmetry-derived surrogate, its validity alone is an application. A substantial contribution would require an additional theorem, a proved efficiency difference, or a new decision capability under precisely matched assumptions. The existing synthetic examples only show that the current sufficient bound can sometimes be positive; they do not establish this separation.

The density-weighted orbit average of Beck and Ochs is also a necessary conceptual comparator. Restricting attention to an unweighted average without explaining that choice could create an artificial difficulty. Extending from one flip to a larger group, adding a calibration layer, varying the quota, or repeating experiments on another dataset are not by themselves the sought innovation.

## Claims to withdraw or keep limited

- Do not claim the first adaptive, selective, budgeted, or safe TTA method.
- Do not claim that distributional asymmetry can harm averaging as a new general phenomenon.
- Do not claim that finite-sample calibration, abstention, or use of unlabeled data is itself novel.
- Do not equate prediction-set coverage, unchanged top-1 decisions, lower expected loss, and absence of individual harmful edits.
- Do not claim that a classifier failing to detect asymmetry supplies an upper bound on every unobserved symmetry defect.
- Do not promote externally unconfirmed distillation to a successful method component.
- Retain the exact conditional scope of the existing bound and its synthetic evidence.

## Decision before further experimentation

The earlier broad research goal should remain provisional. The next decision is whether the proposed structural claim can be stated as a genuinely additional theorem with an informative comparison to the closest results. If it reduces to a standard concentration argument around a known augmentation policy, the current direction should be classified as an implementation or empirical study and reconsidered before another experiment sweep.

This targeted review establishes substantial overlap, not exhaustive priority clearance. It uses primary papers available by the review date, with particular attention to 2025 and 2026 developments. No experiments were started, historical artifacts changed, or research-success claim made during this review.
