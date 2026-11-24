# Primary literature and the novelty boundary

Search date: 21 November 2026. The review was completed before this round's
new baseline development evaluation. It is a targeted primary-source review,
not an exhaustive novelty search. Test-time augmentation (view aggregation)
and test-time adaptation (parameter/state changes) are related but distinct.

## Group averaging and Rao-Blackwell reasoning

Chen, Dobriban, and Lee, **A Group-Theoretic Framework for Data Augmentation**,
JMLR 21(245), 2020, formalize orbit averaging under exact or approximate
distribution invariance and prove variance-reduction results. Section4.1,
including Lemma1 and its Rao-Blackwell discussion, is the relevant conceptual
precedent. Their broad framework concerns augmented estimation and ERM;
our partial exact-quota deployment mask needs its own invariance argument.
Group averaging, Jensen convexity, and the general variance-reduction idea
are not new contributions here.
[Paper](https://jmlr.org/papers/v21/20-163.html),
[full text](https://jmlr.org/papers/volume21/20-163/20-163.pdf).

## Allocation and aggregation already have strong precedents

Kim, Kim, and Kim, **Learning Loss for Test-Time Augmentation**, NeurIPS2020,
use an auxiliary loss-prediction model to select transformations for an
individual input. Sections3.3–3.4 describe prediction and supervised training
of transformation losses. An auxiliary compact network avoids evaluating
the target model on every candidate transformation before deciding.
Input-dependent compute allocation is therefore not new. The current method
instead uses a label-free Jensen table and a specified symmetry assumption;
comparisons must acknowledge the different supervision and compute budgets.
[Paper](https://papers.neurips.cc/paper_files/paper/2020/file/2ba596643cbbbc20318224181fa46b28-Paper.pdf).

Lyzhov et al., **Greedy Policy Search: A Simple Baseline for Learnable
Test-Time Augmentation**, UAI2020, learn test-time augmentation policies and
emphasize an effective simple baseline. This supports comparing to a tuned,
matched-budget symmetric control instead of only a coarse historical table.
It does not establish the safety of our particular exact-quota policy.
[Paper](https://proceedings.mlr.press/v124/lyzhov20a.html).

Shanmugam, Blalock, Balakrishnan, and Guttag, **Better Aggregation in Test-Time
Augmentation**, ICCV2021, learn aggregation weights and show that augmentation
can both correct and corrupt individual predictions. Section4 describes
learned aggregation; Section5 separates its fitting, validation, and test
data, and Section5.4 discusses reducing computation with zero weights.
Their supervised aggregation setting differs from our frozen-logit,
label-free invariant allocation. Neither naive averaging nor an arbitrary
transformation should be presented as universally optimal.
[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf).

## Safety is not a new general objective

Laroche, Trichelair, and Tachet des Combes, **Safe Policy Improvement with
Baseline Bootstrapping**, ICML2019, constrain policy changes where batch-RL
evidence is weak and derive baseline-relative guarantees under their
assumptions. This is a useful conceptual precedent for uncertainty-aware
improvement. Its MDP and behavior-policy assumptions do not supply a theorem
for classification-view allocation; our loss identity and independent
bounded observations must be justified separately.
[Paper](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf).

## Direct overlap with unlabeled action identifiability

Jhawar and Wang, **When Is Test-Time Adaptation Identifiable from Unlabeled
Evidence?**, arXiv:2609.11235v1, 10 September2026, study fixed observation
channels and compatible deployment worlds. Definition1 and Theorem2 say a
common optimal action must exist across compatible worlds. Theorem1 and
Corollary1 give exact and approximate two-world impossibility statements.
Theorem3 separately analyzes a finite-batch KEEP/RECENTER threshold in a
Gaussian shift model. AppendicesA.1–A.5 contain the relevant proofs. The
authors explicitly place general action identification and testing limits
within earlier decision theory.

Our earlier same-input-law, different-hidden-posterior obstruction is a
special case of that information limitation, not a new general impossibility
theorem. The identity involving a conditional imbalance and its numerical
verification are specialized constructive sufficient conditions. Exact
half-turn invariance is another sufficient condition that removes the
hidden-posterior term; it does not overcome arbitrary compatible-world
ambiguity. Their adaptive model interventions and our frozen-model view
allocation also differ. No first label-free guarantee or first identifiability
boundary claim is warranted.
[Full text](https://arxiv.org/html/2609.11235v1),
[record](https://arxiv.org/abs/2609.11235).

## TTA with statistical guarantees also predates this study

Shanmugam, Lu, Sankaranarayanan, and Guttag, **Test-time Augmentation Improves
Efficiency in Conformal Prediction**, CVPR2025, learn augmentation policies
to reduce conformal classification set size. The method separates labeled
policy-training data from labeled conformal-calibration data to preserve
coverage. Its guarantee concerns prediction-set coverage and efficiency,
not a posterior-uniform decrease in logistic risk. This is relevant prior
work on augmentation plus statistical guarantees, with a different
supervision regime and target quantity.
[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Shanmugam_Test-time_Augmentation_Improves_Efficiency_in_Conformal_Prediction_CVPR_2025_paper.html),
[arXiv record](https://arxiv.org/abs/2505.22764).

## What the present study can establish

The defensible contribution is a carefully delimited combination: exact
partial-allocation invariance, an effective input-only Jensen surrogate,
explicit calibration and deployment costs, cheaper independent verification,
and frozen comparisons on a genuinely two-dimensional orbit quotient.
Fresh evidence can establish an advantage over the specified strong controls.
It cannot establish that asymmetric policies generally dominate symmetric
ones, that two invariant coordinates cover arbitrary group geometry, or that
small-model forward-count savings imply a GPU latency improvement.
