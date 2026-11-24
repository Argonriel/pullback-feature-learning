# Literature-informed scope of the symmetry allocation study

Primary-source searches were performed on 21 November 2026, before freezing
the new baseline and confirmation protocol. Search topics were orbit
averaging and Rao-Blackwellization, learned test-time augmentation, safe
policy improvement, unlabeled action identification, and orbit-separating
invariants. This is a focused design review, not an exhaustive novelty audit.
The sources below establish precedents; our study-specific deductions are
marked separately.

Test-time **augmentation** combines predictions on transformed inputs while
keeping the trained model fixed. Test-time **adaptation** changes model
parameters or state using deployment data. This study performs the former.
Adaptation papers inform information limits and experimental design, but
their particular guarantees do not transfer automatically.

## Primary sources and publication status

| Source | Status | Relevance and claim boundary |
|---|---|---|
| Chen, Dobriban, Lee, *A Group-Theoretic Framework for Data Augmentation* | JMLR, 2020 | Orbit averaging and variance reduction are established. The Rao-Blackwell connection is explicit in Section 4.1. A partial mask still requires a separate invariance argument. [Paper](https://jmlr.org/papers/v21/20-163.html) |
| Kim, Kim, Kim, *Learning Loss for Test-Time Augmentation* | NeurIPS, 2020 | Instance-specific transformation selection via predicted losses predates this study. Sections 3.3–3.4 use an auxiliary predictor and supervised loss targets. [Paper](https://papers.neurips.cc/paper_files/paper/2020/file/2ba596643cbbbc20318224181fa46b28-Paper.pdf) |
| Lyzhov et al., *Greedy Policy Search* | UAI, 2020 | Learned test-time augmentation and competitive simple policies are established baselines. [Paper](https://proceedings.mlr.press/v124/lyzhov20a.html) |
| Shanmugam et al., *Better Aggregation in Test-Time Augmentation* | ICCV, 2021 | Learned aggregation and its computational tradeoffs precede the current work; its supervised regime differs. [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf) |
| Laroche, Trichelair, Tachet des Combes, *Safe Policy Improvement with Baseline Bootstrapping* | ICML, 2019 | Baseline-relative safety under uncertainty is established in batch reinforcement learning. Its MDP assumptions do not prove our classification identity. [Paper](https://proceedings.mlr.press/v97/laroche19a.html) |
| Jhawar, Wang, *When Is Test-Time Adaptation Identifiable from Unlabeled Evidence?* | arXiv v1, 10 September 2026; preprint | No first generic obstruction claim: see Theorems 1–2. [Full text](https://arxiv.org/html/2609.11235v1) |
| Shanmugam, Lu, Sankaranarayanan, Guttag, *Test-time Augmentation Improves Efficiency in Conformal Prediction* | CVPR, 2025 | Augmentation with statistical guarantees predates this study. Its target is prediction-set efficiency with coverage, not our logistic-risk quantity. [Published paper](https://openaccess.thecvf.com/content/CVPR2025/html/Shanmugam_Test-time_Augmentation_Improves_Efficiency_in_Conformal_Prediction_CVPR_2025_paper.html), [author record](https://arxiv.org/abs/2505.22764) |
| Dym, Gortler, *Low-Dimensional Invariant Embeddings for Universal Geometric Learning* | Foundations of Computational Mathematics, online 2024; volume 25, 2025 | Separating group orbits is stronger than providing any invariant statistic. Their general embedding results are not needed for our elementary support-specific proof. [Published paper](https://link.springer.com/article/10.1007/s10208-024-09641-2) |
| Hounie, Chamon, Ribeiro, *Automatic Data Augmentation via Invariance-Constrained Learning* | ICML, 2023 | Adaptive training augmentation with invariance constraints is prior art. It motivates distinguishing actual task symmetry from arbitrary transformations. [Paper](https://proceedings.mlr.press/v202/hounie23a.html) |

Fuller source-specific notes and theorem locations appear in
`method/LITERATURE.md` and `task2d/LITERATURE.md`. A preprint is cited as a
preprint rather than as an accepted conference or journal result. We do not
infer novelty from the absence of another method in this small source set.

## Implication for objective 1: a strong symmetric comparison

The new method is a transparent interpolation surrogate for the Jensen gain
of a fixed half-turn average. General group averaging and adaptive view
selection are not its novelty. A fair test must compare against a competitive
symmetric histogram chosen using development data, a nearest-grid control,
and an exact-Jensen upper envelope. Every method's calibration queries count.
The oracle queries both views on every evaluation input, so it cannot be
described as a 1.25-forward method. Equal forward budgets do not imply equal
independent calibration samples or equal wall-clock costs.

Our own identity states that an exact-quota mask unchanged by each
individual input's half-turn has expected gain equal to its selected Jensen
gap, under an invariant input law and posterior. This is a specialized
application of established symmetry and convexity reasoning. The narrow
empirical question is whether the fixed interpolated table captures useful
extra gain over the frozen strong controls on new trained models.

## Implication for objective 2: a genuine two-coordinate quotient

Our new support is

`(r1 cos(phi), r1 sin(phi), r2 cos(2 phi), r2 sin(2 phi))`.

Both amplitudes vary independently and are positive. On this phase-coupled
support, the pair norms `(r1,r2)` separate SO(2) orbits: equal amplitudes and
the first pair's phase difference determine a group element between points.
The support has dimension three and its quotient has dimension two. Total
norm discards information because it fixes only `r1^2+r2^2`; the posterior
is deliberately not constant along those level curves.

This is a deduction for the task, not a claim from the cited embedding paper.
On unrestricted ambient R4, relative phase is an additional invariant; two
pair norms would not separate all orbits. The group representation
`R_theta direct-sum R_(2 theta)` is linear in the input for fixed theta.
The parameterization of the support is nonlinear. Moving from one to two
quotient coordinates is genuine but does not cover arbitrary-dimensional
quotients, arbitrary groups, or unconstrained real-world inputs.

## Implication for objective 3: lower verification cost

For the individually invariant policy family, selected Jensen gaps are
observable without labels. Verification can query both views only on the
selected-set union; deployment separately still predicts every input.
Independent batches supply the statistical units because quota masks couple
items inside a batch. A signed comparison to another safe policy needs its
own confidence bound, and a clipped positive observation only directly
lower-bounds absolute gain. These are our deductions, justified in
`method/THEORY.md` and the certification branch.

This is a cheaper verification path for a restricted symmetric policy family.
It is not an acceleration of the previous orientation-sensitive selector's
conditional-imbalance certificate. It also does not solve arbitrary hidden
posterior ambiguity. The recent action-identification framework is consistent
with this restriction: the symmetry assumption narrows the compatible
worlds. No first label-free guarantee, general identifiability theorem, or
universal safe-improvement principle is claimed.

The admissible conclusion is therefore a controlled comparison and explicit
cost/uncertainty account for one fixed transformation and exact quota. A
logistic-risk certificate does not imply lower classification error, and
offline verification savings do not establish serving-time or GPU speedups.
