# Literature consulted before task design

Search date: 21 November 2026. The search targeted group-invariant learning, orbit-separating invariants, group quotients, and adaptive augmentation. The sources below are primary research publications. This is a focused design review, not an exhaustive novelty search or evidence that the proposed selector is novel.

**Chen, Dobriban, and Lee (2020), A Group-Theoretic Framework for Data Augmentation.** Their framework expresses augmentation through averaging over transformation orbits and distinguishes exact from approximate distributional invariance. Their variance-reduction analysis motivates stating the input-law and label-symmetry assumptions separately. Our task therefore samples phase uniformly independently of the quotient coordinates and defines the label posterior entirely on those coordinates. Their training-augmentation results do not automatically prove a benefit for a learned, selective test-time policy. [JMLR paper](https://jmlr.org/papers/v21/20-163.html).

**Dym and Gortler (2025 journal issue; version of record 2024), Low-Dimensional Invariant Embeddings for Universal Geometric Learning.** They study invariant maps that distinguish different group orbits, and explain their relevance to universal invariant/equivariant learning. This supports using an orbit-separating quotient rather than assuming that an invariant scalar such as total norm retains all relevant information. Their general embedding results are not needed to prove completeness of our simple two-coordinate quotient; that proof is given directly below. [Published paper](https://link.springer.com/article/10.1007/s10208-024-09641-2), [author preprint, v3](https://arxiv.org/abs/2205.02956).

**Hounie, Chamon, and Ribeiro (2023), Automatic Data Augmentation via Invariance-Constrained Learning.** They formulate adaptive augmentation with explicit invariance constraints and discuss how indiscriminate augmentation may introduce bias. This motivates retaining strong symmetric controls, checking the actual task action, and not equating a selected-view loss improvement with an invariance guarantee. Their method concerns adaptive training augmentation and is not presented here as an analysis of our exact-quota inference rule. [ICML/PMLR paper](https://proceedings.mlr.press/v202/hounie23a.html).

## Design deductions, not claims from those papers

Use the phase-coupled support

`x(r1,r2,phi)=(r1 cos(phi), r1 sin(phi), r2 cos(2phi), r2 sin(2phi))`,

with both amplitudes positive and variable. The action is `R_theta direct-sum R_(2theta)`. On this support, the two pair norms `(r1,r2)` separate orbits: equality of the pair norms is necessary, and the phase difference recovered from the first pair supplies a group element mapping either point to the other. The support has dimension three and the quotient has dimension two.

This statement is deliberately restricted to the phase-coupled support. A general point in ambient R4 also has a relative-phase invariant; the two pair norms alone would not separate its orbits. We do not claim otherwise.

Total norm only gives `sqrt(r1^2+r2^2)` and therefore identifies many distinct orbits. A posterior that varies along a constant-total-norm arc tests the scientific limitation left unresolved by the preceding scalar-quotient tasks. Every calibration and evaluation API must retain both quotient coordinates. Neither replacing them by total norm nor checking a single marginal moment would establish conditional balance on this quotient.
