# Primary literature and novelty audit: discrete endpoint effects

Date: 2026-11-21. The initial primary-source screen preceded `PROTOCOL.md` and every new numerical check. This document records the decisive overlaps and the narrower remaining question. It is a bounded search, not a claim of exhaustive priority clearance.

## Directly occupied claims

- [Gunasekar, Lee, Soudry and Srebro, ICML 2018, section 2.3 and Proposition 3a](https://proceedings.mlr.press/v80/gunasekar18a/gunasekar18a.pdf) explicitly distinguish finite-step natural gradient from mirror descent and show that finite natural-gradient steps can change the limiting Bregman-selected solution. Thus loss of the continuous implicit bias under discretization is not new. Their example is a Hessian-metric natural-gradient update in the primal variable; it is not the simultaneous factor congruence with the present additive parameter damping.

- [Kunin et al., Neural Mechanics, ICLR 2021](https://arxiv.org/abs/2012.04728) treats finite-learning-rate breaking of gradient-flow conservation laws. The published original was also located at [the authors' institutional repository](https://ntt-research.com/wp-content/uploads/2023/02/NEURAL-MECHANICS-SYMMETRY-AND-BROKEN-CONSERVATION-LAWS-IN-DEEP-LEARNING-DYNAMICS-1.pdf). Generic broken symmetry, modified flows, and accumulated invariant error are occupied territory.

- [Barrett and Dherin, ICLR 2021](https://arxiv.org/abs/2009.11162) uses backward-error analysis to identify finite-step implicit regularization. A modified-equation explanation alone would not distinguish this project.

- [Nacson et al., ICML 2022](https://proceedings.mlr.press/v162/nacson22a/nacson22a.pdf) studies how step size changes implicit bias in diagonal linear networks through dynamical stability. [Even et al., NeurIPS 2023](https://papers.nips.cc/paper/2023/file/5da6ce80e97671b70c01a2e703b868b3-Paper-Conference.pdf) studies macroscopic steps and trajectory-dependent regularizers in diagonal networks. A general assertion that discrete factor training changes implicit bias is therefore especially weak.

- [Marcotte, Gribonval and Peyre, ICML 2025, section 5 and Proposition 5.1](https://arxiv.org/html/2506.06194v1) gives a Taylor/Hessian-based bound on accumulated conservation error under SGD. The present cancellation is itself a zero of the corresponding invariant curvature at the initial state, so the broad proof mechanism belongs to this established framework. Their proposition does not identify the present damping value or limiting endpoint coefficient.

- [Bartolomaeus and Rauhut, October 2026, Lemma 3.3 and section 3](https://arxiv.org/html/2610.39717v1) represents arbitrary-depth diagonal-network factor steps as perturbed mirror descent and characterizes their limiting bias. The depth-two perturbation already contains a logarithm-minus-linear term. Our exact central invariant increment shares this elementary algebra. The inspected source does not use the additive damped-pullback resolvent `(K_X+lambda I)^-1`, so its stated formulas do not directly give the `lambda=2s` cancellation. Nonetheless, a new trajectory-dependent-regularizer narrative would merely repackage this prior line.

- [The Map Behind the Flow, July 2026, Proposition 14 and section 5](https://arxiv.org/html/2607.04993v1) gives exact factor-imbalance changes, step-induced balancing, and near-edge endpoint selection in solvable models. [Elosegui Borras and Arvanitidis, November 2026](https://arxiv.org/html/2611.06718v1) examines conservation-law breaking through central-flow analysis at the edge of stability. The current check deliberately uses a strict linear stability margin; it is not a claim about edge-of-stability behavior or a new finite-step balancing principle.

## Methods that were rejected without experiments

- [Quispel and Turner, 1996](https://doi.org/10.1088/0305-4470/29/13/006) gives a general framework for direct first-integral-preserving methods.
- [McLachlan, Quispel and Robidoux, 1998](https://doi.org/10.1103/PhysRevLett.81.2399) and [1999](https://doi.org/10.1098/rsta.1999.0363) establish discrete-gradient geometric integration for invariants and Lyapunov functions. A generic projection or discrete-gradient correction of the known central invariants is therefore not a new algorithmic direction.
- [Han et al., NeurIPS 2021, Table 1](https://papers.neurips.cc/paper_files/paper/2021/file/4b04b0dcd2ade339a3d7ce13252a29d4-Paper.pdf) already gives the Bures exponential expression underlying the factor-induced product map. The congruence update is not a new matrix retraction.

## What remains, and what does not

The narrow candidate is the damped-pullback-specific local theorem in `DISCRETE_THEORY.md`: a closed limiting-endpoint expansion at fixed stable step size, a sign reversal across `lambda=2s`, cancellation at that value for every fixed positive quadratic weight, and a certified fourth-order failure showing that the cancellation is not exact neutrality. The equal-trace weight witness prevents the entire effect from being described as a mere global rescaling of learning rate. The noncritical Jordan-closure boundary is a supporting consequence, not a second candidate.

The inspected sources do not state this precise package. Its proof is elementary once the prior continuous invariants are known, and the underlying numerical-analysis mechanism is classical. The appropriate status is a modest, potentially original specialized theorem/corollary, requiring further priority review. It is not sufficient evidence for a practical optimization innovation. Any stronger claim about improving training, changing generalization, or a useful invariant-preserving scheme should be rejected on the current evidence.

The previous `rate-clock` retirement remains in force: the present tests concern fully converged interpolation endpoints, use equal-trace weight matrices, and analytically isolate the small-target coefficient. No fixed-horizon preference change or previously rejected geometry heuristic is revived.
