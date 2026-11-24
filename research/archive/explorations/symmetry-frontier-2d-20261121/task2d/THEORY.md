# A controlled task with two independent invariants

Draw amplitudes R1,R2 independently uniformly on [0.4,1.4] and phase Phi uniformly on [0,2 pi), independently. The input is

`X=(R1 cos Phi, R1 sin Phi, R2 cos(2 Phi), R2 sin(2 Phi))`.

The label posterior is

`p(R1,R2)=1/2+(2/5)tanh(8(R1-0.9))tanh(8(R2-0.9))`.

Both labels have positive probability everywhere. This is a smooth noisy XOR of the two amplitude offsets, with exact prevalence 1/2 by either amplitude's reflection about 0.9. The population quotient density is 1 on the unit square [0.4,1.4]^2, with product uniform CDF and exact rectangle masses. Training uses 128 independent draws from each class-conditional distribution, followed by a shuffle; fixed label counts mean the training sample is stratified, not jointly iid. Calibration and test inputs are iid population draws.

## The group and the complete quotient on the support

The action is `T_theta=R_theta direct-sum R_(2 theta)`. The selected partner is `T_pi=diag(-1,-1,+1,+1)`, an exact involution, not the Euclidean antipode. Uniform independent phase and the invariant posterior give joint input-label invariance. The pair norms `(R1,R2)` are invariant and separate group orbits on this phase-coupled support: equal pair norms imply that the difference of first-pair angles maps one point to the other. The first amplitude is strictly positive, so no angular degeneracy occurs. The support has dimension three and the free circle orbits dimension one; the quotient has dimension two.

This is a statement about the specified support. Two norms do not separate arbitrary ambient R4 orbits; a relative-phase invariant would also be needed without the phase-coupling restriction. The experiment is a bounded synthetic two-dimensional quotient, not validation on unrestricted four-dimensional data.

Total Euclidean norm is insufficient. `(R1,R2)=(0.5,1.3)` and `(sqrt(0.97),sqrt(0.97))` have the same squared norm, 1.94, but their posterior probabilities differ and are on opposite sides of 1/2. Continuity gives variation along norm-level curves, not merely a difference at an isolated pair of points. The model receives only four raw coordinates; calibration policies receive both quotient coordinates and phase. No scalar norm surrogate replaces the quotient.

## Bayes difficulty and learnability limits

The Bayes classification error equals

`1/2-(2/5)[log(cosh(4))/4]^2`.

The Bayes logistic risk is the two-dimensional integral of the Bernoulli entropy, evaluated independently by Gauss quadrature. The constant-posterior predictor has logistic risk log(2) and classification error 1/2. These are separate metrics: a fitted classifier may improve class decisions while overconfident probabilities leave its log loss near or above log(2). The fixed pilot retains all three training recipes and reports this distinction.

## Model and finite bounds

The byte-identical copied model has two 4D residual blocks, width 8, and 144 trainable parameters. Each block is `z -> z + 0.5 B tanh(Az+b)` and the logit is the first output coordinate. The copied file also contains legacy data helpers; none is used for this task. The PB Gram is a sum over all four output-coordinate Jacobians and all training inputs; the logistic objective is a mean. Damping is 0.1. These conventions are unchanged from the previous task.

Let `L=product_blocks(1+0.5||B||2||A||2)` and `C=sum_blocks 0.5 sum_j |B_(first,j)|`. Then `|h(x)-x1|<=C`, `|h(x)|<=1.4+C`, and for `d(x)=(h(x)-h(T_pi x))/2`,

`|d(x)| <= min(1.4+C,1.4 L)`.

These are population bounds on the compact support, with no tail exception. The invariant posterior and half-turn-even selection imply the usual exact cancellation of the posterior-dependent antisymmetric term. A global quota rule must preserve each input's half-turn symmetry with common random keys for that cancellation; the method branch proves and tests that property separately. This task specification alone does not certify an arbitrary orientation-sensitive allocation.

## Fixed calibration and scope

The common grid is 16 by 16 amplitude anchors, each spanning [0.4,1.4], and 256 midpoint full-orbit angles, exactly 65,536 forward examples. The half-turn partner is the angular shift by 128. The paired iid folded control receives 32,768 inputs and their partners at the same model-forward budget. Equal cost does not imply equal information designs; the orbit grid and iid paired sampling differ. Deployment is 256 original inputs and exactly 64 selected partner inputs per quota policy. Raw and full two-view references have different costs and must be labeled separately.
