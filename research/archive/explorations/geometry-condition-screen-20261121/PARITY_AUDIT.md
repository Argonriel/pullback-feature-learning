# Inversion parity and small residual initialization: completed condition screen

Date: November 21, 2026 (America/Los_Angeles). Decision: **reject this condition as a current innovation candidate.** The exact parity identities identify where radial signal first enters this architecture, but their speed/damping consequence is already fixed-tangent spectral filtering. Neither parity nor a small residual scale supplies a non-clock, adaptive-versus-frozen generalization sign over finite training progress.

This note follows the completed full-tangent gate. It does not modify that protocol, expand its failed empirical branches, use its reserved confirmation cohort, or claim that a special zero-initialization witness explains the original nonzero initialization.

## Actual architecture and precise symmetry

Use the original two-block map h_next=h+0.5 B tanh(Ah+b), with fixed first-coordinate readout f. At **all biases zero**, the complete representation h is odd in x. Parameter derivatives with respect to A and B are odd, while parameter derivatives with respect to b are even. This follows by the chain rule: tanh is odd, its derivative is even, and each later state Jacobian is even in x.

Partition parameters into w=(all A,B), with 64 coordinates, and c=(all b), with 16 coordinates. Write the full representation Jacobian as J=[J_w,J_c], and let a_w,a_c be its first-coordinate rows. On a centrally symmetric input distribution, or a finite quadrature containing exactly weighted x/-x pairs,

    E[J_w(x)^T J_c(x)] = 0.

Consequently the full pullback metric is block diagonal in this partition at b=0. This statement concerns inversion parity of functions of x; it is not a statement that the training loss has a b-to-minus-b parameter symmetry.

Let p(x)=Pr(Y=+1|x) be even, as for a radial teacher, and let m=p-1/2. The mean logistic derivative with respect to f is sigmoid(f)-p. Because f is odd, sigmoid(f)-1/2 is odd. Hence

    grad_w L = E[a_w^T (sigmoid(f)-1/2)],
    grad_c L = -E[a_c^T m].

The first expression is independent of the even label posterior. The second carries the radial signal. Replacing p by 1-p leaves the weight gradient unchanged and reverses the bias gradient. These are exact identities under the stated paired-input/even-posterior assumptions.

The same observation appears at the level of restricted risk:

    L(w,b=0) = E[log(2 cosh(f(x)/2))].

The p*f term integrates to zero, so the entire bias-zero submanifold has target-independent risk. Nevertheless its **normal bias derivative need not vanish**. Therefore the zero-bias submanifold is generally not invariant under either GD or PB. Inferring invariance from odd activations or even labels would be wrong. Frozen biases would prevent learning a nonconstant even posterior; trainable biases need not.

For the original randomly sampled finite training set, exact x/-x pairs are absent. The cross Gram and gradient cancellations then need not hold exactly. Nothing here permits averaging the label noise away after the parameters or Jacobian were fitted to those labels.

## Small residual scale reduces to an initial spectral effect

Scale both residual B matrices by epsilon while keeping A fixed and b=0. On a bounded input domain, or under sufficient integrable derivative bounds,

    J_c = epsilon D + O(epsilon^2),
    q_c = grad_c L = epsilon q_1 + O(epsilon^2),
    M_cc = lambda I + epsilon^2 G + O(epsilon^3).

Here the metric uses the original sample sum, while the objective gradient uses the sample mean. For fixed positive lambda,

    u_c^PB = -(epsilon/lambda) q_1 + O(epsilon^2),
    (predicted even logit velocity)^PB
        = -(epsilon^2/lambda) D_readout q_1 + O(epsilon^3).

GD has the same leading expression without 1/lambda. Thus the even signal direction is initially weak, and PB transmits it according to ordinary damped spectral filtering. The complete frozen tangent model has exactly the same initial Jacobian, metric and velocity, so this calculation does not establish adaptive metric feedback. It is not a new mechanism.

If lambda scales as rho*epsilon^2 instead, the bias parameter velocity can scale as epsilon^-1 and the even prediction velocity can remain order one. This is a nonuniform parameter-motion limit; it does not establish a finite-step or finite-progress result for the original fixed lambda. It also remains an initial spectral/damping statement until a trajectory analysis adds something beyond it.

## Zero residual weights: a useful countercheck, not a contribution

At exactly B=0, the initial bias and A Jacobians vanish, while the B Jacobian is odd and nonzero. The entire initial tangent representation is consequently odd for every parameter displacement. It cannot realize a nonzero even prediction. For a centrally symmetric input law and even posterior, every such predictor has population logistic risk at least log(2); equality requires a zero logit almost surely. Whether that zero predictor belongs to this particular finite affine tangent class is a separate approximation question, and is not assumed here.

The nonlinear model need not remain in that class. Its first ordinary GD step changes B through the odd, target-independent gradient. This creates nonzero bias Jacobian columns; the second step can move the biases using the even teacher. In a small-step expansion B=O(eta), b after the second step is O(eta^2), and the first even prediction produced by those two steps is O(eta^3). The numerical checker confirms the cubic scaling in the actual two-block architecture.

This is an example of ordinary nonlinear feature creation from a degenerate initial tangent model. It applies to GD as well as PB and therefore cannot by itself establish a PB-specific adaptive-geometry benefit. It also concerns the singular special initialization B=0, not the project's standard nonzero random B matrices.

## Why growing bias geometry does not imply adaptive harm

An attractive refinement is: B(t) creates useful even bias features, their adaptive metric grows, and adaptive PB therefore suppresses useful signal relative to the initial metric. The narrow same-state fact is true but routine. At a bias-zero state, freezing the metric at a B=0 initialization gives M0_cc=lambda I, whereas the current bias block is lambda I+J_c^T J_c. Therefore

    q_c^T (lambda I+J_c^T J_c)^(-1) q_c
        <= ||q_c||^2/lambda.

This is the standard positive-semidefinite ordering of inverses. It compares predicted training improvement at the same state and scalar rate. It does not compare whole trajectories at equal training progress or determine population-risk change. In a one-dimensional bias block the difference is purely a positive scalar speed change.

There are additional missing terms in the full network. Starting at B=0, B(t)=O(t) and b(t)=O(t^2). The adaptive A/B metric generally changes already at O(t), changing the residual-weight trajectory at O(t^2). The isolated bias-metric correction multiplies an O(t^2) Gram term by an O(t) bias gradient, so it enters the bias velocity only at O(t^3). Changes in the preceding weight trajectory can influence bias learning earlier. Once b is nonzero, the exact parity block separation also disappears, and cross-block effects can enter at the same order as the isolated bias-metric correction. The sign of the isolated term therefore cannot be promoted into the sign of the full matched-progress predictor contrast.

Moreover, if both training and the quantity being called population risk are the same exact population objective, matching their training loss makes the population-risk contrast identically zero. A nonzero generalization contrast requires an empirical-versus-population distinction. There the relevant quantity is a bilinear expression involving different gradients; positive-semidefinite ordering alone does not fix its sign. Reintroducing the unknown population-gradient sign as an assumption would be circular for the requested advance criterion.

## A stronger radial condition is still insufficient

For a one-unit, one-block restriction with readout residual coefficient c, let a be its incoming vector. Under an isotropic input law, angular averaging gives phi_a(r)=E_angle[sech^2(r a^T u)], a nonincreasing function of radius. If the centered posterior m(r) is nondecreasing and has mean zero, then E[m(r) phi_a(r)]<=0 by the elementary covariance ordering for oppositely monotone functions. The initial bias gradient is -0.5*c*E[m phi_a], so its sign is determined relative to c.

This condition identifies an initial signal-fitting direction in this restricted model. It holds equally for GD and the complete initial tangent PB direction. It supplies neither the sign of adaptive-minus-frozen prediction changes nor a finite-progress generalization guarantee. Extending it by assuming all future Jacobian/population alignments preserve that sign would merely assume the desired result. We therefore do not retain it as a new condition.

## Numerical verification

`parity.py` uses the actual analytic core Jacobian, 128 exactly paired inputs on two radii, and an even monotone posterior. These quadrature inputs are distinct from all frozen training/confirmation cohorts. It runs no optimization sweep. The complete numeric record is `parity_results.json`.

- Maximum error across representation parity, Jacobian parity, cross-metric cancellation, label reversal, restricted-risk independence and block solving: **2.36e-16**.
- Nine residual scales from 1 to 1e-4 verify first-order bias-Jacobian/gradient/parameter-velocity scaling and second-order even-prediction-velocity scaling.
- At epsilon=1e-4 and lambda=0.1, the even PB/GD velocity-norm ratio is **9.99999963**, approaching the ordinary 1/lambda factor.
- With exactly zero residual weights, the first bias Jacobian is exactly zero but the residual-weight gradient is nonzero. Two GD steps create an even prediction, with estimated step-size exponent **2.9923**, matching the cubic leading order.
- The single-unit monotone radial covariance is strictly negative in the declared check.

The script records source/core hashes and refuses to overwrite its output. These checks verify the reductions, not predictive generalization or a finite-horizon theorem.

## Closest primary literature and scope

[Daniely and Malach, Learning Parities with Neural Networks, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/eaae5e04a259d09af85c108fe4d7dd0c-Paper.pdf), Sections 2-3 and Theorems 2-3, already proves feature-learning advantages over fixed embeddings on a specially constructed Boolean-parity distribution. Their setting uses ReLU6, hinge loss, symmetric initialization and specified regularization/steps. Boolean parity is not our inversion-parity calculation, and the theorem does not cover this tanh pullback rule. It nevertheless precludes presenting generic nonlinear access to a previously weak/unavailable target feature as a new principle.

[Blumenfeld, Gilboa and Soudry, Beyond Signal Propagation, ICML 2020](https://proceedings.mlr.press/v119/blumenfeld20a/blumenfeld20a.pdf), Sections 1-2, examines zero/identical-feature initialization, residual signal propagation, and subsequent symmetry breaking. Its feature/permutation symmetry and architectures differ from our input-inversion split. The broad story that initially degenerate or zero residual features can become useful during ordinary training is already established; the present two-step Taylor check is not a distinctive contribution.

[Ziyin, Symmetry Induces Structure and Constraint of Learning, ICML 2024](https://proceedings.mlr.press/v235/ziyin24a.html) studies consequences of actual loss-function mirror symmetries under regularization/noise. It should not be applied to assert that our zero-bias submanifold is invariant: its nonzero normal gradient directly disproves the required loss reflection symmetry for a nontrivial even teacher. The rejection here rests on explicit elementary reductions, not a claim that this paper proves the identical residual-network identity.

This is a bounded primary-source check, not an exhaustive literature proof. Prior audit results on lazy training, full tangent kernels, and learned preconditioning remain applicable.

## Final disposition

The parity/small-residual condition is **rejected as an innovation candidate in its current form**. It provides a useful architecture-specific diagnostic of where an even target initially enters, and it catches a false invariant-subspace inference. Its quantitative prediction is initial fixed-feature filtering; its zero-initialization nonlinear effect is generic feature creation; its bias-metric comparison is a same-state spectral inequality. None determines the requested finite-progress adaptive-versus-frozen generalization sign.

A future proposal would need a new structural statement controlling the full coupled metric response and empirical-to-population transfer, with an independently testable sign prediction. The current lane has no such surviving claim and recommends no additional large experiment.
