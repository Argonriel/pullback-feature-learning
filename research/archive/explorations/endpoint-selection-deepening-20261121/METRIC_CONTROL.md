# Established entropy geometry as an independent control

This check isolates the role of the inverse-Lyapunov metric. It is not a proposal for a new optimizer or a claim that changing metrics improves prediction. Original sources were read before fixing and running the small matrix cases.

## Original sources and algebra

[Lesniewski and Ruskai, Monotone Riemannian Metrics and Relative Entropy on Non-Commutative Probability Spaces](https://arxiv.org/pdf/math-ph/9808016) develops the established monotone-metric and relative-entropy framework. [Michor, Petz and Andai, 2000, section 2](https://math.bme.hu/~petz/mpa.pdf) explicitly identifies the Kubo–Mori metric with the Frechet derivative of the matrix logarithm and gives its logarithmic divided-difference eigenvalues. [Wu and Rebeschini, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/abf0931987f2f8eb7a8d26f2c21fe172-Paper.pdf) studies matrix-sensing mirror descent, including entropy-related interpolation bias. These sources prevent treating the following metric replacement or its mirror invariant as a new method.

Let T_X(U) solve XT_X(U)+T_X(U)X=U. The two product-space metrics are

    H_PB(X) = Id + lambda T_X,
    H_ent(X) = Id + (lambda/2) Dlog_X.

The latter is the Hessian of the known strictly convex potential

    Phi(X) = .5 ||X||_F^2 + (lambda/2) tr(X log X-X).

Under a linear measurement loss, P_N[X+(lambda/2)log X] is constant along its continuous flow. When the flow converges to interpolation inside SPD, strict convexity gives a unique Bregman projection, independent of positive loss weights.

In an eigenbasis of X with eigenvalues x_i>0, the operator

    R_X = .5 Dlog_X - T_X

has coefficients

    r_ij = (log x_i-log x_j)/(2(x_i-x_j)) - 1/(x_i+x_j),
    r_ii = 0.

Each coefficient is nonnegative, by the logarithmic/arithmetic mean inequality, and vanishes exactly when the corresponding eigenvalues agree. Thus R_X is a positive-semidefinite operator on symmetric matrices, with R_X(U)=0 whenever [X,U]=0. This statement concerns the operator quadratic form, not positive-semidefinite ordering of its output matrix for every U.

For X=sI+Z near sI, direct convergent matrix series give

    R_X(U) = [Z,[Z,U]]/(24s^3)
             - {Z,[Z,[Z,U]]}/(16s^4) + O(||Z||^4 ||U||),

where {A,B}=AB+BA. The parent independently derived this expansion before comparing it with the geometry and static-loss agents. The numerical check validates the formula, not its originality. Combining the known entropy primitive with the pullback response is a convenient decomposition for the new selection question.

## Fixed experiment, selected before numerical evaluation

Take three-by-three symmetric products, X_initial=I, U=E12+E21, V=E13+E31, and orthonormal measurements U/sqrt(2), V/sqrt(2). The measured targets are sqrt(2)/4 each, so the zero-damping endpoint is X_*=I+(U+V)/4. Only the second squared-loss weight changes, from 1 to 2. Damping is fixed at .2. Compare the pullback metric and the established entropy Hessian on exactly these constraints. RK4 steps .02 and .01 are both evaluated with a maximum measurement residual below 1e-13.

The measurement plane is Lie-triple closed but fails the additional fourth-degree closure condition developed in this round. It is a counterexample to the proposed sufficiency of the cubic condition, not a repetition of the earlier two-by-two example.

| Metric | Nullspace norm of weight-2 minus weight-1 endpoint | Largest endpoint change on halving step |
|---|---:|---:|
| Representation pullback | 2.93947759e-5 | 1.47089e-10 |
| Entropy Hessian | 8.72425e-12 | 1.47104e-10 |

The pullback change in X22 is -2.05084977e-5, in X33 is +2.05371945e-5, and in X23 is -3.29225538e-6. The measured entries agree to the stopping tolerance. The first-order damping/quartic displacement formula is a local asymptotic statement; it is not used to assert exact finite-lambda values for this example.

All trajectories remain safely SPD (the check requires eigenvalues above .5). The matrix-log derivative agrees with an independent central difference to 1.98e-11. Subtracting the quadratic and cubic terms of R leaves residuals with the expected cubic and quartic scaling in four fixed displacement sizes. The raw checks are in `metric_control_results.json`.

## Independent original-parameter replay

`factor_replay.py` imports no spectral or reduced-flow routine. It evolves two unconstrained 3-by-3 factors from the identity, constructs the explicit 9-by-18 representation Jacobian, and solves the damped 18-by-18 parameter system at every derivative evaluation. It uses the same two fixed weights and RK4 steps .01 and .005.

Its refined endpoints agree with the spectral product implementation in Frobenius norm to 1.48e-12 and 1.25e-11 for weights 1 and 2. Measurement residuals are below 1e-13, factor balance and product symmetry satisfy the 1e-10 checks, and all sampled product eigenvalues exceed .5. This confirms that the finite-damping witness is realized by the actual factor pullback model. It is not just an artificial product-space vector field.

Source and data provenance are recorded in `factor_replay_results.json`. These checks establish the example and its controls. They do not establish a population advantage, a new mirror method, or the research originality of the classification theorem.
