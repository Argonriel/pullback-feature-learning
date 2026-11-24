# Independent root proof review

Date: 2026-11-21, America/Los_Angeles. Reviewer: parent agent of this independent research chat. The boundary theorem was drafted by a separate agent with the same inherited model and reasoning settings. This is mathematical cross-checking, not machine-verified formal proof.

## Original chain

The required archived theorem, review, numerical record, primary audit, algebra reduction, and finite-damping audit were read. The original theorem has no identified fatal gap under its stated assumptions. In particular:

1. The unital Jordan projection is positive by testing a negative spectral projector. It preserves measured coordinates and any strict positive lower bound.
2. If the products escape, a normalized limit V is positive semidefinite, central, and normal to S. Since every product commutes with V, all eigenvalues on ran(V) grow to infinity. Both terms of its conserved trace/log coordinate then diverge positively. No strict feasibility was used.
3. On square factors the damping resolvent is analytic everywhere, including singular products. Product boundedness bounds factors. The two gradient/descent estimates justify the analytic finite-length step. This is the standard Lojasiewicz argument, not a new convergence principle. Final editing corrected a wording error: the operator being inverted has eigenvalues at least lambda, while its inverse has norm at most 1/lambda. It also made the standard uniform-continuity justification for a critical omega limit set explicit.
4. The symmetric limiting generator cannot have a negative eigenvalue: its unstable cone would give an unbounded solution, representable as a fixed linear combination of columns of the bounded invertible fundamental matrix. This deterministic argument covers every invertible start.
5. The resulting PSD KKT conditions already prove global minimization for arbitrary data. Exact interpolation and strict feasibility were used only subsequently in the old proof.

## New boundary theorem

The draft `independent_boundary/THEOREM_BOUNDARY.md` was checked line by line. Its replacement of the old strict-feasibility step is valid:

- Strict convexity in measurements makes all global PSD minimizers share the same measurement vector, even when the raw target is inconsistent.
- For any ambient minimizer Z, its Jordan projection is PSD and optimal. The difference N=P_J Z-X* is central and normal.
- On a negative eigenspace of N, positivity of X*+N bounds X* away from zero. A positive N eigenspace containing a zero of X* would force a negative-infinite logarithmic invariant. Contributions cannot cancel. Thus N kills the kernel, and the logarithmic pairing converges on the support of N.
- The full kernel projector E of X* lies in J. The identity tr(E Z)=0 and PSD positivity give ZE=0 for every ambient optimizer. It is the projector, not an arbitrary rank-one kernel projector, that is needed for this argument.
- On the resulting common face, X* is positive definite. The derivative of the anchored objective is X*+(lambda/2)log_P X*-P grad(Phi)(X0)P, which belongs to J by the Jordan quadratic representation. Pairing it with Z-X* reduces to the finite conserved pairing with N and gives zero. Strict convexity then proves the unique ambient Bregman minimum.

There is no ambient logarithm at a singular endpoint in this proof. The logarithm is restricted to its support. Other optimal matrices may have smaller supports; the theorem does not say that every optimal matrix has the same rank.

The stronger all-data claim is an extension of the existing proof, not a separate new mechanism. Inconsistent data generally destroy independence from the observation weight because they change the optimal measurement vector. The exact diagonal KKT control and the coupled-center boundary experiment make these two qualifications independently reviewable.

## Quantitative route and rejected shortcut

Write H_X=I+(lambda/2)Dlog_X and M_X=I+lambda K_X^(-1). The operator C_X=H_X M_X^(-1) is self-adjoint and C_X>=I, because both are spectral multipliers and the logarithmic-mean inequality applies. It fixes central directions and preserves their orthogonal complement. However C_X B need not have nonnegative quadratic form for a general positive definite measurement Hessian B. Therefore the reverse Bregman divergence is not generally Lyapunov. Both independent agents derived explicit 2-by-2 counterexamples.

If B commutes with C_X along the trajectory (in particular, scalar weights on each traceless simple component, arbitrary positive weighting within the measured center, and no mixing between these components), the reverse divergence decreases. Its sublevel gives a computable lower eigenvalue by a one-dimensional entropy inequality, followed by an ordinary PL loss-rate estimate. The root checked the scalar inequality using diagonal entries of Y in an eigenbasis of X and convexity of s log s. This is a useful quantitative corollary of standard tools, not evidence of an independently new spectral-barrier principle.

No explicit uniform-in-time spectral certificate for every arbitrary SPD observation weight has been proved. The counterexample refutes the proposed Lyapunov shortcut, not the possibility of some different certificate.

## Minimal numerical check reviewed

The root independently implemented one new coupled-center boundary problem in `verify_boundary.py`. It is different from the old strictly feasible large-target test. It uses J=Sym(2) direct-sum R direct-sum Sym(2); both matrix blocks have noncommuting observables. The observed central combination is c+z. Its target 1/2, together with a first-block traceless target of spectral radius 1/2, forces the first block onto rank one and the scalar component to zero. The last block retains a free center chosen by its exact logarithmic invariant.

Two SPD weights and two factor-RK4 steps pass numerical consistency checks. The finite-time endpoint error remains about 0.0010074 at time 1600, while step refinement is at most 1.22e-10. This is a boundary transient, not numerical evidence of different endpoints or an exact finite-time solution. The mathematical theorem supplies the infinite-time assertion. No broad benchmark, old experiment rerun, or reserved validation sample was used.

## Review conclusion

The all-data closed-PSD theorem and the geometry-adapted quantitative corollary pass this independent mathematical review. The arbitrary-weight Bregman-monotonicity strengthening is rejected by an explicit counterexample. Novelty remains narrowly scoped and unverified; the closest primary tools and direct reduction obstructions are recorded separately.
