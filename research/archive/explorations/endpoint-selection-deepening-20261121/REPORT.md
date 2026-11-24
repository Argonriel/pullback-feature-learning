# Endpoint-selection deepening: a local classification theorem candidate

November 21, 2026 (America/Los_Angeles). The authorized continuation is complete. This round develops the one retained matrix endpoint question; it does not reopen the previously rejected empirical routes.

## Outcome

The previous round established a two-by-two endpoint example. This round obtains a more general mathematical result: **two algebraic closure conditions characterize when the first-order effect of damping on the interpolating endpoint is independent of all fixed positive quadratic loss weights, for all sufficiently small targets near isotropic initialization.** Failure of the first condition can produce a cubic target-amplitude effect; failure of the second can produce a quartic effect even when the first holds.

The theorem, static-loss witnesses, and small examples have been independently checked within the declared model. A bounded primary-literature review found no directly matching theorem, including in the closest quantum information-geometric integrability results. This is a plausible narrow theorem contribution, not a certified originality claim, a new optimizer, or evidence of better generalization.

## Model and precise claim

The model is balanced two-factor matrix prediction with a positive-definite symmetric product X and a metric built from **all matrix entries**, J^T J+lambda I. Its product flow is

    X_dot = -(Id + lambda T_X)^(-1) grad L(X),
    X T_X(U) + T_X(U) X = U.

The observed linear measurements span a subspace S of real symmetric matrices; N is its Frobenius orthogonal complement. Start at X0=sI for fixed s>0. Choose a small target displacement Delta in S. The objective is a fixed positive-definite quadratic form in measurement residuals, so its zero-loss constraints stay the same when weights change.

For a given fixed loss Hessian B on S, define

    C_B(Delta) = right derivative at lambda=0
                 of the converged product X_infinity(lambda,B,Delta).

Small-target convergence, interpolation, and existence of this derivative follow from an interior-SPD and exponential-decay argument. There is no assumption that differentiation can be exchanged with infinite training time without justification.

**Classification.** C_B(Delta) is independent of B for every sufficiently small Delta if and only if, for every E,V in S, both

    [E,[E,V]] belongs to S,
    {E,[E,[E,V]]} belongs to S,

where [A,B]=AB-BA and {A,B}=AB+BA. The first is equivalent to Lie-triple closure. The second is additional. These are conditions on the measurement subspace, not on whether two matrices happen to commute at one state.

| Algebraic outcome | Consequence for the first damping derivative |
|---|---|
| First condition fails | Some fixed positive loss weights and small target expose a cubic amplitude response |
| First holds, second fails | A quartic response can remain; a cubic test alone gives a false neutrality conclusion |
| Both hold | The response is independent of all fixed positive quadratic weights for all small targets |

The statement allows a full SPD quadratic weight matrix. For any fixed **orthonormal** measurement basis, positive diagonal weights also suffice to witness a failure when the target may be chosen. No such diagonal-weight necessity is claimed for an arbitrary fixed nonorthogonal measurement map. A failed condition does not imply that every target or every pair of weights produces a difference.

The order of limits matters: first take the derivative in damping at zero, then expand in target amplitude. This is not a theorem that exact endpoints agree at every finite damping, nor a limit uniform in s tending to zero.

Proofs are in [general geometry](GENERAL_GEOMETRY.md) and [static quadratic weights](STATIC_WEIGHT_THEORY.md).

## Why the result goes beyond a generic path-dependence observation

The first-order response is a line integral of the normal projected inverse-Lyapunov form on the undamped measurement leaf. A radial calculation forces its potential, if one exists, to be P_N log(X)/2. Comparing the Lyapunov inverse with the established matrix-log derivative reduces exactness to an invariant-subspace question. Two commuting self-adjoint linear operators then show that the cubic and quartic closure conditions are sufficient for all orders of this projected-form exactness. They are not just two necessary Taylor tests.

This is **normal projected-form exactness**. It is not Hessian integrability of the metric restricted to the observed tangent space. For example, if S is the whole symmetric-matrix space, N=0 and the present condition is trivial although the full Bures metric is not Hessian in these coordinates.

The second step proves attainability using fixed losses. Let Delta=epsilon E and take a time-independent Hessian B_tau=Id+tau D, with D self-adjoint and V=DE. The mixed response is

    derivative_tau C_(B_tau)(epsilon E) at tau=0
      = -(5 epsilon^3 / (288 s^3)) P_N [E,[E,V]]
        +(13 epsilon^4 / (576 s^4)) P_N {E,[E,[E,V]]}
        + O(epsilon^5).

Any V can be realized by a self-adjoint D, and sufficiently small fixed tau keeps the loss positive definite. Thus a failed algebraic condition supplies actual fixed weighted quadratic losses. The proof does not replace them with arbitrary time-dependent controls or assume that all abstract loops are realizable by training.

## The failed shortcut and its explicit replacement

The proposed shortcut “Lie-triple closure is enough” is false. In three-by-three matrices, take

    U=E12+E21,  V=E13+E31,  S=span(U,V).

This plane is Lie-triple closed but fails the second condition. Use orthonormal measurements U/sqrt(2), V/sqrt(2), target displacement epsilon(U+V), and fixed weights (1,1) versus (1,2). The first damping derivative differs by

    (epsilon^4/s^4) * [[0, 0, 0],
                      [0, -7/240, -1/210],
                      [0, -1/210, 7/240]] + O(epsilon^5).

The formula is obtained from explicit undamped paths and elementary integrals. The cubic term vanishes exactly, so the example resolves the uncertainty rather than merely showing another noncommuting pair.

Conversely, noncommuting spaces need not be sensitive. The traceless symmetric-matrix subspace passes both conditions. It also has the separate exact invariant trace(X)+(lambda/2)log det(X), which yields weight-independent interpolating endpoints in that particular model whenever the SPD interpolating limit exists. That all-damping invariant is a special additional argument, not an extension of the general local theorem.

## Minimal numerical verification

No large neural training, prior benchmark rescore, population-risk estimate, or reserved confirmation cohort was used.

- Exact undamped-path quadrature verifies the mixed coefficients 5/288 and 13/576 with central-perturbation refinement. The analytic polynomial coefficients are computed independently of damped trajectories.
- For the Lie-triple-closed three-by-three example, the X22 response divided by epsilon^4 approaches -7/240 as epsilon decreases through .2, .1, .05, .025. The corresponding X23 coefficient approaches -1/210.
- Finite polynomial checks and direct derivative calculations validate the geometry on fixed commuting, Jordan, traceless, and failing measurement spaces. These are implementation checks accompanying the proof, not statistical evidence for a universal theorem.
- A finite-damping realization uses s=1, raw amplitude .25, lambda=.2, and weights (1,1)/(1,2). Its unobserved endpoint difference has Frobenius norm **2.93947759e-5**. Both fits have measurement residuals below 1e-13; halving the integration step changes endpoints by at most 1.48e-10.
- An independent original-parameter replay solves the full 18-parameter pullback system. Its endpoints agree with the product-space implementation within **1.25e-11**, while preserving factor balance, symmetry, and positive definiteness numerically.
- Replacing the metric with the established entropy Hessian gives a weight contrast of **8.73e-12**, consistent with its known weight-independent mirror projection. This is a control, not a proposed new method.

See [metric control and original-factor replay](METRIC_CONTROL.md), the individual data files, and [validation checklist](VALIDATION_CHECKLIST.md).

## Primary-literature boundary

The original-source review separates the new target from established ingredients. Bures/Lyapunov geometry, matrix-entropy Hessians, commutator vector fields, and mirror-flow invariants are existing work. [Nagaoka and Fujiwara, 2026 version, Theorem 8.6](https://arxiv.org/html/2307.03431v3) is the closest additional overlap: it characterizes integrability of SLD autoparallel distributions using torsion. Its state-dependent distribution, domain, projection, and quantifiers differ from the fixed affine measurement space and normal projected form here; it is not an immediate fixed-quadratic endpoint classification.

The review also checks G-dual teleparallel connections, Jordan-algebra formulations, affine-invariant Lie-triple geometry, SLD D-invariance, operator Sinkhorn, and matrix-sensing mirror descent. It rejects presenting the contribution as a new metric, generic non-Hessian behavior, generic commutator geometry, or a new entropy-based optimizer. Detailed theorem-by-theorem comparisons and primary links are in [literature audit](LITERATURE_AUDIT.md).

The narrow candidate worth retaining is the **finite algebraic characterization together with static-loss cubic/quartic witnesses**. An exhaustive originality assessment has not been completed. The result is currently restricted to the local first-order damping response of the balanced SPD matrix model; an extension to the original nonlinear residual network or a useful prediction-risk consequence remains unestablished.

## Preservation and next decision boundary

This round closes the all-path-to-fixed-loss gap and rejects the insufficient Lie-triple-only criterion. It does not justify reopening the old tangent gate or enlarging neural experiments. A subsequent contribution should be judged against the exact theorem now written, rather than re-proving this example or renaming the known geometry.

All selected cases, controls, source hashes, and results are retained. The three preceding archives are checked against their existing manifests without modification. The final [archive check](ARCHIVE_CHECK.json) and [data dictionary](DATA_DICTIONARY.md) document the package and its provenance.
