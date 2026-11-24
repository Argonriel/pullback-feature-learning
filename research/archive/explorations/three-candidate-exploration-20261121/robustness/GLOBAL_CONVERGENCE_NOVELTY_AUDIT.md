# Strict primary audit: global convergence on classified measurement spaces

Date: 2026-11-21. Literature audit, with an independent check of the reduction boundaries. Proof construction and numerical validation belong to the parent report; this document does not present the numerical evidence as independently rerun here.

## Status

**Retain as a potentially defensible theorem-level candidate, with moderate novelty confidence and no priority certification.** The strongest statement is global convergence and exact endpoint selection for the damped full-representation flow on the entire classified noncommuting family. Its proof ingredients are classical. The paper-level contribution cannot be advertised as a new global-convergence principle, a new Bregman projection, or a new Jordan structure theorem.

The searched primary results do not directly supply the proposed theorem. The useful distinction is the classified measurement geometry together with a damping-induced central logarithmic invariant. Merely saying “underdetermined” or “without RIP” would be inadequate: some close prior work already allows a loss with a nontrivial kernel.

## Precise proposed statement audited

Let J be a unital Jordan subalgebra of real symmetric matrices, and let S be an associator ideal that generates J together with I. Equivalently in the preceding classification, N_J=J intersect S-perp is central. Let X0 be any strictly positive definite element of J. Use a fixed positive damping lambda and any fixed positive-definite quadratic weight on the measured coordinates. Assume a strictly positive definite feasible target exists. The proposed flow is

    Xdot = -K_X (K_X + lambda I)^(-1) G(X),
    K_X(U)=XU+UX,     G(X) in S.

The claim is global convergence to a strictly positive definite interpolating limit, for every such start and weight, and characterization of that limit as the unique Bregman projection anchored at X0 for

    Phi_lambda(X)=0.5 ||X||_F^2 + (lambda/2) tr(X log X - X).

The statement is not about lambda=0, arbitrary measurement spaces, rank-deficient starts, or inconsistent data. It is also not a new uniform rate claim. Finite-lambda flow is not generally matrix-entropy mirror descent on the whole Jordan algebra.

## Closest original theorems and attempted reductions

### 1. Bah–Rauhut–Terstiege–Westdickenberg

The original [*Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers*](https://www.instmath.rwth-aachen.de/~mwest/files/linear.pdf), arXiv v5 (2020), published 2022, was read at the setup, Theorem 5, Theorems 39–40, and their proof boundaries. Theorem 5 assumes full-rank input covariance XX^T for regression of a full product matrix. The resulting coercivity supplies product boundedness. Global optimality statements are almost-everywhere statements; the paper explicitly warns that balanced initializations form a measure-zero set and require separate treatment. These hypotheses do not cover an underobserved central-nullspace problem with the present damped metric. The bounded analytic trajectory and saddle-avoidance ingredients remain substantial precedents, so those ingredients cannot be claimed as new.

**Reduction outcome:** fails at both the regression observation operator and the metric; the all-positive-definite-start conclusion is also stronger than the applicable almost-everywhere statement.

### 2. Zhang–Peyre (2026)

[*Global Exponential Convergence of Two-Layer Linear Network Training*](https://arxiv.org/html/2610.09356), Assumption 1.1, Theorem 3.1, and Appendix C.3, was inspected directly. The PL assumption explicitly permits a quadratic Hessian with a nontrivial kernel. The global theorem instead depends on a positive conserved signed support gap of the stacked two-factor covariance. Tying the two factors to represent a positive semidefinite product embeds the initial covariance as [[X0,X0],[X0,X0]]. Its signed support gap is zero. Thus the headline convergence theorem does not apply even when X0 is positive definite as a product matrix. The flow also has different damping: the paper's damped extension concerns momentum, not full-representation Euclidean parameter damping.

**Reduction outcome:** fails at the signed support-gap hypothesis for the tied balanced embedding and at the metric. The paper remains a strong precedent for turning conserved quantities into spectral protection and global convergence.

### 3. Preconditioned Burer–Monteiro methods

[Zhang, Fattahi, and Zhang, *Preconditioned Gradient Descent for Overparameterized Nonconvex Burer–Monteiro Factorization with Global Optimality Certification* (JMLR 2023)](https://jmlr.org/papers/volume24/22-0882/22-0882.pdf), Definitions 1–2, Theorem 4, Theorem 8, and Corollary 9 were inspected. The factor-side preconditioner differs from the full representation Jacobian Gram. Linear convergence uses restricted strong convexity and a suitable local region. The global procedure combines perturbed iterations, a strict-saddle condition, and boundedness/coercivity controls. Its general convex global-optimality certification is an important known ingredient, but does not establish deterministic convergence from every positive-definite start in the proposed underobserved family.

**Reduction outcome:** no direct corollary. A claim that “preconditioning globally solves factorization” would nevertheless be grossly overstated in light of this literature.

### 4. Commuting matrix sensing and mirror descent

[Wu and Rebeschini, *Implicit Regularization in Matrix Sensing via Mirror Descent* (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/abf0931987f2f8eb7a8d26f2c21fe172-Paper.pdf), Proposition 5 and Corollary 6, directly prove convergence and an entropy endpoint characterization for commuting measurements under factor gradient flow. The commuting reduction of the present positive-lambda dynamics is likewise standard Hessian-metric optimization, with the quadratic-plus-entropy potential above. That subclass is not a defensible new contribution. The noncommuting associator-ideal family is the part not supplied by their commuting equivalence.

**Reduction outcome:** occupies the commuting mechanism, but the proposed full classification includes noncommuting simple Jordan components, on which the product metric is not the entropy Hessian.

### 5. General Hessian-barrier theory

[Alvarez, Bolte, and Brahic, *Hessian Riemannian gradient flows in convex programming* (2004)](https://www.math.tecnico.ulisboa.pt/~brahic/Legendre10.pdf) develops existence, convergence, Bregman geometry, and dual trajectories for Legendre Hessian metrics. The present metric I+lambda*K_X^(-1) is not generally a Hessian metric on noncommuting spaces. Equality of its normal central block with the entropy Hessian does not imply equality of the whole metric. Consequently this framework does not automatically prove global convergence here, although it fully occupies the ordinary mirror-descent interpretation and much of the commuting case.

**Reduction outcome:** requires an additional argument; a partial conserved potential is insufficient to invoke a full Hessian-gradient theorem.

## What is old and what still needs to carry the contribution

Known ingredients include the analytic/Kurdyka–Lojasiewicz convergence mechanism for bounded trajectories, the convex semidefinite KKT certificate, unstable directions at a nonglobal factor stationary point, Jordan decomposition, logarithmic spectral barriers, and Bregman projection uniqueness. The new claim should not split these into multiple innovations.

The proposed synthesis requires the following problem-specific chain:

1. Loss boundedness controls the measured component P_S X.
2. The classified structure makes every unbounded unmeasured direction central. In a simple block, bounded traceless part forces an unbounded mean to send every eigenvalue of that block to infinity. This prevents cancellation of its linear contribution by a negative log determinant.
3. Conservation of <N,X+(lambda/2)log X>, N in N_J, therefore controls escape despite noncoercivity of the ambient loss.
4. Bounded analytic factor dynamics converge. A limiting negative eigenvalue of the symmetric factor generator would create an unstable solution of its limiting linear equation; invertibility of the initial factor prevents every column combination from lying in the stable subspace. This yields a convex PSD KKT limit.
5. Strict SPD feasibility and centrality exclude a singular interpolating limit: the feasible-minus-limit central difference is positive on every block containing a zero eigenvalue, so its logarithmic invariant would diverge negatively.
6. The earlier exact conservation/projection theorem then identifies the unique endpoint.

Steps 2–3 and 5 are the strongest reasons the result is more than an invocation of a standard bounded-flow theorem. They also explain the necessity of auditing the precise family and feasibility assumptions. The proof must explicitly handle all simple blocks and their representation multiplicities.

## Independence and practical scope

This can reasonably be presented as a third potential direction alongside the universal-kernel classification and the finite-damping loss of integrability. It is, however, an extension of the same classified model, not an unrelated application or an already validated practical optimizer. The global theorem removes the earlier local small-target restriction. Parent-provided validation deliberately leaves that local regime: target displacement 3.2913, strictly feasible minimum eigenvalue 0.3, unconstrained Euclidean affine interpolant minimum eigenvalue -0.508738, and a positive-lambda endpoint remaining positive definite. Those numbers require the parent's experiment record for evidence.

## Remaining priority limits

This search did not locate an equivalent theorem for the precise damped metric and associator-ideal observation family. This is not proof of priority. A broader search of gradient flows on Euclidean Jordan algebras, matrix-valued barrier methods, and factor dynamics with general analytic preconditioners could still produce a direct theorem. The most defensible current description is a potentially new global theorem obtained from a recently established classification, with classical proof machinery and a carefully delimited noncommuting extension.
