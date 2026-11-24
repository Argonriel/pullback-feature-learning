# Three retained mathematical candidates

Date: November 21, 2026 (America/Los_Angeles).

## Outcome and counting rule

Three distinct theorem-level candidates survive this round's original-source audit, derivation review, and bounded validation. They remain within the matrix representation-optimization framework. They are three different questions and results, not three unrelated applications, and not three already certified original discoveries.

1. **Universality across spectral metrics, with arbitrarily delayed detectable obstructions.** Which measurement spaces are neutral does not depend on the particular non-BKM homogeneous analytic spectral correction, even though the first visible endpoint obstruction can occur at arbitrarily high target order.
2. **Mixing neutral geometries destroys conservation generically.** Diagonal observations with off-diagonal initial coherence are integrable at both limiting metrics, but their finite positive blend is generically bracket generating in every matrix dimension. Actual fixed quadratic losses exhibit different endpoints.
3. **Global convergence and exact endpoint selection on the classified family.** Central invariants prevent escape and singular limits, removing the previous small-target or assumed-convergence qualification for every positive damping and strictly feasible target.

The first two have the strongest potential as main mathematical statements. The third provides a different, substantive guarantee for the compatible initialization class. All require further priority scrutiny before publication claims. None establishes better prediction, generalization, or a practical advantage for neural training.

## 1. A universal class can hide arbitrarily late endpoint obstructions

Let M_X have a positive symmetric spectral kernel m(x,y), analytic on positive arguments, homogeneous of degree -1, with m(x,x)=1/x. This includes normalized symmetric monotone quantum metric kernels. Consider the optimization metric I+lambda M_X, starting at sI.

For every m different from the BKM logarithmic kernel, normal projected closedness and first-damping endpoint neutrality under all fixed SPD quadratic weights select exactly the same measurement spaces: the associator ideals previously recognized by the cubic and quartic algebraic tests. BKM itself is exceptional and makes every measurement space neutral.

The additional point is that the first nonzero difference between m and BKM can occur at order 2k for arbitrarily large k. The fixed-weight endpoint obstruction then first appears at target order 2k+1, or 2k+2 when Lie-triple closure holds. Nevertheless positivity of ad_Z^2 reduces every such high-power condition to the same two algebraic closure tests. This supplies an exact universality statement and a limit on finite-order perturbative diagnostics.

A positive mixture of two familiar metrics, two thirds normalized SLD plus one third normalized RLD, cancels the usual leading kernel discrepancy. Its static-weight witnesses begin at fifth and sixth target order. The minimal check recovers exponents 4.929 and 6.002, with quadrature discrepancies below 6.42e-13 relatively. The independently derived coefficients 77/28800 and -29/4800 agree with the small-target derivatives. Positive Gaussian quadrature supplies a proved arbitrary-order family; that construction is not inferred from these two examples.

**Novelty boundary.** The kernel family and integral representations are established in [Lesniewski–Ruskai](https://arxiv.org/pdf/math-ph/9808016); full-state BKM duality uniqueness is established in [Grasselli–Streater](https://math.mcmaster.ca/~grasselli/GrasselliStreater01.pdf); logarithm quadrature is established in [Fawzi–Saunderson–Parrilo](https://arxiv.org/pdf/1705.00812). The candidate is the exact normal-subspace universality plus its fixed-weight response hierarchy. Neither a new monotone metric classification nor a new rational approximation method is claimed.

Proof and checks: [universality/THEOREM.md](universality/THEOREM.md), [independent review](universality/REVIEW.md), [search record](universality/SEARCH_LEDGER.md).

## 2. Two integrable limits with a generically nonintegrable finite blend

For the original full-representation two-factor metric, observe only the diagonal of X. At zero damping the off-diagonal entries are conserved. At the rescaled pure-Bures limit the correlations X_ij/sqrt(X_ii X_jj) are conserved. Each limiting geometry therefore has explicit weight-independent endpoints for nearby compatible targets.

For every finite positive damping, a generic coherent SPD initialization behaves differently. The n diagonal gradient fields and their pairwise first brackets span all n(n+1)/2 symmetric directions. Thus no nonconstant common C^2 product-level conservation law exists. The rank statement holds on an open dense set with measure-zero complement; it does not claim full rank at diagonal or every exceptional state.

For X=[[a,c],[c,b]], d=ab-c^2 and Q=(a+b)d+lambda ab, the exact lifting curvature is

    F_lambda = lambda c(b-a)d/(2Q^2).

It vanishes at both parameter limits and is nonzero at every finite positive lambda when c!=0 and a!=b. This is strengthened beyond a controlled-path argument: a constructed pair of fixed SPD quadratic losses with identical interpolation constraints has a nonzero quadratic endpoint contrast. For X0=[[1,.3],[.3,2]] and lambda=2.865, the predicted coefficient is -.000506650773419. The smallest-target numerical coefficient is -.000502395328628, approaching it at the predicted rate; the actual endpoint gap is -3.14e-7 while step refinement changes endpoints by at most 2.1e-14.

**Novelty boundary.** Lie-rank methods are established by [Marcotte–Gribonval–Peyre](https://arxiv.org/html/2405.12888v1), and pure SLD/Bures integrability is addressed by [Nagaoka–Fujiwara](https://arxiv.org/html/2307.03431v3). Those general methods and limiting geometries are not new. The candidate is the all-dimensional finite-blend destruction result together with the separate static-loss witness. A curvature maximum is not asserted to maximize the finite-target endpoint gap, and bracket generation is not misrepresented as controllability by a single fixed quadratic loss.

Proof, original-source comparisons, and controls: [nonisotropic/REPORT.md](nonisotropic/REPORT.md) and [search record](nonisotropic/SEARCH_LEDGER.md).

## 3. A global theorem replacing the local convergence qualification

Assume S passes the previous exact classification, J=Jordan(S,I), and X0 is any SPD element of J. For every fixed lambda>0, every fixed positive quadratic measurement weight, and every target with an SPD feasible matrix, the original damped flow exists globally, stays SPD, and converges to the unique Bregman projection for

    Phi_lambda(X)=||X||_F^2/2+(lambda/2)tr(X log X-X).

No restricted-isometry or small-target hypothesis is used. This is not a global theorem for arbitrary measurement spaces, incompatible initial states, rank-deficient starts, or zero damping.

The missing argument has two problem-specific parts. First, bounded measured coordinates and central normal invariants preclude unbounded escape even though the measurement loss is noncoercive. Second, after standard analytic factor-flow convergence and convex PSD optimality, a strictly feasible matrix minus a hypothetical singular limit gives a central logarithmic invariant that would diverge. This excludes the boundary. The independent review checked the full logical chain, including the unstable-cone lemma needed for all invertible starts, and found no fatal gap or incorrect constants.

A new large-target case has displacement norm 3.2913 and an unconstrained Euclidean affine interpolant with minimum eigenvalue -.508738, so it lies outside the old local positive-definite continuation argument. At lambda=4, the damped endpoint has minimum eigenvalue .111715. Two positive loss weights agree to 1.41e-11 at the finer step; the largest fine-step prediction error is 1.92e-11.

**Novelty boundary.** The inspected [Bah–Rauhut–Terstiege–Westdickenberg theorem](https://www.instmath.rwth-aachen.de/~mwest/files/linear.pdf) uses different observation/coercivity assumptions. [Zhang–Peyre (2026)](https://arxiv.org/html/2610.09356) already allows underdetermined losses, so underdetermination alone is not a distinction; the tied PSD embedding has zero signed support gap and misses its main global guarantee. Analytic convergence, PSD optimality, and Bregman uniqueness are known tools. The candidate is the global conclusion supplied by classification-based coercivity and boundary exclusion for this precise noncommuting damped family.

Proof: [global_convergence/THEOREM.md](global_convergence/THEOREM.md). Independent proof review: [global_convergence/REVIEW.md](global_convergence/REVIEW.md). Strict primary comparison: [global convergence novelty audit](robustness/GLOBAL_CONVERGENCE_NOVELTY_AUDIT.md).

## Directions deliberately not counted

The discrete lane derived a precise damping value lambda=2s at which the quadratic finite-step endpoint error vanishes for all stable weights, with a nonzero fourth-order certificate showing that neutrality is not exact. It is retained only as a modest supplement: the main mechanism is standard broken-conservation/discretization analysis, and the exact coefficient is not presently an independent research direction. See [discrete/DISCRETE_THEORY.md](discrete/DISCRETE_THEORY.md) and its [primary audit](discrete/PRIMARY_LITERATURE_AUDIT.md).

The robustness lane found a boundary-layer response, but its diagonal baseline is exactly an already studied singular Riemannian barrier system. Generic continuity and near-commuting stability ideas were also insufficient. This lane is not counted; the calculations and rejection are preserved in [robustness/ROBUSTNESS_SCREEN.md](robustness/ROBUSTNESS_SCREEN.md).

No retired feature-learning gate, scalar-depth proposal, poisoning/topology route, or reserved validation cohort was reopened. Reusing elementary matrix bases to test a new metric-order prediction is documented as such, rather than presented as a new empirical dataset.

## Research judgment

The three candidates address different questions: metric universality and what perturbative evidence can detect; the destruction of conservation by finite blending outside a compatible algebra; and global convergence inside the compatible algebra. A coherent manuscript could use the first two as primary statements and the third as a constructive guarantee. That organization does not turn the existing metric, Jordan theory, or conservation framework into new contributions.

Current status is **three rigorously supported potential innovation points**, with proof reviews and minimal consistency checks, subject to unresolved priority and significance. The artifacts do not justify a claim of three certified original discoveries or a practical neural-network improvement. The next publication-level work is a broader priority check and refinement of the mathematical exposition, not larger training experiments to rescue discarded directions.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for reproducibility and [SCREEN_REGISTER.md](SCREEN_REGISTER.md) for the counting and exclusion rules. `ARCHIVE_CHECK.json` records the final package verification and preservation of the five preceding archives.
