# Finite blended geometry: autonomous endpoint freedom and initialization strata

Date: November 21, 2026 (America/Los_Angeles).

**Completion status: COMPLETE. Final disposition: RETAIN the direction, with a strengthened theorem and narrowed novelty attribution.** The requested literature audit, explicit assumptions and quantifiers, independent proof review, bounded validation, failed-attempt records, and reproducible source package are complete. This is a rigorously supported potential contribution, not certification of original discovery.

## What materially changed

The old result showed generic bracket generation in every dimension and one fixed-loss contrast in dimension two. This round proves a stronger autonomous statement:

> For every finite positive damping, at a generic SPD initialization, and for every nonzero sufficiently small target direction, varying one fixed SPD quadratic loss weight produces an open set of interpolating endpoints in all unobserved coordinates.

The weight remains constant throughout each trajectory. The proof does not use switched controls. For n diagonal observations there are n(n-1)/2 hidden coordinates, and the actual endpoint map has that full output rank. The threshold for the target size depends on the initialization, damping, direction, and reference weight. The statement is local and does not make the entire feasible fiber reachable.

The theorem also resolves part of the initialization dependence. With initial coordinate blocks of sizes b_1,...,b_r, the generic attainable hidden dimension is exactly sum b_j(b_j-1)/2 within the invariant block manifold. Cross-block entries remain zero. Full SPD loss weights may couple the observed residuals across those blocks.

The self-contained main draft is [THEOREM.md](THEOREM.md). The detailed independent derivation is [static_loss/THEOREM.md](static_loss/THEOREM.md), and the adversarial review is [exceptional/STATIC_THEOREM_REVIEW.md](exceptional/STATIC_THEOREM_REVIEW.md).

## The new mechanism and its exact limits

At X0, let A0 be the positive observed mobility and P=A0^(1/2). Normalize the target displacement as v=P d and the fixed weight as B=P^-1 C P^-1. The first-order observed path is P(I-exp(-Ct))d. Its signed-area matrix is

    Omega(C)=(CG-GC)/2,       CG+GC=dd^T.

If k is the Krylov dimension of (C,d), the independently derived exact rank is

    rank D Omega(C)=n(n-1)/2-(n-k)(n-k-1)/2.

Thus k>=n-1 suffices for full area variation. Cyclic C exist for every d!=0 and form an open dense set. The hidden endpoint derivative is epsilon^2 times the finite-blend curvature applied to this area derivative, plus a remainder of order epsilon^3 that is uniform with one weight derivative. The generic curvature is invertible, so a nonzero minor persists for actual small nonzero epsilon. Local exponential residual decay justifies the infinite-time derivative and convergence, rather than assuming them.

The exact area image has a useful spectral characterization: dd^T/2+Omega must be real diagonalizable with nonnegative eigenvalues. This is a direct specialization of classical positive-matrix factorization, credited to the original statement in [Ballantine (1968)](https://msp.org/pjm/1968/24-1/pjm-v24-n1-p02-s.pdf). It is not counted as an independent innovation. In two dimensions it gives the sharp leading-area interval |Omega_12|<||d||^2/4. This bounds leading endpoint coefficients; it does not determine the full nonlinear endpoint range at a fixed finite target.

## Exceptional states: proved upgrades and eliminated overclaims

The [exceptional-state report](exceptional/REPORT.md) supplies exact algebraic certificates.

- In dimension two, every coherent state c!=0 has full Lie rank. Unequal diagonal entries need brackets of length two; equal diagonal entries need length three. The only rank-two stratum is c=0. Equal diagonals therefore do not define a conservation leaf, although the quadratic-order curvature vanishes there.
- A fully coherent 3x3 matrix with diagonal entries 2, off-diagonal entries 1/3, and damping 3 has ranks 3,5,6 through bracket lengths 1,2,3. This exactly disproves the proposed strengthening that connected or complete support guarantees two-step full rank everywhere.
- Component partitions of the nonzero-coherence graph are invariant at finite times. They also persist at compact SPD limits when controls have finite integral, as in the local quadratic-loss theorem. Individual zero entries are not invariant: an exact connected-path example creates its missing edge immediately.
- A complete all-state classification for connected blocks of size at least three remains open. Genericity cannot be replaced by every coherent initial state.

A scalar transformed weight C=alpha I has area-derivative rank n-1, so it cannot establish quadratic-order full hidden rank when n>=3. This is a limitation of the leading calculation, not a proof that the exact finite-target endpoint map is rank deficient. Likewise curvature invertibility is sufficient here, not proved necessary.

## Audit of the old determinant and both limits

The all-dimensional leading determinant was rederived in THEOREM.md from the inverse-Lyapunov constraint, including cancellation of the vertical derivative terms. The observed mobility is a positive compression of the full mobility, so normalization is legitimate. No congruence-whitening equivalence is used.

With theta=lambda/(1+lambda), the time-rescaled mobility is

    N_theta=K_X[(1-theta)K_X+theta Id]^-1.

On any compact SPD neighborhood it extends smoothly and uniformly positively to N_0=Id and N_1=K_X. The same small-target bootstrap and derivative estimates therefore give continuous endpoint limits for a fixed compact weight family. At the two ends, off-diagonal entries and correlations, respectively, are conserved. Every fixed-pair endpoint gap tends to zero at both limits. No claim connects a pointwise curvature maximum to a finite-target endpoint-gap maximum.

## Minimal validation and independent checking

One 3x3 state, one finite damping, and one fixed target direction test the new actual endpoint derivative. Three amplitudes assess the predicted remainder; one step refinement tests numerical scale. No training, dataset, reserved cohort, or dimensional performance sweep was used.

| Target amplitude | Scaled endpoint-Jacobian error | Smallest scaled singular value |
|---:|---:|---:|
| 0.08 | 2.12243e-5 | 1.45072e-5 |
| 0.04 | 1.07082e-5 | 1.48286e-5 |
| 0.02 | 5.37837e-6 | 1.49921e-5 |

The predicted smallest scaled singular value is 1.51574e-5. Halving the time step changes the unscaled derivative by 4.43e-13, or 1.11e-9 after amplitude scaling: a diagnostic margin of approximately 13,540. Interpolation and diagonal-sensitivity residuals are below 8e-15. Step differences are not rigorous error bounds; the all-dimensional theorem rests on the proof.

Independent exact-fraction checks establish six selected initialization rank cases. A separate 4x4 algebra check distinguishes Krylov dimensions k=0,1,2,3,4, recovering area ranks 0,3,5,6,6. Centered derivative checks show the expected fourfold error reduction.

The first endpoint finite-difference convergence flag was false: increments 0.002 and 0.001 left discrepancies near 3e-13, where cancellation obscured the expected ratio. The unchanged first source and result are preserved. A targeted larger-increment check gives errors 4.22e-11,1.04e-11,2.66e-12 at increments 0.04,0.02,0.01, recovering the expected behavior while keeping all weights SPD. See [FAILED_ATTEMPTS.md](FAILED_ATTEMPTS.md) and [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Closest original literature and final novelty judgment

The [primary-source audit](literature/AUDIT.md) compares hypotheses and conclusions theorem by theorem. Forty search queries and eight original-source snapshots with hashes are preserved.

- Marcotte--Gribonval--Peyre provide the conservation/Lie-rank framework, and their intrinsic-dynamics work relates recovery to Frobenius structure. Those general implications are already known.
- Nagaoka--Fujiwara and Li--Wang--Lee--Arora supply direct integrability/commuting-flow precedents for the limiting geometry. The finite blend is a different distribution.
- The recent split-Bures covariance-law and isospectral covariance-steering theorems have different state/gradient constraints or allow time-dependent gains. Their control conclusions cannot replace the static-loss proof.
- Lyapunov commutator areas and positive-matrix factorization are established machinery. The exact area-image criterion is explicitly credited as a classical specialization.

The retained contribution is the concrete finite-blend theorem together with **all-dimensional local endpoint submersion under a single autonomous positive quadratic loss**, including its generic block-relative dimension and initialization limitations. No inspected theorem directly covers that combined conclusion. Search non-coverage does not prove priority; a previously published equivalent generic connection theorem would further narrow the attribution. The small exceptional-state propositions are supporting refinements, not additional major innovations.

No retired learning-gate, scalar-depth, poisoning, topology, or reserved-validation route was reopened. No authors were contacted and no external messages were sent. All old archives and shared dependencies were treated as read-only. The reproducibility record and archive integrity check accompany this report.
