# Primary-source priority audit: finite Euclidean--Bures blend

Status: complete bounded audit, November 21, 2026 (America/Los_Angeles).

This audit concerns the real SPD flow `Xdot = -K_X(K_X+lambda Id)^(-1) diag(grad ell(diag X))`, with fixed `0 < lambda < infinity` and `K_X(U)=XU+UX`. It independently inspected original sources before assessing the stronger autonomous-loss claim. Historical research files were read but not changed.

## Judgment

**Retain, with narrowed attribution.** The general conservation/Lie-rank framework, pure-SLD foliation, matrix mobility, and Lyapunov/area machinery are established. The explicit full first-bracket rank for the finite blend is a model-specific calculation within that framework. The stronger candidate is the all-dimensional local submersion of hidden endpoints obtained by varying a single constant SPD quadratic loss weight, with initialization and a small nonzero target direction fixed. None of the inspected theorems directly supplies that conclusion. This is a bounded non-coverage finding, not proof of originality.

The area-map spectral characterization developed during this audit is itself a direct application of a classical SPD-product theorem. It should be used as a credited lemma or alternative proof, not advertised as new matrix theory.

## Theorem-level comparison

1. [Marcotte, Gribonval and Peyre, *Keep the Momentum*, arXiv 2405.12888v1](https://arxiv.org/html/2405.12888v1), Theorem 3.7 and Section 4.5/Appendix L. The theorem counts local independent laws by the codimension of the generated Lie distribution when rank is locally constant, and explicitly covers fields `M grad phi_i`. Thus the inference from a full rank calculation to no common laws is already covered. Appendix L proves preservation for a natural metric built from the same observations, using containment of its image in the original gradient span. The finite blend here uses full-matrix representation geometry with diagonal-only loss; that image-containment argument does not identify its horizontal distribution with the diagonal one. A new conservation-law methodology is not a viable claim.

2. [Marcotte, Peyre and Gribonval, *Intrinsic training dynamics*, arXiv 2508.07370v1](https://arxiv.org/html/2508.07370v1), Theorem 3.3 and Proposition 3.7. These identify intrinsic recoverability through conservation laws and its relation to the Frobenius property. Consequently, loss-independent state recovery from observations and initialization is already a framework question. The paper studies Euclidean parameter gradient fields; its full-matrix linear-network discussion does not compute the present damped diagonal-observation distribution. The unversioned page inspected first has renumbered Theorem 2.17/Proposition 2.21; the preserved v1 and citations here use the earlier numbering. Neither version supplies a constant-quadratic-weight endpoint submersion.

3. [Nagaoka and Fujiwara, *Autoparallelity of Quantum Statistical Manifolds*, arXiv 2307.03431v3](https://arxiv.org/html/2307.03431v3), Theorem 8.6, equation (8.8). For a fixed Hermitian score space, integrability of its SLD-parallel distribution is equivalent to a double-commutator inclusion at every density matrix. Commuting scores make that obstruction vanish. This covers the pure-SLD integrable mechanism (with trace normalization handled separately). It does not apply the resolvent `K(K+lambda Id)^(-1)` or establish the finite-blend transition. The candidate cannot claim that commuting SLD scores define an integrable family for the first time.

4. [Li, Wang, Lee and Arora, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Paper-Conference.pdf), Theorem 4.8, Theorem 4.9 and Corollary 4.11. The sufficient mirror-flow result allows arbitrary initialization and time-dependent losses under regularity and commuting parametrization assumptions. The necessary condition involves higher Lie brackets. Corollary 4.11 already rules out full matrix-factorization mirror-flow representations in general. These are stronger precedents than an informal assertion that noncommuting dynamics can be path dependent. They do not supply the finite-blend diagonal computation or replace a proof for fixed autonomous quadratic weights; the allowed time-dependent losses are a larger class.

5. [Ciaglia and Di Nocera, *Group Actions and Monotone Quantum Metric Tensors* (2022)](https://www.mdpi.com/2227-7390/10/15/2613), Sections 3--4 and Proposition 1. The source constructs metric/group-action correspondences for monotone quantum metrics and the associated expectation gradients, in arbitrary finite dimension. Its hypotheses use the monotone metric kernel family. The present metric kernel `1+lambda/(x+y)` is not homogeneous of degree -1: simultaneous scaling of `x,y` leaves its Euclidean summand unchanged. Therefore membership in that family cannot be silently assumed. This does not prove that no broader group action exists; it only blocks the direct specialization of the inspected theorem.

6. [Zhang and Peyre, *Global Exponential Convergence of Two-Layer Linear Network Training*, arXiv 2610.09356v1](https://arxiv.org/html/2610.09356v1), Appendix D, Proposition D.1 and Corollary D.2. Split pure-Bures covariance fields generate an indefinite-orthogonal Lie algebra, and the complete generic covariance-level laws are functions of a signed spectrum. This is a very close methodological precedent, including explicit orbit dimensions and covariance laws. Its observations are an off-block predictor and its fields are infinitesimal congruences. Replacing them by the finite resolvent fields and a diagonal-only objective changes both algebra and orbit. Its momentum damping is also distinct from the present first-order metric damping.

7. [Sabbagh and Georgiou, *Isospectral Steering*, arXiv 2604.23895v1](https://arxiv.org/html/2604.23895v1), Problem 1, Remark 3 and Section 4. This studies covariance endpoints of the Lyapunov ODE with measurable time-dependent symmetric gains of prescribed spectrum, implemented by rotating their eigenvectors. It supplies substantial covariance reachability theory, and even observes the unique constant symmetric gain joining two SPD covariances without the isospectral constraint. These are not constant loss-Hessian controls: in our flow the gain is state-dependent and constrained by the diagonal residual and fixed weight. Neither arbitrary covariance steering nor controlled-path Lie rank establishes the claimed static-loss submersion.

8. [Friz, Gassiat and Lyons, *Physical Brownian motion in magnetic field as rough path*, arXiv 1302.2531](https://arxiv.org/html/1302.2531), Proposition 1 and Theorem 1. The antisymmetric matrix `(MC-CM*)/2`, with `C` a Lyapunov covariance, already describes an area correction, and the paper explicitly connects area to nonlinear driven systems. Hence the combination of a Lyapunov solution and an antisymmetric area should not be represented as a new construction. Its stochastic homogenization theorem has different controls and covariance forcing; it does not state surjectivity of the deterministic rank-one-impulse area map over SPD decay matrices or the autonomous finite-blend endpoint theorem.

9. [Ballantine, *Products of positive definite matrices II* (1968)](https://msp.org/pjm/1968/24-1/pjm-v24-n1-p02-s.pdf), Corollary 1.1', journal page 8 (PDF page 2). A real matrix is a product of two SPD matrices exactly when it is real diagonalizable with positive eigenvalues. The paper explicitly identifies this as a known consequence of standard diagonalization/inertia facts. This directly yields the spectral feasibility lemma below. Its theorem is not an endpoint-control theorem, but it substantially reduces the algebra needed for the candidate.

## Direct reduction of the cyclic area range to known matrix factorization

This derivation is supplied to prevent overstating the novelty of the intermediate area lemma. Fix a nonzero `d in R^n`, take SPD `C`, and define

`G(C)=integral_0^infinity exp(-Ct) d d^T exp(-Ct) dt`,

`J(C)=(C G(C)-G(C) C)/2`.

The Gramian solves `C G+G C=d d^T`. It is SPD precisely when `d` is cyclic for `C`. Thus on the cyclic domain

`Q := d d^T/2 + J = C G`

is a product of two SPD matrices. Ballantine's criterion gives real positive diagonalizable spectrum. In fact this spectrum is simple: write `S=C^(1/2) G C^(1/2)>0` and `u=C^(1/2)d`, so `CS+SC=uu^T`. Compression to an eigenspace of `S` with eigenvalue `s>0` gives `2s P C P=(Pu)(Pu)^T`; the left side is positive definite on that eigenspace, whereas the right has rank at most one. The eigenspace is therefore one-dimensional.

Conversely, given skew `J` such that `Q=d d^T/2+J=V L V^(-1)` has positive simple real eigenvalues, choose any positive diagonal `H` and set

`C=V L H V^T`, `G=V^(-T) H^(-1) V^(-1)`.

Both are SPD, `CG=Q`, `GC=Q^T`, and hence `CG+GC=dd^T`. By Lyapunov uniqueness this `G` is exactly `G(C)`, so `d` is cyclic for `C` and its area is `J`. Therefore the exact cyclic-area image is

`{J skew: dd^T/2+J has positive simple real spectrum}`.

Simple real positive eigenvalues persist under small real perturbations. Locally choose the eigenvectors smoothly; the displayed factorization then gives a smooth local right inverse for the area map. The factorization can be matched to any given cyclic `C` by choosing the appropriate positive diagonal `H`. This provides an alternative to the derivative-kernel proof of submersion.

The algebra is a direct application of classical factorization plus a short rank-one observation. What remains specific is coupling this map to the finite-blend curvature tensor and controlling the infinite-time endpoint expansion. This argument describes the second-order area image, not the exact nonlinear endpoint set for arbitrary finite targets.

The full image, including noncyclic `d`, has the analogous criterion that `Q` be real diagonalizable with nonnegative eigenvalues. For `Q=V L V^(-1)`, `L>=0`, take `C=V V^T>0` and `G=V^(-T) L V^(-1)>=0`. This is the semidefinite specialization of Ballantine's Theorem 1 and its stated real analogue, rather than a new factorization theorem. The positive eigenvalues are simple by the same compressed rank-one argument; repeated zeros are allowed. The exact differential-rank refinement in `static_loss/THEOREM.md` is more informative at noncyclic points than the cyclic local inverse above, and must be assessed separately from the already classical factorization criterion.

## Limits of the audit

No inspected original gives the proposed finite-blend autonomous endpoint theorem. Searches do not certify novelty. A wider review could uncover an equivalent generic connection result for autonomous quadratic feedback or a pre-existing specialization of the area-map argument. Treat the result as a rigorously scoped potential contribution, with strongest emphasis on autonomous-loss realizability rather than the known Lie-rank formalism.

The two-dimensional equal-diagonal higher-bracket calculation is a useful exact refinement but, by itself, a short model-specific derivative computation; it should not be counted as a separate major innovation. General connected-support controllability must not be asserted from these sources or from dense generic rank.

Local smooth endpoint dependence requires exponential transverse stability in a fixed SPD neighborhood and uniform bounds on derivatives; finite-time smooth ODE dependence alone is insufficient. The audit did not independently prove that analytic estimate; it is an explicit obligation in the theorem/review files.

## Provenance

`SOURCES.json` gives URLs and SHA-256 hashes of downloaded primary-source bytes. The HTML hashes identify the retrieved representation, not a permanent canonical paper hash. MDPI pages were inspected with the browser research tool; direct snapshot requests were denied by their server, so no full-byte hash is claimed for those pages. `SEARCH_LEDGER.md` records the discovery queries and unsuccessful retrievals.
