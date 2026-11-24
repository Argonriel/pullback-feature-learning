# Literature audit: local endpoint selection under parameter damping

Date: 2026-11-21. Status: original-source screening, not a priority certificate. This audit concerns the sharpened candidate developed during this round; it does not reopen the retired generic non-Hessian, scalar-mode, or entropy-mirror claims. No model training or numerical experiment was run for this audit.

## Verdict

The surviving claim is an exact local classification of **first-order-in-damping endpoint neutrality under fixed quadratic loss weights**, together with a cubic/quartic witness hierarchy. I did not find this statement, its finite pair of algebraic closure conditions, or its static-weight attainability bridge in the primary sources examined. It is more specific than their stated theorems and is not an immediate corollary of the closest theorem with unchanged assumptions.

The underlying ingredients are well occupied: Bures/SLD geometry, its double-commutator torsion, Jordan and Lie-triple algebra, matrix-entropy mirror descent, and general non-Hessian implicit-bias obstructions. A contribution framed merely as a new metric, noncommutativity causing path dependence, or repairing Bures with entropy should be rejected. The finite classification and the restriction to fixed losses are the potentially additional results. Their mathematical validity and uniform infinite-time perturbation justification must be established independently of this novelty audit.

## Precise candidate screened

Let the observation operator be surjective onto its measurement coordinates, with

\[
\mathcal S=\operatorname{range}\mathcal A^*,\qquad
\mathcal N=\ker\mathcal A=\mathcal S^\perp,
\qquad X_0=sI,\quad s>0.
\]

For balanced two-factor full-representation pullback flow with Euclidean parameter damping, write

\[
\dot X=-(I+\lambda T_X)^{-1}\nabla L_W(X),\qquad
T_X=(X\cdot+\cdot X)^{-1},
\quad L_W(X)=\tfrac12(\mathcal A X-y)^\top W(\mathcal A X-y).
\]

The normal-valued one-form on \(\Omega=(sI+\mathcal S)\cap\mathrm{SPD}\) is

\[
\omega_X(U)=P_{\mathcal N}T_XU,\qquad U\in\mathcal S.
\]

The team's strengthened proposed equivalence is

\[
d\omega=0
\quad\Longleftrightarrow\quad
\begin{cases}
[Z,[Z,U]]\in\mathcal S,\\
\{Z,[Z,[Z,U]]\}\in\mathcal S,
\end{cases}
\qquad \forall Z,U\in\mathcal S.
\tag{C}
\]

Here the braces are the anticommutator. The first condition is Lie-triple closure after polarization; the second is an additional degree-four condition. Neither a claim that Lie-triple closure alone suffices nor a claim of finite-\(\lambda\) neutrality is being screened as valid.

For the static-weight bridge, the team reports that, around the whitened quadratic operator \(B_0=I_{\mathcal S}\), setting \(B_\delta=I+\delta D\), \(V=DE\), and target \(sI+\varepsilon E\) yields

\[
\left.\partial_\delta C_\delta(\varepsilon)\right|_0
=-\frac{5\varepsilon^3}{288s^3}P_{\mathcal N}[E,[E,V]]
+\frac{13\varepsilon^4}{576s^4}P_{\mathcal N}\{E,[E,[E,V]]\}
+O(\varepsilon^5),
\tag{W}
\]

where \(C\) is the coefficient of \(\lambda\) in the converged endpoint. An arbitrary \(V\) can be realized by a self-adjoint \(D\) for nonzero \(E\). The proposed conclusion is equivalence between (C) and first-order endpoint neutrality for all fixed positive-definite weights and all sufficiently small targets. These protocol-dependent rational coefficients are not universal geometric constants. The proof and sign conventions belong to the mathematical audit, not this literature search.

There are three distinct statements to preserve:

1. Exactness of \(\omega\) on the affine leaf concerns arbitrary paths in that leaf.
2. Realizability by fixed weighted quadratic losses requires the separate bridge (W); arbitrary controlled paths do not establish it.
3. The conclusion concerns the first derivative at \(\lambda=0\), with the optimization time taken to infinity under justified convergence. It does not classify all finite damping strengths.

## Closest primary theorems and their boundaries

| Primary source inspected | Relevant result and assumptions | Effect on the candidate |
|---|---|---|
| [Li, Wang, Lee, Arora, NeurIPS 2022, full paper and appendix](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Supplemental-Conference.pdf) | Theorems 4.8/4.9 concern a single mirror representation valid for all time-dependent losses, with a Lie-algebra obstruction; Corollary 4.11 applies the obstruction to matrix factorization. | Occupies the universal mirror-equivalence obstruction. Failure of that equivalence does not imply different endpoints for two fixed quadratic losses, much less (C) or (W). |
| [Nagaoka and Fujiwara, arXiv:2307.03431v3, 2026 publication](https://arxiv.org/html/2307.03431) | Equation 3.26 gives SLD torsion as \(\frac14[[L_U,L_V],\rho]\). Theorem 8.6 characterizes existence of SLD e-autoparallel leaves through every faithful density matrix for a fixed observable space \(\mathcal B\): \([[A,B],\rho]\in\{\rho\circ C:C\in\mathcal B+\mathbb RI\}\) for every \(\rho\). | Closest integrability theorem. Its varying e-parallel distribution, all-state quantifier, and tangent-space condition differ from a fixed affine leaf and normal projected one-form. No degree-three-plus-four reduction or optimization endpoint classification is stated. |
| [Ciaglia, Di Cosmo, Ibort, Marmo, G-dual teleparallel connections, 2023/2024](https://arxiv.org/html/2207.08694) | Proposition 1 identifies the metric-dual connection using gradient fields; Proposition 2 connects torsion to closure of the dual one-forms. Section 4 identifies Bures gradient commutators with unitary fundamental fields and contrasts commuting BKM gradients. | Generic closedness/torsion machinery and the Bures commutator mechanism are known. It does not supply the special finite algebraic classification on \(sI+\mathcal S\), nor fixed-loss endpoint witnesses. |
| [Tumpach and Larotonda, 2024, Theorem 3.3](https://link.springer.com/article/10.1007/s41884-024-00146-z) | For the **affine-invariant** metric on SPD, a linear symmetric-matrix space is Lie-triple closed precisely when its exponential image is totally geodesic. | Establishes classical status of the Lie-triple criterion. It concerns \(\exp\mathcal S\), not \(sI+\mathcal S\), and a different metric. It cannot justify dropping the quartic condition. |
| [Ciaglia, Jost, Schwachhofer, Information Geometry, Jordan Algebras, and a Coadjoint Orbit-Like Construction, SIGMA 2023](https://arxiv.org/html/2112.09781) | For finite-dimensional formally real Jordan algebras, Section 4 constructs orbit metrics including Bures-Helstrom; Lemma 4.1 and Definition 4.2 use Jordan-multiplication commutators and the associated symmetric-pair structure. | Jordan/Bures and Jordan/Lie-triple connections are established. The paper does not classify arbitrary measurement subspaces by normal-form exactness, nor analyze weighted-loss training endpoints. |
| [Matsuda and Soma, Information Geometry of Operator Scaling, 2021 version / 2022 publication](https://arxiv.org/pdf/2005.01453) | Theorems 3.6 and 4.1 identify operator Sinkhorn iterations as SLD e-projections onto affine partial-trace constraints, assuming positive initial state and, for general marginals, positive marginals. Section 5 compares BKM entropy projections. | Closest existing use of SLD geometry for constrained matrix algorithms. Alternating projections and their divergence interpretation differ from continuous fixed weighted quadratic flows. No local damping hierarchy or finite closure criterion is stated. |
| [Suzuki, Classification and characterization of quantum parametric models, 2018 preprint](https://arxiv.org/pdf/1807.06990) | Definition III.4 and Theorem IV.2 characterize D-invariance of the state-dependent SLD-score tangent space, including agreement of SLD/RLD orthogonality conditions. Theorem III.6 relates it to Holevo/RLD estimation bounds for every positive cost matrix. | The cost weights concern estimation error, not a training loss. The operator and tangent coordinates differ from (C). This is not an equivalent classification under new terminology. |
| [Han, Mishra, Jawanpuria, Gao, NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/file/4b04b0dcd2ade339a3d7ce13252a29d4-Paper.pdf) | Table 1 and Equation 2 express \(g_{BW,X}(U,V)=\frac12\operatorname{tr}(T_XU\,V)\). | The damping metric is Frobenius plus a multiple of established Bures geometry. The metric itself is not an innovation. |

### Why the closest integrability theorem is not automatically the same result

The candidate is **normal projected-form exactness**, not Hessian integrability of the metric induced on \(\mathcal S\). The latter tests the tangent projection \(P_{\mathcal S}T_XdX\). For example, if \(\mathcal S=\mathrm{Sym}_n\), then \(P_{\mathcal N}=0\) and the candidate form is identically zero, even though Bures is not a Hessian metric in mixture coordinates. Calling (C) a classification of “Bures Hessian subspaces” would therefore be inaccurate.

The Nagaoka–Fujiwara theorem does provide a precise reason to expect double commutators in SLD integrability questions. To derive (C) from it one would still have to change the distribution, projection, domain, and quantifiers and prove the finite reduction. That is additional work, not a substitution of symbols into its theorem. Its all-state leaf criterion is stronger than existence of one special leaf, a distinction the authors themselves demonstrate with real density matrices in Propositions 8.11/8.12.

D-invariance is also distinct. For a real symmetric noncommuting family, the commutation operator \(\mathcal D_\rho=(1/i)T_\rho\operatorname{ad}_\rho\) sends real symmetric scores into imaginary-skew Hermitian directions. Thus D-invariance can fail for the full real trace-one family, while the candidate normal form on that affine family is exact. This comparison follows directly from the definitions and is not a new quantum-estimation claim.

## What the entropy comparison contributes

The BKM/SLD distinction is classical. [Lesniewski and Ruskai](https://arxiv.org/pdf/math-ph/9808016) develop monotone metrics and their operator-mean representation; the logarithmic and arithmetic means already give the two operators compared here. [Grasselli and Streater](https://math.mcmaster.ca/~grasselli/GrasselliStreater01.pdf) establish uniqueness of BKM under the specified monotonicity and duality of mixture/exponential connections on faithful finite-dimensional density matrices. Those assumptions should not be silently extended to uniqueness among arbitrary metrics on an unnormalized SPD cone.

The team's comparison

\[
R_X=\tfrac12D\log_X-T_X\succeq0,
\qquad R_XU=0\text{ when }[X,U]=0
\]

is a useful way to expose the closure conditions. Near \(X=sI+Z\), its first two nonzero terms are

\[
R_XU=\frac{[Z,[Z,U]]}{24s^3}
-\frac{\{Z,[Z,[Z,U]]\}}{16s^4}+O(\|Z\|^4\|U\|).
\]

The matrix-function comparison, its commutative limit, and the existence of a matrix-entropy Hessian are not by themselves novel. The possible extra result is that **the first two nonzero normal closure conditions suffice for exact normal-form closedness throughout the affine SPD leaf**, followed by the static-weight equivalence.

Likewise,

\[
\widetilde H_X=I+\tfrac\lambda2D\log_X
=\nabla^2\!\left[\tfrac12\operatorname{tr}X^2+
\tfrac\lambda2\operatorname{tr}(X\log X-X)\right]
\]

is a known-type mirror construction. Matrix entropy updates and Bregman projections appear in [Tsuda, Ratsch, Warmuth 2005](https://jmlr.org/papers/volume6/tsuda05a/tsuda05a.pdf). [Wu and Rebeschini 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/abf0931987f2f8eb7a8d26f2c21fe172-Paper.pdf) explicitly analyze matrix-sensing mirror descent with spectral entropy/hypentropy and characterize its interpolating solution; their factorized-flow equivalence uses commuting sensing matrices. The entropy metric should be a positive control with known endpoint behavior, not a separately claimed new optimizer.

## Contribution boundary and failure conditions

A defensible candidate statement would be: “For balanced damped full-representation two-factor SPD flow from an isotropic point, two finite algebraic closure conditions characterize local independence of the first damping derivative of the interpolating predictor from all constant positive quadratic observation weights; failure is detected by a cubic or quartic small-target response.”

The potentially useful distinction is between three cases: non-Lie-triple measurement spaces with cubic witnesses; Lie-triple spaces failing the additional condition with quartic witnesses; and spaces satisfying both with exact first-order neutrality. The team's three-dimensional star example is valuable because it disproves the tempting but false truncation to Lie-triple closure alone. This audit did not independently rerun that example.

The following would still invalidate or materially narrow the result:

- An unproved interchange of the damping derivative with the infinite-time limit. Finite-time perturbation theory alone is insufficient.
- A mistaken replacement of all fixed SPD loss weights with arbitrary time-dependent control. Static attainability is part of the candidate, not a consequence of generic nonintegrability.
- A diagonal-weight claim without specifying the measurement basis and whether the theorem is existential over targets. Whitening a general SPD weight usually changes the coordinate interpretation.
- Treating the classification as finite-damping path independence or as a universal data-independent strictly convex regularizer theorem.
- A source proving the same normal projected Lyapunov-form classification under equivalent coordinates. None was found in this bounded search; the absence of a match is not a proof of priority.

The earlier caution from [Azulay et al. 2021, Section 5](https://proceedings.mlr.press/v139/azulay21a/azulay21a.pdf) still applies: an integrating factor/time change can remove a non-Hessian obstruction without changing endpoints. Consequently the explicit static witness, rather than a generic non-Hessian label, is essential. [Mirrorless Mirror Descent](https://proceedings.mlr.press/v130/gunasekar21a/gunasekar21a.pdf) already supplies the ambient non-Hessian-gradient-flow viewpoint.

## Search record and remaining uncertainty

Queries covered Bures/SLD Hessian integrability and affine subspaces; Lie-triple/Jordan closure and anticommutators; SLD autoparallelity and D-invariant quantum models; fixed weighted quadratic implicit bias; damping, endpoint sensitivity, and higher-order matrix-factorization response. Technical conclusions above use primary papers or author manuscripts, not summaries from aggregators.

An additional current primary preprint, [Aladrah et al., arXiv:2601.06597v3](https://arxiv.org/html/2601.06597), now titled *Inverse design of implicit bias by symmetry, stochasticity and holomorphic parametrizations*, treats finite-step stochastic charge selection and constructed parametrizations. It is distinct from deterministic damped-flow endpoint derivatives and does not close this gap. The current version was checked because the earlier title sounded closer.

The search located no original result with the combination of fixed affine mixture-coordinate measurement space, normal projection, Bures-versus-entropy operator difference, a degree-three-plus-four necessary-and-sufficient reduction, and static positive quadratic-weight witnesses. The justified disposition is **retain this narrow theorem candidate for proof review and minimal validation**. It is not yet a claim of broad algorithmic benefit, practical superiority, or publication-level novelty.
