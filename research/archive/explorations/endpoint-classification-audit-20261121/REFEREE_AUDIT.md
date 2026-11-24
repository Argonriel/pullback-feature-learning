# Referee audit: static-weight endpoint classification

Date: 2026-11-21. The two prior reports were reread in full, and their SHA-256 hashes were compared with the archived copies under `explorations/endpoint-selection-deepening-20261121/`. The read copies match the archive. Repository `AGENTS.md` was read. No old report, output, or source was modified.

## Referee verdict on the established first-order result

No substantive mathematical gap was found in the stated classification of the **first damping derivative**, with variable small targets and fixed positive quadratic loss weights. The proof is materially more specific than a generic non-Hessian or arbitrary-control argument. The two constants, the invariant-subspace sufficiency argument, endpoint differentiation, and the fixed-weight realizability argument survive this review.

The result should still be presented with strict scope. Neither numerical examples nor failure to find matching papers establishes originality. The algebraic interpretation has now been completed and independently checked: it makes finite-damping neutrality a consequence of central conserved coordinates. `FINITEDAMPING_AUDIT.md` gives that stronger result and its limits. This substantially narrows what can be claimed as distinctive about the first-order statement.

## Items checked adversarially

| Potential failure | Review outcome |
|---|---|
| A missing factor from Frobenius coordinates or symmetric matrix units | The prior scalar null coordinate and the raw-versus-orthonormal star-plane amplitudes are consistently distinguished. The quartic raw coefficient gains `1/4` when both orthonormal target coordinates use the same stated amplitude. |
| Wrong signs or constants in the static-weight expansion | Independent differentiation reproduces `-5/(288s^3) P_N ad_E^2(V)` and `+13/(576s^4) P_N {E,ad_E^2(V)}`. Equivalently the cubic term is positive `5/(288s^3)` times `P_N[[E,V],E]`. |
| Higher Taylor terms not controlled by the two closure conditions | For fixed `Z`, `A=ad_Z^2` and `C={Z,·}` are commuting self-adjoint operators. `S` reduces `A`; `CA S subset S` makes `C` preserve `S intersect range(A)`. The exact matrix-function difference vanishes on `ker(A)`, so joint spectral calculus controls every higher term. This is a complete sufficiency argument, not an extrapolation from two numerical coefficients. |
| The claimed logarithmic primitive is chosen without justification | On the convex affine SPD leaf, the radial integral is forced to be `P_N[log(X)-log(s)I]/2`. Closedness therefore forces the asserted derivative. |
| Illegal interchange of the infinite-time limit and damping differentiation | The compact-SPD continuation argument gives uniform exponential residual decay on a compact positive-definite weight family and a bounded damping interval. The exact normal velocity divided by damping is integrably dominated, so dominated convergence suffices for the first derivative. |
| A covert use of arbitrary time-dependent losses | The explicit perturbation `B_tau=I+tau D` is frozen throughout each flow. Its exact matrix-exponential path has variation `t exp(-t) DE`. Any desired `DE=V` is realized by a self-adjoint `D`; a sufficiently small fixed `tau` keeps the loss positive. |
| Claiming a diagonal-weight witness in the wrong coordinates | The diagonal witness is valid for a fixed orthonormal measurement basis and a target that may be chosen. The reports correctly withhold it for arbitrary nonorthogonal measurement coordinates. |
| Claiming every target detects an obstruction | The reports explicitly give a fixed-target counterexample: diagonal weights can leave the path radial while full SPD weights expose the obstruction. The classification quantifies over all small targets. |
| Treating Lie-triple closure as sufficient | The star plane in `Sym(3)` passes the cubic condition but fails the quartic one, with an actual fixed-diagonal-weight witness. The correction is genuine. |

One wording correction is appropriate in future versions: the sentence saying “the path and its weight derivative decay exponentially” should say **the path velocity, the residual, and the weight derivative decay exponentially**. The path itself converges to a nonzero target. The integral and remainder arguments use the decaying quantities, so this is not a flaw in the theorem.

The uniformity statement must remain as written in the detailed proofs: uniform estimates apply on compact SPD weight families. There is no asserted uniform damping neighborhood over weights of arbitrarily large condition number. The necessity argument only needs a neighborhood of the identity, while the exact closed-form response extends to other fixed regular weights.

## Independent positivity lemma from the two conditions

The following useful structural consequence was derived without invoking an abstract Jordan classification. Let `P_N` denote the Frobenius-orthogonal normal projection and assume, for all `Z,U in S`,

\[
A_ZU=[Z,[Z,U]]\in S,
\qquad C_ZA_ZU=\{Z,[Z,[Z,U]]\}\in S.
\]

Then

\[
N_Z:=P_N(Z^2)\quad\text{commutes with every }U\in S.
\]

Proof: self-adjointness of `A_Z` means `P_N` commutes with `A_Z`. Since `A_Z(Z^2)=0`, we have `A_ZN_Z=0`. Polarizing the second condition at `U+tZ`, with tangent argument `Z`, gives

\[
P_N(C_ZA_UZ-C_UA_ZU)=0.
\]

The product-rule identities

\[
A_U(Z^2)=C_ZA_UZ+2[U,Z]^2,
\qquad
C_UA_ZU+2[U,Z]^2=A_Z(U^2)
\]

then imply

\[
\langle N_Z,A_U(Z^2)\rangle
=\langle N_Z,A_Z(U^2)\rangle=0.
\]

Because `A_U` preserves `S`, this equals `⟨N_Z,A_U N_Z⟩`. Positivity gives

\[
0=\langle N_Z,A_U N_Z\rangle=\|[U,N_Z]\|_F^2.
\]

Polarization shows that every normal defect `P_N(Z circ U)` of a Jordan product commutes with all generators. It therefore commutes with the entire unital Jordan algebra `J=Jordan(S,I)`. The algebra agent independently obtained the same lemma by a differentiated commutator argument.

This lemma is stronger than the original local curvature observation. By itself it does **not** prove that all of `J intersect S-perp` is central: extending from degree-two defects to the entire generated algebra requires an additional argument. The algebra agent subsequently supplied that argument through the orthogonal irreducible modules of `[S,S]`; this review independently checked its scalar-multiplication and central-idempotent steps. The completed bridge is summarized in `FINITEDAMPING_AUDIT.md`. It must not be replaced by the generally unjustified assertion that multiplication by every generator preserves `S`.

## Existing conservation-law machinery

[Marcotte, Gribonval, and Peyre, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/c7bee9b76be21146fd592fc2b46614d5-Paper-Conference.pdf), Proposition 2.5, characterize conserved functions by orthogonality of their gradients to the relevant vector fields. Their paper uses Lie-algebra and Frobenius methods to determine conservation laws and explains that a sufficient collection can determine implicit bias without a full mirror representation. These principles already cover the general strategy of deriving endpoint invariance from partial conserved coordinates.

[Marcotte, Gribonval, and Peyre, ICML 2024](https://arxiv.org/html/2405.12888v1), sections 2–3, explicitly extends this framework to non-Euclidean gradient flows. In particular the mobility-weighted span of gradients, rather than the Euclidean gradient span alone, governs conservation. Its dataset-independent smooth-mobility assumptions fit a fixed matrix measurement problem with the mobility used here. The general conservation method is therefore prior work; any distinct contribution must be the specific algebraic classification and its consequences, not the existence of conservation-law analysis.

Neither paper, in the portions inspected, states the exact two polynomial closure conditions for this inverse-Lyapunov correction. This bounded observation is not an originality certificate. The broader algebraic and information-geometric literature audit remains decisive.

## Audit disposition

Retain the first-order result as mathematically sound within its stated assumptions. The algebraic reduction is now complete; the primary-source priority comparison remains essential before describing this as a new theory. The completed central-invariant argument excludes a passing-space second-order-damping counterexample whenever the flows converge to SPD interpolants. No broad numerical search was run.

Archived report hashes:

- `GENERAL_GEOMETRY.md`: `22fcbfc0118e43951148d8f03c6aafcdf47dec6a5a99a53e74b23d276a2ca5a0`
- `STATIC_WEIGHT_THEORY.md`: `a210f3bb14bbb4b5dd60bf12cd9ebd05f8f899e36947b7c57f72d00dc643a410`
