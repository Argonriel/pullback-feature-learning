# Static quadratic weights: cubic and quartic endpoint selection

Date: 2026-11-21. This is the fixed-loss attainability component of the deeper endpoint screen. Original papers were read before the numerical checks. It strengthens the previous worked example without using time-dependent loss controls. All statements concern the first derivative of the interpolating endpoint with respect to damping at zero damping, unless explicitly qualified.

## Result and quantifiers

Let `S` be a subspace of real symmetric matrices, `N=S^perp`, and `X0=sI` with fixed `s>0`. Consider full-row-rank linear measurements with `range(A*)=S`, a sufficiently small attainable target displacement `Delta in S`, and a **fixed positive-definite quadratic loss weight matrix** `W`. The undamped Hessian on `S` is `B=A*WA`. The reduced metric is `H_X=I+lambda S_X^-1`, where `S_X(U)=XU+UX`.

Write

\[
C_B(\Delta)=\left.\frac{\partial X_\infty(\lambda,B,\Delta)}{\partial\lambda}\right|_{\lambda=0^+}.
\]

The following two algebraic conditions, required for every `E,V in S`, characterize loss-weight independence of this response for **all sufficiently small targets and all fixed SPD weights**:

\[
\text{(A)}\quad [E,[E,V]]\in S,
\qquad
\text{(CA)}\quad \{E,[E,[E,V]]\}\in S.
\]

Here `[P,Q]=PQ-QP` and `{P,Q}=PQ+QP`. Condition (A) is equivalent by polarization and Jacobi to Lie-triple closure. Condition (CA) is an additional condition; Lie-triple closure alone does not suffice.

If (A) fails, some fixed SPD weight perturbation and some small target give a **cubic** residual-amplitude response. If (A) holds but (CA) fails, a fixed SPD perturbation and target give a **quartic** response. If both hold, the response is weight-independent. For any fixed orthonormal measurement basis, positive **diagonal** weights suffice for the existential witnesses when targets may vary. For an arbitrary fixed nonorthogonal measurement map, only the full-SPD-weight necessity statement is established here.

This theorem does not say every target or every pair of weights exposes a failed condition. It also does not establish independence of the exact endpoint at arbitrary nonzero damping. The dependence order is: differentiate the endpoint in damping first; expand in small target amplitude next. A sufficiently small, fixed weight perturbation is selected to make the relevant coefficient nonzero; it is not changed over optimization time.

## Literature boundary

[Li, Wang, Lee, and Arora, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Supplemental-Conference.pdf) characterize mirror-flow equivalence using commuting parametrizations. Their universal-flow statements do not automatically imply a fixed-quadratic-weight endpoint witness. The argument below explicitly constructs such fixed losses instead of invoking arbitrary controls. [Han et al., NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/file/4b04b0dcd2ade339a3d7ce13252a29d4-Paper.pdf), Table 1, identifies the inverse-Lyapunov Bures-Wasserstein metric; it is existing geometry. [Azulay et al., ICML 2021](https://proceedings.mlr.press/v139/azulay21a/azulay21a.pdf), section 5, shows why failure of a direct Hessian representation alone cannot establish endpoint sensitivity: time reparameterization can preserve implicit bias.

The ingredients used here—matrix-function expansions, ODE sensitivity, polynomial polarization, and invariant-subspace arguments—are standard. The specific optimization criterion and static-weight attainability proof must still be compared with the closest original implicit-bias and matrix-geometry results before claiming publication novelty. The present audit establishes a mathematical result within the screened model, not a generalization claim or a completed novelty certificate.

## Endpoint derivative and its small-target expansion

Set `Delta=epsilon E`. For a fixed positive operator `B` on `S`, the exact undamped path is

\[
X_B(t)=sI+\epsilon Z_B(t),\qquad
Z_B(t)=(I-e^{-Bt})E.
\]

Because the eigenvalues of `I-e^{-Bt}` lie in `[0,1]`, `||Z_B(t)||_F<=||E||_F`. Thus all these undamped paths stay uniformly SPD for `|epsilon| ||E||_F < s/2`, independent of the weight choice. For each fixed `B`, exponential residual decay and the previous interior-SPD perturbation argument justify the damping derivative and give

\[
C_B(\epsilon E)=-P_N\int_{\gamma_B}S_X^{-1}\,dX.
\]

No uniform small-damping neighborhood over arbitrarily ill-conditioned weights is claimed. Uniform estimates over a compact SPD weight family are sufficient for the expansions used below.

Expanding the Lyapunov inverse around `sI` gives

\[
C_B(\epsilon E)
=\frac{\epsilon^2}{4s^2}P_N(E^2)
-\frac{\epsilon^3}{8s^3}P_N\left(E^3+\int_0^\infty Z_B\dot Z_BZ_B\,dt\right)
+O(\epsilon^4).
\]

The first-order term vanishes because the path lies in `S`. The quadratic term is the endpoint differential `d(Z^2)` and therefore is weight-independent. This proves that the first possible weight contrast near isotropic initialization is cubic, rather than quadratic.

For direct coefficient evaluation, choose an orthonormal eigenbasis `F_i` of `B` in `S`, eigenvalues `b_i>0`, and write `E=sum_i e_i F_i`. Then

\[
\int Z_B\dot Z_BZ_B\,dt
=\sum_{ijk}e_ie_je_kF_iF_jF_k\left[
1-\frac{b_j}{b_i+b_j}-\frac{b_j}{b_k+b_j}
+\frac{b_j}{b_i+b_j+b_k}\right].
\]

This is a finite expression for the cubic coefficient; it needs neither ODE training nor a small-damping numerical difference.

## A useful exact differential and the two mixed coefficients

Define

\[
R_X=\tfrac12D\log_X-S_X^{-1}.
\]

Two paths with the same endpoints have damping-response difference equal to the difference of their integrals of `P_N R_X dX`. The radial path has `R_X dX=0` exactly. Around `X=sI+Z`,

\[
R_X(U)=\frac{[Z,[Z,U]]}{24s^3}
-\frac{\{Z,[Z,[Z,U]]\}}{16s^4}
+O(\|Z\|^4\|U\|).
\]

This also makes the order distinction transparent: projected Lie triples govern cubic path effects, but a projected anticommutator can survive at the next order.

Take the fixed quadratic family

\[
B_\tau=I_S+\tau D,\qquad D=D^*,\qquad V=DE.
\]

For sufficiently small fixed `tau`, every `B_tau` is SPD. At `tau=0`, the path perturbation is

\[
Z_\tau(t)=f(t)E+\tau h(t)V+O(\tau^2),
\qquad f=1-e^{-t},\quad h=te^{-t}.
\]

Differentiating the path integral yields the explicit formula

\[
\boxed{
\left.\partial_\tau C_{B_\tau}(\epsilon E)\right|_{\tau=0}
=\frac{5\epsilon^3}{288s^3}P_N[[E,V],E]
+\frac{13\epsilon^4}{576s^4}P_N\{E,[E,[E,V]]\}
+O(\epsilon^5).}
\]

The numerical constants follow from

\[
\int_0^\infty ff'h\,dt=\frac14-\frac19=\frac5{36},
\qquad
\int_0^\infty f^2f'h\,dt=\frac14-\frac29+\frac1{16}=\frac{13}{144}.
\]

For the cubic term, the variation of `ad_Z^2(dZ)` is `f(fh'-hf') ad_E^2(V)`, whose integral is `-3 int ff'h`. For the quartic term, the extra outer `Z` supplies another `f`; the corresponding integral is `-4 int f^2f'h`. These signs give the positive `5/288` coefficient multiplying `[[E,V],E]=-ad_E^2(V)` and the positive `13/576` coefficient multiplying the anticommutator.

The remainder and its weight derivative are controlled on a small compact interval of `tau`: the path and its weight derivative decay exponentially, and the SPD neighborhood permits a uniformly convergent matrix-function expansion. A nonzero derivative of the cubic or quartic coefficient at `tau=0` ensures that some sufficiently small fixed nonzero `tau` has a nonzero coefficient. One then takes small residual amplitude; there is no assumption that an arbitrary geometric loop can be generated by fixed losses.

## Realizing the perturbations with fixed positive weights

For any nonzero `E` and any `V in S`, a self-adjoint operator satisfying `DE=V` exists. In vector notation on `S`, one choice is

\[
D=\frac{VE^*+EV^*}{\|E\|^2}
-\frac{\langle E,V\rangle}{\|E\|^4}EE^*.
\]

For a general full-row-rank measurement map, choose `W0=(AA*)^-1`. Then `A*W0 A=I_S`. Every self-adjoint `D` on `S` is realized by

\[
W_\tau=(\mathcal A\mathcal A^*)^{-1}
+\tau(\mathcal A\mathcal A^*)^{-1}\mathcal A D\mathcal A^*(\mathcal A\mathcal A^*)^{-1},
\]

which is positive definite for sufficiently small `tau`. The interpolation set stays fixed throughout.

For an orthonormal measurement basis, an existential diagonal-weight witness also follows. A nonzero obstruction is a polynomial condition and persists under a small perturbation of `E`. Choose such an `E` with all basis coordinates nonzero, then set `D_ii=V_i/E_i`. This diagonal `D` realizes `V=DE`, and `I+tau D` remains positive for small `tau`.

The target must be allowed to vary in this assertion. A fixed target supported on a single measurement eigenvector can make every diagonal-weight path radial, even when general SPD weights expose an obstruction.

## Why the two algebraic conditions suffice

This section records the finite-dimensional closure argument coordinated with the depth agent, so that the static-weight equivalence does not rest on an assumed all-path theorem.

For fixed `Z in S`, let `A_Z=ad_Z^2` and `C_Z(U)=ZU+UZ`. These operators are commuting and self-adjoint on the Frobenius space of symmetric matrices. If `A_Z S subset S`, then

\[
A_ZS=S\cap\operatorname{range}A_Z.
\]

If also `C_Z A_Z S subset S`, commutation shows that `C_Z` preserves this same range subspace. In an eigenbasis of `Z`, the multiplier of `R_{sI+Z}` is

\[
\frac{\log(s+z_i)-\log(s+z_j)}{2(z_i-z_j)}
-\frac1{2s+z_i+z_j},
\]

with the continuous diagonal value zero. It is a joint function of `A_Z` and `C_Z`, and annihilates `ker A_Z`. Therefore the two invariance conditions force `R_{sI+Z}(S) subset S`. Equivalently,

\[
P_NS_X^{-1}(U)=\tfrac12P_ND\log_X(U)\qquad(U\in S).
\]

The projected one-form on the affine SPD leaf is exact, with primitive `P_N log(X)/2`. Every undamped static-weight path then gives the same response

\[
C_B(\Delta)=-\tfrac12P_N\bigl[\log(sI+\Delta)-\log(sI)\bigr].
\]

Conversely, if responses are independent of all fixed SPD weights for every sufficiently small target, the mixed formula forces the cubic projected coefficient to vanish for every `E,V`, which is (A). It then forces the quartic coefficient to vanish, which is (CA). This proves necessity using only attainable static quadratic paths. It also shows why Lie-triple closure alone is insufficient.

## Fixed-target distinction: diagonal versus general SPD weights

In `Sym(2)`, use the orthonormal measurement basis

\[
F_1=\begin{pmatrix}1&0\\0&0\end{pmatrix},\qquad
F_2=\frac1{\sqrt3}\begin{pmatrix}0&1\\1&1\end{pmatrix},
\quad E=F_1.
\]

The projected repeated triple with `V=F2` is nonzero. A general SPD Hessian `[[1,0.25],[0.25,1]]` produces a cubic response. For every positive diagonal Hessian, however, the undamped path remains on the `F1` ray and its response contrast is zero. In fact, for this target the damped flow also remains on that ray, so the endpoint is fixed. Thus one must not replace the theorem's existential target quantifier by a statement about every given target, or silently replace general SPD weights by diagonal weights for an arbitrary measurement representation.

## Explicit quartic witness despite Lie-triple closure

Let

\[
U=E_{12}+E_{21},\qquad V=E_{13}+E_{31},\qquad
S=\operatorname{span}\{U,V\}\subset\operatorname{Sym}(3).
\]

This is a Lie-triple-closed plane: `[[U,V],U]=-V` and `[[U,V],V]=U`. Choose orthonormal measurements `U/sqrt(2), V/sqrt(2)`, initial state `sI`, and target displacement `epsilon(U+V)`. Compare the fixed positive diagonal weights `(1,1)` and `(1,2)`.

For weights `(1,2)`, the undamped path is `X=sI+epsilon(aU+bV)`, with `a=1-exp(-t)` and `b=1-exp(-2t)=a(2-a)`. The equal-weight reference is radial. Writing `v=(a,b)`, the relevant quartic anticommutator has zero first row/column and lower block

\[
Q=\|v\|^2d(vv^T)-2(v\cdot dv)vv^T.
\]

Since `b da-a db=a^2 da`, integration from `a=0` to `1` gives

\[
\int Q=\begin{pmatrix}7/15&8/105\\8/105&-7/15\end{pmatrix}.
\]

Consequently the damping-response contrast is

\[
C_{(1,2)}-C_{(1,1)}
=\frac{\epsilon^4}{s^4}
\begin{pmatrix}
0&0&0\\
0&-7/240&-1/210\\
0&-1/210&7/240
\end{pmatrix}
+O(\epsilon^5).
\]

The projected cubic term vanishes; the quartic term does not. This is an attainable fixed-diagonal-weight counterexample to a Lie-triple-only endpoint criterion. The displayed amplitude is in the **raw** matrices `U,V`. If each orthonormal measurement coordinate instead has target displacement `epsilon`, replace raw amplitude by `epsilon/sqrt(2)`, multiplying the quartic coefficient by `1/4`.

## Minimal verification and retained outputs

All numerical checks used exact undamped matrix-exponential paths; no neural model or damped ODE was trained. The predefined amplitude grid was `0.2, 0.1, 0.05, 0.025`, with 32/64-node Gauss-Legendre quadrature on fixed intervals through time 64. A stable spectral series evaluated `R_X` directly, avoiding subtraction of large common endpoint terms.

| Check | Result |
|---|---|
| Mixed cubic factor `5/288` | Central-weight errors at perturbations 0.02, 0.01, 0.005: `2.38e-7`, `5.94e-8`, `1.48e-8` |
| Fixed-target diagonal-weight control | Response contrast exactly zero in the computation |
| General SPD cubic contrast | The contrast divided by `epsilon^3` converges to the finite spectral coefficient; largest component at the smallest amplitude is `0.00141855`, versus predicted `0.00146698` |
| LTS quartic `X22` coefficient | Contrast divided by `epsilon^4`: `-0.0312949`, `-0.0296754`, `-0.0292925`, `-0.0291980`, approaching `-7/240` |
| LTS quartic `X23` coefficient | Smallest-amplitude ratio `-0.00476616`, approaching `-1/210` |
| Mixed quartic factor `13/576` | Central-weight errors at 0.02, 0.01, 0.005: `4.28e-6`, `1.07e-6`, `2.68e-7` |
| Quadrature refinement | Below `2.0e-19` for the response grid; `4.17e-17` for the mixed quartic derivative |

The second mixed-factor calculation is in a separate script and output, so the first script and its records remain unchanged. Every configured amplitude, weight perturbation, and quadrature order is retained. Both outputs report `passed` and record source hashes; the second also records its imported dependency hash. No failed or selected-away numerical run occurred in this round.

Repository sources:

- `experiments/endpoint_selection_deepening_20261121/static_weight_response.py`
- `experiments/endpoint_selection_deepening_20261121/static_weight_quartic_mixed.py`

Staged outputs:

- `static_weight_response_results.json`
- `static_weight_quartic_mixed_results.json`

The independent depth derivation and root review checked the two mixed constants and the quartic witness's signs and normalization. The remaining research question is novelty and scope beyond this first-order damping criterion—not whether the static-weight examples require arbitrary loss controls.
