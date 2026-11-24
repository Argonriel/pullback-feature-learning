# Discrete factor steps: a damping cancellation and its precise boundary

Date: 2026-11-21. This is one narrow research candidate, not several candidates. The continuous classification and its Bregman endpoint law are reused from the preceding archive, not counted as new work. Generic finite-step drift, step-size-dependent implicit bias, and invariant-preserving projection schemes were rejected as novel directions before the numerical checks.

## Assessment

The surviving statement is a local, finite-step theorem for the actual simultaneous balanced two-factor pullback update: its quadratic small-target endpoint error changes sign with damping, and vanishes for every stable fixed positive quadratic loss weight at `lambda=2s`, where `X0=sI`. This cancellation is not exact endpoint neutrality. An equal-trace loss-weight example has a strictly positive fourth-order endpoint difference at the cancellation value, certified by an exact rational calculation.

The elementary mechanism is that the continuous invariant has zero quadratic curvature in the balanced factor update directions at this initialization and damping. It falls within established conservation-error analysis. The inspected original papers do not state the particular damped-pullback cancellation, its all-weight quantifier, or the coefficient below, but absence from this bounded search is not a priority certificate. The defensible scope is a potentially original specialized theorem/corollary. No useful new optimizer, generalization benefit, or large-target guarantee is established.

## Setup and exact product step

Let `S` be a real subspace of symmetric matrices satisfying the preceding continuous-neutrality classification, and let `J=Jordan(S,I)`. Thus `N_J=J intersect S-perp` consists of central matrices. Write `P_N` for Frobenius projection on `S-perp`; expressions below remain in `J`.

Set

\[
X_0=sI,\qquad
L_B(X)=\tfrac12\langle P_S(X-sI)-\epsilon E,
 B[P_S(X-sI)-\epsilon E]\rangle,
\]

where `E in S`, `B:S->S` is fixed positive definite, and `s>0`. Let `G=grad L_B`, `K_X(U)=XU+UX`, and damping `lambda>0`. The full factor-coordinate pullback metric is `J_factor^T J_factor+lambda I`; damping is independent of the loss weight `B`.

For transpose-balanced factors `X=W2 W1`, `W2=W1^T`, the pullback direction is

\[
U=(K_X+\lambda I)^{-1}G,\qquad
\Delta W_1=-W_1U,\quad \Delta W_2=-UW_2.
\]

One simultaneous factor step of length `eta` therefore induces exactly

\[
X_+=(I-\eta U)X(I-\eta U)
   =X-\eta K_XU+\eta^2UXU.                 \tag{1}
\]

This is a congruence/Bures exponential expression, not a new retraction. It differs from Euler applied directly to the product flow. Jordan closure ensures `(1)` remains in `J`, although it need not remain in `sI+S`.

For central `N in N_J`, put

\[
F_N(X)=\langle N,X+\tfrac\lambda2\log X\rangle.
\]

When `I-eta U` is positive definite, blockwise log-determinant identities and `<N,G>=0` give the exact one-step drift

\[
F_N(X_+)-F_N(X)
=\eta^2\operatorname{tr}(NXU^2)
 +\lambda\operatorname{tr}N[\log(I-\eta U)+\eta U].       \tag{2}
\]

Only the central projection of the logarithmic identity is used. There is no assertion that `log(AXA)=log X+2log A` for noncommuting matrices.

## Finite-step small-target endpoint theorem

Define

\[
d=2s+\lambda,\quad m=2s/d,\quad
R=I_S-\eta mB,
\]

and suppose `0<eta*m*lambda_min(B)` and `eta*m*lambda_max(B)<2`. Keep `s,lambda,eta,B,E` fixed and let `epsilon` tend to zero. For sufficiently small `|epsilon|`, the iterates stay SPD on the positive congruence branch, converge to an interpolating endpoint, and that endpoint depends smoothly on `epsilon`.

Let `X_cont(epsilon E)` be the continuous-flow Bregman endpoint, which is independent of `B` in this classified space. Then

\[
\boxed{
X_{\infty}^{\rm factor}(\epsilon;B)-X_{\rm cont}(\epsilon E)
=\epsilon^2\eta^2m\frac{s-\lambda/2}{d^2}
 P_N\sum_{k=0}^{\infty}(BR^kE)^2+O(\epsilon^3).
}                                                        \tag{3}
\]

The order of limits is the converged discrete endpoint first, followed by `epsilon->0`; there is no `eta->0` limit. Constants depend on the stability margin and fixed parameters. No uniform neighborhood over weights approaching instability or arbitrary conditioning is claimed.

Proof. To first order, `X_k=sI+epsilon(I-R^k)E+O(epsilon^2)`, and `U_k=-epsilon BR^k E/d+O(epsilon^2)`. Expanding `(2)` gives a quadratic invariant increment

\[
\epsilon^2\eta^2(s-\lambda/2)d^{-2}
 \langle N,(BR^kE)^2\rangle.
\]

At `sI`, the derivative of `X+lambda log(X)/2` is `m^{-1}I`. Both limiting points interpolate the same measurements, so their difference is normal. Summing the increments and multiplying by `m` yields `(3)`.

The infinite sum and smooth endpoint require justification. In a sufficiently small neighborhood of `sI`, the measurement residual update is `r_+=Rr+O((||X-sI||+||r||)||r||)`, while `||X_+-X||<=C||r||`. Choose a contraction factor strictly between `||R||` and one. Starting with `r_0=-epsilon E`, a small-target bootstrap gives geometric residual decay, an `O(epsilon)` bound on total displacement, SPD preservation, and a convergent endpoint. Differentiating the recursion through any fixed finite order gives geometrically decaying forcing multiplied by a polynomial in the iteration index; these series are summable. This validates the endpoint Taylor expansion and its remainder for fixed stable parameters.

Consequences:

1. At `lambda=2s`, all quadratic endpoint discretization error vanishes, for every fixed stable positive `B` and every small target direction `E` in the classified space. The error is `O(epsilon^3)`, not identically zero.
2. Against a positive semidefinite central normal `N`, the leading error relative to the continuous endpoint has the sign of `s-lambda/2`, provided at least one summand is visible to `N`. This is a leading small-target sign, not a global finite-target inequality.
3. The universal cancellation is a property of simultaneous factor steps and the stated damping convention. It must not be transferred to product-space Euler, other factorization depths, other damping conventions, or arbitrary initializations without a separate calculation.

## Structural distinction, away from the cancellation value

Within the continuously neutral classified spaces, fix `s>0`, `eta>0`, and `lambda>0` with `lambda!=2s`. Exact endpoint independence from every stable fixed positive quadratic weight for every sufficiently small target holds if and only if `S` itself is Jordan closed.

Sufficiency: when `X=sI+Z`, `Z in S`, the resolvent of `K_X+lambda I` maps `S` to `S`. Thus `U in S`, `K_XU in S`, and `UXU=sU^2+UZU in S`; the last inclusion follows from the Jordan quadratic-representation identity. All iterates stay in `sI+S`, whose interpolating point is unique.

Necessity: if `S` is not Jordan closed, choose `E in S` with `P_N E^2!=0`, and take `B=beta I_S`. Formula `(3)` reduces to

\[
X_\infty^{\rm factor}-X_{\rm cont}
=\epsilon^2\frac{\eta(s-\lambda/2)}{d^2}
\frac{\beta}{2-\eta m\beta}P_NE^2+O(\epsilon^3).
\]

Two distinct stable positive values of `beta` give different quadratic coefficients. This proves the mathematical necessity, but its witness is an ordinary loss/step-size rescaling and is not promoted as an independent innovation. The result does not classify exact neutrality at the exceptional value `lambda=2s`.

## Equal-trace weight witness and exact critical counterexample

Use `Sym(2)`, `s=1`, the orthonormal basis

\[
A_1=\operatorname{diag}(1,-1)/\sqrt2,\qquad
A_2=\begin{pmatrix}0&1\\1&0\end{pmatrix}/\sqrt2,
\]

`S=span(A1,A2)`, target `epsilon(A1+A2)`, step `eta=1/2`, and compare `B0=I` with `B1=diag(1/2,3/2)`. These weights have identical trace; both runs fully interpolate. Hence this comparison is neither a fixed-training-time comparison nor a scalar loss-rate rescaling. The measured coordinates of both endpoints are exactly `(epsilon,epsilon)`; only their scalar identity component differs.

For general damping, the quadratic coefficient of `a_infty(B1)-a_infty(B0)`, where `a=tr X/2`, is

\[
\frac{\eta(s-\lambda/2)}{2d^2}
\left[\frac{1/2}{2-\eta m/2}
 +\frac{3/2}{2-3\eta m/2}
 -\frac{2}{2-\eta m}\right].                 \tag{4}
\]

The bracket is positive by strict convexity in the weights. At damping `0.5`, `(4)` is `+0.00297619047619048`; at damping `4`, it is `-0.000188217579521926`.

At `lambda=2`, orthogonal conjugation that negates both Pauli directions makes the scalar endpoint an even function of `epsilon`. The quadratic coefficient vanishes and the next possible term is fourth order. It is exactly

\[
a_\infty(B_1)-a_\infty(B_0)
=\frac{155954674477253}{261630534078728100}\epsilon^4
 +O(\epsilon^6).                              \tag{5}
\]

The coefficient is approximately `0.000596087436913324`, and is strictly positive.

For an independent algebraic certificate of `(5)`, write `X=aI+v1 A1+v2 A2`, `U=u0I+u1 A1+u2 A2`. At critical damping, the first-order coefficients are `v_i(k)=1-q_i^k`, `u_i(k)=-b_i q_i^k/4`, where `q_i=1-b_i/4`. Quadratic conservation gives `a2(k)=||v(k)||^2/8`; the scalar part of the second-order `U` is `u0,2=-(v dot u)/4`. The fourth-order scalar invariant increment is

\[
D_4=\eta^2\left(\frac{||v||^2||u||^2}{16}
 -\frac{(v\cdot u)^2}{4}\right)
 +\frac{\eta^3}{4}(v\cdot u)||u||^2
 -\frac{\eta^4}{8}||u||^4.
\]

All its terms are geometric sequences. Their exact sums are `-121/29600` for `B0` and `-6552163007881/2262750565005216` for `B1`. The scalar invariant has endpoint derivative two, so half their difference yields `(5)`. Higher-order trajectory coefficients cancel from this calculation.

## Minimal validation and preserved records

`factor_step_cancellation.py` executes the fixed protocol: three damping values, three target amplitudes, and two weights, for 18 tiny direct recurrences. It uses standard-library 80-digit decimal arithmetic. A separately implemented matrix Taylor-jet recurrence predicts orders two through four, and an eight-coordinate full factor-Jacobian solve checks `(1)` independently.

Results: all direct residuals were below `9.81e-66`; all matrix eigenvalues stayed above `0.90137`; every positive congruence factor stayed above `0.97763`. The exact invariant drift telescoped with maximum error `4.66e-78`. Factor-coordinate directions agreed within `6.94e-18` and induced products within `1.11e-16`. Quadratic formula/jet discrepancies were below `1.16e-14`.

| Damping | Expected limiting ratio | Ratio at epsilon=0.025 |
|---|---:|---:|
| 0.5 | `Delta/epsilon^2 -> 0.002976190476` | 0.002975936305 |
| 2 | `Delta/epsilon^4 -> 0.000596087437` | 0.000595931843 |
| 4 | `Delta/epsilon^2 -> -0.000188217580` | -0.000188000413 |

The two larger prespecified amplitudes approach the same limits consistently. `critical_quartic_certificate.py` performs only exact rational geometric summation and independently confirms the critical fourth-order coefficient. These checks validate the claimed narrow mechanism; they do not establish usefulness outside this model.

Files: `PROTOCOL.md`, `FAILED_ATTEMPTS.md`, the failed initial dependency snapshot, `run01/results.json`, `run01/verification.json`, and `critical_exact.json` preserve the full record. The only failed attempt was an unavailable optional package before any calculation ran; it was replaced by standard-library decimal arithmetic. No model, damping value, step size, target, or output was discarded.
