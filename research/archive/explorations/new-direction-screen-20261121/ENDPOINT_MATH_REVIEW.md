# Independent review of the damped-matrix endpoint response

Date: 2026-11-21. Scope: mathematical review of the depth agent's fixed two-by-two endpoint lead. No damped trajectory or training experiment was rerun. The independent computation integrates a perturbation coefficient along exact undamped matrix-exponential paths.

## Review decision

The leading response and its sign are mathematically valid under an interior-SPD condition, and that condition can be proved for the specified undamped family. A small-positive-damping bootstrap supplies the local endpoint expansion. In the fixed two-measurement family below, every second-loss weight `w>1` has a strictly positive first-order endpoint contrast against `w=1`.

The generic integral formula is ordinary ODE sensitivity combined with projection onto an unobserved subspace. Its nonzero sign in this example comes from the exterior derivative of a non-exact scalar one-form. This is a **valid narrow lead**, not by itself a research-novel implicit-bias law. It does not establish a general measurement-geometry criterion, a global-in-damping result, a generalization benefit, or a limit uniform in vanishing initialization. Calling this calculation Bures holonomy would be unjustified without a separately defined connection and parallel-transport result.

## Conventions and the generic perturbation calculation

Work on real symmetric matrices with Frobenius inner product. Let `A` be a linear measurement map of full row rank, `W` a positive-definite loss-weight matrix, and assume interpolation is feasible. Define

\[
L_W(X)=\tfrac12\|\mathcal A X-y\|_W^2,\qquad
B_W=\mathcal A^*W\mathcal A,\qquad
N=\ker\mathcal A,
\]
\[
S_X(U)=XU+UX,\qquad T_X=S_X^{-1},\qquad
H_X=I+\lambda T_X.
\]

For the balanced two-factor model, the reduced full-representation pullback flow is

\[
\dot X_\lambda=-(I+\lambda T_{X_\lambda})^{-1}\nabla L_W(X_\lambda).
\]

The Euclidean endpoint `X_*` is the orthogonal projection of `X_0` onto the affine interpolation set; it does not depend on `W`. The exact undamped path is

\[
\bar X_W(t)=X_*+e^{-B_Wt}(X_0-X_*).
\]

The initial difference lies in `range(A*)`, so the zero eigenvalues of `B_W` cause no nondecaying residual term. Formally differentiating at `lambda=0` gives

\[
\dot\xi=-B_W\xi+T_{\bar X_W(t)}B_W(\bar X_W(t)-X_*),\qquad \xi(0)=0.
\]

Projection onto `N`, followed by an integrable infinite-time limit, yields

\[
C_W=P_N\int_0^\infty T_{\bar X_W(t)}\nabla L_W(\bar X_W(t))\,dt
=-P_N\int_{\gamma_W}T_X\,dX,
\]
\[
X_\infty(\lambda,W)=X_*+\lambda C_W+O(\lambda^2).
\]

The sign in the first expression is **positive**. The sign in the path expression is **negative**, because `dX=-grad L dt` along the undamped path. The sensitivity has no range component at the endpoint because both endpoints interpolate.

There is also an exact identity before expanding. Multiplying the flow by `H_X` and projecting gives

\[
P_N\dot X_\lambda=-\lambda P_N T_{X_\lambda}\dot X_\lambda,
\qquad
P_N(X_\infty(\lambda,W)-X_0)
=-\lambda\int_{\gamma_{\lambda,W}}P_NT_X\,dX.
\]

None of these identities is specific to the Lyapunov inverse: replace `T_X` by any sufficiently smooth matrix-valued metric perturbation and the same argument applies.

## Why the infinite-time expansion is justified locally

Assume the undamped path stays in `X >= s I` for fixed `s>0`. Let `mu` and `beta` be the smallest positive and largest eigenvalues of `B_W`. While `X_lambda >= (s/2) I`, `||T_X|| <= 1/s`; for `0<=lambda<=s`, the operator `M_lambda=(I+lambda T_X)^-1` lies between `(1/2)I` and `I`. Thus

\[
\dot L=-\langle G,M_\lambda G\rangle\leq-\mu L,
\quad
\|G(t)\|\leq\sqrt{2\beta L(0)}e^{-\mu t/2},
\quad
\int_0^\infty\|G(t)\|dt\leq K:=\frac{2\sqrt{2\beta L(0)}}\mu.
\]

For `e=X_lambda-bar X`, Duhamel's formula gives

\[
\dot e=-B_We+\lambda T_{X_\lambda}M_\lambda G_\lambda,
\qquad \sup_t\|e(t)\|_F\leq\lambda K/s.
\]

Choosing `lambda` sufficiently small, for example below `min(s,s^2/(4K))`, closes the interior-SPD bootstrap with a margin. The flow exists for all time, its gradient is integrable, and it converges to an interpolating point. On the observed range, the semigroup also gives `||P_range e(t)|| <= C lambda exp(-mu t/2)`, hence the time integral of `||G_lambda-G_0||` is `O(lambda)`.

The Lyapunov inverse is uniformly Lipschitz in this neighborhood, by the inverse-operator identity. Replacing `T_Xlambda`, `M_lambda`, and `G_lambda` in the exact projected integral by `T_barX`, `I`, and `G_0` therefore incurs `O(lambda)` inside the integral, or `O(lambda^2)` in the endpoint. This supplies the claimed local remainder rather than assuming interchange of an infinite-time limit and a derivative.

The constants depend on `s`, `mu`, `beta`, and the initialization/loss. They deteriorate as `s` tends to zero. This proof does not exchange the small-damping and small-initialization limits.

## Fixed two-by-two family and normalization

Let `0<s<1`, `X_0=s I`, and use measurements

\[
A_1=\begin{pmatrix}1&0\\0&0\end{pmatrix},\quad
A_2=\tfrac12\begin{pmatrix}1&1\\1&1\end{pmatrix},\quad
\mathcal A X=(\langle A_1,X\rangle,\langle A_2,X\rangle),\quad y=(1,1).
\]

The loss is `(r1^2+w*r2^2)/2`, with `w>0`. The orthonormal Frobenius coordinates are

\[
z=(X_{11},X_{22},\sqrt2X_{12}),\qquad
\mathcal A=\begin{pmatrix}1&0&0\\1/2&1/2&1/\sqrt2\end{pmatrix}.
\]

The null direction is `u=(0,-2,sqrt(2))`, corresponding to the matrix

\[
U=\begin{pmatrix}0&1\\1&-2\end{pmatrix},\qquad \|U\|_F^2=6.
\]

In particular, a null displacement `c U` changes the literal matrix entry `X12` by `c`, with no extra square-root factor. Along every undamped path, the null coordinate is conserved, so

\[
X=\begin{pmatrix}a&b\\b&s+b\end{pmatrix},\qquad
X_*=\begin{pmatrix}1&(1-s)/3\\(1-s)/3&s+(1-s)/3\end{pmatrix}.
\]

Define the scalar one-form

\[
\alpha=\frac{\langle U,S_X^{-1}(dX)\rangle}{6}.
\]

Writing `D=(a+s+b)(a(s+b)-b^2)`, direct solution of the two-by-two Lyapunov equation yields

\[
\alpha=-\frac{b(s+2b)}{6D}\,da
+\frac{a(s+2b-a)+b^2}{6D}\,db.
\]

Its exterior derivative is

\[
d\alpha=
\frac{s(s+3b-a)}{6(a+s+b)^2\,[a(s+b)-b^2]}\,da\wedge db.
\]

The `X12` endpoint response is `c_w=-int_gamma_w alpha`, with `C_W=c_w U` in the generic matrix formula.

## Exact undamped paths, SPD region, and strict sign

Set `q=1-s`, `d_w=sqrt(w^2-w+1)`, and `mu_±=(1+w±d_w)/2`. For `w>1`, the exact residuals are

\[
r_1(t)=-\frac q{2d_w}\left[(d_w-1)e^{-\mu_-t}+(d_w+1)e^{-\mu_+t}\right],
\]
\[
r_2(t)=-\frac q{2d_w}\left[(d_w-w)e^{-\mu_-t}+(d_w+w)e^{-\mu_+t}\right].
\]

The matrix coordinates are `a=1+r1` and `b=(1-s+2*r2-r1)/3`. Both coefficients in `r1` are strictly positive before the overall minus sign, so `a` increases strictly from `s` to `1`. Further,

\[
r_2-r_1=\frac{q(w-1)}{2d_w}
\left(e^{-\mu_-t}-e^{-\mu_+t}\right)>0\quad(t>0).
\]

Hence the path for `w>1`, viewed as a graph `b=b_w(a)`, lies strictly above the line `b=(a-s)/3` in its interior. The `w=1` path is precisely that straight segment: `r1=r2=-q exp(-3t/2)`.

The SPD condition is not merely a numerical observation. Write `X-sI=u_1(t)A1+v_2(t)A2`. Then `u_1'=-r1>0`. Also `v_2'=-w*r2`; since `d_w<w`, `r2` changes sign exactly once from negative to positive. Thus `v_2` first increases from zero and then decreases to its positive endpoint `2q/3`. Both coefficients are nonnegative, proving `X(t)>=sI`. The straight segment also lies above `sI`. Every vertical segment between the two curves is a convex combination of SPD matrices above `sI`, so the entire enclosed region satisfies the same bound.

Orient the loop by traversing the lower straight path `gamma_1` from initialization to endpoint, then traversing the upper path `gamma_w` backward. This is counterclockwise in the `(a,b)` plane. Its enclosed region has `s+3b-a>0`, positive determinant, and positive area. Consequently Stokes' theorem gives

\[
c_w-c_1
=\int_{\gamma_1}\alpha-\int_{\gamma_w}\alpha
=\iint_{R_w}\frac{s(s+3b-a)}{6(a+s+b)^2\,[a(s+b)-b^2]}\,da\,db
>0.
\]

For each fixed `s in (0,1)` and `w>1`, the local endpoint expansion therefore implies a positive endpoint contrast for all sufficiently small positive damping. It provides no explicit useful size of that damping interval, no sign theorem for every positive damping value, and no uniform statement as initialization vanishes.

## Independent numerical verification

`endpoint_response_review.py` constructs the exact undamped path by diagonalizing `B_W` on its positive-eigenvalue range. It integrates the response over fixed intervals ending at time 96, with 64 and 128 Gauss-Legendre nodes per interval. No damped ODE is integrated. At `s=0.1`:

| Quantity | Value |
|---|---:|
| `c_1` | 0.12053287230899491 |
| `c_9` | 0.12724274012884180 |
| `c_9-c_1` | **0.006709867819846879** |
| Largest 64-to-128-node change | 8.97e-14 |
| Largest bound on omitted response tail | 2.23e-20 |
| Explicit one-form versus direct Lyapunov solve error | 3.89e-16 |
| Exact curl at `(a,b,s)=(0.6,0.25,0.1)` | 0.031300374039469774 |
| Central finite-difference curl | 0.03130037405305042 |

An initial 16/32-node check failed the required refinement tolerance: its largest coefficient change was `4.798742591882732e-5`, against a required `1e-11`. Quadrature order was increased without relaxing the tolerance. `endpoint_response_quadrature_attempts.json` preserves all 16/32/64/128-node values and the exact evaluation expression. These calls were replayed with the unchanged coefficient function after the failure; the original full-file hash was not captured because the check failed before result assembly. Only the quadrature orders in `verify()` and an explanatory comment changed. The final script and main JSON retain the refined checks. The depth agent independently checked the one-form normalization, loop orientation, and strict-sign argument.

The coefficient agrees with the depth agent's small-damping endpoint contrast, whose reported ratio at damping 0.003 is about 0.00670979. The actual finite-damping trajectories remain numerical evidence. The proof here concerns the derivative and the resulting existence of a sufficiently small positive-damping interval.

## Interpretation and boundaries

For simultaneously diagonal trajectories, `S_X^-1 dX` has coordinates `dx_i/(2x_i)`. The metric is the Hessian of `sum_i[x_i^2/2+(lambda/2)(x_i log x_i-x_i)]`. The corresponding mirror invariant and unique Bregman projection explain loss-weight-independent interpolants in that commuting case. This is a sufficient mechanism for independence, not an if-and-only-if classification. Noncommuting measurement families can also have structural endpoint invariants, such as an invariant measurement span; that remains a necessary control.

A nonzero Hessian-integrability defect alone does not establish loss-weight-dependent endpoints. In particular, [Azulay et al., ICML 2021, section 5](https://proceedings.mlr.press/v139/azulay21a/azulay21a.pdf) show that a time reparameterization can recover a mirror description and preserve implicit bias in some non-Hessian cases. The strict loop-integral sign above supplies the missing endpoint statement for this family.

The name “Bures holonomy” should not be used for this calculation. The exterior derivative of the projected one-form is not, by definition, the curvature of the Levi-Civita connection, and no vector was parallel-transported around the loop. The Lyapunov inverse is associated with Bures-Wasserstein geometry, but that fact does not identify this integral with a connection's holonomy.

What survives is a concrete, analytically checked local endpoint response in a restricted matrix family. What remains missing for a substantial research claim is a broader criterion that separates weight-sensitive from invariant measurement geometries, with a precise distinction from existing mirror/reparameterization and Riemannian implicit-bias results. The generic perturbation formula and this worked sign calculation should not be represented as evidence of such novelty.

Artifacts: repository script `experiments/new_direction_screen_20261121/endpoint_response_review.py`; staged `endpoint_response_review_results.json` and `endpoint_response_quadrature_attempts.json`. The final results include the script SHA-256 and all settings. Final numerical status: `passed`.
