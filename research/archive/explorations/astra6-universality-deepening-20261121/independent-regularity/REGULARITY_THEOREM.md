# Exact universality without analyticity, and a sharp locality obstruction

Date: November 21, 2026 (America/Los_Angeles).

Status: independent mathematical audit and strengthening. This note does not certify priority. The associator-ideal classification used below is inherited from the previously reviewed algebraic reduction; the new arguments here concern continuous spectral kernels, exact response inversion, and sharp counterexamples to weakening the nondegenerate-germ assumption.

## 1. Setup and quantifiers

Let `E=Sym(n,R)` with the Frobenius inner product, `S` any real linear subspace, `P` its orthogonal projection, `Q=I-P`, and `s>0`. Let

    m(x,y)=a^(-1) h(v),   a=(x+y)/2,   v=(x-y)/(x+y),

where `h:(-1,1)->R` is continuous and even, `h(0)=1`. Positivity `h>0` is required for the optimization interpretation, but not for the algebraic cancellation argument. Let `M_X` be the corresponding symmetric spectral multiplier. Put

    L(v)=atanh(v)/v,  L(0)=1,
    g(v)=L(v)-h(v),
    R_X=Dlog_X-M_X.

The key nondegeneracy assumption is

    (G) For every eta>0, some |v|<min(eta,1) satisfies g(v)!=0.

Thus the discrepancy is not identically zero on any neighborhood of the diagonal. Condition (G) allows a discrepancy flat to every order at zero. Real analyticity and globally non-BKM imply (G), but (G) is substantially weaker.

Under (G), the following are equivalent:

1. There is `r>0` such that `Q R_(sI+Z) U=0` for every `Z,U in S` with `||Z||_F<r`.
2. For every `Z,U in S`, both `ad_Z^2 U` and `{Z,ad_Z^2 U}` belong to `S`.
3. With `J=Jordan(S,I)`, `S` is an associator ideal of `J`, equivalently `J intersect S-perp` is central in `J`.
4. `Q R_X U=0` for every `X in (sI+S) intersect SPD` and `U in S`.

If `h` is additionally `C^1`, these are also equivalent to local closedness of the normal one-form `Q M_X dX`, to global closedness on the affine SPD slice, and to the following precisely quantified fixed-weight condition:

5. There exist `r>0` and `0<eta<1` such that, for every `Delta in S` with `||Delta||_F<r` and every self-adjoint SPD `B:S->S` with `||B-I||<eta`, the first-damping endpoint responses satisfy `C_B(Delta)=C_I(Delta)`.

An equivalent weaker-looking condition is: there exists a single `r>0` such that, for every such target and every self-adjoint `D:S->S`, `partial_tau C_(I+tau D)(Delta)|_0=0`. Thus all first variations at the identity on a common target ball suffice. Once the space is classified, response equality holds for every compact family of SPD weights on a target ball that may depend on that family. Arbitrary fixed SPD weights follow as a corollary. No uniform finite-damping claim over unbounded weight condition numbers is needed.

The theorem is about exact identities. It neither gives a uniform condition number nor supplies a finite Taylor-order recognition rule in this larger smooth class.

## 2. A continuous spectral-pair separation lemma

For `a>0` and real `b`, define the germ

    F_(a,b)(t)=(1+bt)^(-1) g(at/(1+bt)).

The zero-gap class `a=0` has identically zero germ. Under (G), no positive-gap class has zero germ, and distinct pairs `(a,b)` have distinct germs.

Proof. Suppose `F_(a,b)=F_(c,d)` for all sufficiently small signed `t`, with `a,c>0`. Set

    w=at/(1+bt),   r=c/a>0,   delta=(d-b)/a,
    H(w)=w g(w),   T(w)=r w/(1+delta w).

Direct substitution, without differentiating `g`, gives

    g(w)=(1+delta w)^(-1) g(T(w)),
    H(T(w))=r H(w).

If `0<r<1`, sufficiently small `w` has iterates

    T^j(w)=r^j w/[1+delta w(1-r^j)/(1-r)],

all staying in the identity's neighborhood and converging to zero. Since `g(0)=0`,

    H(w)=[T^j(w)/r^j] g(T^j(w)) -> 0.

This forces `g` to vanish on a neighborhood of zero, contradicting (G). If `r>1`, exchange the two pairs to obtain the previous case.

If `r=1` but `delta!=0`, then for `delta w>0`,

    T^j(w)=w/(1+j delta w) -> 0,
    H(T^j(w))=H(w).

Continuity forces `H(w)=0` on one side; evenness of `g` forces the same on the other side. Equivalently, one can use inverse iterates on the opposite side. Again (G) fails. Hence `r=1` and `delta=0`, which means `a=c,b=d`. Finally, if a positive-gap germ were identically zero, its argument map is a local bijection, so `g` would be zero near zero. This proves the lemma.

The lemma rules out both flat and oscillatory counterexamples; no finite nonzero derivative, sign condition on `g`, or asymptotic expansion was used.

## 3. Necessity of both polynomial closure conditions

Fix `Z in S`, with distinct eigenvalues `z_i` and spectral projectors `P_i`. Along `X=sI+tZ`, the multiplier of `R_X` on the symmetric `(i,j)` Peirce component is

    s^(-1) F_(a_ij,b_ij)(t/s),
    a_ij=|z_i-z_j|/2,  b_ij=(z_i+z_j)/2.

Zero-gap components all have multiplier zero. Positive-gap components are distinguished by the preceding lemma unless they have the same unordered pair of eigenvalues. There are only finitely many such classes.

Normal cancellation implies that every sufficiently small `R_(sI+tZ)` leaves `S` invariant. These operators are commuting self-adjoint operators. For each pair of different joint classes, select a sufficiently small `t` that distinguishes them. The resulting finite family has distinct joint eigenvalue vectors on distinct classes. A real linear combination of this finite family can be chosen to have distinct eigenvalues on all those classes, by avoiding finitely many proper hyperplanes of coefficients. Lagrange interpolation then recovers each joint spectral projector as a polynomial in that linear combination. Every such projector preserves `S`.

On a positive-gap class, the operators `A_Z=ad_Z^2` and `C_Z A_Z`, where `C_Z(U)=ZU+UZ`, have scalar values

    4 a_ij^2,   8 b_ij a_ij^2,

respectively. Both vanish on the combined zero-gap class. They are therefore linear combinations of the recovered projectors, and both preserve `S`. Since `Z` was arbitrary, the two closure conditions follow.

This is an exact finite-dimensional argument. It remains valid with repeated matrix eigenvalues. It does not assume that one sample value of `t` distinguishes all classes; that shortcut is unjustified for merely continuous kernels.

## 4. Sufficiency and the Peirce-invariance check

The inherited algebraic bridge is the equivalence of the two polynomial generator conditions with `S` being an associator ideal of `J=Jordan(S,I)`. It is proved in the old `endpoint-classification-audit-20261121/ALGEBRA_REDUCTION.md`; it is not inferred just from the classical theorem that all derivations of `J` are inner.

For completeness, the spectral step used after that bridge needs no analyticity. If `X in J` is SPD, its spectral projectors are polynomials in `X`, so belong to `J`. For a projector `P`, the quadratic representation `U_P(Y)=PYP` preserves `J`. Consequently

    P_i Y P_j+P_j Y P_i
      =U_(P_i+P_j)(Y)-U_(P_i)(Y)-U_(P_j)(Y)

belongs to `J` whenever `Y in J`. Every symmetric spectral multiplier, including `M_X` and `Dlog_X`, therefore preserves `J`.

If `N in J intersect S-perp` is central, it commutes with `X`, and normalization gives

    M_X N=X^(-1)N=Dlog_X N.

Self-adjointness yields `<N,R_X U>=0`. Components of `S-perp` outside `J` pair to zero because `R_X U in J`. Thus normal cancellation holds globally on the affine SPD slice.

For positive kernels, the same argument gives conservation along any well-defined damped trajectory in `J`:

    P_(J intersect S-perp)[X+lambda log X]=constant.

If such a trajectory converges to an SPD interpolating point, its endpoint is the unique Bregman projection for

    Phi_lambda(X)=||X||_F^2/2+lambda tr(X log X-X).

This is a conditional finite-damping statement. No general global-convergence claim for all continuous positive kernels is added here.

## 5. Closedness and the exact primitive

When `h is C^1`, the finite-dimensional operator map `X->M_X` is `C^1`. One clean representation on all real matrices is

    A_X=(L_X+R_X)/2,
    V_X=(L_X-R_X)(L_X+R_X)^(-1),
    M_X=A_X^(-1)h(V_X).

The operators commute and are self-adjoint, `spec(V_X) subset (-1,1)`, and finite-dimensional `C^1` scalar functional calculus is `C^1`. Evenness preserves the symmetric-matrix subspace. This is a standard functional-calculus ingredient, not a new regularity theorem.

On either a small convex ball about `sI` in the affine slice or the entire affine SPD slice, any closed normal form has its primitive fixed by radial integration. Along `X(u)=sI+uZ`, the tangent commutes with `X`, so every normalized kernel acts as `X^(-1)` on that tangent. Hence the primitive is exactly

    Q(log X-log(s)I).

Differentiating it shows that closedness is equivalent to normal cancellation. For merely continuous `h`, one can instead state the corresponding exact-primitive identity, avoiding an undefined classical exterior derivative.

## 6. Exact fixed-weight response inversion

Choose orthonormal measurement coordinates on `S` and let `B:S->S` be any fixed self-adjoint positive-definite operator. The optimization flow is

    Xdot=-(I+lambda M_X)^(-1) B(P(X-sI)-Delta),
    X(0)=sI.

The needed local endpoint statement follows from an elementary bootstrap, with constants uniform for `B` in any compact set of SPD operators. Choose a closed Frobenius ball of radius `rho<s/2` about `sI`. On this ball `M_X` is bounded. For all sufficiently small signed `lambda`, the inverse `K_X=(I+lambda M_X)^(-1)` has uniform positive bounds `kappa_- I<=K_X<=kappa_+ I`. Let `beta_- I<=B<=beta_+ I` on the chosen compact weight family and set `r(t)=P(X(t)-sI)-Delta`. As long as the trajectory stays in the ball,

    d/dt [<r,B r>/2]=-<B r,K_X B r>
      <=-kappa_- ||B r||^2
      <=-2 kappa_- beta_- [<r,B r>/2].

Hence `||r(t)||<=sqrt(beta_+/beta_-)||Delta|| exp(-kappa_- beta_- t)`. The velocity is bounded by `kappa_+ beta_+ ||r(t)||`, so its entire path length is at most a constant times `||Delta||`, uniformly in this weight family and damping interval. Choosing the target ball sufficiently small closes the bootstrap strictly inside the chosen SPD ball. The trajectory exists for all time and converges to an SPD interpolating endpoint.

The normal velocity obeys the exact identity

    Q Xdot=lambda Q M_X K_X B r.

After division by `lambda`, its integrand has an integrable exponential bound uniform in the same parameters. Continuous dependence of finite-time solutions and dominated convergence therefore justify differentiation of the endpoint at `lambda=0`, without assuming a differentiable stable-foliation theorem. Tangential endpoint coordinates are exactly `Delta` for all these parameters. The zero-damping path is

    Z_B(t)=(I-exp(-Bt))Delta,

and the normal endpoint derivative is

    C_B(Delta)=-integral_0^infinity Q M_(sI+Z_B(t)) Zdot_B(t) dt
      =-Q(log(sI+Delta)-log(s)I)
        +integral_0^infinity Q R_(sI+Z_B(t)) Zdot_B(t) dt.

These formulas are therefore actual endpoint derivatives in the local regime, and involve no assumed finite-damping endpoint universality. They are also well-defined path integrals whenever the zero-damping path remains SPD.

Let `B_tau=I+tau D`, with `D` self-adjoint, `Delta=epsilon E`, and `V=DE`. At `tau=0`,

    Z=epsilon f E,   partial_tau Z=epsilon h_0 V,
    f=1-exp(-t),     h_0=t exp(-t).

For every symmetric `Z`, normalization implies `R_(sI+Z) Z=0`. Differentiating this identity, rather than differentiating a matrix eigenbasis, gives

    (DR_X[V])Z=-R_X V.

It follows exactly that

    partial_tau C_(B_tau)(epsilon E)|_0
      =epsilon integral_0^1 K(u) Q R_(sI+epsilon u E)V du,
    K(u)=1+log(1-u)/u < 0,    0<u<1.

Every `V in S` is realizable by a self-adjoint `D` for fixed nonzero `E`; explicitly

    D=(V tensor E+E tensor V)/||E||^2
      -<E,V>(E tensor E)/||E||^4.

For sufficiently small fixed `tau`, `I+tau D` is SPD. Thus this is a genuine static-weight variation, not an arbitrary controlled path.

### An elementary injectivity lemma

For any scalar continuous `f:[0,d]->R`, define

    (Tf)(x)=integral_0^1 kappa(u)f(xu)du,
    kappa(u)=-K(u)=sum_(n>=1) u^n/(n+1).

If `Tf(x)=0` for every `0<x<=d`, then `f` is zero. No analyticity, finite nonzero jet, or fixed sign is required.

To prove it, put `x=d exp(-t)` and

    q(t)=exp(-t) f(d exp(-t)),   t>=0.

Then `q in L^2(0,infinity)`. Multiplying the transformed equation by `exp(-t)` yields

    sum_(n>=1) (n+1)^(-1) R_n q=0,
    (R_n q)(t)=integral_0^infinity exp(-nv)q(t+v)dv.

The series defines a bounded operator because `||R_n||<=1/n` and `sum 1/[n(n+1)]=1`. If `r=R_nq`, then `r'=nr-q` in the weak sense, `r in H^1 intersect L^2`, and integration by parts gives

    integral_0^infinity q r
      =n||r||_2^2 + |r(0)|^2/2 >=0.

Taking the inner product of the operator equation with `q` is justified by absolute convergence. Every summand is nonnegative; in particular the `n=1` term vanishes, forcing `R_1q=0`, then `q=0`. Continuity recovers `f=0` everywhere, including zero.

Apply this lemma componentwise to `f(x)=Q R_(sI+xE)V`, and also to the negative ray when needed. Vanishing weight variations for all sufficiently small targets imply exact local normal cancellation. The converse follows directly from the path-integral formula. This completes the static-weight equivalence under `C^1` regularity.

## 7. Smooth flat kernels: an infinite-jet limitation outside the monotone class

Define

    psi(v)=exp(-1/v^2) for v!=0,  psi(0)=0,
    h(v)=L(v)+psi(v).

This gives a positive, symmetric, normalized, homogeneous `C^infinity` kernel with a nonzero discrepancy germ. Every Taylor coefficient at the diagonal equals BKM, yet the theorem says that its exact local neutral spaces are precisely the associator ideals.

For any fixed SPD `B`, the discrepancy response is flat in target amplitude: for a bounded target direction and small `epsilon`, the zero-damping path stays within `O(epsilon)` of `sI`, all spectral ratios are `O(epsilon)`, and

    ||R_X|| <= C exp(-c/epsilon^2).

The path has length `O(epsilon)`, so the first-damping endpoint contrast is `O(epsilon exp(-c/epsilon^2))`. Thus all target-amplitude Taylor coefficients agree with BKM although exact nonneutrality persists for every non-associator space.

This statement is deliberately restricted to the larger smooth class. The constructed kernel is not claimed to be a quantum monotone/Petz kernel; normalized operator-monotone kernels are analytic. It does not strengthen the analytic-class conclusion into an assertion that arbitrary finite diagnostics fail.

## 8. A positive smooth plateau defeats even global universality

The nonzero-germ assumption cannot be replaced by mere global inequality `m!=BKM`, even if closedness is required on the entire affine SPD slice.

Fix `delta=1/2`, and define a smooth even nonnegative function

    psi_delta(v)=0                         if v^2<=delta^2,
    psi_delta(v)=exp(-1/(v^2-delta^2))      if v^2>delta^2,
    h(v)=L(v)+psi_delta(v).

This kernel is positive, symmetric, normalized, homogeneous and globally non-BKM, but it equals BKM for `|v|<=delta`.

Let `D=diag(1,-1)`, `H=[[0,1],[1,0]]`, `K=4`, `n=6`, `s=1`, and

    S={diag(a D+b(I_2+H), K a,-K a,K b,-K b): a,b real}.

For `X=I_6+Z` in the full affine SPD slice, positivity of the four scalar blocks forces `|a|,|b|<1/K`. The two active-block eigenvalues are

    1+b +/- sqrt(a^2+b^2),

so their ratio coordinate satisfies

    |v|=sqrt(a^2+b^2)/(1+b)
       <sqrt(2)/(K-1)=sqrt(2)/3 < 1/2=delta.

Every tangent in `S` is block diagonal with the same block structure. The active-block multiplier therefore equals BKM, and each scalar block has a commuting tangent. All cross-block tangent entries are zero. Consequently

    R_X U=0

for every `X in (I_6+S) intersect SPD` and every `U in S`. In particular the normal one-form is globally exact.

Nevertheless, with the two generators

    Z_1=diag(D,K,-K,0,0),
    Z_2=diag(I_2+H,0,0,K,-K),

one has

    ad_(Z_1)^2 Z_2=diag(4H,0,0,0,0) notin S.

The scalar tails force the coefficients of any candidate linear combination of the generators to be zero, while the active block is nonzero. Thus even Lie-triple closure fails. This is an exact counterexample, not a numerically inferred one.

The role of the scalar tails is essential: they bound the active spectral ratios throughout the entire affine SPD slice. A two-by-two plateau example only disproves the local-to-global implication; this six-by-six construction also disproves the proposed global classification without (G).

## 9. What remains known, new, and unclaimed

Known or inherited ingredients: normalized homogeneous spectral coordinates; finite-dimensional functional calculus and Peirce invariance; the previously established generator-to-associator-ideal bridge; radial primitives; ordinary first variation of an ODE endpoint; and elementary positive resolvent energy identities.

Potential contribution developed here: a continuous-germ separation theorem replacing finite-jet extraction, its combination with exact inversion of the static-weight response, and a sharp global counterexample when the nonzero-germ assumption is dropped. These answer a meaningful question: whether exact neutrality classification is a consequence of analyticity and whether arbitrarily flat kernels can truly evade the static-weight test. They show it is not a consequence of analyticity, while also identifying a precise obstruction to removing all local distinguishability assumptions.

No claim of a previously unknown associator-ideal class, a new Petz classification, a new quadrature scheme, a numerical performance advantage, or publication priority is made. No global convergence theorem for arbitrary kernels is supplied. A literature audit of exact equivalent formulations remains necessary.
