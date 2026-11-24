# A finite criterion for first-order endpoint sensitivity

November 21, 2026. Scope: real symmetric matrices, an isotropic initial product, the full representation pullback metric, and the first derivative of the interpolating product with respect to damping at zero.

**Result:** Lie triple closure alone is insufficient. The corrected criterion consists of two polynomial closure conditions, of degrees three and four. They are necessary and sufficient for the relevant normal-projected one-form to be closed on the whole positive-definite affine leaf. Together with a fixed-quadratic-loss realization argument, they also characterize local invariance of the first damping response under fixed positive-definite loss weighting. The result has a proof within the stated model; originality remains a candidate subject to the separate literature audit.

This is not a classification of finite-damping endpoints, a Hessian characterization of the whole Bures metric, or a generalization theorem for the original residual network.

## 1. Literature boundary checked before validation

The inverse Lyapunov operator is established matrix geometry. [Malagò, Montrucchio, and Pistone, *Wasserstein Riemannian Geometry of Positive Definite Matrices*](https://arxiv.org/abs/1801.09269) explicitly express the Wasserstein metric tensor through the Lyapunov equation. No novelty is claimed for that tensor or its differential.

The closest apparent Lie-triple theorem has a different domain and metric. [Tumpach and Larotonda, *Totally geodesic submanifolds in the manifold SPD of symmetric positive-definite real matrices*](https://arxiv.org/html/2405.20784v1), Theorem 3.3, presents the classical equivalence between Lie triple closure and total geodesicity of **exp(S)** for the affine-invariant metric. Here the domain is the **affine** section sI+S and the object is a normal projection of an inverse-Lyapunov one-form. Transferring their equivalence would be incorrect; the explicit counterexample below shows why.

The distinction between a general reparameterized gradient flow and a loss-independent mirror flow is already central to [Li, Wang, Lee, and Arora, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Paper-Conference.pdf). Our statement is narrower: it concerns the measurement-normal part of one metric correction, rather than an ambient mirror representation. Its mathematical ingredients are standard differential and spectral calculus. The potentially distinct object is the finite algebraic criterion plus its realization by fixed quadratic losses, not noncommutativity or path dependence by themselves.

## 2. Setup and the quantity being classified

Let V=Sym(n) with Frobenius inner product, let S be any real linear subspace, and let P_N be the orthogonal projection onto N=S-perp. Fix s>0 and define the convex open affine domain

    Omega_s = (s I + S) intersect SPD(n).

For X positive definite, define the invertible Lyapunov operator and its inverse by

    K_X(U)=XU+UX,                 T_X=K_X^(-1).

The N-valued one-form of interest is

    omega_X(U)=P_N T_X(U),       U in S.

For balanced two-factor full-representation pullback, the product flow is

    X_dot=-(Id+lambda T_X)^(-1) G(X),      G(X) in S.

At lambda=0, the trajectory lies in sI+S. Under conditions justified locally in Section 5, the first damping response of its endpoint is

    C_G = d/dlambda X_infinity(lambda)|_(lambda=0+)
        = - integral_gamma omega,

where gamma is the zero-damping trajectory. Its measured component is zero because all interpolating endpoints have the same measurements. Thus closedness of omega concerns whether the **first normal endpoint response** depends on the path.

This normal projection is essential. If S=Sym(n), then P_N=0 and the criterion below holds trivially, although the ambient Bures metric is generally not Hessian. Closedness here must not be described as a full mirror-flow theorem.

## 3. Exact finite criterion

For Z symmetric, write

    A_Z(U)=[Z,[Z,U]],
    C_Z(U)=ZU+UZ.

Here C_Z is an operator, unrelated to the endpoint-response symbol C_G. The following conditions are equivalent:

1. omega is closed on Omega_s, equivalently its integral is independent of every piecewise C1 path with fixed endpoints in Omega_s.
2. For all X in Omega_s and U in S,

       P_N T_X(U) = (1/2) P_N Dlog_X[U].

3. For every Z,U in S, both polynomial closure conditions hold:

       [Z,[Z,U]] in S,                                      (A)
       {Z,[Z,[Z,U]]} = [Z^2,[Z,U]] in S.                     (CA)

Condition (A) is the usual Lie triple condition in its repeated-variable form. Condition (CA) is additional. Neither condition requires I to belong to S. Because the conditions do not depend on s, closedness for one positive isotropic scale implies closedness for every positive isotropic scale.

When these conditions hold, the primitive normalized to vanish at sI is

    F(X)=(1/2) P_N [log X - (log s) I].

### Proof: the primitive is forced

Convexity makes Omega_s simply connected. If omega is closed, integrate it along the radial segment X(t)=sI+tZ. Since X(t) commutes with Z,

    T_(sI+tZ)(Z)=(1/2)(sI+tZ)^(-1) Z.

The integral is exactly F(X). Differentiating the forced primitive gives condition 2. Conversely condition 2 says omega=dF, so it is closed. This also excludes choosing an alternative primitive to evade the subsequent conditions.

### Proof: necessity of the two polynomial conditions

Define

    R_X=(1/2)Dlog_X - T_X.

Expanding at sI gives, for sufficiently small Z,

    R_(sI+Z)(U)
      = A_Z(U)/(24 s^3)
        - C_Z A_Z(U)/(16 s^4)
        + O(||Z||^4 ||U||).

For completeness, the coefficient of degree k in Z is

    (-1)^k/(2 s^(k+1)) *
       [ (1/(k+1)) sum_(j=0)^k Z^j U Z^(k-j)
         - (1/2^k) sum_(j=0)^k binomial(k,j) Z^j U Z^(k-j) ].

Degrees zero and one vanish. If P_N R_(sI+tZ)(U)=0 for every sufficiently small t, its degree-two and degree-three Taylor coefficients vanish. This is precisely (A) and (CA). Every Z in S can be scaled into the positive-definite neighborhood, so the quantifiers are over all Z,U in S, not merely small Z.

### Proof: sufficiency, with no higher conditions needed

Fix any Z in S for which X=sI+Z is positive definite. On Sym(n), the operators A=A_Z and C=C_Z are self-adjoint and commute. A is positive semidefinite.

Condition (A) implies that S reduces A. Therefore the spectral pseudoinverse A-plus preserves S, and

    S = (S intersect ker A) direct-sum (S intersect range A).

Let S_active=S intersect range A. Condition (CA) implies that C preserves S_active: for v in S_active,

    Cv = CA A-plus v in S,

and commutation of A and C also keeps Cv in range A. Thus both A and C preserve S_active.

In a basis diagonalizing Z, the symmetric matrix units simultaneously diagonalize A and C. If their eigenvalues are a=(z_i-z_j)^2 and c=z_i+z_j, respectively, the eigenvalue of R_X is

    h_s(a,c) = atanh(sqrt(a)/(2s+c))/sqrt(a) - 1/(2s+c),

with the continuous value zero when a=0. Positivity of X guarantees 2s+c>sqrt(a). Hence R_X is a joint spectral function of A and C and vanishes on ker A. It preserves S_active and annihilates S intersect ker A. Consequently R_X(S) is contained in S, proving condition 2.

This step is why the degree-three and degree-four matrix conditions suffice: invariance under A and CA controls the entire joint functional calculus on the noncommuting part. An infinite list of Taylor checks is unnecessary.

## 4. Lie triple closure fails at the next order

Set n=3 and use unnormalized symmetric matrix units

    U=E12+E21,      V=E13+E31,      W=E23+E32,
    S=span(U,V).

Writing J=[U,V]=E23-E32 gives [J,U]=-V and [J,V]=U. Thus S is Lie triple closed. However,

    A_U(V)=V,       C_U A_U(V)=W outside S.

Condition (CA) fails. This space is a counterexample to the conjectured Lie-triple-only classification, despite its standard symmetric-space interpretation under another metric.

The exterior derivative convention used here is

    domega_X(U,V)
      = P_N[-T_X K_U T_X(V) + T_X K_V T_X(U)].

Its leading term at an isotropic point is

    domega_(sI+Z)(U,V)
      = P_N [[U,V],Z]/(8s^3) + O(||Z||^2).

The sign and coefficient agree with direct expansion. The leading term vanishes on a Lie triple system. In the star example, the next term does not vanish: at X=sI+aU, with 0<|a|<s,

    domega_X(U,V)
      = [ a^2(8s^2+a^2) /
          (2(s^2-a^2)(4s^2-a^2)^2) ] W.

This is a positive nonzero coefficient. One can also directly compare the candidate primitive. The coefficient of W in omega_X(V)-(1/2)P_N Dlog_X[V] is

    -a/(4s^2-a^2) - log(1-a^2/s^2)/(4a),

whose leading term is a^3/(16s^4).

## 5. Fixed quadratic losses, rather than arbitrary path controls

Let E in S be a target direction and B a fixed positive-definite self-adjoint operator on S. Consider

    loss_(B,epsilon)(X)
      = (1/2)<e,B e>,
    e=P_S(X-sI)-epsilon E,       X_initial=sI.

An orthonormal measurement basis identifies B with a fixed positive-definite matrix of residual weights. General B permits mixed residual weights. Diagonal B corresponds to independent positive weighting in that basis.

At zero damping, the exact path is

    X_0(t)=sI+epsilon(Id-exp(-Bt))E.

### Local existence, interpolation, and endpoint differentiation

The first damping response need not be assumed for sufficiently small targets. Fix B in any compact positive-definite family (in particular, a neighborhood of Id for the necessity argument) and lambda in [0,lambda_0]. While ||X-sI||_F<s/2, all eigenvalues of X exceed s/2. The mobility (Id+lambda T_X)^(-1) has eigenvalues between s/(s+lambda_0) and 1. Therefore the loss decays exponentially at a uniform positive rate. The velocity is bounded by ||B e||, whose time integral is at most a constant times |epsilon| ||E||.

Choose |epsilon| ||E|| small enough that this total path length is less than s/2. A continuation argument keeps the whole trajectory in the positive-definite neighborhood. It converges, and its measured residual tends to zero. These bounds are uniform in the stated B and lambda sets.

The exact normal velocity is

    P_N X_dot
      = lambda P_N (K_X+lambda Id)^(-1) B e.

After division by lambda, the integrand is bounded by a uniform integrable exponential. Finite-time continuity of the ODE in lambda and dominated convergence give the right derivative

    C_B(epsilon E)
      = integral_0^infinity P_N T_(X_0(t)) B e_0(t) dt
      = - integral_(gamma_B) omega.

This argument justifies the first derivative and an o(lambda) endpoint remainder locally. No unproved exchange of differentiation and infinite-time convergence is required. Stronger uniform higher-order lambda estimates are not needed for this result.

### Equivalence with local invariance under fixed weights

The two algebraic conditions are equivalent to the following local statement: for every sufficiently small target in S and every fixed B in a neighborhood of Id in SPD(S), C_B is independent of B. Sufficiency follows directly from the closed one-form and its forced primitive. In particular,

    C_B(epsilon E)
      = -(1/2)P_N[log(sI+epsilon E)-(log s)I]

under the criterion. This common response can be nonzero; independence of weighting does not mean absence of a damping effect.

Necessity uses fixed losses, not time-varying controls. Let B_delta=Id+delta D, where D is self-adjoint, and put V=DE. For all sufficiently small delta, B_delta is positive definite. With f(t)=1-exp(-t) and h(t)=t exp(-t), the path perturbation is

    Z_delta(t)=epsilon[f(t)E+delta h(t)V]+O(delta^2 epsilon).

The mixed response derived in the companion [static-weight analysis](STATIC_WEIGHT_THEORY.md) is

    d/ddelta C_(B_delta)(epsilon E)|_(delta=0)
       = -5 epsilon^3/(288s^3) P_N A_E(V)
         +13 epsilon^4/(576s^4) P_N C_E A_E(V)
         +O(epsilon^5).

Both constants were independently checked. They use

    integral_0^infinity f f' h dt = 5/36,
    integral_0^infinity f^2 f' h dt = 13/144.

Any V can be realized as DE by a self-adjoint D when E is nonzero. Failure of (A) therefore produces a fixed-weight witness at cubic order in target size. If (A) holds but (CA) fails, a fixed-weight witness appears at quartic order. This establishes necessity with actual frozen positive-definite quadratic weights and arbitrarily small targets.

For any fixed **orthonormal** measurement basis, one can also obtain an existential witness using diagonal weights: choose an obstruction direction E with every coordinate nonzero, possible by a small perturbation because the obstruction is polynomial and nonzero on an open set, and take D=diag(V_i/E_i). This claim allows selection of the target. It does not say that every target witnesses failure. It also does not establish necessity for diagonal weights in an arbitrary fixed nonorthogonal measurement basis.

## 6. Controls and stronger special cases

**Commuting but not Jordan closed.** In Sym(3), take S spanned by diag(1,0,1) and diag(0,1,1). These commute, while their Jordan product is outside S. Both conditions nevertheless hold because all commutators vanish. Omega is closed and generally nonzero. The corresponding diagonal factor flow also has an ordinary separable mirror potential, giving an independent stronger explanation in this special case.

**Traceless but not Jordan closed.** Let S be all traceless symmetric matrices. Conditions (A) and (CA) hold because both expressions are commutators and hence traceless. The normal form is exact by the trace identity

    tr T_X(U)=(1/2)tr(X^(-1)U).

There is also a finite-lambda conserved quantity whenever the gradient is traceless:

    tr X + (lambda/2) log det X = constant.

For a fixed traceless interpolating component Y, write X=Y+cI. The derivative of the conserved quantity with respect to c is n+(lambda/2)tr((Y+cI)^(-1))>0 on the SPD interval. Thus any interpolating limit is uniquely determined by the constraint and initial invariant, independently of loss weighting. This stronger finite-lambda conclusion uses a separate invariant, not the general first-order theorem.

**Jordan closed without containing I.** If S is Jordan closed, K_(sI+Z)=2s Id+K_Z preserves S. Its positive-definite inverse also preserves S, so omega is identically zero. It is unnecessary to assume I is in S. A concrete control is the symmetric top-left 2-by-2 block embedded in Sym(3). Its finite-lambda product flow stays on sI+S, so its measurement-normal component remains fixed whenever the trajectory exists.

These examples separate three notions: zero projected motion, a nonzero but exact first-order response, and independently justified neutrality at finite damping. They must not be conflated.

## 7. Minimal validation and record preservation

The identities were derived before their numerical checks. `general_geometry_check.py` performs full nine-dimensional Kronecker Lyapunov solves, a spectral matrix-log derivative, direct exterior derivatives, and finite differences on predefined matrices. It also checks all polynomial coefficients by finite polarization: quadratic dependence on Z uses basis elements and pairwise sums; cubic dependence additionally uses pairwise differences and triple sums. The tangent argument is linear, so checking its basis completes these finite certificates. Floating-point certificates support the explicit algebraic proofs; they do not replace them.

For the star counterexample at s=1, the exact curvature coefficients are .0106813080 at a=.2 and .0527033730 at a=.4. Full-matrix formula errors are below 7.2e-17; finite-difference errors are below 1.2e-11. Lie triple closure residual is 3.2e-16, while the unnormalized extra-condition obstruction has norm sqrt(2). The complete polarization check accepts condition (A) and rejects (CA).

The commuting, traceless, and nonunital Jordan controls pass both polynomial conditions, with residuals at most 1.54e-15. Their projected matrix-log comparison residuals are at most 7.4e-17. The two non-Jordan controls have nonzero omega, confirming that exactness is weaker than zero motion. A fixed non-Lie-triple control rejects the cubic condition and verifies the sign and coefficient of the linear curvature expansion at three shrinking scales.

The initial check output and its exact source were preserved in `general_geometry_checks/`. The finite-polarization addition used a fresh `general_geometry_certificates/` output directory, with source SHA-256 recorded in both JSON files. No training experiment or numerical search was used for these geometry checks. The companion static-weight and parent-level finite-damping checks are recorded separately.

## 8. What is established and what remains open

Within the stated isotropic SPD model, the two-condition local classification has a complete proof, a minimal counterexample to the weaker Lie-triple conjecture, and a realization by fixed positive-definite quadratic losses. This is a substantive advance over the previous single 2-by-2 numerical example.

The result concerns the normal endpoint derivative at lambda=0. It does not classify all finite-lambda endpoint effects, arbitrary nonisotropic initialization, restricted diagonal weighting in every measurement coordinate system, general deep nonlinear networks, or predictive benefit. The exact two-condition statement has not been identified in the primary sources reviewed so far, but this is bounded novelty evidence, not a claim of exhaustive priority. The separate literature audit should govern how strongly the research contribution is presented.
