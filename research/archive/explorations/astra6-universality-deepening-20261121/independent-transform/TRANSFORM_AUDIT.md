# Exact fixed-weight response transform and independent audit

Date: November 21, 2026 (America/Los_Angeles).

## Scope and verdict

The first damping derivative, differentiated once in a **fixed** SPD loss weight, is an injective integral transform of the normal spectral-kernel defect along each radial line. Injectivity holds for bounded continuous defects; an analytic Taylor expansion is unnecessary. Consequently a smooth defect flat at the isotropic point is still identifiable from the full small-amplitude response function. This is an exact-data statement, not a stable finite-sample recovery guarantee.

The proof below independently establishes the transform identity and injectivity. It also independently checks the separate regularity agent's spectral-germ separation argument. No numerical experiment is needed for these identities. The positivity-of-exponential-kernels argument is standard analysis; the potentially useful result is its application to this particular optimizer response.

## 1. The exact transform

Let S be a real linear subspace of Sym(n), N=S-perp, s>0, and R_X=Dlog_X-M_X. Assume M is differentiable where this calculation is used, symmetric as an operator in the Frobenius inner product, and normalized on commuting tangents. Thus

    R_X I=0,   R_X X=0,   R_X(X-sI)=0.

The loss is one half of <P_S(X-sI)-Delta, B[P_S(X-sI)-Delta]>, and the flow mobility is (I+lambda M_X)^-1. Write C_B(Delta) for its endpoint's first lambda derivative at zero. In the regime in which the endpoint derivative exists,

    C_B(Delta) = - integral_0^infinity P_N M_(X_B(t)) dX_B(t).

This is a normal vector because all compared endpoints interpolate the same measurements. The exact Dlog integral depends only on the endpoints, so its weight derivative vanishes.

For nonzero E in S and arbitrary V in S, choose a self-adjoint D on S with DE=V. B_tau=I+tau D is SPD for all sufficiently small |tau|, and is held fixed throughout each individual flow. At lambda=0 and Delta=epsilon E,

    X_(B_tau)(t)=sI+epsilon(I-exp(-B_tau t))E,
    X_(B_0)(t)=sI+epsilon f(t)E,
    partial_tau X|_0=epsilon h(t)V,
    f(t)=1-exp(-t),   h(t)=t exp(-t).

Differentiating R_X(X-sI)=0 gives

    (D R_X[V])E = -R_X V/(epsilon f)

at X=sI+epsilon f E, for f nonzero. Therefore

    partial_tau C_(B_tau)(epsilon E)|_0
      = epsilon integral_0^infinity (h'-h f'/f) P_N R_(sI+epsilon f E)V dt
      = epsilon integral_0^1 K(u) P_N R_(sI+epsilon u E)V du,

where

    K(u)=1+log(1-u)/u
        =- sum_(m=2)^infinity u^(m-1)/m,       0<u<1.

The integrable logarithmic singularity at u=1 is harmless. Near u=0, K(u)=-u/2+O(u^2). The transform therefore remains well defined for bounded continuous normal defects. Differentiability is only needed in deriving the response identity from the flow, not in the following injectivity theorem.

## 2. Continuous local injectivity

**Theorem.** Let q:[0,T)->R be continuous (or locally bounded measurable). Suppose, for every 0<t<T,

    integral_0^1 K(u)q(tu)du=0.

Then q(t)=0 for every 0<t<T in the continuous case, and almost everywhere in the measurable case. In particular this applies to C-infinity functions with every derivative zero at t=0. The value q(0)=0 is natural for spectral defects but is not required for injectivity.

**Proof.** Fix T_0<T and set Q(x)=q(T_0 exp(-x)), x>=0. Q is bounded. With u=exp(-v), the hypothesis becomes

    integral_0^infinity k(v)Q(x+v)dv=0,       x>=0,
    k(v)=-exp(-v)K(exp(-v))
        =sum_(m=2)^infinity exp(-mv)/m.

Put P(x)=exp(-x)Q(x), x>=0, and extend P by zero to the negative half-line. P belongs to L2(R). Multiply the last integral equation by exp(-2x)Q(x), integrate over x>=0, and symmetrize. One obtains

    0 = (1/2) sum_(m=2)^infinity (1/m)
          integral_R integral_R exp(-(m-1)|x-y|)P(x)P(y) dx dy.       (1)

All exchanges are justified absolutely: Young's inequality bounds the absolute double integral by

    2 ||P||_2^2 sum_(m=2)^infinity 1/[m(m-1)] = 2 ||P||_2^2.

For any a>0, the kernel exp(-a|x-y|) is strictly positive definite on real L2(R). An elementary proof avoids a spectral theorem: set

    w_a(x)=integral_(-infinity)^x exp(-a(x-y))P(y)dy.

Then w_a belongs to H1(R), w_a'+a w_a=P, and symmetrization plus integration by parts gives

    integral_R integral_R exp(-a|x-y|)P(x)P(y) dx dy
       =2 integral_R P w_a
       =2a ||w_a||_2^2 >=0.

Equality implies w_a=0 and hence P=0. Every summand in (1) is nonnegative, so already its m=2 term forces P=0. Thus Q=0 almost everywhere and q=0 on (0,T_0), almost everywhere or everywhere according to its regularity. T_0 was arbitrary. QED.

Vector- and matrix-valued versions follow by applying the scalar statement to each coordinate. Thus vanishing of all infinitesimal static-weight responses for a fixed radial E and all V implies exact normal cancellation P_N R_(sI+tE)V=0 along that radial line, without a finite nonzero jet assumption.

## 3. A single fixed 2-by-2 measurement space identifies the kernel

Define

    E=diag(1,-1),
    H=[[0,1],[1,0]],
    V=I+H,
    N=H-I,
    S=span{E,V}.

N spans S-perp. Define the self-adjoint weight direction

    D(U)=(V<E,U>+E<V,U>)/2,    U in S.

Since <E,V>=0 and ||E||_F^2=2, DE=V. In the orthonormal basis E/sqrt(2), V/2, D has off-diagonal entries sqrt(2) and zero diagonal. Its eigenvalues are plus/minus sqrt(2). Therefore B_tau=I+tau D is a concrete fixed SPD weight whenever |tau|<1/sqrt(2).

For a positive symmetric degree-minus-one homogeneous normalized kernel, write

    m(x,y)=a^-1 h(v),
    a=(x+y)/2,    v=(x-y)/(x+y),
    g(v)=atanh(v)/v-h(v),    g(0)=0.

At X=sI+epsilon u E,

    R_X V = s^-1 g(epsilon u/s) H.

For z=epsilon/s, define the scalar mixed response

    F(z)= (1/2)<N, partial_tau C_(B_tau)(s z E)|_(tau=0)>.

The preceding exact identity becomes

    F(z)=z integral_0^1 K(u)g(zu)du.                         (2)

The factors are correct because <N,H>=2. The statistic is independent of the isotropic scale s. Each B_tau is fixed along its trajectory; no time-dependent weight is used.

Equation (2), together with continuous injectivity, has two consequences:

1. The complete response profile on any neighborhood 0<z<r identifies g on that neighborhood, including nonanalytic flat defects. Exact zero response on that interval is equivalent to exact BKM agreement there.
2. If the profile is available for 0<z<1, it identifies the full symmetric homogeneous kernel: evenness recovers negative v, and homogeneity recovers arbitrary positive means a. For endpoint interpretation at an individual |z|<1, use a sufficiently small lambda neighborhood and weight neighborhood depending on the positive distance s(1-|z|) of the base path from the SPD boundary. This does not assert global convergence at a common finite damping value over all z.

An independent algebra check shows that this fixed S is nonneutral: ad_E^2 V=4H is not in S. Thus it is a valid universal witness space, not a passing space that accidentally erases the kernel.

## 4. Taylor coefficients and exact-data stability limits

If g(v)=sum_(j>=1) b_j v^(2j), termwise integration gives

    F(z)=-sum_(j>=1) a_j b_j z^(2j+1),
    a_j=[H_(2j+1)-1]/(2j)>0.

This independently reproduces the general-k constants after accounting for ad_E^(2k)V=4^k H. In particular a_1=5/12 and a_2=77/240. No coefficient is lost, but a_j tends to zero like log(j)/(2j).

The transform is bounded in the uniform norm:

    sup_(0<=z<=r) |F(z)| <= r (pi^2/6-1) sup_(0<=v<=r)|g(v)|.

**Stronger conditioning statement within the monotone class.** The root agent proposed the following refinement, independently verified here. Let nu be the BKM probability measure on [0,1] with moments 1/(2j+1), and let nu_m be its positive m-node Gaussian quadrature measure. Write

    b_(m,j)=integral u^j dnu-integral u^j dnu_m.

Then b_(m,j)=0 for j<2m, and b_(m,j)>0 for every j>=2m. To verify the sign, let p be the degree-at-most-2m-minus-one Hermite interpolant of u^j at the m quadrature nodes, matching both value and derivative. The interpolation remainder is a positive multiple of the squared node polynomial because the 2m-th derivative of u^j is nonnegative on [0,1], and is positive away from zero when j>=2m. Integrating and using exactness for p proves the sign; strictness follows from the full support of nu.

For the corresponding normalized symmetric monotone kernels,

    g_m(v)=sum_(j>=2m) b_(m,j) v^(2j)>0,       0<v<1.

All series terms are positive. Since a_j=integral_0^1 (-K(u))u^(2j)du decreases to zero,

    ||F_m||_[0,r]
      =r sum_(j>=2m) a_j b_(m,j) r^(2j)
      <=r a_(2m) ||g_m||_[0,r].

Also 0<||g_m||_[0,r]<=r^(4m)/(1-r^2), so these kernels approach BKM. The inverse from the exact response profile to the kernel defect is therefore **not locally Lipschitz at BKM in these uniform norms, even restricted to normalized symmetric monotone metrics**. This is stronger than the general-positive-kernel example below. It does not refute continuity or a weaker modulus of continuity, and it is not a new Gaussian quadrature construction. The possible added content is the precise conditioning obstruction for the optimizer-response map.

There is no converse bound with a constant independent of the kernel on this full positive homogeneous analytic class. Fix 0<r<1 and take

    g_j(v)=-(v/r)^(2j),
    h_j(v)=atanh(v)/v+(v/r)^(2j).

Each h_j is positive, even, real analytic on (-1,1), and normalized at zero; hence it defines a permitted positive spectral kernel. It is not claimed to be a monotone quantum metric. On [0,r], ||g_j||_infinity=1 but ||F_j||_infinity=r a_j tends to zero. Exact injectivity therefore must not be advertised as uniformly stable recovery from noisy data. The example establishes a specific response-norm obstruction, not an impossibility for every conceivable diagnostic.

## 5. Independent check of spectral-germ separation

The regularity agent proposes replacing real analyticity by the condition that continuous even g, with g(0)=0, is nonzero on every neighborhood of zero. The key scalar lemma was independently checked as follows.

For s normalized to one, put

    F_(a,b)(t)=(1+bt)^-1 g(at/(1+bt)),    a>0.

Suppose F_(a,b)=F_(c,d) as two-sided germs. With w=at/(1+bt), r=c/a>0, and delta=(d-b)/a, their equality is equivalent to

    g(Tw)=(1+delta w)g(w),
    H(Tw)=rH(w),
    T(w)=r w/(1+delta w),    H(w)=w g(w).

If 0<r<1,

    T^j(w)=r^j w/[1+delta w(1-r^j)/(1-r)].

Choose w in a sufficiently small symmetric neighborhood so all denominators stay nonzero and the iterates remain within the equality domain. Since g(T^j(w)) tends to zero while the denominator tends to a nonzero number, the identity forces g(w)=0. The case r>1 reduces to this by swapping pairs.

If r=1 and delta is nonzero, choose delta w>0. Then T^j(w)=w/(1+j delta w) tends to zero while H(T^j(w))=H(w), so H(w)=0 on one side. Evenness of g makes it zero on both sides. Alternatively inverse iteration handles the opposite side. This contradicts the nonzero-germ assumption. Therefore r=1 and delta=0, giving a=c and b=d. A zero-gap block has identically zero multiplier, while any positive-gap block does not, again by the nonzero-germ assumption.

The finite-dimensional conclusion is valid: for fixed Z, the commuting self-adjoint family R_(sI+tZ) distinguishes all finitely many distinct positive-gap/sum pairs of its Peirce components. Its generated algebra contains their joint spectral projectors, and therefore both ad_Z^2 and (L_Z+R_Z)ad_Z^2. A subspace reducing every R_(sI+tZ) reduces these two operators. Repeated eigenvalues and the zero-gap block cause no gap in this argument.

The algebraic statement only requires continuous g. The regularity agent's completed note supplies the missing justification for C1 classical closedness and flow statements: on all real matrices, put A_X=(L_X+R_X)/2 and V_X=(L_X-R_X)(L_X+R_X)^-1. Then M_X=A_X^-1 h(V_X); these are commuting self-adjoint operators, spec(V_X) lies in (-1,1), and finite-dimensional C1 scalar functional calculus is C1. Evenness preserves the symmetric subspace. Thus C1 is sufficient; this does not claim that it is a necessary differentiability threshold for every formulation.

## 6. Independent check of the global guarded plateau counterexample

The regularity agent's completed `REGULARITY_THEOREM.md`, section 8, was independently checked in full. Let delta=1/2 and h=L+psi_delta, where psi_delta vanishes for |v|<=delta and is a positive smooth bump above it. With D=diag(1,-1), H having off-diagonal entries one, and K=4, define

    S={diag(aD+b(I_2+H),Ka,-Ka,Kb,-Kb):a,b real} subset Sym(6).

Every X in (I_6+S) intersect SPD has |a|,|b|<1/4 because of the four scalar guard entries. The active 2-by-2 block has eigenvalues 1+b plus/minus sqrt(a^2+b^2). Its spectral ratio is strictly less than sqrt(2)/3<1/2. Thus the defect vanishes on every active-block spectral entry. The scalar tangents commute with their scalar blocks, and cross-block tangent entries are zero, so R_X U=0 for every such X and U in S. Coinciding eigenvalues across different blocks do not invalidate this: one may use a block-respecting eigenbasis, and the resulting spectral operator is basis independent.

On the other hand, for the two obvious generators Z_1 and Z_2,

    ad_(Z_1)^2 Z_2=diag(4H,0,0,0,0).

If this belonged to S, its zero scalar guards would force a=b=0, contradicting its nonzero active block. Hence even the first closure condition fails. This proves that global non-BKM inequality alone does not imply the universal classification, even with exactness on the whole affine SPD slice. It does not assert convergence of all positive-damping flows for all feasible targets; the asserted exact global one-form counterexample is sufficient and correct.

## 7. Failed shortcuts deliberately excluded

- K(u)<0 alone does not prove injectivity for a sign-changing q. The positive-definite energy identity above supplies the missing argument.
- Nonvanishing of Mellin multipliers on real powers only proves coefficientwise analytic recovery, not recovery of flat functions.
- Complete amplitude-profile identifiability is not a finite diagnostic and does not overturn the finite-jet noncertification result.
- Arbitrarily small response in a norm does not imply small kernel defect uniformly without complexity or regularity control; the explicit h_j family refutes that inference.
- Differentiability of the infinite-time endpoint requires the compact-SPD and residual-decay argument from the dynamics audit, not merely formal differentiation of the ODE.

## Completion status

Complete as an independent exact derivation and proof audit. The response transform, its continuous injectivity, the canonical fixed-weight realization, and the regularity agent's pair-separation lemma are verified. Priority is not certified; the root literature audit remains responsible for comparison with original sources.
