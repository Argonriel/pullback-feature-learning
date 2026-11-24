# Autonomous quadratic losses locally fill the hidden endpoint fiber

Status: independently derived and reviewed algebraic result; November 21, 2026 (America/Los_Angeles). This note belongs only to the new blended-geometry exploration. All older outputs are read-only.

## 1. Precise scope

Let n >= 2, lambda in (0,infinity), and X0 in SPD(n). Equip Sym(n) with the Frobenius pairing, define K_X(U)=XU+UX and M_lambda(X)=K_X(K_X+lambda Id)^(-1), and observe q=diag(X). A loss has the autonomous, time-independent form

    ell_B(q) = (q-y)^T B (q-y)/2,        B in SPD(n),
    X_dot = -M_lambda(X) diag(B(q-y)).

Write z for the n(n-1)/2 strict-upper-triangular entries of X, and m=n(n-1)/2. Let F_i=M_lambda E_ii and A_ji=(F_i)_jj. Then A is SPD: v^T A v=<diag(v),M_lambda diag(v)> > 0 for v!=0. The normalized horizontal frame is H_i=sum_k F_k(A^-1)_ki, so diag(H_i)=e_i. In the coordinates (q,z), write H_i=partial_qi+beta_i.partial_z. Define

    R_ij = H_i(beta_j)-H_j(beta_i),        i<j,
    R: skew(n) -> R^m,    R(Omega)=sum_(i<j) R_ij Omega_ij.

Here a skew matrix is identified with its entries indexed by i<j; no extra factor of two is implicit. Assume R(X0) is invertible. The earlier generic bracket theorem supplies an open dense, full-measure set of such X0 for each finite positive lambda. This note does not replace that earlier determinant proof, and does not assert invertibility at every non-diagonal state.

**Theorem A (fixed-target autonomous endpoint submersion).** For every nonzero v in R^n and every SPD C0 for which

    d=A(X0)^(-1/2)v,
    dim span{d,C0 d,...,C0^(n-1)d} >= n-1,

put B0=A(X0)^(-1/2) C0 A(X0)^(-1/2). There is epsilon0>0 such that, for every 0<|epsilon|<epsilon0 and target y=diag(X0)+epsilon v, all losses with B in a neighborhood of B0 have a globally defined trajectory contained in a common compact SPD neighborhood, converge to an interpolating endpoint, and the map

    B -> offdiag(X_infinity(B,y))

has differential of rank m at B0. Consequently its image contains an open neighborhood of offdiag(X_infinity(B0,y)) in R^m. Equivalently, the endpoint image has nonempty relative interior in the affine fiber diag(X)=y intersected with SPD(n).

For every nonzero v, admissible C0 exist: choose distinct positive eigenvalues and an orthogonal eigenbasis in which every component of d is nonzero. The cyclic choices (Krylov dimension n) form an open dense subset of SPD(n).

The weights are fixed for the entire trajectory. No time switching, target switching, arbitrary control, or concatenation is used. The theorem is local: the open set can shrink like epsilon^2, and need not be the entire feasible fiber. It does not assert uniform epsilon0 over unbounded or singular weight families, global reachability, or full endpoint rank at every exceptional initialization.

## 2. A standalone rank theorem for the area of an exponential path

For C in SPD(n) and d in R^n fixed, let

    u_C(t)=(I-exp(-Ct))d,
    G(C)=integral_0^infinity exp(-Ct) dd^T exp(-Ct) dt,
    Omega(C)=1/2 integral_0^infinity [u_C du_C^T-du_C u_C^T].

Thus CG+GC=dd^T. Direct integration gives

    2 Omega(C)=CG-GC.

In an orthonormal eigenbasis C=diag(c_i), this becomes

    G_ij=d_i d_j/(c_i+c_j),
    Omega_ij=d_i d_j(c_i-c_j)/(2(c_i+c_j)).

These Lyapunov/Gramian and signed-area identities are established elementary tools. They are not claimed as a new control-theoretic construction.

**Theorem B (exact differential rank).** Let k=dim span{d,Cd,...,C^(n-1)d}. Then

    rank D Omega(C) = k(k-1)/2 + k(n-k)
                   = m-(n-k)(n-k-1)/2.

This includes d=0, when k=0 and the rank is zero. In particular the differential is onto skew(n) if and only if k>=n-1.

### Full cyclic case

If k=n, G is positive definite. Indeed h^T G h is the integral of (h^T exp(-Ct)d)^2, so its vanishing is equivalent, by analyticity and Cayley-Hamilton, to h being orthogonal to the Krylov space.

For a symmetric variation E of C, let Z=DG(C)[E]. Differentiation gives

    CZ+ZC=-(EG+GE),
    2 D Omega(C)[E]=EG-GE+CZ-ZC.

The area derivative vanishes exactly when EG+CZ=0 (and its transpose GE+ZC=0). The unique candidate is Z=-C^-1 EG. This is symmetric exactly when

    EGC=CGE.

Set T=C^-1/2 E C^-1/2 and S=C^1/2 G C^1/2. The last equation is [T,S]=0. Moreover S is SPD and

    CS+SC=ww^T,       w=C^1/2 d.

S has simple spectrum. If an eigenvalue s>0 had eigenspace V of dimension at least two and P were its orthogonal projector, compressing the displayed equation would give

    2s (P C P)|_V = (P w)(P w)^T|_V.

The left side is positive definite on V and the right side has rank at most one, a contradiction. The symmetric matrices T commuting with the simple-spectrum symmetric matrix S have dimension n. Since E<->T is an isomorphism, dim ker D Omega=n, and rank D Omega=n(n+1)/2-n=m.

This proof needs no total-positivity theorem or assertion that a numerical Cauchy determinant is nonzero.

### General Krylov dimension

The Krylov space V is C-invariant, and its orthogonal complement is also invariant because C is symmetric. In an orthonormal basis adapted to V,

    C=diag(C1,C2),   d=(d1,0),   G=diag(G1,0),

where G1 is positive definite and (C1,d1) is cyclic in dimension k. The V-V part of the area derivative has rank k(k-1)/2 by the preceding proof. For E12, the cross-block derivative Z12 solves

    C1 Z12+Z12 C2=-G1 E12,
    (2 D Omega[E])12=2 C1 Z12.

The first map is invertible by the positive-spectrum Sylvester equation, G1 is invertible, and C1 is invertible. Thus the cross block gives rank k(n-k). The V-perp to V-perp block of the area derivative is zero. The two nonzero output blocks are independent, giving the exact formula. For k=0 the same conclusion is immediate.

## 3. From signed area to endpoint selection

Let P=A(X0)^(1/2), choose target displacement v=P d, and parameterize weights by B(C)=P^-1 C P^-1. On linearizing the observed residual around X0,

    q(t)=q0+epsilon x_C(t)+O(epsilon^2),
    x_C(t)=P(I-exp(-Ct))d.

The normalized lift obeys dz=beta(q,z)dq. Expanding beta around X0, with all displayed beta and R evaluated there, gives the endpoint expansion

    z_infinity(C,epsilon)
      = z0 + epsilon sum_i beta_i v_i
        + epsilon^2 [ S0(v)+R(P Omega(C) P^T) ]
        + O_C1(epsilon^3),

where

    S0(v)=1/2 sum_(i,j) [H_i(beta_j)] v_i v_j

is independent of C. To check the coefficient and sign, write

    integral x_i dx_j = v_i v_j/2 + (P Omega P^T)_ij.

The symmetric part gives S0, while pairing the antisymmetric part with H_i(beta_j) gives sum_(i<j) R_ij (P Omega P^T)_ij. The second-order correction to q(t) makes no additional endpoint term because q_infinity=q0+epsilon v exactly; its contribution under the constant beta is the endpoint displacement of that correction, which is zero.

The error is uniform together with one C derivative on a sufficiently small compact neighborhood of C0. Hence

    D_C z_infinity(C0,epsilon)
      = epsilon^2 R o [Omega -> P Omega P^T] o D Omega(C0)
        + O(epsilon^3).

The first two maps are isomorphisms and Theorem B makes the last map onto. A full-row-rank minor remains nonzero for sufficiently small nonzero epsilon. Since C<->B is a linear isomorphism, the ordinary submersion theorem proves Theorem A.

### Infinite-time remainder and smoothness

This argument does not exchange an unjustified finite-time expansion with t=infinity. Here is a direct local justification. Choose a compact SPD neighborhood of X0 and a compact weight neighborhood. Let A(X)>=a I, B>=b I uniformly, and let all required derivatives of the analytic mobility be uniformly bounded there. With r=q-y and V=r^T B r/2,

    V_dot=-r^T B A(X) B r <= -2ab V.

Uniform norm equivalence gives |r(t)|<=K |epsilon| exp(-ab t), and |X_dot|<=K |r|. The total path length is at most K'|epsilon|. A standard first-exit bootstrap, taking epsilon sufficiently small, therefore keeps the whole trajectory inside the chosen neighborhood. The velocity is integrable, its limit is SPD, and r tends to zero.

For clarity about parameter derivatives, differentiate the residual and hidden equations

    r_dot=-A(y+r,z) B r,
    z_dot=-beta(y+r,z) A(y+r,z) B r.

A derivative of r satisfies a uniformly exponentially stable equation plus terms bounded by |r| times the corresponding derivative of z and bounded lower-order terms. A derivative of z has integrable forcing made from that residual derivative and terms with a factor r. Variation of constants and Gronwall with integral |r|=O(|epsilon|) bound the z derivatives uniformly and give residual derivative bounds of the form polynomial(t) exp(-gamma t), with gamma>0 after slightly weakening the decay rate. This starts at first order (the epsilon derivative has a fixed bounded initial residual derivative); induction gives every fixed finite mixed order in epsilon and C. All derivatives of X_dot needed for three epsilon derivatives and one C derivative are integrable uniformly on the parameter neighborhood. Dominated differentiation of X0+integral X_dot therefore yields a C^4 endpoint map and the stated C^1 Taylor remainder. Smoothness here follows from local stability, not from an unproved global convergence theorem.

## 4. Counterexamples and necessary qualifications

1. **Zero target displacement.** If v=0, every quadratic-loss trajectory is stationary at X0 and every endpoint derivative vanishes. The theorem explicitly excludes this.
2. **Scalar transformed weight.** At C=alpha I and d!=0,

       D Omega(C)[E]=(Edd^T-dd^T E)/(4alpha),

   so its rank is n-1. For n>=3 this is strictly less than m. Thus a proof using only an isotropic transformed reference weight cannot give all hidden endpoint directions at quadratic order. Noncyclicity is not always fatal: in dimension two n-1=m, and Theorem B says the derivative is still onto.
3. **Diagonal X0.** Then the diagonal submanifold is invariant for all the losses, so every hidden endpoint is zero. A theorem for every SPD initialization would be false. Invertible curvature is a substantive hypothesis.
4. **Scalar loss rescaling.** Multiplying B by a positive scalar merely changes time, so an endpoint map is never injective in all weight parameters. Theorem B has at least n kernel dimensions in the cyclic case; there is no conflict between full hidden-output rank and noninjective weights.
5. **Open local image is not global controllability.** The proof supplies an open image near one endpoint and small target displacement. It neither reaches every SPD matrix with that diagonal nor concatenates different fixed objectives.
6. **Uniformity in B.** Taking eigenvalues to zero or infinity destroys the compact-family constants. No weight-uniform neighborhood is asserted over the whole SPD cone.

The earlier dimension-two witness is recovered: for C0=I, C1=diag(1,2), d=(1,1), Omega(C1)_12=-1/6 and Omega(C0)=0. Thus the hidden endpoint contrast is -epsilon^2 sqrt(det A0) R_12/6+O(epsilon^3), including its sign.

## 5. Independent review of the two damping limits

Put theta=lambda/(1+lambda) and rescale time by the positive factor 1+lambda. The mobility becomes

    N_theta(X)=K_X[(1-theta)K_X+theta Id]^-1,
    theta in [0,1].

It is jointly analytic in X and smooth in theta on any compact SPD neighborhood, with N_0=Id and N_1=K_X. If k ranges over the positive eigenvalues of K_X, its scalar eigenvalues are k/((1-theta)k+theta), bounded above and below uniformly over the compact set and theta in [0,1]. The observed mobility inherits uniform positivity. For one fixed compact family of SPD loss weights, the same local stability and path-length bootstrap therefore work with one sufficiently small target neighborhood for every theta. Endpoint maps, and their required local derivatives, extend continuously (indeed smoothly) to both limits.

The endpoint gap between fixed losses tends to zero at each limit because off-diagonal entries are conserved at theta=0 and correlations X_ij/sqrt(X_ii X_jj) at theta=1. The time rescaling changes no endpoints. No statement about where a finite-target gap is maximal follows from this compactification or from maximizing a pointwise curvature.

## 6. Novelty boundary

The normalized horizontal lift, curvature, local signed-area expansion, Lyapunov Gramian, and submersion theorem are known mathematical tools. The independently derived addition is an exact differential-rank lemma for this fixed-target exponential-path family and its use to prove full-dimensional local endpoint variation under autonomous SPD quadratic losses in every dimension at the generic finite-blend states. It is a theorem-level strengthening of the old single two-dimensional loss pair. This note makes no priority claim without the separate primary-source audit.

The scope is suitable for retention if the algebra and prior-art comparison survive review; no training experiment or external dataset is required to decide the statement.

## 7. Exact range of the leading signed area

The following strengthening was proposed independently by the primary-source reviewer and checked here. It gives an exact algebraic range, while retaining the distinction between a leading coefficient and a finite-target endpoint.

**Theorem C (exact area image).** Fix d!=0 and let Omega be a real skew-symmetric n-by-n matrix. There exists C in SPD(n) whose exponential path has area Omega if and only if

    Q=dd^T/2+Omega

is diagonalizable over R and all its eigenvalues are nonnegative. Every positive eigenvalue of such a Q is automatically simple; zero eigenvalues may repeat.

Necessity follows from Q=CG, which is similar to S=C^1/2 G C^1/2>=0. For sufficiency write Q=V Lambda V^-1 with V real invertible and Lambda diagonal nonnegative, and put

    C=V V^T,       G=V^-T Lambda V^-1.

Then C>0, G>=0, CG=Q, GC=Q^T and CG+GC=dd^T. Uniqueness of the positive-spectrum Lyapunov equation makes G exactly the integral Gramian, and (CG-GC)/2=Omega. More generally C=V H V^T and G=V^-T H^-1 Lambda V^-1 work for any diagonal H>0. Simplicity of positive eigenvalues follows from the same compressed rank-one Lyapunov identity used in Theorem B, with no need for S to be positive definite. The associated Krylov dimension equals rank G, the number of positive eigenvalues of Q.

This image is bounded. For example, since trace Q=||d||^2/2 and its eigenvalues are nonnegative,

    ||Omega||_F^2 = ||d||^4/4-trace(Q^2)
                 <= (||d||^4/4)(1-1/n).

For n>=2, equality would require all Q eigenvalues equal and positive, contradicting the simplicity just proved; hence the bound is strict. The bound is an outer bound, not an assertion that every skew matrix satisfying it is attainable.

In dimension two, rotate coordinates so d=(r,0), r=||d||. Write Omega_12=omega. The eigenvalues of Q are

    r^2/4 +/- sqrt(r^4/16-omega^2).

Thus the exact range is

    -||d||^2/4 < omega < ||d||^2/4.

At equality Q has a repeated positive eigenvalue and is not diagonalizable, so the endpoints are excluded. At omega=0 its zero eigenvalue is allowed. For a dimension-two initial state with nonzero curvature F=R_12, the set of attainable *second-order coefficients*, after subtracting the common first-order and symmetric second-order terms, is precisely the open interval of radius

    |F| sqrt(det A0) ||A0^-1/2 v||^2 / 4.

Every member of this coefficient set is realized by one fixed SPD weight as epsilon tends to zero. This does not justify taking an epsilon-dependent sequence of unbounded or singular weights inside a uniform Taylor expansion, and does not determine the entire endpoint range at a fixed noninfinitesimal target.

## 8. Independent bounded validation and preserved failure

`validation_v1/results.json` checks only the new algebraic rank claims, not the previously run endpoint example. For one four-dimensional diagonal C=diag(1,2,4,7), directions supported on k=0,1,2,3,4 eigenvectors give ranks 0,3,5,6,6, exactly as Theorem B predicts. The scalar-weight control gives rank 3. A centered derivative comparison at increments .01,.005,.0025 gives Frobenius errors 6.0681e-7, 1.5170e-7, 3.7925e-8, with the expected factor four improvement. The cyclic Lyapunov residual is 1.57e-16.

Source: `experiments/astra6_blended_geometry_deepening_20261121/static_loss/check_area_rank.py`; SHA-256 `d216cbd6fe00df9915cd933360a5387a00533c9ad83362e2da2a56393fd81cdc`. The result contains all inputs, singular values, source hash, and Python/NumPy versions. The first attempt to invoke the unqualified `python` command failed because the WindowsApps alias was inaccessible; no result was produced and no package was installed. The successful run used the bundled Codex runtime, Python 3.12.14 / NumPy 2.3.5.
