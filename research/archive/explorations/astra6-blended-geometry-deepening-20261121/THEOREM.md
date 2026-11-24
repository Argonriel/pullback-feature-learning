# Finite metric blending permits autonomous losses to fill hidden endpoint fibers

Date: November 21, 2026 (America/Los_Angeles). Status: completed theorem draft with independent review and bounded checks. All matrices are real. No theorem below asserts priority, global reachability, or a learning-performance benefit.

## 1. Model and quantifiers

Let n>=2, let SPD(n) denote the positive definite symmetric matrices, and use the Frobenius inner product on Sym(n). For X in SPD(n), set

    K_X(U)=XU+UX,
    M_lambda(X)=K_X(K_X+lambda Id)^(-1),
    q=diag(X),    z=(X_ij)_(i<j),    m=n(n-1)/2.

Fix a finite lambda>0. A fixed positive quadratic loss means

    ell_B(q)=0.5(q-y)^T B(q-y),    B in SPD(n),
    X_dot=-M_lambda(X) diag(B(q-y)).                         (1)

The weight B and target y do not change during a trajectory. The geometry is constructed from the full matrix representation, while the objective observes only its diagonal. It is not a metric built from only these observed outputs.

For E_i=E_ii, write F_i=M_lambda E_i, A_ji=(F_i)_jj, and

    H_i=sum_k F_k(A^-1)_ki.

A is SPD because v^T A v=<diag(v),M_lambda diag(v)> >0 for v!=0. Thus H_i is analytic and diag(H_i)=e_i. Write H_i=partial_qi+beta_i.partial_z. Adopt [H_i,H_j]=D H_j[H_i]-D H_i[H_j] and define the curvature map

    R_X(Omega)=sum_(i<j) offdiag([H_i,H_j](X)) Omega_ij,
    Omega in skew(n).

Here offdiag takes strict upper entries once, without sqrt(2); a skew matrix is identified with its strict upper entries. These conventions fix all factors and signs.

**Theorem 1 (generic autonomous endpoint submersion).** For each n>=2 and each fixed lambda in (0,infinity), there is an open dense full-measure set U_lambda in SPD(n) with the following property. For every X0 in U_lambda, every v in R^n with v!=0, and every C0 in SPD(n) satisfying

    P=A(X0)^(1/2),    d=P^-1 v,
    k=dim span{d,C0 d,...,C0^(n-1)d} >= n-1,
    B0=P^-1 C0 P^-1,

there are epsilon0>0 and a neighborhood V of B0 in SPD(n) such that for all |epsilon|<epsilon0 and B in V, (1), with y=diag(X0)+epsilon v, exists for all t>=0, stays in a common compact SPD neighborhood, and converges to an interpolating endpoint. For every 0<|epsilon|<epsilon0, the map

    B -> offdiag(X_infinity(B,y))

has rank m at B0. Its image consequently contains a relatively open neighborhood of X_infinity(B0,y) in the feasible SPD fiber diag(X)=y.

For each v!=0 the permitted C0 include an open dense set (the cyclic choices, k=n). The exceptional set U_lambda^c is allowed to depend on lambda. No one common generic set valid simultaneously for every lambda is asserted. The size of epsilon0 may depend on X0,lambda,v,C0. The open image is local and typically has scale epsilon^2. It is not the whole feasible fiber.

**Corollary 1 (initialization-dependent dimension).** Fix a coordinate partition Pcal with block sizes b_1,...,b_r, and its block-diagonal SPD manifold. For each lambda>0, at a generic X0 in that manifold and every v!=0, the same conclusion holds for the within-block hidden coordinates, of dimension

    m_Pcal=sum_j b_j(b_j-1)/2.

The endpoint image has relative interior in the block-diagonal feasible fiber; all cross-block entries stay zero. Full SPD loss weights, including coupling of residuals across blocks, are allowed. This includes v supported in just one block. If m_Pcal=0, the relative fiber is a singleton. No comparable conclusion is asserted for all exceptional states in a block of size at least three.

## 2. Generic curvature rank: independent reconstruction

Put X=diag(a)+Z with distinct a_i>0 and Z symmetric with zero diagonal. Since M_lambda^-1=Id+lambda K_X^-1, the normalized lift satisfies

    offdiag(H_i+lambda K_X^-1 H_i)=0.

Expand the inverse Lyapunov operator at diag(a). For p<r,

    beta_i^(pr)
       = lambda Z_pr/[2(a_p+a_r+lambda)]
         (delta_ip/a_p+delta_ir/a_r) + O(||Z||^2).           (2)

For the bracket [H_i,H_j], the terms from vertical differentiation of the first-order expression cancel. The ordinary q derivatives give

    offdiag([H_i,H_j])
      = lambda Z_ij(a_j-a_i)/[2a_i a_j(a_i+a_j+lambda)^2] e_(ij)
        + O(||Z||^2),       i<j.                           (3)

Its diagonal is exactly zero because diag H_i is constant. Set Z=delta C with every C_ij nonzero. The determinant of the m-by-m vertical curvature matrix is

    delta^m product_(i<j)
       {lambda C_ij(a_j-a_i)/[2a_i a_j(a_i+a_j+lambda)^2]}
       + O(delta^(m+1)).                                  (4)

The leading coefficient is nonzero. Small enough |delta| also leaves X SPD. All denominators defining the fields are nonvanishing on SPD, so this determinant is analytic and not identically zero on the connected cone. Its nonzero set U_lambda is open dense; the zero set has Lebesgue measure zero. This proves R_X is an isomorphism there. Equivalently the n fields and their m first brackets span Sym(n).

A common C^2 scalar law for every diagonal-only loss has differential annihilating these fields and brackets. The differential vanishes on U_lambda, then everywhere by continuity; connectedness makes the function constant. This is an application of the established conservation/Lie-rank framework, not a new general criterion; see [Marcotte et al., Theorem 3.7](https://arxiv.org/html/2405.12888v1).

For a fixed block partition, all F_i and H_i are supported on their own blocks. Their Lie algebra is a direct product. Apply (4) independently in each nonsingleton block: R has rank m_Pcal generically on the block manifold. This is relative openness and relative full measure, rather than ambient full measure.

## 3. The static-loss area map

For fixed d and C>0, define

    u_C(t)=(I-exp(-Ct))d,
    G(C)=integral_0^infinity exp(-Ct)dd^T exp(-Ct)dt,
    Omega(C)=0.5 integral_0^infinity (u_C du_C^T-du_C u_C^T).

Then CG+GC=dd^T and

    Omega(C)=(CG-GC)/2.                                   (5)

This is ordinary Lyapunov/Gramian and signed-area machinery. In an orthonormal eigenbasis of C, G_ij=d_i d_j/(c_i+c_j) and Omega_ij=d_i d_j(c_i-c_j)/(2(c_i+c_j)).

**Lemma 2 (exact differential rank).** If k=dim Krylov(C,d), then

    rank D Omega(C)=m-(n-k)(n-k-1)/2.                       (6)

Proof. If k=n, G>0: a vector has zero quadratic form under G precisely when it is orthogonal to the Krylov space. For E=E^T, put Z=DG[E]. Then

    CZ+ZC=-(EG+GE),
    2 D Omega[E]=EG-GE+CZ-ZC.

Vanishing of D Omega is equivalent to EG+CZ=0. Its unique candidate Z=-C^-1 EG is symmetric exactly when EGC=CGE. Equivalently T=C^-1/2 E C^-1/2 commutes with S=C^1/2 G C^1/2.

The SPD matrix S has simple spectrum. Indeed CS+SC=ww^T, w=C^1/2 d. Compression to an eigenspace V for a positive eigenvalue s yields 2s (PCP)|_V=(Pw)(Pw)^T|_V. The left side is SPD on V and the right side has rank at most one, so dim V=1. Therefore the symmetric commutant of S has dimension n. The kernel dimension is n and the rank is m.

For general k, split into the C-invariant Krylov space and its orthogonal complement. Then C=diag(C1,C2), G=diag(G1,0), with G1>0. The first block contributes k(k-1)/2 to rank. The cross block obeys

    C1 Z12+Z12 C2=-G1 E12,
    (D Omega[E])12=C1 Z12.

Both Sylvester inversion and multiplication by G1,C1 are invertible, so the cross block contributes k(n-k). The remaining block vanishes. Their output coordinates are independent, proving (6), including k=0. QED.

For d!=0, cyclic C form an open dense set: in a simple-eigenvalue decomposition they are exactly those with no zero component of d in the eigenbasis. Thus Lemma 2 supplies more than existence of one weight pair. Scalar C=alpha I has only k=1 and rank n-1, deficient at quadratic order when n>=3. That deficiency alone does not establish deficiency of the exact finite-epsilon endpoint map; higher orders might restore rank.

## 4. Endpoint expansion and proof of Theorem 1

Parameterize B(C)=P^-1 C P^-1 and y=q0+epsilon v, v=P d. The first observed displacement is

    x_C(t)=P(I-exp(-Ct))d.

The normalized lifting equation is dz=sum_i beta_i dq_i. Its second-order expansion, evaluated at X0, gives

    z_infinity(C,epsilon)
       =z0+epsilon sum_i beta_i v_i
        +epsilon^2 [S0(v)+R_X0(P Omega(C) P^T)]
        +O_C1(epsilon^3),                                (7)

where S0(v)=0.5 sum_(i,j) H_i(beta_j) v_i v_j is independent of C. In fact integral x_i dx_j=v_i v_j/2+(P Omega P^T)_ij; the antisymmetric term pairs with H_i beta_j-H_j beta_i. The constant beta multiplying the second-order observed correction contributes zero, since the observed endpoint is exactly q0+epsilon v at every weight. This also checks the sign: C=diag(1,2), d=(1,1) gives Omega_12=-1/6.

The remainder is uniform with one weight derivative near C0. Here is the infinite-time justification. On a compact SPD neighborhood and compact weight neighborhood take A>=aI, B>=bI and bounded mobility derivatives. For r=q-y and V=r^TBr/2,

    V_dot=-r^T B A B r <= -2ab V.

Hence |r|<=K |epsilon| exp(-ab t), and |X_dot|<=K'|r|. Total path length is O(|epsilon|). A first-exit bootstrap keeps all small-target trajectories in the chosen neighborhood, proves global existence, and gives an SPD interpolating limit. In residual and hidden coordinates, parameter derivatives satisfy an exponentially stable residual equation coupled to the hidden derivative by coefficients proportional to r. Variation of constants and Gronwall with integral |r|=O(|epsilon|) bound hidden derivatives and give polynomial(t) exp(-gamma t) bounds for residual derivatives. Induction through three epsilon derivatives and one C derivative gives uniformly integrable derivatives of X_dot. Differentiating X0+integral X_dot is therefore valid and gives the C1 Taylor remainder in (7). This uses local transverse exponential stability, not finite-time smooth dependence alone and not an assumed global convergence theorem for arbitrary targets.

Differentiate (7):

    D_C z_infinity(C0,epsilon)
      =epsilon^2 R_X0 o (Omega -> P Omega P^T) o D Omega(C0)
        +O(epsilon^3).                                   (8)

On U_lambda, the first two maps are isomorphisms. Lemma 2 makes the third onto whenever k>=n-1. A full-rank minor in (8) remains nonzero for sufficiently small nonzero epsilon. The submersion theorem proves Theorem 1. For Corollary 1, the curvature has surjective range of dimension m_Pcal and the same composition proves full relative rank; block invariance supplies the matching upper bound.

The proof is autonomous throughout: it never replaces one fixed quadratic loss by a time-dependent sequence of controls.

## 5. Known spectral range tool and a sharp leading-order bound

For fixed d!=0, the exact image of Omega(C) is

    {Omega skew: Q=dd^T/2+Omega is real diagonalizable
                 with all eigenvalues nonnegative}.       (9)

Necessity: Q=CG is similar to C^1/2 G C^1/2>=0. Conversely, if Q=V Lambda V^-1 with real V invertible and Lambda>=0 diagonal, set C=VV^T and G=V^-T Lambda V^-1. Then CG=Q, GC=Q^T and CG+GC=dd^T, so Lyapunov uniqueness recovers the required Gramian. This is a specialization of the classical real positive-matrix factorization facts in [Ballantine (1968), Theorem 1 and Corollary 1.1'](https://msp.org/pjm/1968/24-1/pjm-v24-n1-p02-s.pdf), not a separately claimed new matrix theorem. Positive eigenvalues are automatically simple by the compression proof above; repeated zeros are allowed.

For n=2 the exact signed-area interval is |Omega_12|<||d||^2/4; equality gives a defective repeated positive eigenvalue of Q. At zero area a noncyclic C is allowed. Thus, after subtracting the weight-independent terms in (7), the possible second-order hidden coefficients in dimension two form the open interval of radius

    |R_12(X0)| sqrt(det A0) ||A0^-1/2 v||^2/4.

This characterizes coefficients for individual fixed weights as epsilon->0. It does not determine the complete nonlinear endpoint interval at a fixed target, because the remainder is not uniform over weights approaching the boundary or infinity.

## 6. Exceptional initialization and the two integrable limits

For n=2, X=[[a,c],[c,b]], let D=ab-c^2 and Q=(a+b)D+lambda ab. Then

    beta_a=lambda c b/(2Q),    beta_b=lambda c a/(2Q),
    [H_a,H_b]=F partial_c,
    F=lambda c(b-a)D/(2Q^2).

At a=b=s and c!=0, F=0 but [H_a,[H_a,H_b]]=-lambda c(s^2-c^2)/(2Q^2) partial_c is nonzero. Thus the complete Lie-rank classification is: rank 2 on c=0; rank 3 at bracket length 2 for c!=0,a!=b; rank 3 at length 3 for c!=0,a=b. Theorem 1 uses invertible first curvature, so this higher-bracket fact alone does not extend its quadratic-order statement to a=b.

In higher dimensions, block component partitions are invariant at finite times, and also at compact SPD limits under integrable controls. Individual missing edges are not invariant. A fully coherent 3x3 state with diagonal 2, off-diagonal 1/3 and lambda=3 has ranks 3,5,6 through lengths 1,2,3. Hence complete nonzero support does not ensure two-step full rank. The exact certificates and proofs are in [exceptional/REPORT.md](exceptional/REPORT.md). Full Lie rank at every connected state in n>=3 remains unresolved.

At lambda=0, off-diagonal entries are constant. At the time-rescaled pure-Bures limit, X_dot=-K_X diag(g) and X_ij/sqrt(X_ii X_jj) is constant. Both limiting endpoint laws are independent of B for nearby targets. The latter mechanism has established commuting-parametrization/SLD precedents, detailed in [the primary audit](literature/AUDIT.md).

To justify the limiting endpoints rather than just vector fields, put theta=lambda/(1+lambda). Multiplying the mobility by 1+lambda gives

    N_theta=K_X[(1-theta)K_X+theta Id]^-1,
    N_0=Id,    N_1=K_X.

On a compact SPD set its eigenvalues are uniformly positive and bounded for all theta in [0,1]. The local convergence and derivative argument for (7) therefore works uniformly for a fixed compact family of weights and sufficiently small targets. Endpoints extend continuously to both limits, and every fixed-weight-pair gap tends to zero there. This asserts neither a global damping-unimodality law nor an equality between a curvature maximizer and a finite-target endpoint-gap maximizer.

## 7. Attribution and validation boundary

The generic determinant is rechecked from the archived candidate, not counted as a new discovery in this round. The strengthened candidate is (8): full-dimensional local variation from autonomous quadratic weights, in every dimension and for every nonzero small target direction at generic finite-blend initial states, with the block-relative version. The exact area differential rank is a supporting lemma. Conservation counting, Frobenius, signed areas, Gramian formulas, matrix factorization, and submersion are established tools.

See [the primary theorem-level comparison](literature/AUDIT.md), [independent review](exceptional/STATIC_THEOREM_REVIEW.md), and [DATA_DICTIONARY.md](DATA_DICTIONARY.md). No novelty conclusion is inferred from unsuccessful searching. No arbitrary-target convergence, global fixed-loss controllability, all-connected-state classification, or performance advantage is claimed.
