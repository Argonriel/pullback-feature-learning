# Finite damping destroys diagonal-observation conservation laws

Date: November 21, 2026. This is a new, bounded exploration of initial states outside the measurement-generated Jordan algebra. Historical runs and reports were not changed.

## Disposition

**Retain one scoped candidate:** an interpolation between two geometries with loss-independent endpoint laws can have no nonconstant common product-level conservation law at any finite positive mixing parameter. For the full-representation damped pullback metric, diagonal observations suffice: arbitrary off-diagonal initialization makes the distribution generically bracket generating in every dimension. A fixed pair of positive quadratic losses also selects different interpolants; this last claim is proved locally and checked numerically in dimension two.

The candidate is the combination of the all-dimensional rank theorem and its fixed-loss consequence, not a new metric, a new Frobenius theorem, or a claim that generic non-Hessian dynamics are surprising. The closest original sources provide those tools and the two limiting geometries. No inspected theorem directly supplies the finite-blend rank result, but the search is bounded and does not establish priority or publication significance. No prediction or generalization improvement is claimed.

## Model and distinction from the previous classification

Let X be real symmetric positive definite, with the Frobenius inner product. Write

    K_X(U)=XU+UX,
    M_lambda(X)=K_X(K_X+lambda Id)^(-1),
    X_dot=-M_lambda(X) diag(grad_q ell(q)),    q=diag(X).

This is the same specified balanced two-factor full-representation metric, with positive scalar parameter damping. The representation metric uses every matrix entry. The objective observes only the n diagonal entries. Its gradient span S=Diag(n) is commuting and its generated Jordan algebra is just Diag(n).

An initialization with nonzero off-diagonal entries lies outside that algebra. The previous isotropic/within-J theorem therefore does not apply. This is not the Gauss-Newton metric constructed only from the n observed predictions. Replacing the representation metric by that different metric changes the question and its conservation laws.

Two initial ideas were rejected before validation:

1. Merely extending the prior theorem to X0 in its generated Jordan algebra is already in the prior report.
2. A universal criterion based only on S being an ideal in Jordan(S,X0,I) is too strong without qualifications: a single measurement permits only one direction field and positive loss reweighting is just a clock change. No arbitrary-initial-state classification is claimed here.

The old result that commuting *metrics* need not give a scalar clock also remains retired. The present statement concerns complete loss-family conservation laws, commuting *measurements*, and two neutral limiting geometries, not an eigenvector-rotation heuristic.

Whitening is not a valid reduction to the old isotropic problem. For X=CYC with symmetric SPD C,

    C^(-1) K_X(CVC) C^(-1)=Y C^2 V+V C^2 Y,

which is not K_Y(V) in general. The Euclidean term is also changed by congruence. Moreover the transformed diagonal observation matrices C E_ii C need not commute. Orthogonal equivariance does not imply general congruence equivariance.

## The two neutral limiting geometries

At lambda=0, X_dot=-diag(grad ell), so every off-diagonal entry is conserved. If a nearby positive-definite interpolating limit exists, it is uniquely specified by the target diagonal and these initial entries.

After multiplying time by a positive constant, the lambda-to-infinity limit is the pure Bures/factor-gradient flow

    X_dot=-K_X(diag(g)).

Its entries obey q_i_dot=-2q_i g_i and X_ij_dot=-(g_i+g_j)X_ij. Hence each correlation

    rho_ij=X_ij/sqrt(q_i q_j)

is conserved. The endpoint with target diagonal y>0 is D X0 D, where D_ii=sqrt(y_i/q0_i). Thus both limits have n-dimensional leaves and n(n-1)/2 independent hidden-coordinate laws.

These are established ingredients. In particular [Li, Wang, Lee and Arora (2022), Theorem 4.8 and Remark 4.5](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Paper-Conference.pdf) cover commuting parametrizations and arbitrary initialization for Euclidean parameter gradient flow. The finite-blend product flow below is not that pure factor-gradient flow.

## All-dimensional finite-blend theorem

**Theorem.** Fix n>=2 and 0<lambda<infinity. For diagonal observations under the displayed full-representation metric, the distribution

    D_X={M_lambda(X) diag(v):v in R^n}

is bracket generating in two steps on an open dense subset of SPD(n), whose complement has Lebesgue measure zero. Consequently there is no nonconstant C^2 product-level function on SPD(n) conserved for every differentiable loss depending only on diag(X). There is likewise no such nonconstant function on a neighborhood of a generic state.

This is a generic-state theorem, not a statement at every state. In particular diagonal initial states remain in the diagonal submanifold. Factor-space gauge/balance invariants are not contradicted: the theorem is about functions of X.

### Proof

Let F_i(X)=M_lambda(X)E_ii. The observed mobility A(X), with entries A_ji=(F_i)_jj, is symmetric positive definite. Normalize the frame by

    H_i=sum_k F_k (A^(-1))_ki,

so diag(H_i)=e_i. The original and normalized fields generate the same distribution and the same distribution after adjoining first brackets modulo that distribution. In coordinates q_i=X_ii and z_ij=X_ij for i<j, write

    H_i=partial_(q_i)+sum_(p<r) beta_i^(pr) partial_(z_pr).

Around X=diag(a)+Z, with Z symmetric and zero diagonal, the inverse-Lyapunov expansion in the normal constraint

    offdiag[H_i+lambda K_X^(-1)H_i]=0

gives

    beta_i^(pr)
      =lambda z_pr/[2(a_p+a_r+lambda)]
        (delta_ip/a_p+delta_ir/a_r)+O(||Z||^2).

Use the bracket convention [H_i,H_j]=D H_j[H_i]-D H_i[H_j]. The vertical-derivative contributions cancel to first order, while the q derivatives give, for i<j,

    [H_i,H_j]
      =lambda z_ij(a_j-a_i)/[2a_i a_j(a_i+a_j+lambda)^2]
        partial_(z_ij)+O(||Z||^2).

All other vertical entries in this bracket are O(||Z||^2), and its diagonal is exactly zero. Take distinct positive a_i and Z=epsilon C with every C_ij nonzero. The matrix of the n(n-1)/2 brackets on vertical coordinates has determinant

    epsilon^[n(n-1)/2]
      product_(i<j) {lambda C_ij(a_j-a_i)
                       /[2a_i a_j(a_i+a_j+lambda)^2]}
      +O(epsilon^[n(n-1)/2+1]),

which is nonzero for sufficiently small nonzero epsilon. Together with the normalized H_i these brackets span Sym(n).

All coefficients are real analytic (indeed rational with nonvanishing denominators) throughout SPD(n). The determinant just exhibited is not identically zero on this connected domain. Its zero set has measure zero and empty interior, proving the generic rank statement. A common C^2 conserved function has differential annihilating every H_i and every bracket, so its differential vanishes on the dense full-rank set and hence everywhere by continuity. It is constant on the connected cone. QED.

No assertion is made that arbitrary hidden endpoints can be obtained using a *fixed* positive quadratic loss. Bracket generation permits broader controls. The next result supplies a separate fixed-quadratic consequence.

## Exact two-dimensional curvature and a fixed-loss witness

Write X=[[a,c],[c,b]], d=ab-c^2>0, t=a+b, and Q=td+lambda ab. The normal equation is exactly

    dc=beta_a da+beta_b db,
    beta_a=lambda cb/(2Q),   beta_b=lambda ca/(2Q).

Its vertical curvature is

    F_lambda=partial_a beta_b-partial_b beta_a
                 +beta_a partial_c beta_b-beta_b partial_c beta_a
            =lambda c(b-a)d/(2Q^2).

Thus every finite positive lambda has nonzero curvature whenever c!=0 and a!=b. At a=b this particular first bracket vanishes; no conclusion of neutrality follows from that pointwise degeneracy. The report does not need to classify that exceptional set.

For fixed (a,b,c), the magnitude of this curvature tends to zero at both parameter endpoints and reaches its maximum at

    lambda_star=(a+b)(ab-c^2)/(ab).

This maximizes the displayed *local curvature*, not necessarily the finite-target endpoint gap. No global unimodality of endpoint sensitivity is claimed.

To obtain actual autonomous losses, let A0=A(X0), P=A0^(1/2), and set

    B0=A0^(-1),
    B1=A0^(-1/2) diag(1,2) A0^(-1/2),
    y=q0+epsilon P(1,1)^T,
    ell_j(q)=0.5(q-y)^T B_j(q-y).

Both weights are fixed SPD matrices, and both losses have exactly the same zero set. The first-order observed paths are

    q^(1)_0(t)=P(1-e^(-t),1-e^(-t))^T,
    q^(1)_1(t)=P(1-e^(-t),1-e^(-2t))^T.

Expanding the lifting equation to second order in epsilon, the exact differential part cancels between paths with the same endpoints. Their signed area difference is -det(P)/6. Therefore

    c_infinity(B1)-c_infinity(B0)
      =-F_lambda(X0) sqrt(det A0) epsilon^2/6+O(epsilon^3).

This is nonzero for sufficiently small epsilon whenever c!=0 and a!=b. The statement holds at every specified finite positive lambda, with the constructed weights and direction allowed to depend on it. It does not claim that every target or every weight pair produces a difference. The weights are full SPD quadratic forms, rather than only scalar rescalings of the original coordinate losses.

For completeness, the local endpoint expansion is legitimate: on a small compact SPD neighborhood the mobility and observed mobility are smooth and uniformly positive. A small target perturbation yields exponential decay of the positive quadratic loss and an integrable velocity bound proportional to its square root. A bootstrap keeps the trajectory in that neighborhood. Differentiating this stable system in epsilon yields exponentially decaying residual derivatives and convergent endpoint derivatives. The second-order lifting expansion thus has the stated infinite-time remainder.

There is also a genuine two-limit consequence for a *fixed* small target and fixed weight pair chosen at one finite lambda. Normalize time with (1+lambda)M_lambda. This mobility extends smoothly to both lambda=0 and lambda=infinity after compactifying the parameter, with limits Id and K_X. In a fixed small SPD neighborhood it is uniformly elliptic over that whole parameter interval. The same local convergence argument gives continuous endpoints, so the fixed-pair endpoint gap vanishes at both limits while being nonzero at the constructed finite lambda. This conclusion does not require a parameter sweep or a claim about where its maximum lies.

## Minimal validation

The source is `experiments/three_candidate_exploration_20261121/nonisotropic/coherent_blend.py`. It independently implements the mobility by spectral solves and its analytic derivative by

    D M_lambda(X)[V]U
      =lambda (K_X+lambda Id)^(-1)
          K_V (K_X+lambda Id)^(-1)U.

The exact 2-by-2 curvature agrees with the full operator derivative to 8.7e-18; independent centered differences have errors 1.60e-10, 4.00e-11, and 1.01e-11 when the increment is halved. A single predefined 3-by-3 case, diag(1,2,3)+epsilon C with C12=.2, C13=-.3, C23=.4, has rank six at epsilon=.02,.01,.005. Its scaled bracket-matrix errors decrease from 3.79e-5 to 1.89e-5 to 9.44e-6, as predicted by the O(epsilon) remainder. This is a check of the general proof, not evidence from a dimension sweep.

For X0=[[1,.3],[.3,2]] and lambda=2.865, F_lambda=.00625. The fixed-loss formula predicts a quadratic gap coefficient -0.000506650773419.

| Target amplitude epsilon | Observed gap / epsilon^2 |
|---:|---:|
| .1 | -.000489830629219 |
| .05 | -.000498173044994 |
| .025 | -.000502395328628 |

The errors approach zero linearly in epsilon, as the O(epsilon^3) endpoint remainder predicts. The smallest actual gap is -3.1399708e-7. Halving the RK4 step from .01 to .005 changes either endpoint by at most 2.1e-14; residuals are below 5e-14 and the minimum sampled eigenvalue is .91690.

The Euclidean control has exactly zero off-diagonal weight gap. The pure-Bures control initially has a 2.89e-10 numerical gap; the separately saved step refinement reduces it to 1.73e-11, a factor 16.77, and both endpoints agree with the exact correlation law to 1.73e-11. Both records are preserved. The main source and first output were not overwritten. `refine_bures_control.py` performs only this additional control refinement.

Data: `validation_v1/results.json` and `bures_refined/results.json`. Each contains the source SHA-256, full inputs, endpoints, residuals, and numerical checks.

## Originality boundary

[Marcotte, Gribonval and Peyre (2024)](https://arxiv.org/html/2405.12888v1), Sections 2-3, supply the general non-Euclidean conservation-law/Lie-rank framework. Applying that framework is not new. Their Section 4.5 also preserves Euclidean-model laws under a natural gradient built from that model's observations; here the metric is built from the larger full representation, while the loss is diagonal-only.

[Nagaoka and Fujiwara, version 3 (2026), Theorem 8.6](https://arxiv.org/html/2307.03431v3) characterizes integrability of SLD score distributions. For commuting scores it supports the pure-Bures limiting foliation. It does not insert an additional Euclidean metric or give the displayed finite-blend rank result. [Ciaglia and Di Nocera (2022)](https://www.mdpi.com/2227-7390/10/15/2613) study pure monotone quantum metrics and compatible group actions; the Euclidean-plus-Bures metric is not one of their homogeneous monotone metrics.

[Zhang and Peyre (October 2026), Appendix D](https://arxiv.org/html/2610.09356v1) is a particularly recent conservation-law comparison: it computes the Lie algebra and complete covariance laws for split pure-Bures fields. Its state space, allowed gradients, and mobility differ from the present diagonal-observation finite blend. [Sabanayagam and Lucey (October 2026), Appendix E.9](https://arxiv.org/html/2610.04249v1) gives the aligned scalar exact joint-GN damping filter. This rules out advertising the filter itself as new; it does not give an off-diagonal coherence or endpoint-selection theorem.

The result is therefore retained as a potentially useful **loss of conservation under mixing of neutral geometries**, with an all-dimensional generic statement and a controlled static-loss consequence. Its general mechanism is known differential geometry, and a tiny matrix example alone would be insufficient. Remaining research risk is significance and priority of this specific combination, not failure of the checked local mechanism. Large training experiments are not warranted by this evidence alone.
