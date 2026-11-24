# Independent adversarial review: autonomous endpoint submersion

Reviewer: exceptional-state subagent, independently of the static-loss derivation.
Date: November 21, 2026 (America/Los_Angeles).
Disposition: **pass with scope qualifications; no fatal mathematical gap found**.

## Material reviewed

- `../static_loss/THEOREM.md`, SHA-256 at review: `a2f25f062dbb4bc8e4fd258ef82f4acfe585158a0666fcfc4b26059480416390`.
- Root `validate_static_range.py`, source SHA-256 `31abe9fc7757cfaa792c1936f9136844d49fa8e187ddd0ef96d71b0442ba9de5`.
- `../validation_v1/results.json`, SHA-256 `31438dc88aa5e2f04dca3f013af9c3b219a985f6100ad421654a985ab54ee679`.
- `../endpoint_difference_refinement/results.json`, SHA-256 `45a6fc13b41bd0801541c823e2beac988dee8f615f26da478f1503230be8c5db`.

The root source hash matches the hash recorded in the original results. The refinement input hash matches the original results file. The review is mathematical and methodological; it is not a proof-assistant certificate or independent priority certification.

## 1. Observed normalization and differential-rank lemma

A is positive definite by compression of the SPD mobility M to diagonal matrices. The use of A^(-1), its positive square root P, and the transformed positive loss weight C=P B P is therefore justified. This is normalization of the diagonal observation speed, not congruence whitening of X or replacement of the full representation metric.

I independently rederived the signed-area identity for u=d-exp(-Ct)d. Since du=C exp(-Ct)d dt,

    integral u du^T = dd^T-GC,
    integral du u^T = dd^T-CG,

and therefore Omega=(CG-GC)/2, with exactly the stated sign and factor one half.

For E=E^T, differentiating the Lyapunov equation yields CZ+ZC=-(EG+GE). Vanishing of the area derivative is equivalent to EG+CZ=0. The candidate Z=-C^-1 E G is symmetric if and only if EGC=CGE. Upon the stated congruence this is exactly [T,S]=0, where T=C^-1/2 E C^-1/2 and S=C^1/2 G C^1/2.

The crucial simple-spectrum claim is valid. When d is cyclic, G and S are positive definite. The rank-one identity CS+SC=ww^T forbids a positive S eigenvalue of multiplicity >=2: compression onto that eigenspace gives a positive definite matrix of rank >=2 equal to a rank-at-most-one matrix. This uses positivity of C and S and does not assume that C and G commute. Thus the symmetric commutant of S has dimension n, producing rank n(n-1)/2.

For Krylov dimension k<n, the C-invariant Krylov space has an invariant orthogonal complement. The derivative's 12 block obeys

    C1 Z12+Z12 C2=-G1 E12,
    D Omega[E]12=C1 Z12.

The map from E12 to this output is invertible. The 11 block has rank k(k-1)/2 and the 22 block is zero. Therefore the exact claimed formula

    rank D Omega = k(k-1)/2+k(n-k)

is correct, including k=0 and k=n-1. Repeated eigenvalues of C cause no overlooked exception because k already incorporates their excitation structure. The all-positive-spectrum assumption on C is essential to the argument.

## 2. Endpoint expansion and infinite time

For the normalized lift dz=sum beta_j dq_j, the second-order coefficient is

    sum_(i,j) H_i(beta_j) integral x_i dx_j.

The identity integral x_i dx_j = v_i v_j/2 + (P Omega P^T)_ij gives the symmetric weight-independent term and the curvature term with no additional factor of two. Pairing ordered i,j indices with an antisymmetric matrix produces exactly sum_(i<j)(H_i beta_j-H_j beta_i)(P Omega P^T)_ij. The old two-dimensional contrast is consequently -F sqrt(det A) epsilon^2/6 for C=diag(1,2), d=(1,1), relative to C=I. The reported sign is consistent.

The second-order observed path correction is allowed to depend on the weight. Its contribution under constant beta is zero at the endpoint because the observed endpoint is exactly q0+epsilon v. The proof correctly uses this endpoint identity instead of assuming that the whole second-order observed path vanishes.

The local convergence and C1 remainder are adequately justified. The differential inequality V_dot<=-2ab V is correct because B^2>=b B. It gives a uniform integrable velocity bound O(|epsilon|) for a compact weight neighborhood. The first-exit bootstrap therefore supplies global-in-time trajectories in a fixed compact subset of SPD for sufficiently small |epsilon|. No separate arbitrary-target global convergence result is needed.

For parameter differentiability, the residual block is uniformly exponentially stable in the B norm. Its coupling to derivatives of the hidden variables carries a factor r=O(epsilon exp(-gamma t)). Differentiated hidden variables may approach nonzero limits, but their forcing remains integrable. Variation of constants and Gronwall with an integrable coupling coefficient give bounded hidden derivatives and exponentially decaying residual derivatives. Repeating this for the finite mixed orders needed for three epsilon derivatives plus one C derivative gives an integrable derivative of the velocity. This supports a uniform O_C1(epsilon^3) endpoint remainder. The argument does not require the full state Jacobian to be exponentially stable; hidden directions are neutral. A future journal version could package this calculation as a separate local endpoint-smoothness lemma, but this is an exposition improvement, not an identified gap.

Once the leading C derivative is onto, its selected nonzero minor persists for sufficiently small nonzero epsilon. The submersion theorem then concerns the actual fixed-target, finite-epsilon endpoint map. It is more than a statement about the formal leading coefficient.

## 3. Quantifiers and attempted counterexamples

The following restrictions are essential and correctly retained:

- n>=2, fixed finite lambda>0, and invertible curvature at the specified X0.
- Every nonzero target direction v is permitted, but the positive reference weight C0 must satisfy k>=n-1. Cyclic choices exist for every v!=0 and form an open dense set.
- The small-target threshold can depend on X0, lambda, v, C0 and the bounded weight neighborhood. No uniform assertion over the entire SPD weight cone follows.
- The conclusion is a local open image around an attained endpoint within the target fiber; it is not the entire feasible fiber.

I tried the natural failure cases. For v=0 all trajectories are stationary. For diagonal X0 all hidden entries remain zero. For scalar C=alpha I, the area derivative is (Edd^T-dd^T E)/(4alpha), of rank n-1, which is deficient for n>=3. For the coherent equal-diagonal 2x2 state the curvature is zero but a third bracket is nonzero. For the equicorrelated 3x3 state the first-bracket rank is deficient despite complete support. Each case is excluded or properly qualified by the theorem.

Two converse statements would be false or unproved and should not be inferred:

1. Rank deficiency of the leading area derivative does not establish rank deficiency of the actual finite-epsilon endpoint map. Higher-order terms may supply missing directions.
2. Singularity of the curvature map is not proved to prevent finite-epsilon endpoint submersion. Invertible curvature is a sufficient hypothesis used here, not an established necessary one.

Scalar loss rescaling provides one endpoint-kernel direction. The area's cyclic kernel has dimension n, so full hidden-output rank does not conflict with this symmetry. No claim of injectivity in the loss weight is made.

## 4. Exact image of the leading signed area

The spectral characterization is algebraically correct. Necessity follows because Q=dd^T/2+Omega=CG is similar to C^1/2 G C^1/2>=0. In particular Q is real diagonalizable and has nonnegative spectrum, including semisimple zero eigenvalues.

Conversely, if Q=V Lambda V^-1 with real V and nonnegative diagonal Lambda, set C=VV^T and G=V^-T Lambda V^-1. Then CG=Q, GC=Q^T, C>0, G>=0, and the rank-one symmetric part gives CG+GC=dd^T. Uniqueness of this positive-spectrum Lyapunov solution identifies G with the exponential integral. This proves attainability of that area, including noncyclic cases with zero eigenvalues. Positive eigenvalues are automatically simple by the same compression argument; no extra assumption is silently omitted.

The Frobenius outer bound follows from trace Q^2=||d||^4/4-||Omega||_F^2 and the nonnegative real eigenvalues. Equality at the n-dimensional Cauchy bound would force repeated positive eigenvalues, so it is excluded for n>=2. This remains only an outer bound, as the theorem states.

For n=2, the discriminant of Q in coordinates d=(r,0) gives precisely |Omega_12|<r^2/4. At equality the repeated positive eigenvalue is defective, so a positive weight cannot realize the boundary. At zero area, the allowed zero eigenvalue is semisimple. Multiplication by det P converts this to the stated interval for the curvature-dependent second-order hidden coefficient.

This image theorem describes leading coefficients as epsilon tends to zero with the chosen weight fixed. It does not determine the full endpoint range at a fixed noninfinitesimal target, and it cannot justify interchanging that limit with weights tending to the SPD boundary. The note correctly makes that distinction. The spectral factorization has a close classical predecessor and should be positioned according to the separate primary-source audit rather than advertised as an independently new matrix theorem.

## 5. Block-partition corollary independently confirmed

The parent proposed the following extension during this review. Let P be a coordinate partition and m_P=sum_r b_r(b_r-1)/2. Restrict to the invariant block-SPD manifold. At a generic state within it, its within-block curvature map

    R_P: skew(n) -> R^(m_P)

is onto, with cross-block curvature columns zero. For every v!=0 choose any cyclic transformed C0. The composition R_P o Lambda^2(P) o D Omega is onto: the first and last maps are onto and the middle map is invertible. The same C1 expansion proves actual local endpoint submersion onto every within-block hidden coordinate for sufficiently small nonzero epsilon. This is precisely the maximal possible output dimension because the cross-block entries stay identically zero.

This remains true when v is supported in only one initial block, provided the permitted loss weight B is a full SPD matrix. Its cross-block entries can couple observed residual coordinates and cause temporary excursions of observed coordinates in other blocks. It would not follow under an additional restriction to block-diagonal B. The initial geometric block structure is preserved throughout. No new numerical experiment is needed for this direct corollary.

## 6. Numerical review and limits of the margin

The variational equation in `validate_static_range.py` differentiates both the mobility and the residual-gradient term with respect to the loss weight. The derivative of M is implemented with the positive resolvent formula. The three perturbation directions are right singular vectors of the area derivative, and the predicted endpoint derivative applies the curvature map and the exterior-square coordinate transformation. The code's coordinate conventions are consistent with the theorem.

The one 3x3 experiment has nonzero curvature determinant -1.3640844e-8. Its smallest predicted scaled endpoint singular value is 1.5157404e-5. At epsilon=.08,.04,.02, the actual scaled derivative errors are 2.12243e-5,1.07082e-5,5.37837e-6, showing the expected first-order decrease in epsilon after division by epsilon^2. At epsilon=.02 the smallest measured scaled singular value is 1.4992140e-5. Halving the integration step changes the unscaled derivative by 4.4288235e-13, or 1.1072059e-9 after scaling, yielding the recorded margin ratio 13540.5. Diagonal interpolation and diagonal sensitivity residuals are around 1e-14, and the minimum sampled eigenvalue is .9406163.

The first centered endpoint-difference check used increments .002 and .001, where cancellation left errors 3.15e-13 and 2.49e-13. Its factor-four-convergence check is explicitly false in the preserved first results. The additional record uses .04,.02,.01; errors 4.21756e-11,1.04377e-11,2.65805e-12 recover the expected behavior, and all perturbed weights remain SPD. This is a reasonable refinement of a roundoff-limited check, not an erased failure.

These are strong consistency checks. Step-difference magnitudes are not rigorous upper bounds on the exact ODE integration error, and sampled minimum eigenvalues are not continuous-time certificates. The all-dimensional existence and submersion claims rest on the proof; the finite experiment does not independently certify an explicit epsilon0.

## Final assessment

The exact area-rank calculation and its autonomous-loss consequence survive this independent adversarial review. The new endpoint result is materially stronger than the original controlled-bracket statement or one two-dimensional pair of losses. Retain it with the stated generic initialization and local-target scope, and with the classical spectral image characterization clearly separated from any originality claim. The block-partition corollary gives a valid precise initialization-dependent output dimension. No all-state arbitrary-dimensional accessibility theorem or entire-fiber fixed-loss reachability theorem has been established.

## Integrated theorem comparison

The final root `../THEOREM.md` was also read after integration. Reviewed SHA-256: `7a564dcbd069dc23eafa9f65f943ad8477382d3fa76a13af9ef6d234089bd1df`.

No discrepancy from the reviewed static-loss theorem was found. I additionally checked the generic determinant reconstruction: the first inverse-Lyapunov term gives beta_i^(pr)=lambda Z_pr(delta_ip/a_p+delta_ir/a_r)/(2(a_p+a_r+lambda)); its vertical derivative terms cancel in the first-order bracket; the q derivatives give the displayed coefficient lambda Z_ij(a_j-a_i)/(2a_i a_j(a_i+a_j+lambda)^2). Distinct positive a_i and nonzero coherences supply a nonzero analytic determinant, with the stated generic and relative-block quantifiers. The integrated theorem correctly limits the generic set separately for each lambda, includes the reviewed block-relative submersion, and does not infer actual endpoint-rank deficiency from a deficient quadratic coefficient. Its spectral-range section now explicitly attributes the positive-matrix factorization tool to its classical source. The integrated theorem passes this independent consistency check without a requested correction.
