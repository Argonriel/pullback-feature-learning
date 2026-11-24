# Independent mathematical and implementation review

Date: November 21, 2026. Reviewed `THEOREM.md`, `spectral_delay_check.py`, and all entries of `validation/results.json`. No extra experiment was needed. The reviewer independently checked the algebraic closure theorem in the previous round and has now rederived the general-k response coefficients rather than extrapolating from the k=1 case.

## Verdict

No fatal mathematical defect was found in the narrowly stated theorem. The analytic spectral-kernel universality, the first-damping static-weight equivalence, and arbitrarily delayed detection by positive monotone kernels are supported by the supplied proofs. The finite-damping endpoint statement remains conditional on an SPD interpolating limit, with local convergence as stated. This review does not establish priority or generalization benefit.

One wording refinement is recommended: the impossibility of certification at a uniformly bounded Taylor order is proved for **target-amplitude jets of the first-damping endpoint-weight response**. It should not be generalized to every conceivable finite diagnostic, to a full symbolic description of the metric, or to a finite-damping target expansion without an additional argument.

## General-k coefficients checked independently

Let f=1-exp(-t), h=t exp(-t), Z=epsilon(fE+tau hV)+O(tau^2), and U=dZ/dt. Since [fE,f'E]=0, differentiation at tau=0 gives

    partial_tau ad_Z^(2k) U
       =epsilon^(2k+1) f^(2k-1)(f h'-h f') ad_E^(2k)V.

Multiplication by C_Z adds epsilon f C_E. For p>=0, integration by parts and u=f give

    integral f^p(f h'-h f') dt
      =-(p+2) integral f^p f' h dt
      =-[H_(p+2)-1]/(p+1).

Thus the coefficients in the theorem are exactly

    -r_k [H_(2k+1)-1]/(2k),
    +r_k [H_(2k+2)-1]/(2s).

The sign is consistent with C_B=-integral M_X dX: after the exact logarithmic endpoint term cancels between losses, the varying term is +integral(Dlog_X-M_X)dX. For k=2 and r_2=-1/(120s^5), this yields +77/(28800s^5) and -29/(4800s^6), respectively.

If A_E^k preserves S for every E, the self-adjoint positive operator A_E preserves S by spectral calculus. If A_E already preserves S, invariance under C_E A_E^k is equivalent to that under C_E A_E: apply the pseudoinverse of A_E^(k-1) on its range and note that A_E vanishes on its kernel. Consequently a failure of either old closure condition supplies a nonzero displayed coefficient for some E,V. A symmetric D with DE=V exists for every nonzero E, and I+tau D remains SPD for small tau. This validates the fixed, positive-quadratic quantifiers.

## Analytic and algebraic scope

Homogeneity makes the kernel a^-1 h(v) on a>0, -1<v<1. Symmetry makes h even; normalization fixes h(0)=1. The real-analytic identity theorem on this connected interval ensures a finite first nonzero coefficient unless the kernel is exactly BKM. Smoothness without analyticity would not justify this step, so the assumption is substantive and correctly explicit.

The affine SPD domain is convex and contains sI. A closed normal one-form therefore has the stated radial primitive, since commuting tangents see only m(x,x)=1/x. This is normal projected closedness, not full Hessian integrability.

The sufficiency proof applies to all stated kernels. Spectral projectors of X belong to the generated unital matrix Jordan algebra; the symmetric Peirce components preserve that algebra by quadratic representations. For a central N, only diagonal spectral blocks occur and M_XN=X^-1N. Components orthogonal to J cannot be generated. This establishes both the primitive identity and the finite-damping invariant without asserting that every kernel is globally Hessian.

The normalized symmetric Petz family is included: its spectral kernel has the required positivity, symmetry, homogeneity, normalization, and analyticity. The Euclidean addition is not asserted to remain channel-monotone.

## Positive quadrature construction

The resolvent parameterization can be checked directly:

    ((1+t)/2) [1/(x+ty)+1/(tx+y)]
      =1/[a(1-u v^2)],   u=((1-t)/(1+t))^2.

Positive probability mixtures therefore remain in the normalized symmetric monotone family. This is supported by [Lesniewski and Ruskai, equations 36-40 and Theorems 2.9-2.13](https://arxiv.org/pdf/math-ph/9808016), which were independently inspected for this review.

A 2m-node Gauss-Legendre rule on [-1,1] is exact through degree 4m-1. Pairing its positive and negative nodes under the normalized uniform measure gives positive weights summing to one on u=r^2. Moments u^j match for j<=2m-1. The error at j=2m is strictly positive: the square of the monic node polynomial has degree 4m, positive integral, and zero quadrature sum, while its lower terms are integrated exactly. Hence k=2m, as claimed.

The quadrature/Padé ingredient is classical; [Fawzi, Saunderson and Parrilo, Appendix B.1, Proposition 5](https://arxiv.org/pdf/1705.00812) explicitly relates positive Gaussian quadrature, moment matching, and Taylor matching. The report appropriately excludes this construction itself from originality claims.

## Code and numerical evidence

- The response routine integrates R=Dlog-M along the exact zero-damping path. Its unweighted reference path is radial, where R annihilates its tangent, so not explicitly subtracting that reference is correct.
- The bases used in both cases are Frobenius orthonormal, so the coded normal projection is correct.
- `r_apply` uses the correct moment sequence: zero for normalized SLD beyond degree zero, and 1/3 for the SLD/RLD mixture at all positive moments. The first v^2 coefficient cancels exactly.
- The 79-term sum is amply sufficient for these declared targets. A conservative bound gives eigenvalues in [.76,1.24] and |v|<=.24, making the omitted kernel tail smaller than 1e-98. This is not a source of the reported fifth/sixth-order effects.
- The saved refinement discrepancies, observed powers, mixed-derivative residuals, and positive moment tables match the prose. The mixed-derivative relative errors fall by approximately four on target halving, consistent with the two retained orders and the extra symmetry in the star case.
- Minor implementation qualification: the aggregate `passed` Boolean checks powers, quadrature refinement, and moment matching, but does not gate on the mixed-derivative residuals. Their saved values were inspected separately here and are consistent. Do not describe that Boolean alone as a gate covering every reported check.

The minimal computations support the formulas, while the all-kernel and arbitrary-delay claims rely on the analytic proof. No additional simulation or numerical sweep is required to address the points reviewed above.
