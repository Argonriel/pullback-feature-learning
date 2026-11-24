# Candidate: universality of neutral spaces across spectral metrics

## Precise scope

Let m(x,y) be positive, symmetric, real analytic on the connected positive quadrant, homogeneous of degree -1, and normalized by m(x,x)=1/x. Let M_X be its spectral multiplier on real symmetric matrices. The normalized BKM kernel is

    ell(x,y)=(log x-log y)/(x-y), with ell(x,x)=1/x.

These hypotheses include every normalized symmetric monotone quantum metric kernel. The actual optimization metric considered here is I+lambda M_X; no claim is made that its additional Euclidean term preserves quantum-channel monotonicity.

For m not identically ell, any S subset Sym(n), and isotropic X0=sI, the following are equivalent:

1. The normal one-form P_(S-perp) M_X dX is closed on (sI+S) intersect SPD.
2. For all Z,U in S, ad_Z^2 U and {Z,ad_Z^2 U} belong to S.
3. S is an associator ideal of J=Jordan(S,I), equivalently J intersect S-perp is central.
4. The first damping derivative of the endpoint is independent of fixed SPD quadratic loss weights for every sufficiently small target.

The BKM exception is exact: if m=ell, the one-form is a projected logarithmic differential and every S passes, without either algebraic condition.

For passing S, finite-damping conserved coordinates are P_(J intersect S-perp)[X+lambda log X]. Every SPD interpolating limit is the corresponding Bregman projection for

    Phi_lambda(X)=||X||_F^2/2+lambda tr(X log X-X).

This uses the normalized kernel convention; the preceding two-factor audit used M=(L_X+R_X)^-1, one half of normalized SLD, and therefore had lambda/2 instead of lambda. The general-metric conclusion here is conditional on an SPD interpolating limit, with local small-target convergence available by uniform ellipticity. The separate global theorem concerns the original inverse-Lyapunov model, not every kernel in this larger family.

## Proof of universality

Put a=(x+y)/2 and v=(x-y)/(x+y). Homogeneity and symmetry give m(x,y)=a^-1 h(v), with h even and real analytic, h(0)=1. The BKM kernel is a^-1 atanh(v)/v. Unless the kernels agree identically, there is a finite first nonzero coefficient:

    ell(x,y)-m(x,y)=a^-1 [b_k v^(2k)+O(v^(2k+2))], k>=1, b_k!=0.

Real analyticity on the connected domain rules out a nonzero difference flat to all orders at v=0.

Closedness forces the primitive by integrating along the radial segment from sI. On that segment X commutes with the tangent, and every normalized kernel acts as X^-1 on commuting tangents. Thus the primitive is P_N(log X-log(s)I), and closedness is equivalent to

    P_N (Dlog_X-M_X) U=0, for X in sI+S, U in S.

Let A_Z=ad_Z^2 and C_Z=L_Z+R_Z. The first two homogeneous terms of R_X=Dlog_X-M_X at X=sI+Z are

    R_(sI+Z) U = r_k A_Z^k U
                   -(2k+1) r_k/(2s) C_Z A_Z^k U
                   +O(||Z||^(2k+2)||U||),
    r_k=b_k/(2^(2k) s^(2k+1)).

No intervening term comes from h, which is even. Hence exact normal cancellation forces A_Z^k S subset S and C_Z A_Z^k S subset S. A_Z is positive self-adjoint. Invariance under A_Z^k implies invariance under A_Z by finite-dimensional spectral calculus, including its zero eigenspace. On range(A_Z), use its pseudoinverse to recover C_Z A_Z S subset S from C_Z A_Z^k S subset S. Thus the same two old closure tests are necessary regardless of how late the kernel difference first appears.

For sufficiency use the proved algebraic classification. Symmetric spectral multipliers preserve a matrix Jordan algebra: spectral projectors of X belong to J, and each symmetric Peirce component P_i U P_j+P_j U P_i belongs to J by the quadratic representation. For N central in J, M_X N=X^-1 N=Dlog_X N. Pairing with every N in J intersect S-perp establishes normal cancellation; the component outside J vanishes by invariance. The same pairing gives the exact finite-damping conserved coordinates.

The static-weight equivalence is not inferred solely from arbitrary controlled paths. Its explicit witness is below. The general-k kernel argument was independently cross-checked by the algebra lane, including the pseudoinverse and Peirce-invariance steps.

## Static weights and delayed detectable order

Use a fixed weight perturbation B_tau=I+tau D, with D self-adjoint. Put V=DE and target Delta=epsilon E. At zero damping, the path is

    X(t)=sI+epsilon(I-exp(-B_tau t))E.

At tau=0 its variation is epsilon t exp(-t) V. Write H_j=sum_(i=1)^j 1/i. Direct differentiation of the path integral gives

    partial_tau C_(B_tau)(epsilon E)|_(tau=0)
      =-r_k [H_(2k+1)-1]/(2k) epsilon^(2k+1) P_N A_E^k V
       +r_k [H_(2k+2)-1]/(2s) epsilon^(2k+2) P_N C_E A_E^k V
       +O(epsilon^(2k+3)).

The scalar integrals are beta-function derivatives; for example, with f=1-exp(-t) and h=t exp(-t), the first is

    integral f^(2k-1)(f h'-h f') dt = -[H_(2k+1)-1]/(2k).

Any V can be realized as DE for nonzero E by a self-adjoint D; sufficiently small fixed tau keeps B_tau SPD. If Lie-triple closure fails, invariance of A_E^k fails for some E,V, giving a nonzero order 2k+1 witness. If Lie-triple closure holds but the second condition fails, invariance of C_E A_E^k fails and gives order 2k+2. Thus delayed Taylor detection does not enlarge the exact neutral class.

## Positive monotone kernels with arbitrarily delayed obstruction

The normalized symmetric resolvent kernels have the form

    c_u(x,y)=1/[a(1-u v^2)], 0<=u<=1.

They are the standard extremal kernels after the parameter change u=((1-t)/(1+t))^2. Any positive probability mixture is a normalized symmetric monotone metric. BKM is the mixture with moments

    integral u^j dnu_BKM(u)=1/(2j+1),

namely u=r^2 for uniform r in [0,1].

A concrete simple mixture is

    m=(2/3)c_SLD+(1/3)c_RLD,
    c_SLD=2/(x+y), c_RLD=(x+y)/(2xy).

It agrees with BKM through v^2, but ell-m starts at -(2/15)a^-1 v^4. Therefore k=2, r_2=-1/(120s^5). Its explicit static-weight coefficients are

    +(77/28800s^5) epsilon^5 P_N ad_E^4 V,
    -(29/4800s^6) epsilon^6 P_N {E,ad_E^4 V}.

More generally, use the positive nodes of a 2m-node symmetric Gauss-Legendre rule and their paired normalized weights. The resulting m-atom distribution of u matches BKM moments j=0,...,2m-1. At j=2m the quadrature error is strictly nonzero, since the squared node polynomial has positive integral and zero quadrature value. Hence k=2m and the first kernel discrepancy has order 4m. The corresponding endpoint obstructions can begin only at target orders 4m+1 or 4m+2.

Gaussian quadrature and rational approximation of the logarithm are established tools. The proposed additional conclusion is the optimizer-identifiability consequence: no fixed finite order of the target Taylor coefficients of the first-damping endpoint-weight response, uniformly over this metric class, can certify neutrality, even though the exact algebraic recognition test and neutral spaces are universal. This consequence and the universality theorem form one candidate, not two items counted toward the user's requested three.

## Minimal validation

The positive SLD/RLD mixture was tested on one non-Lie-triple measurement space and one Lie-triple space failing the second condition. These reuse elementary matrix bases as controls for a new metric and new fifth/sixth-order prediction; the previous rejected learning experiments were not reopened.

At the finest target halving, the observed exponents were:

| Space | Normalized SLD | Matched positive mixture |
|---|---:|---:|
| Non-Lie-triple | 2.95923 | 4.92900 |
| Lie-triple star | 4.00083 | 6.00182 |

Independent quadrature orders 24 and 48 differed by at most 6.42e-13 relatively. The explicit mixed-derivative formula was also checked without fitting the exponent: at epsilon=.01, relative truncation discrepancies were 8.31e-4 and 1.91e-4, decreasing with target size. Positive moment rules with 1, 2, 3, and 4 atoms matched the stated moments to 6.11e-16 or better and had nonzero first unmatched moments.

All results and response matrices are retained in `validation/`. These computations validate scaling and constants, not the universal theorem or its priority.

## Original-source comparison and novelty assessment

[Lesniewski and Ruskai, Monotone Riemannian Metrics and Relative Entropy](https://arxiv.org/pdf/math-ph/9808016), equations 36-40 and Theorems 2.9-2.13, provide the spectral/resolvent representation and convex metric family. Those ingredients are known. [Grasselli and Streater, On the Uniqueness of the Chentsov Metric](https://math.mcmaster.ca/~grasselli/GrasselliStreater01.pdf), sections 3-4, identify BKM through duality of the specified mixture/exponential connections on the full state manifold. That is not the classification of arbitrary affine measurement subspaces by normal projected closedness. In particular full S makes our normal condition trivial for every kernel while full-metric mixture/exponential duality remains restrictive.

[Fawzi, Saunderson, and Parrilo, Semidefinite Approximations of the Matrix Logarithm](https://arxiv.org/pdf/1705.00812), sections 2-3, use integral representations and Gaussian quadrature for operator-compatible logarithm approximations. The moment-matching construction is not an innovation claim. Their approximation problem does not state the static-weight obstruction hierarchy described here.

The previous Jordan/associator classification and general conservation machinery remain prior ingredients. The potentially additional result is the all-kernel neutral-space universality with its explicit delayed static-weight witness. A bounded search of monotone metric, normal projection, Lie-triple/Jordan, Hessian, and integrability formulations did not locate the exact theorem. That observation does not certify priority. This is a proved, specialized candidate of moderate mathematical interest; no optimization performance benefit is established.
