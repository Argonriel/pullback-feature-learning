# Depth, damping, and interpolation selection: bounded audit

November 21, 2026. Literature was checked before the corresponding numerical probe. No previous nonlinear-versus-tangent gate was reopened and no population-performance experiment was run.

**Decision:** reject scalar depth/balance and cumulative-prefix-metric proposals as innovation claims. A narrower matrix question survives the numerical screen: positive damping can make the interpolating product depend on positive loss weights even when the interpolation constraints and representation metric are identical. This is a specific unresolved research lead, not an established innovation. Generic non-Hessian geometry, a single counterexample, and a first-order perturbation integral are insufficient publication claims.

## Primary literature and exact novelty boundaries

| Original source | Relevant assumptions and result | Boundary for this audit |
|---|---|---|
| [Kerekes, Mészáros, Huszár, *Depth Without the Magic*](https://arxiv.org/abs/2111.11542) | Natural-gradient implicit bias in deep linear classification and matrix factorization, with rank and likelihood/Fisher assumptions. | Broad claims that curvature correction removes depth-induced bias are occupied. Logistic Fisher is not the unweighted representation metric used here; its classification theorems cannot simply be transferred. |
| [Gunasekar et al., ICML 2018, *Characterizing Implicit Bias in Terms of Optimization Geometry*](https://proceedings.mlr.press/v80/gunasekar18a.html) | Infinitesimal natural gradient with a Hessian metric is mirror flow. Under interpolation/convergence conditions, the Bregman projection characterizes the selected solution. | The diagonal model below is an exact application, not a new implicit-bias principle. Finite steps require separate treatment. |
| [Woodworth et al., COLT 2020, *Kernel and Rich Regimes in Overparametrized Models*](https://proceedings.mlr.press/v125/woodworth20a.html) | Initialization-dependent implicit bias in homogeneous diagonal linear models, including depth dependence. | An initialization/damping crossover inferred from the explicit mirror potential is not enough to establish novelty. |
| [Bah et al., *Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers*](https://www.instmath.rwth-aachen.de/~mwest/files/linear.pdf) | Balanced factor gradient flow induces a Riemannian flow on fixed-rank matrix manifolds. | The matrix geometry itself is established. This result does not by itself determine the finite-damping endpoint under arbitrary underdetermined measurements. |
| [Wu and Rebeschini, NeurIPS 2021, *Implicit Regularization in Matrix Sensing via Mirror Descent*](https://proceedings.neurips.cc/paper/2021/file/abf0931987f2f8eb7a8d26f2c21fe172-Paper.pdf) | Matrix mirror descent; equivalence to factorized gradient dynamics in a symmetric commuting measurement setting. | A commuting example is particularly unlikely to supply a new direction. The matrix probe below uses two noncommuting rank-one measurements. |
| [Li, Wang, Lee, Arora, NeurIPS 2022, *Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent*](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Paper-Conference.pdf) | Sufficient and related necessary conditions for reparameterized gradient flow to be a loss-independent mirror flow, formulated using commuting parameter vector fields. | Failure of a mirror representation and path dependence are not new concepts. The precise damped representation metric and its interpolation response need a more specific theorem to become a contribution. |
| [Xu et al., 2023, *The Power of Preconditioning in Overparameterized Low-Rank Matrix Sensing*](https://users.ece.cmu.edu/~yuejiec/papers/ScaledGD_Overparam.pdf) | Damped factorwise ScaledGD, small random initialization, low-rank sensing with measurement conditions, and recovery guarantees. | Generic “damping plus preconditioning retains low-rank bias” is occupied. Its factorwise preconditioner and recovery assumptions differ from the full joint representation metric here. |
| [Sabanayagam and Lucey, October 2026, *Optimizer Geometry Sets the Pace*](https://arxiv.org/html/2610.04249v1) | Explicit spectral phase laws assume full observation, balanced factors, alignment with the target, and compatible optimizer state. Appendix E.9 also gives exact joint damped Gauss–Newton scalar mobility. | This directly occupies the scalar depth/damping calculation below, including its distinction from block K-FAC. The underdetermined, nonaligned, loss-weight endpoint question is not its aligned phase-law theorem. Its general discussion already links optimizer motion to unobserved directions, so that description alone is not novel. |

The final row was found during the audit and strengthens the scalar rejection; it was not ignored because it is recent. Search coverage is bounded, and absence of an exact theorem in these sources does not establish originality.

## 1. Rejected: diagonal depth creates a new implicit regularizer

Let beta_i be the product of L positive diagonal factors, and let the full representation be diag(beta)x. Use the summed representation metric with coordinate input energies c_i, damping lambda, and arbitrary differentiable loss gradient q_i=d loss/d beta_i. Balanced initialization gives

    beta_dot_i = -q_i / [c_i + (lambda/L) beta_i^(2/L-2)].

For each coordinate, the factor Jacobian is a rank-one block with entries beta_i/w_li. A rank-one inverse gives the displayed formula. Squared factor differences are conserved because every layer has the same derivative of w_li squared.

The induced metric is the Hessian of the separable potential

    phi_i(beta) = c_i beta^2/2
                 + lambda L/[2(2-L)] beta^(2/L),          L != 2,
    phi_i(beta) = c_i beta^2/2
                 + (lambda/2)(beta log beta-beta),        L = 2.

Thus, conditional on convergence to interpolation inside the stated domain, ordinary mirror-flow results identify the Bregman projection from the initialization. The coefficient of the potential can be negative at L>2 while its second derivative remains positive; there is no convexity contradiction.

With factor initialization epsilon, beta_initial=epsilon^L. The Bregman anchor contains the scales lambda epsilon^(2-L) for L>2 and lambda log(1/epsilon) for L=2. The resulting nonuniform limits are real but follow immediately from this formula. A fixed depth-three, one-measurement example, beta1+2 beta2=1 with c=(1,4) and lambda=1e-4, moves its Bregman-projection beta1 from 0.49927 at epsilon=.1 to approximately 1.56e-14 at epsilon=1e-5. This is a calculation of an explicit convex projection, not a newly demonstrated learning advantage.

The direct full-parameter solve agrees with the predictor formula to at most 1.11e-16 for depths 2, 3, and 4; layer-balance derivative discrepancies are at most 4.31e-16. These checks validate the reduction and therefore support rejection of the broad novelty claim.

## 2. Rejected: accumulating intermediate representations gives a new zero-damping mechanism

For a scalar chain with nonzero factors, write prefix products a_l=w_l...w_1. The map from factors to the full prefix tuple is locally invertible. Its zero-damping pullback metric becomes Euclidean in a-coordinates. If the loss depends only on the final prefix, the earlier prefixes remain fixed and only the last changes. This is an ordinary coordinate-gradient calculation.

Depths 2, 3, and 4 verify the prefix velocities to at most 1.65e-16. Positive damping changes the transformed metric to c I + lambda (D w/D a)^T(D w/D a); writing this expression is not a new theory of useful feature learning. No claim is made that the scalar invertibility argument extends unchanged through rectangular bottlenecks or singular prefix maps.

## 3. Narrow unresolved lead: loss weights select different matrix interpolants

This is a different mathematical model from the original 80-parameter residual network. It is a two-factor, 2-by-2 matrix model X=W2 W1 with balanced factors and an SPD product. The representation is **every matrix entry**, with parameter metric J^T J+lambda I, where J=d vec(X)/d parameters. It is not the Gauss–Newton metric of only the two observed measurements. The representation metric is held fixed in definition when the objective's loss weights change.

Balanced factors imply the product Jacobian Gram operator

    K_X(A)=XA+AX,
    X_dot = -K_X(K_X+lambda I)^(-1) gradient_X loss
          = -(I+lambda K_X^(-1))^(-1) gradient_X loss.

Balance conservation follows from the common effective matrix driving all factor updates. The formula applies while the SPD reduction is valid. The effective metric is a Euclidean term plus the inverse Lyapunov operator, a familiar factorization geometry.

It is generally not a Hessian metric in the ambient symmetric-matrix coordinates. At X=diag(a,b), use the orthonormal coordinates (X11,X22,sqrt(2)X12). For H=I+lambda K_X^(-1), the necessary Hessian identity fails by

    partial_X11 H_zz - partial_z H_z,X11
        = lambda (b-a)/(2a(a+b)^2).

At a=.8, b=1.7, lambda=.3, this equals .027; three finite-difference resolutions recover it. The full eight-parameter and reduced matrix velocities agree to 2.23e-16 at the checked state. **This obstruction alone is not the research claim.**

The actual endpoint probe asks whether two positive weightings of the same interpolation conditions can yield different unobserved predictions:

    X_initial = .1 I,
    A1 = [[1,0],[0,0]],       A2 = .5 [[1,1],[1,1]],
    loss_w(X) = .5 [(<A1,X>-1)^2 + w (<A2,X>-1)^2],
    w in {1,9}.

Both losses have the identical constraints X11=1 and (X11+2X12+X22)/2=1. Their interpolants can have different X12, a genuinely unobserved linear functional. This is not an off-training population-risk comparison or a claim that either interpolant is better.

The fixed discovery probe used lambda in {0,.003,.03,.3}, RK4 steps .02 and .01, and maximum constraint residual 1e-11. After a nonzero contrast appeared, numerical refinement used the same cases and tolerance with steps .005 and .0025. No case was added or selected for a better effect. The refinement was a numerical follow-up, not an unannounced prospective test.

| Damping | X12, weight 1 | X12, weight 9 | Difference | Difference / damping |
|---:|---:|---:|---:|---:|
| 0 | .3000000000 | .3000000000 | 7.06e-12 | — |
| .003 | .3003596812 | .3003798105 | 2.01294e-5 | .00670979 |
| .03 | .3034306115 | .3036312368 | 2.00625e-4 | .00668751 |
| .3 | .3226999442 | .3243452276 | 1.64528e-3 | .00548428 |

All final residuals are below 1e-11. The minimum sampled path eigenvalue is .1, so this example stays away from rank collapse. Halving the final step changes any recorded X12 by at most 8.88e-9. Hence the displayed effect is not supported merely by the coarse integration.

An independent audit integrated the original eight factor parameters and explicitly solved the full damped parameter system, rather than reusing the reduced matrix flow. At lambda=0,.003,.3 its refined weight contrasts were 1.72e-10, 2.01296337e-5, and .001645283596. They agree with the reduced results within approximately 3e-10. Constraint residuals were below 1e-12, symmetry and factor balance were preserved numerically, and the SPD checks passed. Refinement from .0025 to .00125 changed X12 by at most 2.72e-9. The initial coarse parameter integration had a zero-damping weight gap around 5e-8; those outputs and source version were retained. The tighter result followed an explicit numerical refinement, not deletion of the less accurate control. Records are in `endpoint_parameter_validation/` and `endpoint_parameter_refined/`.

A separate nontrivial diagonal control has three positive coordinates, two overlapping constraints beta1+beta3=beta2+beta3=1, beta_initial=(.1,.1,.1), and the same damping/weight choices. Its interpolation set also has dimension one. Its separable mirror potential predicts weight-independent endpoints; the largest refined weight discrepancy is 3.66e-9. This control was added after the matrix observation to check the proposed distinction.

The small-lambda candidate response law is

    P_N [X_infinity(lambda,w)-X_infinity(0)]
      = lambda P_N integral_0^infinity
          K_{X_0,w(t)}^(-1) gradient loss_w(X_0,w(t)) dt
        + O(lambda^2),

where P_N projects onto the measurement nullspace and X_0,w is the zero-damping trajectory. The sign is positive because -(I+lambda K^-1)^-1 G=-G+lambda K^-1 G+O(lambda^2). Interchanging differentiation with the infinite-time limit requires uniform interior and decay conditions; numerical SPD checks are not a proof of those conditions. In this example the null vector in orthonormal coordinates is u=(0,-2,sqrt(2)), and dividing its projected integral by u^T u=6 gives the response of X12. An independent mathematical review evaluates this coefficient rather than fitting it to the finite-damping endpoints.

That independent calculation uses the exact zero-damping matrix-exponential paths and quadrature, giving C1=.120532872308995, C9=.127242740128842, and C9-C1=.00670986781985. The finite-damping ratio at lambda=.003 is .00670978855329, differing by 7.93e-8. Quadrature refinement from 64 to 128 nodes per interval changes the coefficients by at most 8.97e-14, with a bounded tail below 2.23e-20. Initial 16/32-node checks did not satisfy the declared quadrature tolerance and were refined; the tolerance was retained.

There is also a concrete sign calculation, rather than only an observed coefficient. Write the zero-damping affine plane as X=[[a,b],[b,s+b]], where s=.1, and D=(a+s+b)[a(s+b)-b^2]. The one-form alpha=<U,K_X^(-1)dX>/6, where U=[[0,1],[1,-2]], has

    alpha_a = -b(s+2b)/(6D),
    alpha_b = [a(s+2b-a)+b^2]/(6D),
    partial_a alpha_b-partial_b alpha_a
       = s(s+3b-a)/[6(a+s+b)^2(a(s+b)-b^2)].

For weight 1 the trajectory is the straight segment a=s+3b. For weight w>1 the explicit residual solution puts the trajectory strictly above that segment and keeps X at least s I. The loop following the straight segment forward and the weighted trajectory backward has positive orientation. Its enclosed region has positive curl, so Cw-C1=integral_loop alpha>0. The one-form's normalization and this sign were independently checked. The completed mathematical review also supplies an interior-SPD bootstrap and infinite-time remainder for each fixed 0<s<1 and w>1, resolving the local regularity condition for this family. This proves a positive first-order damping response locally; it does not prove monotonicity for all positive damping or a new generalization result. The details and independent numerical coefficient are in `ENDPOINT_MATH_REVIEW.md` and `endpoint_response_review_results.json`.

## Keep/reject boundary

The endpoint effect survives this bounded test. It is not implied by an aligned scalar phase law, and a general Riemannian framework is not, by itself, a theorem that settles this exact endpoint question. However, the first-order integral is also a generic perturbation expansion; noncommuting mirror obstructions are established. The surviving question would need a substantive condition or quantitative law predicting interpolation sensitivity beyond displaying one matrix or renaming that obstruction. A second issue is whether that law adds value outside a hand-built matrix model. Neither is established here.

Noncommuting measurement matrices alone are not a sufficient condition. Let their linear span S contain X_initial and be closed under the Jordan product AB+BA. For SPD X in S, K_X maps S into S. Its positive-definite restriction is invertible, so K_X^(-1) also preserves S. Since every loss gradient lies in S, the entire damped flow remains in S and its measurement-nullspace component stays fixed. All interpolating endpoints then coincide, even when the generators of S do not commute. This elementary exception is another reason that a useful characterization must concern the projected inverse-Lyapunov response, not just pairwise matrix commutators.

Therefore retain only the precisely stated **research question**, not an innovation claim or authorization for larger experiments. The generic perturbation and path-integral machinery is established; a theorem-shaped explanation for this example does not by itself establish original research. If the independent originality review identifies the specific endpoint consequence as a direct existing corollary, retire it too. Do not revive rejected parity, replica-noise, local-curvature-sign, unread-coordinate, or tangent-superiority claims under this label.

## Reproducibility

Source files: `experiments/new_direction_screen_20261121/depth_screen.py`, `depth_endpoint_probe.py`, and `depth_commuting_control.py`. Each requires a fresh external output directory and records its source SHA-256. Initial source and output records were preserved.

Data: `depth_checks/depth_results.json`, `depth_endpoint/depth_endpoint_results.json`, `depth_endpoint_refined/depth_endpoint_refined_results.json`, and `depth_commuting/depth_commuting_results.json` in this report's staging directory. Refined endpoint runs call the same unmodified `integrate` routine with smaller steps and retain the original probe's source hash. Parent-level independent parameter integration and the separate response/novelty reviews are additional checks, not evidence silently substituted for this report's original computations.
