# A different symmetry task with a tractable quotient

The input is a two-harmonic waveform represented by four real coefficients:

`x(r,phi) = (cos(phi), sin(phi), r cos(2phi), r sin(2phi))`.

The phase phi is uniform on [0,2pi), independently of r. With equal class prior, u=r-.4 has Beta(2,5) density for the negative class and Beta(5,2) density for the positive class. Thus r lies in [.4,1.4], its marginal density is

`rho(r) = 15 u(1-u)[u^3+(1-u)^3]`,

and its posterior is `p(r)=u^3/[u^3+(1-u)^3]`. Population prevalence is exactly one half and the Bayes classification error is exactly 7/64=.109375. Training samples use 128 observations from each class, shuffled; conditional observations are independent, but the fixed class count is a stratified design. Calibration and test inputs use the iid population mixture. Labels remain stochastic after conditioning on the input, except at probability-zero endpoints.

The true continuous symmetry is `T_theta = R_theta direct-sum R_(2theta)`. It preserves harmonic amplitude while shifting phase. In particular,

`T_pi x = (-x_1,-x_2,x_3,x_4)`.

This is not the Euclidean antipode -x. The correct second view preserves the second harmonic. Pair averaging projects only onto this valid half-turn subgroup, not the full continuous group: remaining second-harmonic phase dependence is still possible.

This is a transfer of ambient dimension, orbit geometry, group representation, partner transformation, input law, and posterior law. It is not an ellipse or a relabeling of the same two-dimensional ring data. Its limitations are equally explicit: the quotient is still one-dimensional, and r is a monotone function of the full input norm sqrt(1+r^2). It does not establish transfer to a multidimensional quotient, general unknown manifolds, or real-world data.

## Fixed model and training context

The model has two ordinary four-dimensional residual blocks, each

`z_next = z + .5 B tanh(A z+b)`,

with eight hidden units, giving 144 parameters in total. The logit is the first output coordinate. This is the direct dimensional generalization of the earlier residual model, not a network given the invariant r as an additional input. Training uses raw four-dimensional inputs and labels.

The pullback direction solves `(J^T J + .1 I) v = -gradient`, where J contains all four output coordinates for every training input and the Gram matrix is a sum. The logistic loss is a mean. The independent GD implementation uses reverse-mode gradients. Two learning rates per optimizer were evaluated on four separate pilot seeds, retaining all 16 fits. PB8 and GD.3 were chosen using supervised pilot conditional risk, with 500 updates each. PB is the primary confirmation training recipe; the visibly underfit GD recipe is context only. Equal update counts are not equal compute, and neither this limited pilot nor transfer-task results justify an optimizer superiority claim.

## What transfers unchanged in the rank family

The deployed score remains

`S(r,phi;k,t) = a(r) + w frac(k phi/(2pi)+t)`.

The profile a(r) is the calibrated half-turn Jensen gain; w is its standard deviation on independent calibration radii, multiplied by the previously frozen tau=1. The same 129 radial anchors, 512 orbit angles, frequencies {1,2,3,4}, quarter-grid action phases, zero-moment upper-hull program, and radius-only randomized interpolation are retained. No task pilot chooses a new selector hyperparameter. The implementation changes only task geometry, radial sampling/support, model evaluation, and geometry-specific diagnostic constants.

Because every integer-frequency action has a uniform fractional phase conditional on r, the score distribution does not depend on the fitted actions. For exact global top64-of256 ranking, the inclusion marginal is still

`Q(F_S(S))`, with `Q(u)=P[Binomial(255,u)>=192]`.

All selection uses coordinates, stored tables, and independent keys. It does not use unseen partner logits. Every deployed batch uses 256 initial and exactly 64 partner example forwards. The even-frequency control is invariant under T_pi for fixed keys and has exact conditional balance at every radius. The learned zero-anchor-moment policy is only approximately balanced in the continuous population; its actual rank marginal must be used in verification.

## Exact population CDF for verification

The polynomial radial CDF and first-moment primitive on the support are

`F_R(r)=7.5u^2-20u^3+22.5u^4-9u^5`,

`M_R(r)=.4 F_R(r)+5u^3-15u^4+18u^5-7.5u^6`,

with u clipped to [0,1]. The marginal score CDF is

`F_S(s) = integral rho(r) clip((s-a(r))/w,0,1) dr`.

Since a is piecewise linear, split each radial interval where a(r)=s or a(r)=s-w. Constant portions integrate with F_R, and linear portions integrate with F_R and M_R. `rank_policy.population_cdf` implements this exact integral formula; floating-point arithmetic is still subject to rounding. Independent split Gauss-Legendre integration checks the formula on all eight chosen pilot states. Calibration continues to use the same empirical CDF as the frozen family; the exact population formula is evaluator-only and does not alter the policy.

## Geometry constants for continuous bounds

For the residual network, `L=product(1+.5||B||_2||A||_2)` is a global spatial Lipschitz constant and `C=sum .5||B[0,:]||_1` bounds |h(x)-x_1|. Define d=(h(x)-h(T_pi x))/2. The partner separation is exactly two, giving `|d|<=min(1+C,L)`. The radial input derivative has norm one; the angular input derivative has norm sqrt(1+4r^2). Therefore

`|partial_r d|<=L`, and `|partial_phi d|<=L sqrt(1+4r^2)`.

The previous DKW convolution argument, bounded-variation angular quadrature rule, and radial interpolation bound remain valid after substituting these constants. The code applies the angular bound to the sawtooth jumps, not an invalid smooth-midpoint rule. These global bounds may be loose. Task-specific local interval verification is separate work and must preserve the deployed score and selection distribution. A numerical integral of a pointwise bound is not itself a certified integrated bound without controlling the integration error.
