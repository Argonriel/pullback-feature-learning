# Global ranking with an action-invariant score law

Let the continuous input have independent radius R and uniform angle Phi. Write h=h(X), h_pi=h(-X), m=(h+h_pi)/2, d=(h-h_pi)/2, and A=softplus(h)-softplus(m). For a radial posterior p(r), an allocation with inclusion probability q(x) has gain

`Delta(p) = E[q A] - E[p(R) b(R)]`, where `b(r)=E[q d | R=r]`.

All expectations below include independent action keys and the other 255 inputs in the batch. Calibration is independent of these inputs. The input law, not any label posterior, is used to calibrate allocation. Exact conditional balance means b(r)=0 for almost every r. Zero moments on finitely many calibration radii do not establish this property.

## A genuinely larger feasible allocation family

Choose a radial profile a(r), width w>0, and an action (k,t) with nonzero integer k. The action distribution may depend on r but not on angle. Define

`S(r,phi;k,t)=a(r)+w frac(k phi/(2 pi)+t)`.

For every action, the fractional term is uniform on [0,1). Therefore `S | R=r` is uniform on [a(r),a(r)+w], independently of the fitted action distribution. The score law F is unchanged by fitting arbitrary radius-dependent mixtures of these actions. We choose the 64 highest scores among 256 iid inputs, using independent keys for the probability-zero mathematical ties. This uses exactly 64 partner calls. No rejection sampling, quota repair, or unseen partner logits enter selection.

Given an input and its action, the number of other scores below S is Binomial(255,F(S)). The true inclusion marginal is

`q(S)=Q(F(S)), Q(u)=P[Binomial(255,u)>=192]=I_u(192,64)`.

The derivative is the Beta(192,64) density. Its maximum is `L_Q=14.759059713171533`, attained at u=191/254. This formula is the actual global-rank marginal, not a normalization of the previous grouped propensity. Marginals can approach zero or one, and mean allocation can vary across radii. Only the population average remains 1/4. The earlier grouped field was restricted to [.125,.375] and mean 1/4 at every radius.

## Label-free anchor optimization

The profile a is the half-turn Jensen gain estimated from 512 equispaced orbit logits at each of 129 radii on [0,3]. Width is tau times the standard deviation of a(R) on 131,072 independent coordinate-only calibration draws, with a 1e-8 scale floor. The final tau is selected only from {.25,1,4}, using held-out old-model centered logistic gain, without labels or a posterior.

At each radius, 2,048 actions use k in {1,2,3,4} and phase `(j+.25)/512`. For each action estimate its centered moment E[q d] and objective E[q A]. Their upper concave hull gives the maximum objective at zero moment, using at most two actions. The `rank_even` control chooses the highest-objective even-frequency action; `rank_unbalanced` chooses the highest-objective action without a moment constraint. Between radial anchors, randomize between the adjacent anchor mixtures with the linear interpolation weights. This preserves action independence from angle and the score law. Outside radius 3 use the even action k=2, phase=0.

FFT calibration is an exact rearrangement of the discrete sums. The fixed quarter-grid phase offset avoids sampling the discontinuities of the sawtooth. This numerical correction was made after an independent old-only audit; the original sources, choices, and full pilot remain preserved. It changes neither the score-law theorem nor the evaluation seeds.

## Exact control and approximate learned balance

For any even k, the score is invariant under x -> -x, for every fixed action key. A canonical half-plane implementation makes that identity bitwise for ordinary finite coordinates. The global competitor score distribution is the same under the reflection, so q is even. Since d is odd, `b(r)=0` exactly for all radii. Mixtures of even actions depending only on r retain this guarantee. No CDF approximation is needed for this result.

The learned `rank_balanced` policy may use odd frequencies. Its calibration anchor moments are zero in the discrete empirical-CDF problem. Its continuous true moments need not be zero. We report those moments separately using refined, independent known-input-law quadrature and the actual rank marginal. No claim of exact balance is transferred from the control to the learned policy.

## Uniform finite-calibration and radial error bound

The following bound is implemented in `conditional_bound`. Condition on the model and orbit-derived profile a. For N independent radii let Fhat_A be the empirical CDF of a(R). With probability at least 1-alpha,

`sup_z |Fhat_A(z)-F_A(z)| <= eps = sqrt(log(2/alpha)/(2N))`.

Convolving with any uniform width w contracts this error. The event is uniform in w, so the bound also permits the data-dependent width. Fitting actions does not alter F. Consequently `|Q(Fhat(s))-Q(F(s))| <= min(1,L_Q eps)` uniformly over every fitted action and radius. The default alpha=.05 is per model; use alpha=.05/16 for a simultaneous 16-model bound. No independence is asserted between the 512 orbit points.

For this two-block residual network, globally valid constants are

`L_h = product_blocks (1 + .5 ||B||_2 ||A||_2)`,

`C_h = sum_blocks .5 ||B[0,:]||_1`.

Then |d(r,phi)| <= D(r)=min(r+C_h,r L_h), |partial_r d| <= L_h, and |partial_phi d| <= r L_h. These are deliberately conservative global bounds.

At radial anchor j and a fixed action k, let q_lo=Q(Fhat(a_j)) and q_hi=Q(Fhat(a_j+w)). The angular function q has total variation at most `2k(q_hi-q_lo)` over a full circle, including all sawtooth jumps. Thus the product q d has total variation at most

`V_jk = 2k(q_hi-q_lo)D_j + 2 pi r_j L_h q_hi`.

The equispaced quadrature error is bounded conservatively by V_jk/M, M=512. This is a bounded-variation argument, not an invalid smooth-midpoint rule across discontinuities. For a mixture, add these errors weighted by its odd-action probabilities; even-action moments vanish structurally. Add the absolute discrete mixture residual and `gamma_j D_j min(1,L_Q eps)`, where gamma_j is the odd-action mass. Call the resulting anchor error E_j.

For r=(1-t)r_j+t r_(j+1), write Delta_r=r_(j+1)-r_j, Delta_a=|a_(j+1)-a_j|. The true score CDF has density at most 1/w. Applying the spatial Lipschitz bound to d and L_Q/w to changes of the score location gives

`|b(r)| <= (1-t)E_j+t E_(j+1)`

` + t(1-t) [ L_h Delta_r (gamma_j+gamma_(j+1))`

`              + (L_Q Delta_a/w)(gamma_j D_j+gamma_(j+1)D_(j+1)) ].`

Cap by D(r); outside [0,3] the even fallback gives zero. This explicitly accounts for changes of both network logits and score location across a radial interval. There is no quota-repair bias: global ranking is already included through Q. The bound is uniform in r on the DKW event, although its numerical size may be uninformative. The exact-even control receives bound zero.

## Numerical diagnostics and limits

`population_cdf` integrates the known Rice-mixture radius law within every anchor interval using Gauss-Legendre rules. It is evaluator-only and does not supply free network calls to deployment. The integral on [0,3] omits a radial tail below 5.4e-32 for the specified input law. Increasing the integration order measures numerical stability but is not an analytic quadrature certificate. Conditional moment diagnostics must also refine angular resolution because the policy has sawtooth jumps. They must use the actual `Q(F)` marginal and cannot reuse the earlier group-of-four formula.

Neither old centered-gain development nor an increase in E[q A] proves improvement for all hidden radial posteriors. The imbalance term remains relevant. A useful practical improvement, useful rigorous bound, and exact continuous balance are three separate conclusions.
