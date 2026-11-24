# Local imbalance bounds for the actual global-rank marginal

All statements in this note are real-arithmetic statements. The implementation uses float64, conservative padding and independent numerical checks. It is not a directed-rounded computer-assisted proof.

## Fixed deployed policy and population inclusion

Inputs are IID with a uniform orientation conditional on radius. Independent action keys choose one of the fixed integer-frequency actions at the two adjacent radial anchors. For any nonzero integer frequency and any phase, the fractional phase is uniform on [0,1). Therefore, conditional on radius, every action has score distribution a(r)+w U, where U is uniform and w>0. Mixtures of the actions preserve this conditional distribution. The population score CDF is

F(s) = E[clip((s-a(R))/w, 0, 1)].

The deployed batch selects the largest 64 scores from 256 IID inputs. Scores have a continuous distribution because w>0. Conditional on a focal input's score s, its exact inclusion probability is Q(F(s)), where Q is the Beta(192,64) CDF. Integration over its action key gives q(x). The policy continues to rank scores directly; verification never changes scores, action probabilities, masks, calibration data or the number of deployed predictions.

Let d(x)=(h(x)-h(Tx))/2 for the half-turn T. Conditional imbalance is b(r)=E[q(X)d(X)|R=r]. In rings, T=-I. In the waveform task, T=diag(-1,-1,1,1), so it is not full input negation. For every even-frequency action, its score is invariant under T and d is antisymmetric. Its conditional contribution is exactly zero at every radius. This structural simplification does not apply to an odd learned action merely because an anchor fit was balanced.

## Known-law CDF enclosure

Partition the radial support into cells that do not cross a knot of the fixed piecewise-linear a(r). Let their exact probabilities be m_i and endpoint location extrema be a_i^- and a_i^+. Monotonicity gives

sum_i m_i clip((s-a_i^+)/w,0,1) <= F(s)
<= sum_i m_i clip((s-a_i^-)/w,0,1).

The grid is allocated in proportion to variation of a(r), so cells with a large change receive more subdivisions. The law is used only for verification. The rings CDF uses the positive Poisson–Gamma mixture truncated after term 160, with an explicit omitted-mixture tail bound. The missing subprobability can be placed at an endpoint at a cost no greater than that tail bound for any CDF query. The waveform radial CDF is the polynomial 7.5u^2-20u^3+22.5u^4-9u^5 on u in [0,1]. Outside the fitting grid, the score location is clamped and the action is exactly even; the corresponding constant-location mass is retained in F.

Write G for the distribution CDF of A=a(R). Almost everywhere,

F'(s) = P(s-w < A < s)/w.

The same endpoint masses enclose G. On a score interval [s_l,s_u], a lower derivative bound is max(0,G_lower(s_l)-G_upper(s_u-w))/w, and an upper bound is min(1,G_upper(s_u)-G_lower(s_l-w))/w. These bounds do not assume a density for A. Flat a segments may create atoms in A; F remains Lipschitz, and almost-everywhere derivatives suffice below.

Q is monotone. Its derivative is the beta density, which increases to u=191/254 and then decreases. Endpoint values and the possible interior maximum therefore enclose Q' on every interval. Q values are interpolated on 262,145 points with a real-arithmetic interpolation error at most sup|Q''| times the squared mesh width divided by 8. The code bounds |Q''| by the sum of the separate maxima of its two beta-monomial terms. This interpolation error is included in every probability bracket.

## Optional empirical-law verification

An independent verification-only sample of radii can replace the analytic law. The fixed transformation a(R) then gives IID observations of A. A DKW band of radius sqrt(log(2/alpha)/(2n)) is uniform for G. Integration against the monotone conditional-uniform kernel gives the same error radius for F. The derivative interval uses two CDF endpoints and therefore includes twice that error. This event is uniform in score, radius and all fixed actions, so the numerical integration does not need a union bound over its grid nodes. Its confidence event must nevertheless be allocated separately from the later expected-gain confidence sample. Calibration radii already used to choose the policy are not substituted for this independent verification sample.

## Jump-split angular integral

For one action, split [0,2pi] at every sawtooth wrap. Within each resulting interval, q(phi)=Q(F(a+w u(phi))) is nondecreasing. Further divide each interval into short cells of width h. At a midpoint, enclose d_0, d'_0 and d'' throughout the cell using interval automatic differentiation through the actual residual network.

Monotone left/right sums of the enclosed q values bound its cell average q_bar. The centered first moment satisfies

|mean_cell[(phi-midpoint) q(phi)]| <= (q_right-q_left) h/8.

To see this, subtract the left endpoint value and maximize the resulting centered integral over monotone functions with a fixed range; the extremum is a single jump at the midpoint. Taylor's remainder contributes at most sup|d''| q_right h^2/24. Thus the cell average of q*d differs from d_0 times the midpoint of its q_bar interval by at most

|d_0| error(q_bar) + error(d_0) q_right
+ |d'_0| (q_right-q_left) h/8
+ sup|d''| q_right h^2/24.

Sum signed midpoint contributions and positive error terms with the exact cell lengths. Wrap discontinuities lie on cell boundaries, so no smoothness assertion crosses them. Contributions are then mixed using the deployed anchor and action probabilities. This yields a signed enclosure of b at every node in the refined radial grid.

## Local radial variation without F''

Within one original radial interval, the score location a and the action mixture weights are affine in r. For a fixed action, write I(r)=E_phi[q(r,phi)d(r,phi)]. It is absolutely continuous and almost everywhere

I'(r)=E_phi[q d_r + Q'(F) F' a' d].

Interval network jets enclose d and d_r over each full polar cell. CDF and beta-density intervals enclose q and its score derivative. Signed interval integration gives a range for I'. Endpoint integral enclosures and this derivative range also bound I throughout the radial band; their intersection with the direct polar-box enclosure tightens the result. Finally, apply the product rule to the actual affine action weights. The resulting [L,U] encloses b' almost everywhere throughout the band, including both changes in logits and in score location and action weights.

For an absolutely continuous function whose derivative lies in [L,U] on [r_0,r_1], its value at t=(r-r_0)/(r_1-r_0) differs from linear interpolation of its exact endpoint values by at most

(U-L)(r_1-r_0)t(1-t).

This follows by bounding the two secant slopes from the interior point to the endpoints in [L,U] and writing the interpolation residual as their difference times (r_1-r_0)t(1-t). Combining the endpoint error intervals gives a valid signed enclosure at every radius, not just anchors or radial averages. A convenient uniform band envelope uses t(1-t)<=1/4. Exact-even bands and out-of-range even actions have zero imbalance analytically.

## From imbalance to a hidden-posterior guarantee

For a radial label posterior p(r), expected improvement over the raw predictor is C-E[p(R)b(R)]. Centering p at 1/2 gives a lower bound E[W-B(R)/2] whenever |b(r)|<=B(r), where W is the antithetic centered-softplus gain observation formed with the actual marginal q. When q itself is bracketed, each coefficient's sign selects the conservative probability endpoint. Known labels or posterior values are not used to construct this lower observation.

If |d|<=D, each centered-softplus difference is bounded by D/2 in magnitude and the two differences sum to a nonnegative Jensen gap. Consequently the antithetic lower observation remains in [-D/4,D/2] even when conservative endpoints in [0,1] are chosen separately. An independent bounded-variable confidence bound can then be applied to W_lower-B(R)/2. That statistical step, its confidence allocation and its sample cost are owned by the root certificate. A positive lower confidence bound means uniform expected-risk improvement for all measurable radial posteriors under the stated input symmetry; it does not establish b=0 or exact identification.
