# Approximate symmetry and identifiable selective gains

Let A(t)=log(1+exp(t)); binary conditional risk is A(t)-pt. All logits below are finite; the population statements require integrability. The model, transformations and input-only policy are fixed before the unknown target posterior is chosen.

## 3. Sharp antipodal mismatch bound

For an equiprobable pair with logits a,b and posteriors p,q, put m=(a+b)/2 and J=[A(a)+A(b)]/2-A(m). Averaging the two original risks and subtracting the risk after averaging logits gives

    G = J - (p-q)(a-b)/4.

Thus |p-q|<=epsilon implies G>=J-epsilon|a-b|/4. For epsilon in [0,1], equality is attained by p=(1+epsilon sign(a-b))/2 and q=(1-epsilon sign(a-b))/2. This is sharp with no further restrictions on p,q. Equal logits give zero. Otherwise the inequality certifies mismatch up to min(1,4J/|a-b|). A negative lower bound does not establish harm: choosing p=q still gives J>=0.

For p_epsilon(r,phi)=(1-epsilon)p0(r)+epsilon(1+cos(phi))/2, the antipodal difference is epsilon cos(phi), whose absolute value is at most epsilon. Inputs retain their original radial distribution and uniform angle. The sweep changes labels while keeping the fitted models fixed; it is a distribution-shift sensitivity study, not retraining or a matched-Bayes-difficulty experiment.

### A certified continuous-input gate without a second deployment query

An offline polar cell and its antipode are enclosed in Cartesian rectangles. Standard interval propagation through each residual tanh block encloses every possible pair of logits in [aL,aU] and [bL,bU]. Matrix interval multiplication handles positive and negative coefficients separately. The residual output interval is the Minkowski sum of its input interval and its residual interval; dropping their dependence is conservative, not an equality assumption.

The minimum Jensen gap on these two logit intervals is zero if they overlap. If aL>bU, it is J(aL,bU); if bL>aU, it is J(aU,bL). To see this, partial J/partial a=(sigmoid(a)-sigmoid(m))/2 has the sign of a-b, and the derivative with respect to b has the opposite sign. Consequently a valid cell lower bound is

    J_min - epsilon * max(|aL-bU|,|aU-bL|)/4.

The two terms can attain their extrema at different points, so this cell bound may be substantially conservative. It uses no fresh second-view prediction. The table is indexed by radius and angle modulo pi. Rotating any one batch input by pi leaves its table index, tie key and selection decision unchanged. Conditioning on other inputs and tie keys therefore permits pairwise orientation averaging, even when p and q differ. The true pair gain exceeds the stored bound for each selected orbit.

The optional-quota rule selects at most 64 of 256 inputs, taking only strictly positive lower bounds. A forced-quota rule takes the largest 64 even if negative; its quota gives no non-harm guarantee. Inputs outside the covered radius interval are left unselected by the certified rule. Full pair evaluation of held-out risks is an offline scoring device, not a permitted way for a deployment gate to obtain free scores.

## 4. Identification with unknown radial posteriors

Let g be an input-only selection indicator (possibly randomized and batch-dependent), a the first logit and m its proposed replacement. Define

    C = E[g(A(a)-A(m))],
    b(r) = E[g(a-m) | R=r].

Other batch observations and tie randomness have no extra information about a point's label conditional on its input. If the unknown posterior is p(R), iterated expectation gives the exact gain

    Delta(p) = C - E[p(R)b(R)].

For every measurable radial posterior 0<=p<=1, the gain is identified by unlabeled observations if and only if b=0 almost surely. Sufficiency follows immediately. For necessity, compare p=0 to indicators of sets on which b is positive or negative; if the functional is constant all these integrals vanish. The sharp range is

    [C-E b_+, C+E b_-],

where b_+=max(b,0), b_-=max(-b,0). Endpoints use p=1 on positive or negative b, respectively, and p=0 elsewhere. Non-identification of the value need not prevent sign certification: the lower endpoint can remain positive.

### Known class prevalence

If E p(R)=pi with 0<pi<1, identification holds if and only if b is almost surely constant; the identified value is then C-pi*b. Necessity follows because an integrable b orthogonal to every bounded mean-zero perturbation around the interior posterior p=pi must be constant. One can use bounded indicator contrasts of two positive-mass sets with different mean b, with sufficiently small perturbations to remain inside [0,1]. This also handles unbounded integrable b without assuming an unbounded perturbation is admissible.

Let Q_b be the increasing quantile function of b(R). The sharp prevalence-constrained range is

    [C - integral_(1-pi)^1 Q_b(u)du,
     C - integral_0^pi Q_b(u)du].

The lower gain puts the available positive-label probability mass on the largest b values; the upper gain puts it on the smallest. An exchange of probability mass from a smaller to a larger b cannot decrease E[pb], proving optimality. At an atom at the threshold, use a fractional p to attain the exact prevalence. On a finite weighted grid this is a fractional-knapsack calculation, not an approximation or an integer constraint on p.

The same formulas apply to a difference between two fixed policies by replacing C and b with their differences. Opposite gains under observationally indistinguishable posteriors invalidate any claim that unlabeled observations alone always rank those policies correctly.

### Four-point, fixed-prevalence counterexample

Use two equally likely radii, each with two equally likely antipodes; logits are (0,2) and (3,3). Select only the logit-zero point and replace it by 1, using exactly one extra call among four. The per-input gain is [A(0)-A(1)+p(first radius)]/4. Assigning labels 1 to the first radius and 0 to the other gives +0.094971373; reversing them gives -0.155028627. Both have prevalence 1/2 and identical unlabeled observations, classifier, policy and quota.

### Finite-grid conditional balance with an exact quota

Our constructive population has 16 fixed radii and 16 equally likely orientations per radius. Every stratified batch contains each of its 256 support points once, in independently permuted order. At each radius, enumerate every subset of four orientations. Every such action uses exactly four calls. Its pair (b_S,C_S) is known from the offline support logits. A mixture over these actions is a linear program maximizing C subject to E b_S=0 and probabilities summing to one. Feasibility is guaranteed by a subset consisting of two complete antipodal pairs.

The upper concave envelope of the finite (b_S,C_S) points determines the optimum at b=0. An optimum can use at most two actions. Mixing them with the linear interpolation probability balances signed change in expectation while selecting exactly four points in each realization. Independent choices across the 16 radii give exactly 64 second calls per batch. The marginal selection probabilities can differ at antipodes: orientation sensitivity is checked directly, rather than inferred from individual randomized realizations. This is balance in expectation, not necessarily balance within each realized batch.

This construction is an exact finite-support demonstration. Offline predictions cover all 256 support points once. A cache of all those outputs could also answer future support-only predictions without model calls; we do not claim a fundamental compute saving in that artificial setting. The executable deployment deliberately retains only score/selection tables and reevaluates the first and chosen second views, to verify its 256+64 call implementation. It is not evidence of continuous-input generalization.

### Why balancing a radius bin is insufficient

Let R be uniform on [0,1], with two equiprobable orientations of logits 2 and 0 at every radius. Select only the logit-2 orientation for r<1/2 and only the logit-0 orientation for r>=1/2. Then b(r)=1/2 in the first half and -1/2 in the second: one broad bin has mean b=0 although b is not zero conditionally. The two opposite half-interval label assignments both have prevalence 1/2 and give gains differing by 1/2 nat. Smooth approximations to these posteriors still give different gains.

If p is known to be L-Lipschitz and the conditional imbalance integrates to zero in every bin of width at most w, comparison with the bin-center posterior gives |E[pb]|<=L*w*E|b|/2. Such a credible smoothness assumption controls the remaining ambiguity; finite-bin cancellation by itself does not identify the gain. Population identification, approximation under smoothness, and sampling uncertainty must remain distinct.
