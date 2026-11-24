# Tighter rectangle bounds and cost-accounted adaptive certificates

For logits a,b and an external bound epsilon on the difference in the two antipodal label probabilities, define

    F(a,b) = [A(a)+A(b)]/2 - A((a+b)/2) - epsilon |a-b|/4,

where A(t)=log(1+exp(t)). The exact pair-averaged gain is at least F. Our earlier implementation separately minimized the Jensen term and maximized the absolute separation over a rectangle of enclosed logits. That is valid but can combine extrema that do not occur together. Joint minimization of F cannot produce a weaker mathematical lower bound.

## A finite list contains every rectangle minimum

Consider a closed rectangle [aL,aU] by [bL,bU]. The corners are candidates. If the rectangle meets the diagonal a=b, zero is also a candidate. The remaining candidates are stationary points along its four edges, split where they cross the diagonal.

To rule out a non-diagonal interior minimum, write m=(a+b)/2 and d=(a-b)/2, with d nonzero. Changing m holds the separation penalty fixed. The derivative of the Jensen term with respect to m has the sign of minus m:

    sigmoid(m+d)+sigmoid(m-d)-2 sigmoid(m)
      = sinh(m) [1-cosh(d)] /
        [(cosh(m)+cosh(d))(cosh(m)+1)].

At nonzero m there is a direction that decreases F. At m=0, moving either way decreases the Jensen term strictly when d is nonzero. Therefore no non-diagonal interior point is a local minimum. Equivalently, the only smooth interior stationary pair for 0<epsilon<1 is a=-b=log((1+epsilon)/(1-epsilon)), and it is a saddle rather than a minimum. Degenerate rectangles are covered by their edges and corners.

On an edge with fixed endpoint a=A, let c=exp(A/2), t=exp(b/2)>0, and s=sign(b-A). The edge derivative vanishes exactly when

    epsilon c t^3 + (epsilon-2s)t^2
      + c(epsilon+2s)t + epsilon = 0.

Retain every positive real root whose b=2log(t) lies on that edge and on the indicated side of A. Evaluate F at all such roots, all corners, and any diagonal candidate; their minimum is the exact rectangle minimum. For epsilon=0, overlapping logit intervals yield zero, and disjoint intervals attain the minimum Jensen gap at their closest endpoints.

The implementation uses batched cubic companion eigenvalues, checks stationary residuals, and falls back to the previous conservative bound on detected numerical ambiguity. A margin of 1e-10 times one plus the maximum absolute rectangle endpoint is subtracted before combining with the old bound. This is a numerically checked implementation of the exact mathematical characterization, not a formally verified directed-rounding certificate. Polynomial work and fallback counts are recorded in addition to interval propagation work.

## Interval enclosure and deployment guarantee

Every polar cell is enclosed by a Cartesian rectangle that includes endpoint extrema and the sine maximum when relevant. Each model and its antipodal input rectangle are propagated through the residual tanh network with signed interval matrix products. Tanh is monotone. Adding interval bounds for the residual update is conservative even when the input and residual are dependent. This dependency relaxation remains a source of conservatism; the present study does not claim to have removed it.

The pair of resulting logit intervals is an enclosure for every input in the cell. The joint minimum is therefore a lower bound for every such input's pair-averaged gain. For inputs outside radius 3, the global residual amplitude gives |h(x)-h(-x)|<=2|x_1|+2C, where C is the sum of half the first-row absolute output-weight sums of the two blocks. The fallback bound is -epsilon(2|x_1|+2C)/4 and never supplies a positive certificate.

Lookup uses canonical half-plane coordinates, radius and angle modulo pi. Negating any one batch input leaves its lookup score unchanged. Ties are independent of inputs and labels and fixed when comparing the two orientations. Consequently choosing the top at most 64 positive scores among 256 is invariant to that flip. Integrating over the uniform antipodal orientation of each input gives a nonnegative expected pair/orbit risk improvement for the selected policy. The guarantee concerns this expectation and requires the external mismatch bound. It does not apply to every realized labeled example, and it does not infer epsilon from unlabeled predictions.

The deployment function predicts once on every input, selects using the precomputed table, and queries only the selected partners. It does not obtain a free second prediction before allocation. A diagnostic oracle that computes exact F for every new input has already used two predictions per input and is charged accordingly.

## Adaptation with all parent work counted

The adaptive method begins with 64 radial by 128 folded-angular cells: 8,192 cells and 16,384 interval propagations, because each cell and its antipode are propagated. Each binary split evaluates two new children and costs four further propagations. Twenty-four rounds of 512 splits add 49,152 propagations, for exactly 65,536 total. There are 32,768 evaluated nodes but only 20,480 final leaves. Parent computations remain charged.

The equally expensive uniform control uses 128 by 256 cells, or 32,768 final cells and 65,536 propagations. The original 128 by 128 control costs 32,768 propagations. Both independent and joint objective bounds are evaluated on the fixed grids, separating tighter objective minimization from greater spatial resolution. Fixed-grid bounds are shared physically by their two scoring rules, while each rule's standalone setup requirement is stated.

Adaptive priorities use independent unlabeled radius draws and the known uniform-orientation input law to estimate cell mass. For epsilon 0.1, only currently uncertified cells have positive priority. Their priority is mass times a conservative upper bound on possible positive F. Each selected cell is split along its larger radial or mean-radius arc diameter. This is a fixed heuristic, not an optimal partition theorem. The full objective and lookup computation time is measured separately from the interval propagation count, so equal propagation counts do not imply equal latency.

## Prospective success criterion

At epsilon 0.1 on the locked fresh 16-model cohort, practical success requires all of:

1. Mean selected-input fraction at least 2%.
2. Mean actual gain at least 50% of the positive mean gain of the charged two-call pointwise oracle.
3. A positive lower endpoint of the pointwise paired 95% model-bootstrap interval for improvement over the original fixed-grid independent bound.

The equal-cost uniform joint bound is mandatory and all tested epsilon values remain reported. A failure of any criterion is reported as a failed practical target. This is a cohort-average criterion, not a per-model guarantee. Historical models are development checks and are not counted as new confirmation replications.
