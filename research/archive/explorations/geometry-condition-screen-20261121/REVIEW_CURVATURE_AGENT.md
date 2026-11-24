# Independent curvature-lane review of the noise calculation

Reviewed `NOISE_AUDIT.md`, Section 3 of `REPORT.md`, and the relevant implementation in `noise.py`. Decision: the signs, projection, and loss-clock normalization are consistent. No mathematical correction is required for the displayed identities or their stated interpretation.

## Raw label expectation

For Bernoulli targets T, q=A^T(s-T)/n and v=-Pq. Therefore the centered-label coefficient is **positive** w_i=P A_i^T/n. The raw adaptive-minus-frozen initial acceleration is -P DM[v]v: the two vector fields share -P Dq[v], which cancels, while DP[v]=-P DM[v]P supplies the remaining term. Defining Q(u,w)=-P DM[u]w makes it bilinear without requiring symmetry. Independent centered Bernoulli labels eliminate off-diagonal and linear terms, leaving exactly sum_i p_i(1-p_i)Q(w_i,w_i). The correlated-label extension with a double covariance sum is also correct.

## Projection and squared-speed factor

Let d=q^T Pq>0 and v=-Pq, so q=-Mv, q^T v=-d, and d=v^T Mv. If b is the raw acceleration difference, matching training loss gives

    b_perp = b - v*(q^T b)/(q^T v)
           = b - v*(v^T M b)/(v^T Mv).

Thus the audit's M-weighted projection has the correct minus sign, and q^T b_perp=0. It is not the Euclidean orthogonal projection unless the metric happens to give that equivalence.

For the positive loss-decrease clock rho=L0-L, initially drho/dt=d. Differentiating the normalized field v/d yields the adaptive-minus-frozen second derivative

    delta(d^2 theta/d rho^2) = b/d^2 + v*(q^T b)/d^3
                            = b_perp/d^2.

Hence the leading parameter difference is rho^2*b_perp/(2*d^2), and the leading risk coefficient is 0.5*grad R(theta0)^T E[b_perp/d^2] when the population risk is fixed across label replicas. Neither E[b_perp] nor a projection of E[b] supplies this quantity generally. The implementation applies precisely this normalization before taking the weighted expectation and before contracting with the population gradient.

The local expansion needs nonzero descent. All sixteen enumerated label configurations have positive recorded initial speed, and there are finitely many, so the fixture avoids a stationary-point singularity. No uniform claim near arbitrary stationary states is justified, and the audit explicitly excludes one.

## Code and scope checks

- `derivative_metric` differentiates the analytic full representation Jacobian and includes both product-rule terms DJ^T J+J^T DJ. Damping is constant and correctly has zero derivative.
- The independent central difference checks operate on the whole metric, not just one Jacobian factor.
- `progress_field` uses the full nonlinear current predictive gradient for both methods and changes only the inverse metric. Its sign is -Pq/(q^T Pq), consistent with increasing loss decrease.
- Population logistic risk uses the fixed Bernoulli expectation log(1+exp f)-p f. Changing the unseen conditional probability changes the population gradient but none of the observed-input replica statistics.
- The nonlinear-flow comparison is correctly described as a tiny local expansion check, not a new predictive mechanism, independent future-horizon validation, or a practical performance effect.
- The conditional-information witness is limited to rules lacking the full population law. Section 3 clearly preserves the known-law rings possibility and does not claim an exhaustive impossibility result.

The phrase "independent short-flow check" refers to direct integration of the nonlinear vector fields rather than reusing the acceleration approximation. It does not mean a separately implemented network Jacobian or an interval-certified ODE error bound. Current wording does not require such a stronger interpretation.

The main report correctly classifies the covariance identity, loss-clock projection, and non-identifiability construction as ordinary bookkeeping and elimination evidence. No new-innovation claim is hidden in Section 3.
