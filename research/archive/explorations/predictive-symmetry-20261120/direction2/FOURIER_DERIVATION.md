# Fourier transfer and the exact log-loss benchmark

Fix radius r and write the real logit function as

`h(r,θ) = μ(r) + Σ_{k≠0} c_k(r) exp(ikθ)`, with `c_{−k}=conjugate(c_k)`.

The two-view operator is `S_αh(θ)=[h(θ)+h(θ+α)]/2`. It maps coefficient c_k to

`c_k (1+exp(ikα))/2 = c_k exp(ikα/2) cos(kα/2)`.

Consequently each mode's squared energy is multiplied by `cos²(kα/2)`. At α=π, odd modes disappear and even modes remain. For a radius-dependent α(r), the same identity holds separately at each radius, and the radial mean μ(r) is unchanged. A uniform random angle retains half the expected energy of each nonzero mode. The implementation's finite uniform-angle group has the same half-energy property for each nonzero discrete Fourier mode; the Nyquist frequency must be counted once.

This does **not** mean that the removed fraction of Fourier energy equals the removed fraction of logistic risk. Let `φ(h)=log(2cosh(h/2))`. For labels Y∈{−1,+1},

`ell(h,Y)=φ(h)−Yh/2`.

Under a rotation-invariant joint law, the original and rotated pairs have the same joint distribution. Therefore

`R(h)−R(S_αh) = E[φ(h)−φ(S_αh)]`

because `E[Yh]=E[Yh_rotated]`. Equivalently this is the expectation of the nonnegative symmetric Jensen gap

`[φ(h)+φ(h_rotated)]/2−φ([h+h_rotated]/2)`.

These equalities also hold for α(r), because radius is unchanged by rotation. They need not hold for an arbitrary first-view-dependent selector. A finite IID difference of two pointwise losses retains the signed linear term; a complete finite calibration orbit cancels it exactly for every orbit-preserving candidate. This is why the experiment uses complete angular orbits, not an unpaired unlabeled difference as a claimed exact risk measurement.

Now write ε=h−μ. Since both ε and S_αε have zero angular mean, a second-order expansion around μ gives

`G(α) ≈ (1/2) E_r[φ″(μ(r)) Σ_{k≥1} P_k(r) sin²(kα/2)]`,

where `P_k=2|c_k|²` for paired positive/negative modes, and `φ″(μ)=sigmoid(μ)[1−sigmoid(μ)]`. This formula uses a radius-dependent logistic-curvature weight. Omitting it produces a squared-logit energy objective with different units, not an approximation expressed in nats.

For truncation K, the omitted quadratic contribution is bounded by

`0 ≤ G_quadratic,all−G_quadratic,K ≤ (1/2) E_r[φ″(μ) Σ_{k>K} P_k]`.

The logistic nonlinearity creates a separate error. Since `sup_h |φ‴(h)|=1/(6sqrt(3))`, Taylor's theorem and Jensen's inequality imply

`|G_exact−G_quadratic,all| ≤ E|ε|³ / (18sqrt(3))`.

This is a coarse, globally valid sufficient bound under integrability; it is not a claim that observed errors attain that scale. It demonstrates that increasing K alone cannot remove the finite-amplitude approximation error.

The fresh experiment separates the two effects. Pullback's K=8 curvature-weighted selector understates its selected-angle gain by 0.002476 nats on average; retaining all 64 discrete modes still understates gain by 0.001674. The unweighted energy selector is weaker in actual risk. At the same time, the curvature-weighted K=8 argmax is close to the exact Jensen argmax, so an inaccurate gain scale need not imply a poor discrete decision. Calibration sampling error is another distinct contribution; exact Jensen is exact for the calibration orbit law, while its estimate of population gain has finite-radius sampling noise.

All spectral power, omitted-tail bounds, angular logit amplitudes and nonlinear gain comparisons are archived in `fresh_results/spectrum_diagnostics.csv`. The group summaries of per-seed fractions in that diagnostic file must not be confused with the ratio-of-means angular fractions in the primary results table.
