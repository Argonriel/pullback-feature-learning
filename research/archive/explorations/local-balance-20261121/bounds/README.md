# Useful local bounds for continuous conditional imbalance

The new local analysis makes the previously loose conditional-balance bounds useful on all 16 fresh models. The mean input-mass integrated bound falls from 1.06028 to 0.0000814162, a factor of 13,022.95. An independent finite-sample calculation then gives a positive expected-risk improvement for every hidden radial posterior on all 16 models, with at least 95% family confidence in the real-arithmetic theorem. The smallest lower bound is 0.000755336 nats per input; the mean is 0.00278791.

The deployed field, trained predictor, and exact 64-extra-calls-per-256-inputs rule are unchanged. This is progress in what can be justified about an existing policy, not a claim that the policy's predictions or empirical gains improved. It also does not establish exact conditional identification: the learned moment b(r) is small but generally nonzero.

## What changed

The previous analysis used global spectral norms to bound angular quadrature and radial interpolation. Those constants discarded local behavior and made the error penalty much larger than the gain. The new analysis retains signed angular quadrature and propagates local intervals for the network value and its first and second radial/angular derivatives.

Only odd Fourier harmonics of the field can contribute to the imbalance moment. After removing the even part analytically, the product of the odd field and antipodal logit difference is periodic over a half circle. Local angular second derivatives bound midpoint quadrature error. Signed local radial second-derivative intervals then bound the interpolation error at every radius, including between calibration knots. The proof never substitutes average cancellation within a broad radial bin for pointwise conditional balance.

The frozen resolution is four subdivisions per original radial interval and 256 folded angular cells. That gives 512 radial bands and 131,072 polar cells. A separate global-odd-only control checks how much improvement comes from the parity simplification alone. Its result shows that the main improvement comes from the local derivative analysis.

See THEORY.md for the derivation and PROTOCOL.md for the frozen design. The mathematical bounds are valid in real arithmetic. The float64 implementation uses small interval padding and numerical checks; it is not formally verified directed-rounding arithmetic.

## Fresh confirmation

The method froze at 2026-11-21 09:28:53 UTC. The parity attribution control froze at 09:30:11 UTC. Root authorized new-model generation at 09:39:19 UTC, and the fresh fitted-field tables were committed at 09:39:58 UTC. All local bounds were completed and committed at 09:40:32 UTC. No new model or outcome was inspected to choose the bound resolution.

The 16 fresh PB500 models use training seeds 921001–921016 and initialization seeds 922001–922016. The unchanged continuous field is fitted separately to each frozen model. Historical confirmation below uses the prior 821xxx models and is not counted as additional fresh training replications.

| Quantity: mean across models | Prior 16 models | Fresh 16 models |
| --- | ---: | ---: |
| Previous global integrated bound | 0.937214 | 1.060279 |
| Global bound after removing even harmonics | 0.516778 | 0.562384 |
| New local integrated bound | 0.0000745481 | 0.0000814162 |
| Numerically observed integral of absolute moment | 0.0000376458 | 0.0000422446 |
| Previous-global / local ratio of means | 12,571.93 | 13,022.95 |
| Odd-global / local ratio of means | 6,932.14 | 6,907.53 |
| Local / numerical-observed ratio of means | 1.9803 | 1.9273 |

All 16 fresh models have tighter local bounds than either global control. The prespecified target required the local mean to be at most 1% of the previous-global mean; the achieved ratio is 0.0076788%. The comparison to the observed moment is diagnostic and does not replace the proof.

The fresh mean local bound decomposes into 0.0000135820 from angular quadrature, 0.0000121290 from radial interpolation curvature, and 0.0000557051 from the signed-node envelope. The mean uniform-in-radius bound is 0.00183288; the largest model-specific uniform bound is 0.00442031. The signed pointwise envelope is often tighter than the whole-band bound used in the integrated table.

## Independent all-hidden-posterior certificate

Root separately froze and evaluated the centered lower bound

    Delta(p) >= E[W0 - B(R)/2]

for every measurable radial posterior p, where W0 is an antithetic, label-free centered gain and B is the local pointwise bound. The centered formulation halves the ambiguity penalty relative to C-E|b|. It uses the varying B(R), not its uniform maximum as the average penalty. The uniform maximum enters only the finite-sample range bound.

Each model receives 262,144 new independent unlabeled inputs, seeds 941001–941016. The antithetic pair is one observation. A one-sided empirical Bernstein bound with delta=0.05/16 provides family confidence of at least 95% across the 16 model-specific certificates. The confidence sample fits neither the field nor the bound. No hidden labels or posterior probabilities are queried.

All 16 lower bounds are positive. The minimum is 0.000755336 nats per input and the mean is 0.00278791. These statements concern expected risk under the assumed uniform conditional orientation and the sampled deployment input law. They do not promise improvement on every labeled example, arbitrary distribution shifts, or every future trained model.

This component independently checked the root's derivation against the cited empirical-Bernstein theorem, recomputed every saved bound value and confidence statistic, and regenerated the full antithetic observations for models 1, 8, and 16 using an independent centered-softplus expression. Maximum discrepancies were 8.67e-19 for B, 2.17e-19 for the final lower confidence bound, and 2.78e-16 for an individual antithetic gain. See certificate_independent_audit.json and ../root_audit/CERTIFICATE_REVIEW.md. Root's raw certificate observations and results remain under ../root_audit/certificates/.

## Cost and numerical checks

The main bound requires, per model:

- 262,144 antipodal interval-jet propagations, each carrying the value and four derivative components.
- 262,656 ordinary network point forwards for signed quadrature at radial nodes.
- About 0.8043 seconds of measured setup on average for the fresh cohort, using one BLAS thread.

An interval jet is more expensive than an ordinary forward pass; these counts are deliberately separate. The additional moment diagnostics use 2,624,512 point forwards per model. Derivative spot checks add 128 interval jets, 512 exact analytic point jets, and 3,200 finite-difference point forwards per model. The root confidence calculation separately costs 8,388,608 ordinary forwards across 16 models. The independent three-model confidence audit adds 1,572,864. These are offline costs, not free deployment information.

Every saved local cell and radial node remains available. All old and fresh interval-enclosure, continuous-radius moment, and finite-difference convergence diagnostics passed. The maximum observed moment-enclosure violation is zero within the recorded numerical tolerance. The radial probability masses from the Poisson-Gamma formula agree with independent 24-point true-density quadrature to 5.85e-16. Its exact-arithmetic truncation remainder is at most 1.14e-57.

The development sweep retains all four combinations of radial subdivisions 4/8 and angular cells 256/512 on four prior models. The larger configurations improve the bound further but were unnecessary for the bounded confirmation. The initial coarse finite-difference audit exceeded absolute tolerances on high-curvature outer-radius points; all six step sizes, their convergence, and the resulting roundoff behavior are preserved. The interval construction was not changed to fit those diagnostic errors. A command-parser failure produced no numerical result and is also recorded in pilot/attempts.md.

## Limits and interpretation

The main remaining approximation is the learned field's actual conditional imbalance, not a global-norm penalty thousands of times larger. The local bound is still conservative, especially where the true moment is nearly zero or in low-mass outer radii. Exact zero is asserted only where the odd coefficients vanish structurally throughout a radial band.

This analysis applies to the existing group-of-four field whose marginal inclusion probability is 1/4+u/8. It must not be applied without a new derivation to a different global-ranking policy. The separate global-allocation component uses its own true rank marginal and error analysis. Its empirical performance and balance claims are distinct from this component's positive certificate.

The construction is specific to the present two-block tanh architecture, a piecewise-linear Fourier field, and rotational input symmetry. Extending it to a different architecture or geometry requires corresponding interval and symmetry arguments. No GPU or larger-model efficiency claim is made here.

## Reusable outputs

- `freeze.json`, `THEORY.md`, `PROTOCOL.md`: immutable method specification and proof.
- `local.py`, `experiment.py`, `checks.py`: frozen implementation and checks.
- `old/` and `fresh/`: all bounds, numerical moment curves, summaries, checks, and commitments.
- `parity_control.py` and its freeze: global odd-only attribution control.
- `analysis_summary.json`, `combined_summary.csv`: exact old/fresh aggregates.
- `local_bounds.png` and `.pdf`: visually inspected scientific figure.
- `reproduce.py`, `REPRODUCE.md`: non-destructive reproduction from preserved fitted fields.

Independent reviews of the global-rank formula, LP, FFT implementation, and repaired quarter-grid phase convention are preserved separately under ../root_audit/rank_independent/, including the original failure and corrected checks. No original archive or frozen primary policy was modified by this investigator.
