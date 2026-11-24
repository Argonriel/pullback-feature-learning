# New-direction screen: one narrow endpoint-selection lead, no established innovation

November 21, 2026 (America/Los_Angeles). This is a bounded, literature-first exploration. It is not a preregistered predictive-performance study or an exhaustive originality search.

## Decision

Four settings distinct from the retired empirical routes were screened against original papers and small falsification checks. Three settings are rejected as innovation proposals: stochastic metric/gradient coupling, input-map topology, and geometry-only poisoning. Within depth/damping, scalar depth laws and cumulative intermediate representations are also rejected. Only a precisely stated underdetermined matrix question survives: **how does positive damping change the interpolating solution selected when the same training constraints are assigned different positive weights?**

The surviving observation is real in a fixed 2-by-2 model and has been independently reproduced in the original eight factor parameters. Its novelty remains unresolved. Generic non-Hessian geometry, path dependence, and a perturbation integral are known mathematics; an identical paper was not found in the bounded search, which does not prove originality. No learning advantage or population-risk improvement has been established.

The original full-tangent gate remains failed at its declared effect floor. Its threshold, cohorts, data, and conclusions were not changed. The prior structural-condition screen also remains unchanged. See the [prior inventory](INVENTORY.md) and [screen register](SCREEN_REGISTER.md) for the explicit exclusion boundary.

## Results by setting

| New setting | Original-literature threat | Minimal validation | Disposition |
|---|---|---|---|
| Depth, damping, and interpolation | Natural-gradient/mirror-flow implicit bias; October 2026 exact joint-GN scalar damping law | Direct factor solves, mirror projection, prefix-coordinate reduction; then fixed noncommuting matrix endpoint probe | Reject the broad scalar proposals; retain only the specific matrix endpoint question |
| Stochastic metric and gradient sampling | Benzing 2021, SketchySGD, Nayak et al. 2026 directly address coupling and independent batches | Exact binomial enumeration for batch sizes 1–1024; exact conditional-label enumeration | Reject: the large mean-direction reversal is ordinary sample reweighting, and cross-fitting is occupied |
| Input-map topology | Augmented Neural ODEs 2019, iResNet 2019, Topology of DNNs 2020 | Exact residual-network slice with constant parameter metric but input-map fold | Reject: broad topology story is known; a universal parameter-spectrum fold warning is false |
| Separately sampled geometry poisoning | Covariance/PCA poisoning, robust covariance, adversarial linear-system perturbation | Rank-one, fixed-size mixture, and target-direction identities checked to floating-point precision | Reject: distinct sampling interface does not make the elementary matrix mechanism novel |

Detailed original-source comparisons: [depth](DEPTH_AUDIT.md), [stochastic](STOCHASTIC_AUDIT.md), [topology](TOPOLOGY_AUDIT.md), and [poisoning](POISONING_AUDIT.md).

## The one surviving question

Use a balanced two-factor matrix model X=W2 W1 and the full matrix representation, with parameter metric J^T J+lambda I. On the symmetric positive-definite product domain, its induced product flow is

    X_dot = -(I + lambda S_X^(-1))^(-1) grad L(X),
    S_X(U) = XU + UX.

The representation metric uses all four matrix entries, even though the objective observes two linear measurements. Reweighting the objective does not redefine this metric. This is a different model from the original residual network and from the Gauss–Newton metric formed from the two observed predictions.

Start from X0=.1 I and minimize

    L_w(X) = .5 [(X11-1)^2 + w ((X11+2X12+X22)/2-1)^2],
    w in {1,9}.

Both losses have exactly the same zero-loss constraints. X12 is an unobserved linear functional: those constraints do not determine it uniquely. Thus two different limiting values are a genuine selection effect, not a comparison at different training progress.

| Damping lambda | Final X12, weight 1 | Final X12, weight 9 | Weight-9 minus weight-1 |
|---:|---:|---:|---:|
| 0 | .3000000000 | .3000000000 | 7.06e-12 |
| .003 | .3003596812 | .3003798105 | 2.01294e-5 |
| .03 | .3034306115 | .3036312368 | 2.00625e-4 |
| .3 | .3226999442 | .3243452276 | 1.64528e-3 |

These are the refined reduced-product calculations. Constraint residuals are below 1e-11; halving the final integration step changes any X12 by at most 8.88e-9. The sampled product path remains strictly positive definite, with minimum eigenvalue .1.

An independent implementation integrates the original eight factor parameters using the explicit parameter metric, without importing the reduced flow. Its refined gaps at lambda=.003 and .3 are 2.01296337e-5 and .001645283596; they agree with the reduced calculation within 3e-10. Its zero-damping control gap is 1.73e-10, residuals are below 1e-12, and factor balance and product symmetry remain within the stated numerical checks. The first, coarser parameter integration had a visibly larger zero-damping artifact; both the initial record and the refinement are preserved. The effect is not justified from that coarse record alone.

A three-coordinate diagonal control has two overlapping constraints and a one-dimensional interpolation set. Its mirror-flow prediction is weight-independent; refined weight gaps are at most 3.66e-9. This supports the intended distinction without reducing the control to a uniquely determined solution.

The independent [mathematical review](ENDPOINT_MATH_REVIEW.md) derives the small-damping response from the exact zero-damping paths, rather than fitting a coefficient to these endpoints. The [independent prior-work review](ENDPOINT_PRIOR_AUDIT.md) checks whether the specific fixed-loss endpoint consequence is already a theorem in the closest sources. These reviews are part of the decision, not optional background.

The mathematical review also establishes a restricted local result: for every fixed initialization X0=s I with 0<s<1 and every fixed w>1 in this same two-measurement family,

    X12_infinity(lambda,w) - X12_infinity(lambda,1)
        = lambda (c_w-c_1) + O(lambda^2),    c_w-c_1 > 0.

It proves positive definiteness of the undamped paths, the response sign by an explicit one-form integral, and a small-damping infinite-time remainder by a uniform interior estimate. At s=.1 and w=9, independent exact-path quadrature gives c_9-c_1=.006709867819846879; increasing quadrature order changes the coefficient by less than 9e-14. The initial coarse quadrature failed its precision criterion and its replayed values are preserved. This is a checked local result for one constructed family, with no claim for every damping value or for vanishing initialization. The use of standard perturbation and integration arguments still leaves its research originality and broader significance unestablished.

## What is and is not different from prior work

[Sabanayagam and Lucey, October 2026](https://arxiv.org/html/2610.04249v1), Appendix E.9, directly covers the exact joint-GN aligned scalar damping mobility. That recent result is a decisive reason to reject the broad depth proposal. Its aligned, fully observed phase-law setting does not determine this underdetermined fixed-loss endpoint comparison.

[Li, Wang, Lee and Arora, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Paper-Conference.pdf) gives commuting/mirror-flow conditions and related obstructions across losses. Failure of a single loss-independent mirror representation does not itself prove that these two fixed weighted quadratic losses have different endpoints. Conversely, that distinction alone does not make the example publishable. The additional prior audit also checks the established Bures–Wasserstein geometry and time-warped implicit-bias results, which prevent overstating the non-Hessian calculation.

The matrix metric is Frobenius plus a multiple of the familiar Bures–Wasserstein metric. It is not a new geometry. The first-order nullspace response is a standard perturbation identity. The possible contribution would have to lie in a substantive, predictive selection law or a useful characterization of when that response vanishes, with explicit assumptions and remainder control.

Noncommuting measurements alone are not sufficient: if their span contains the initial product and is closed under Jordan products, the Lyapunov operator and its inverse preserve that span. The unobserved component can then remain fixed despite noncommutation. This exception rules out a tempting overbroad explanation before any larger experiment.

## Follow-up gate and stopping point

This round is complete. Retain one research question, not a verified innovation or a recommendation for a large neural sweep. A justified continuation would need all of the following:

1. A quantitative selection condition or theorem that adds content beyond the general non-Hessian/path-integral framework, with the fixed example as a check rather than the whole claim.
2. A sharper original-source comparison for that exact theorem. If an existing result directly implies it, retire the lead.
3. A meaningful consequence outside the constructed example, with an explicit prediction and information model. Different interpolants alone do not show which one predicts better.

No reserved confirmation cohort, old benchmark rescore, topology sweep, stochastic training run, or poisoning training experiment was used. No external messages were sent. Numerical data, source snapshots, refinements, and the rejected routes are retained, so the next round can exclude them rather than rediscover them.

See [data dictionary](DATA_DICTIONARY.md), [validation checklist](VALIDATION_CHECKLIST.md), and [archive check](ARCHIVE_CHECK.json) for reproducibility and preservation checks.
