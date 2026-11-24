# Independent decision and novelty review

Date: 2026-11-21 (America/Los_Angeles).

## Decision

Stop expansion of C0 for the two declared task families under the recorded gate. Do not run the reserved confirmation cohort, fixed-initial-metric intervention, or stage-two spectral study to rescue this candidate. The completed comparison does not meet its prespecified effect-size criterion. This is a resource-allocation decision under a declared screening rule, not a finding of equivalence or absence of nonlinearity.

No validated innovation candidate remains from this experiment. C1, the early label-replica metric-variability hypothesis, was conditional on a surviving gate and is not advanced by these results. The overall research question remains open.

## Material inspected and arithmetic independently checked

Read PROTOCOL.md, COVERAGE_AMENDMENT.md, REFINEMENT_AMENDMENT.md, discovery_refined_scores/gate_decisions.json, and discovery_refined_scores/contrasts.csv in the learned-geometry-gate-20261121 archive. Recomputed means, sample standard deviations, and train-logit discrepancies from the four refined primary-endpoint rows per task. This review is independent of the score-aggregation function; it is not a second replay of all model trajectories or an audit of every source hash.

For the primary training-loss fraction 0.25, damping 0.1, and refined controller (logit target 0.005, parameter cap 0.05):

| Task | B mean +/- sample SD | D mean +/- sample SD | Mean PB nonlinear/tangent train-logit RMS difference |
|---|---:|---:|---:|
| Moons | +0.009598 +/- 0.016095 | +0.002191 +/- 0.021375 | 1.999435 |
| Rings | -0.008501 +/- 0.007757 | -0.004390 +/- 0.011650 | 0.993037 |

B=R_tangentPB-R_nonlinearPB; positive B favors nonlinear PB. D=(R_nonlinearGD-R_nonlinearPB)-(R_tangentGD-R_tangentPB); positive D means PB's comparative advantage is greater with nonlinear predictions. Risks are in nats. Both tasks have four completed pairs at this endpoint after the recorded supplements.

The coarse-controller B means are +0.009642 (moons) and -0.008545 (rings). Both B signs agree in 3/4 pairs at both resolutions. Both fail the unchanged |mean B| >= 0.01 floor. Coarse/refined mean differences are only about 0.000044 nat. Moons is close to the cutoff, but lowering the floor after observing that closeness would invalidate the recorded decision rule. D also fails its floor; moons D has only 1/4 signs agreeing with its positive mean. No formal significance or equivalence test follows from these four pairs.

The original 6000-update protocol and the supplemental completed comparison must remain separately labeled. Original rings censoring is a missing-endpoint result, not evidence of nonlinearity or lack of capacity. The coverage supplement and the jointly reduced logit/parameter-step refinement were recorded before predictive evaluation. Their existence and the limitation of the original cap-only refinement should be visible in the final report rather than collapsed into a single supposedly pristine frozen experiment.

## Interpretations supported and excluded

Supported: within the declared architecture, damping, task laws, and matched training-progress endpoint, the observed nonlinear-versus-tangent PB risk contrasts are below the operational retention threshold and have substantial between-pair variation. The controller refinement changes their means little. The measured training prediction vectors differ appreciably even though their scalar losses are matched.

Excluded: these results do not show that the network is lazy, that nonlinear and tangent predictions coincide, that all observed behavior is a scalar clock change, or that initial geometry fully explains the trajectories. A scalar loss can agree for different prediction vectors, and similar risk can coexist with substantial functional differences. Step refinement is numerical sensitivity evidence, not proof of an exact gradient-flow limit.

Also excluded: the four arms cannot identify the causal contribution of adapting the metric. Their nonlinear/tangent difference changes both the prediction map and the representation Jacobian. The unrun fixed-M0 arm is the necessary next intervention only if the empirical gate justifies expansion. Its absence here is the declared stopping outcome, not a missing experiment to fill by lowering the standard. No cost-superiority conclusion is supported by the controller's Jacobian-heavy implementation or unequal coverage caps.

## C1 novelty critique

The replica-variability hypothesis is not validated, and its generic form faces a direct prior-work threat. [Zhang et al. (ICML 2025), Section 3 equations (4)-(5)](https://arxiv.org/html/2502.01763v1) already uses learned F^T F geometry and independently sampled update batches. [Han et al. (ICML 2025), Sections 3-4](https://arxiv.org/html/2505.18909v1) already analyzes signal-dominated learning followed by noisy-sample memorization under logistic GD. Combining independent-label diagnostics, learned geometry, and a sign reversal is not enough. A successful future condition must predict the adaptive-versus-frozen metric difference beyond generic prediction instability and known signal/noise dynamics. See LITERATURE_AUDIT.md for exact restrictions and additional sources.

## Next narrow analytical gap, without additional training

Screen the following structural question before proposing another empirical candidate:

> For an explicitly restricted residual architecture and known task law, can one identify an architecture/data condition under which label-dependent metric curvature produces a sign-definite, non-clock difference between adaptive and frozen-initial-metric learning over a nonzero interval of matched training progress?

The condition must be stated using initialization, task structure, and permitted early information. An assumption that the unknown future population derivative already has the desired sign is circular. A population-gradient/update inner product is a diagnostic, not the sought advance criterion.

A useful algebraic starting point, explicitly not a new result, separates the local metric term. Let P0=(M0+lambda I)^(-1), q0=grad L(theta0), a=-P0 q0, and M(theta)=J(theta)^T J(theta). For smooth continuous-time adaptive and frozen-P0 flows with the same initialization, their difference in initial parameter acceleration is

    delta_b = P0 DM(theta0)[a] P0 q0.

If q0^T a is nonzero, removing the second-order scalar-clock difference by matching attained training loss gives the transverse coefficient

    delta_b_perp = delta_b - a (q0^T delta_b)/(q0^T a).

The leading population-risk contrast is one half t^2 grad R(theta0)^T delta_b_perp, plus higher-order terms. These statements follow from the derivative of an inverse matrix and Taylor expansion. They do not constitute an innovation, a finite-horizon sign theorem, or a deployable predictor. In particular, zero-mean label noise does not automatically eliminate quadratic feedback terms because a itself depends on labels.

The actual analytical target is a nontrivial structural restriction that determines the conditional signal/noise contributions to this transverse term and prevents their sign from immediately cancelling or reversing as learning proceeds. Start with symbolic low-parameter restrictions of the actual two-block map, rather than constructing another unread-out-motion witness. Test whether task symmetry, curvature sign, or readout coupling imposes a usable condition. Do not assume that more label noise must hurt adaptive PB.

Terminate this analytical direction if its sufficient condition reduces to fixed-feature whitening, a scalar field rescaling, the inverse-matrix/Taylor identity above, an assumed population-gradient sign, or a generic stability/remainder bound. Also terminate a universal sign claim as soon as a simple admissible architecture/data counterexample reverses it. Only a surviving specific condition with a new falsifiable finite-progress prediction should justify a new protocol and independent cohort.

This section formulates the next screen. It does not assert that such a condition exists or that any candidate has passed it.

## Review checklist

- [x] Applied the declared effect-size, sign, refinement, and endpoint rules without post-result changes.
- [x] Independently recomputed refined summary arithmetic from paired rows.
- [x] Distinguished original censoring from pre-evaluation supplements.
- [x] Preserved failed-cutoff versus equivalence distinction.
- [x] Preserved nonlinear prediction difference versus metric attribution distinction.
- [x] Did not use secondary endpoints to rescue the primary gate.
- [x] Did not promote conditional C1 or run additional training.
- [x] Identified a narrow analytical screen and explicit elimination rules.
- [ ] Established or independently validated a new mechanism: not achieved.
