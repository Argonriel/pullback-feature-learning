# Independent mathematical review

Reviewer scope: main `REPORT.md`, `COMMUTING_AUDIT.md`, `PARITY_AUDIT.md`, the declared `SCREENING_PROTOCOL.md`, and the supporting curvature note where the main report summarizes its reduction. The reviewer did not treat its own noise derivation as independent validation. No additional training or numerical tests were run for this review.

**Disposition: the stated bounded rejection decisions are supported. No blocking mathematical overclaim was found.** This review verifies formulas and interpretation, not every recorded numerical value or source timestamp.

## Commuting-metric witness

The claimed embedding is correct: trainable A1 indices 0 and 3 with the specified fixed B yield logits 1+tanh(u) and tanh(v) on e1/e2. The restricted full representation metric is diagonal with entries sech(u)^4+lambda and sech(v)^4+lambda. Hence all such metrics commute, yet changing their relative diagonal entries need not preserve collinearity of their action on the gradient. This directly rejects commutation as a scalar-clock criterion.

The loss-progress field -Pq/(q^T Pq) gives dL/ds=-1 wherever q is nonzero. Different prediction vectors at a common attained loss therefore exclude a mere loss-clock difference on this example. Population weights (0.8,0.2) and (0.2,0.8) both assign probability 0.32 to the unordered two-sample event containing one of each input. Opposite risk signs conditional on that observed sample are an ordinary information limitation, and the report correctly says known population weights distinguish the cases.

The restriction freezing 78 coordinates is explicit. It is not asserted to be invariant under the full 80-parameter flow. The matrix calculation and mixture reversal are correctly classified as elementary counterexamples, not innovations.

## Parity and small-residual calculation

At zero biases, the residual composition is odd in x; A/B derivatives are odd and bias derivatives even. Central input symmetry therefore cancels the weight-bias metric block. With even p(x), the weight gradient contains only sigmoid(f)-1/2, while the bias gradient contains the even teacher p-1/2. The target-independent restricted risk does not force its normal derivative to vanish; the report correctly avoids an invariant-submanifold inference.

For fixed positive lambda, J_bias=O(epsilon), grad_bias=O(epsilon), and the bias metric is lambda I+O(epsilon^2), giving even predictive velocity O(epsilon^2/lambda). This is an initial spectral/damping result shared by the full tangent comparator. It does not establish adaptive-metric benefit. The zero-B feature-creation example applies to ordinary GD as well, as explicitly noted.

The same-state inverse-ordering inequality concerns a quadratic training-improvement form. It supplies no sign for a cross-gradient population expression or a matched-progress trajectory contrast. The report also correctly distinguishes exact paired quadrature from an ordinary finite random training sample and notes that matching an exact population objective would make that same objective's risk contrast zero tautologically.

## Main report and curvature summary

The main report preserves the essential scope boundaries: the parity/noise local calculations use the full architecture, while curvature and commutation witnesses use constrained slices. It does not claim that the training-only indistinguishability examples rule out predictors given the known rings law. It distinguishes small-progress derivative checks from a validated advance predictor and does not repackage identities as contributions.

The summarized rank-one reduction M=g g^T+D and Sherman-Morrison formula are correct. Its conclusion is appropriately limited: adaptive versus a particular frozen M0 can differ while adaptive flow itself equals another constant-preconditioner flow up to time change. Therefore a frozen-M0 intervention is informative but not alone sufficient for an irreducible-adaptation novelty claim.

The recorded sign reversal rejects persistence based solely on an initial favorable coefficient and activation-concavity/mixed-partial signs. It does not contradict the ordinary sufficiently small interval guaranteed by a nonzero Taylor coefficient, nor joint concavity, neither of which is asserted. Counterexample-search selection is disclosed and not represented as confirmatory performance evaluation.

## Final interpretation

The conclusion that these four formulations fail the innovation screen is warranted at the stated scope. The broader known-law, architecture-specific finite-progress transfer question remains open. No exhaustive impossibility result, broad full-model generalization theorem, or positive novel mechanism follows from these screens. The report states those limits explicitly.
