# Structural conditions for learned geometry: completed screening round

November 21, 2026 (America/Los_Angeles).

**Decision: none of the four screened conditions supports a defensible innovation claim.** Some exact architectural identities are valid, but they reduce to ordinary symmetry/spectral calculations. The proposed universal or persistent predictive-sign implications fail by explicit counterexamples. The broad research question remains open; this round does not establish that all possible learned-geometry mechanisms are impossible.

This is the analytical follow-up to the failed full-tangent empirical gate. It does not alter that gate's threshold, consume its reserved confirmation seeds, or add a task sweep to rescue the old claim. The [screening protocol](SCREENING_PROTOCOL.md) declared four bounded lines of investigation and elimination rules. All four are now resolved at the level claimed below.

## Decision ledger

| Proposed condition or explanation | What was actually established | Decision |
|---|---|---|
| Even radial labels plus small residual initialization creates a special useful feedback channel | Exact odd/even Jacobian blocks; fixed-lambda even predictive velocity scales as residual amplitude squared. The initial behavior is already fixed-tangent spectral filtering. | Valid identity, insufficient and not novel |
| Initial favorable transverse curvature and a fixed activation-concavity sign preserve a later adaptive-metric benefit | An exact two-parameter slice has favorable initial risk curvature but later harmful risk difference on the same fixed population, while every active tanh argument stays positive. | Persistent-sign claim rejected |
| Early label-replica variability or zero-mean noise determines the benefit/harm sign | Raw feedback has a standard nonzero covariance contraction; loss-clock normalization is nonlinear. Training-only-identical laws can have opposite local population-risk signs. | Universal training-only sign rule rejected |
| Metric eigenvector rotation is necessary for predictive differences beyond a clock | An exact residual slice has diagonal commuting metrics throughout yet different predictions at matched training loss. | Necessity/clock-equivalence claim rejected |

There is no surviving new mechanism or prospectively validated predictor. The counterexamples are elimination evidence, not innovations in their own right.

## 1. Parity supplies an exact decomposition, not the missing prediction

At zero biases, the actual two-block residual tanh map is odd in x. Its A/B parameter-Jacobian columns are odd, and its bias columns are even. With centrally symmetric inputs and an even conditional label probability p(x), the summed metric's weight-bias cross-block vanishes. The A/B loss gradient is independent of p; the even signal enters through the bias gradient. These statements use the complete 80-parameter architecture, not a different network.

Scaling all residual-output B matrices by epsilon makes the bias Jacobian O(epsilon). At fixed positive damping, the resulting even predictive PB velocity is O(epsilon^2/lambda). The numerical check recovers this scaling and the limiting PB/GD even-velocity ratio 1/lambda=10 for lambda=0.1. This is the usual damped initial spectral filter, also present in the full tangent model. It is not evidence of a learned-metric advantage.

Exactly zero residual B does not create a permanently invariant signal-blind subspace: label-independent odd dynamics can immediately create B, making bias directions available later. Attempting to turn this into a sign theorem fails at the next step. Shrinkage of the isolated bias update is only a same-state, same-clock training statement; other metric blocks change earlier, biases break the parity block structure, and finite-sample population alignment has no forced sign. If training is instead done on the exact population objective, matching that objective makes the same population-risk contrast zero by definition.

See [parity audit](PARITY_AUDIT.md) and `parity_results.json` for derivation, source overlap, and numerical checks. This eliminates a parity-only predictive explanation without dismissing parity as useful bookkeeping.

## 2. A stronger warning about the frozen-metric intervention

An exact slice of the existing residual model has two active original parameters and representation

    z(x;u,v) = (0.5*v*tanh(x2+c*u+b), x2+c*u),
    c = 0.5*tanh(1).

All other 78 parameters are fixed. The training point is x=(0,0), with label +1. The fixed population has three input atoms, all labeled +1. The comparison integrates in training-logit coordinates, so a common logit is exactly common training logistic loss.

For a constructed and fully recorded example, the adaptive-minus-frozen population-risk contrast changes from approximately -0.000111831 at the early logit to +0.00000643151 at logit 3. The population law stays fixed. All active tanh arguments stay positive, retaining the same concavity sign. Three RK4 resolutions preserve the reversal. The initial matched-logit risk coefficient is also favorable, so this is not merely an ambiguous first observation. Local sign and activation concavity do not provide persistence.

However, even this difference does **not** isolate an irreducible learned-geometry mechanism. On this slice,

    M(theta) = g(theta) g(theta)^T + D,
    D = diag(c^2+lambda, lambda),

where g is the training predictive gradient. Sherman-Morrison gives

    M(theta)^(-1) g = D^(-1)g / (1 + g^T D^(-1)g).

Consequently adaptive PB has exactly the same continuous parameter orbit as nonlinear GD under the fixed preconditioner D, after a scalar clock change. It can differ from the particular frozen initial metric M0 while still reducing completely to another fixed preconditioner. Thus the fixed-M0 intervention is necessary for separating two ingredients, but passing it would not by itself prove a new adaptive-geometry mechanism.

This reduction is standard matrix algebra and is explicitly not claimed as new. The slice is not invariant under unrestricted 80-parameter training. The finite sign reversal was found by a small deterministic construction search, not a preregistered population-performance experiment; the complete search grid/results and verification are retained. See [curvature audit](CURVATURE_AUDIT.md).

## 3. The noise calculation exposes a bookkeeping trap

At a fixed initialization, the full representation Jacobian is independent of sampled labels. Let P=(J^T J+lambda I)^(-1), v=-P grad L, and define the bilinear map

    Q(a,b) = -P DM[a] b.

For independent Bernoulli labels with conditional probabilities p_i, write v=v_bar+sum_i w_i(Y_i-p_i). Then the raw adaptive-versus-frozen initial acceleration satisfies the exact standard identity

    E Q(v,v) = Q(v_bar,v_bar)
               + sum_i p_i(1-p_i) Q(w_i,w_i).

The covariance contribution need not vanish even though the label noise has mean zero. Enumerating all sixteen label configurations on four inputs in the actual 80-parameter model verifies the formula to 7.63e-17 in maximum coordinate error; the covariance contribution has norm about 0.06433 in this example.

This is not yet the matched-progress quantity. Removing the loss clock uses a projection depending on v, followed by division by the squared initial loss speed. Expectation cannot simply be commuted with those operations. In the constructed example, projecting the mean differs from averaging the projected acceleration by norm 0.005924. Treating the raw covariance identity as a matched-loss signal/noise theorem would therefore be wrong.

Two population laws that agree on the observed inputs and all their conditional-label replica statistics, but differ at an unobserved input, give opposite expected local risk coefficients: about -0.70360 and +0.69323 per squared training-loss decrease. Very short full-model flow checks at loss decreases 1e-4 and 5e-5 confirm the predicted opposite signs and approach the local coefficients. These tiny flows validate a derivative expansion; they are not a new benchmark training sweep or an advance predictor.

The negative result is restricted to the declared training-only information. It does **not** refute a rule that is given the known rings population law, including unseen conditional probabilities. Nor is the covariance identity, local population contraction, or basic non-identifiability argument new. See [noise audit](NOISE_AUDIT.md).

## 4. Eigenvector rotation is neither the missing condition nor a clock test

A second exact two-parameter residual slice has predictive logits (1+tanh u,tanh v) on inputs e1,e2. Its damped metric is diagonal with entries sech(u)^4+lambda and sech(v)^4+lambda. Every learned metric commutes with the initial metric and the eigenaxes remain fixed.

Despite that, adaptive and frozen-metric flows have different prediction vectors at identical training loss. At a declared loss decrease of 0.18, predictive RMS difference is 0.0103410, while the maximum numerical loss-clock error is 4.74e-12. Changing relative eigenvalues is enough to change direction; eigenvector rotation is unnecessary. The relevant orbit condition is collinearity on the current gradient, not commutation of the metric matrices.

Two population input mixtures compatible with the same observed training sample give risk differences +0.00100836 and -0.00100836. That is ordinary conditional non-identifiability of a geometry-only sign rule. The population weights, if supplied as known task information, distinguish the two cases. This is not a counterexample to every possible data-informed condition. See [commuting audit](COMMUTING_AUDIT.md).

## What remains unresolved

The rejected shortcuts omit the difficult part: how a finite training sample produces a task-aligned, label-dependent geometry change that transfers to the population over a nonzero interval of progress. Neither symmetry, a local curvature sign, a scalar replica-variance measure, nor eigenvector motion provides that bridge by itself.

A future admissible condition must state the empirical sampling law and permitted population information, address finite-progress transport of predictive directions, and survive reduction to fixed preconditioning plus a clock. Assuming the desired future population-gradient alignment is circular. A generic Taylor remainder or stability inequality would remain a routine bound. No condition satisfying these requirements was found in this bounded round.

The justified action is to stop expanding these four formulations. No additional large training sweep, cost comparison, or new-innovation claim follows from them. This is not an exhaustive no-go theorem for the broader research question, and it is not a request to reinterpret the earlier failed gate as a success.

## Archive and validation

All four analyses include reproducible English source and numerical data. The parity and noise calculations use the original full architecture at initialization; the curvature and commuting witnesses explicitly freeze 78 parameters. Counterexample-search outputs are construction records, not estimates of success frequency or generalization.

The report retains original-source novelty checks and an independent mathematical review. A fresh archive holds this protocol, reports, data, source snapshot, validation checklist and SHA-256 manifest. The earlier 105-file learned-geometry gate archive and all older rejected directions remain untouched. No messages were sent to other people.
