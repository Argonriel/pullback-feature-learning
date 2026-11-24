# Independent mathematical review

This review is prepared before inspecting the new outcome tables. It distinguishes analytic claims from numerical checks and empirical generalization.

## Equal actual training loss decrease

Fix one parameter state theta, its training set, and a direction v with DL(theta)[v] < 0. Let alpha(delta) be the local root of L(theta + alpha v) = L(theta) - delta, with alpha(0) = 0. For differentiable L and Q, the inverse-function theorem gives

    alpha'(delta) = -1 / DL(theta + alpha(delta)v)[v]
    d Q(theta + alpha(delta)v) / d delta
        = -DQ(theta + alpha(delta)v)[v] / DL(theta + alpha(delta)v)[v].

At delta = 0 the first-order prediction is delta DQ(theta)[v] / (-DL(theta)[v]). Comparing two methods requires subtracting the two normalized directional responses from the same theta. This normalization is unchanged by a positive scalar rescaling of v. An actual solved alpha times DQ(theta)[v] is a different linearized prediction that uses the finite training-loss line search; it remains outcome-blind if Q is not used to solve alpha.

The expansion is asymptotic in delta. For a finite target, derivatives change along the ray; a small denominator can cause a large step, the ray loss can turn upward, and multiple roots may exist. A bracketed line search checks a selected crossing, not absence of an earlier narrow crossing between scanned points. Failure to bracket must remain in the coverage record. Repeated recomputation on different evolving states does not imply the original derivative continues to predict a multistep comparison. Equality of one training-loss scalar also does not identify the predictor or population risk.

## Matching radial and angular tasks

A valid controlled transfer task shares the full distribution of X. Matching only label balance or a displayed Bayes-error number is insufficient: equality in distribution of posterior eta(X) supplies common class prevalence, Bayes classification error, Bayes logistic risk, and all posterior-only Bayes scores. A deterministic measure-preserving swap between uniform radial rank and uniform angle can establish this equality exactly. The same sampled X may be used in paired tasks even when the label laws differ. Joint rotation symmetry must be checked against eta(gX)=eta(X), not against the marginal law of X alone. A radial task is fully rotation invariant; a general angular task is not. Averaging under a group that changes its posterior is a deliberate misspecified-symmetry control and has no unconditional Jensen risk guarantee.

## Finite cyclic averaging and Fourier modes

For fixed radius, C_K h(phi) = K^-1 sum_j h(phi + 2 pi j/K) retains precisely Fourier frequencies divisible by K. This is an exact function identity. A finite angular DFT only verifies its sampled and aliased counterpart. Low retained energy can predict reduced angular variation, but it does not identify logistic gain without amplitudes, phases and the orbit mean: logistic curvature varies with the mean logit. Gains for K and K+1 need not be ordered. Nested groups, such as K and 2K, satisfy an extra Jensen comparison under the required symmetry because C_2K C_K = C_2K.

## What unlabeled calibration identifies

Write B(a)=log(1+exp(a)). If the joint law is G invariant, the loss decrease from the complete orbit mean equals the unlabeled Jensen gap E[mean_g B(h(gX)) - B(mean_g h(gX))]. Conditional versions require invariant conditioning variables and invariant selection, including invariance to changing each selected observation's orbit orientation while holding other batch members and tie randomness fixed. An invariant quota gate can therefore use calibration estimates to rank conditional expected gains. Optimality is only among gates with the same observables and costs, with true conditional gains; finite calibration estimates can misrank them.

An orientation-sensitive first-view gate leaves a signed label-dependent term. Its true risk change generally cannot be determined from unlabeled predictions alone even when the joint law is invariant. An angular label task without the averaging group's joint symmetry has the same problem. Independent labels or known posterior may score those policies after selection, but that is evaluation evidence, not a label-free deployment guarantee.

## Separation and uncertainty

Model/protocol selection, unlabeled gate calibration and final risk evaluation must be separate. Shared seeds across methods are beneficial paired designs; repeated states or group sizes within one seed are not independent replicates. Bootstrap whole seed records, preserve common pairing, report failed/excluded settings and subgroup coverage, and label secondary intervals pointwise. Check that all budgets count actual additional forwards; calibration, offline oracle scoring and deployment costs are distinct. A new sample evaluated on models from an earlier study is new evaluation, not fresh training replication.
