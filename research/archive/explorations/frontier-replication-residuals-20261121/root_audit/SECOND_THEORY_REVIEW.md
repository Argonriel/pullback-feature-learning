# Independent second mathematical review

Reviewed structured/THEORY.md, structured/PROTOCOL.md and the first
root_audit/THEORY_REVIEW.md, with read-only inspection of math_utils.py and the
certificate event enumeration in run.py. No numerical experiments were run and
no frozen source or scientific setting was changed.

**Assessment: no material mathematical or inferential defect found within the
explicit assumptions.** The main restrictions are substantive assumptions, not
facts established by the finite samples: the invariant reference law, residual
orthogonality and support-wide envelope, pointwise invariant quota scores,
parameter-derived logit range, and constant-cell posterior model.

## Endpoint and orbit argument

With other batch coordinates and tie keys fixed, single-coordinate mask
invariance makes the two displayed even/odd decompositions valid. Symmetrizing
only that coordinate under mu gives the claimed identity without changing the
law of the other coordinates. An exact quota by itself would not suffice, but
the stronger invariant pointwise-score condition is stated.

For r >= 0, decomposing p and pT into common mass and disjoint TV residuals
maximizes the positive terms at max d and the negative term at min d. This gives
rb + [r + (1-r)epsilon]S/2; exchanging the posteriors and negating d handles
r < 0. Hence h(r) = epsilon S/2 + (1-epsilon)|r|S/2 - rt is the appropriate
joint penalty. It is convex in r, so its interval maximum is attained at an
endpoint. The interval reversal under T makes that maximum even. Nonnegativity
follows from |t| <= S/2 and epsilon(1-|r|)S/2 >= 0.

## Residual identification and random range

Oddness and a residual sup-norm bound alone do not identify the retained
coefficients. The additional orthogonality conditions explicitly present in
equation (4) make E_P cos(j phi) = theta_j/2. A fifth harmonic satisfies these
conditions; understating its supplied envelope remains an assumption failure
that more samples of the first two moments cannot repair.

For every realized evidence sample, including a failed coefficient CI,
the constructed interval endpoints obey
|endpoint| <= min(1, sum_j(|theta_hat_j| + q_j) + rho) = rbar.
Therefore H <= D[epsilon/2 + (1-epsilon/2)rbar] is a valid deterministic range
conditional on that evidence. Independent verification permits conditional
concentration with this random, evidence-dependent range. The truth-in-interval
event is needed to connect the resulting statistic to risk, not to make the
statistic's own range valid. It is paid for separately in the union bound.

## Finite-cell posterior evidence

Under the stated within-cell posterior constancy, conditioning on observed
cell counts yields Bernoulli labels with the cell probability. The count-based
Hoeffding intervals and [0,1] empty-cell rule give unconditional per-cell
coverage by averaging conditional coverage. Their dependence across cells is
irrelevant to the union bound. Maximizing both directed differences of each
paired rectangle correctly bounds the binary pointwise TV defect.

The stated limitation is essential: without constancy, the same labels only
identify cell averages. For p(phi) = (1 + sin(3phi))/2, the right and left
half-circle sine means vanish under the cosine-only densities, while partner
TV reaches one. Independent radial bins do not resolve this obstruction.
Productive labeled auditing here does not establish unrestricted posterior
regularity on natural data.

## Concentration and 8,622-event accounting

A nonnegative selected item weight in [0,L] gives a quota-64 batch mean in
[0,L/4]. With fixed pointwise scores, replacing one input/key changes at most
one selected item, so the selected sum changes by at most L, not 2L.
Consequently the average over n batches of 256 independent records has the
stated McDiarmid radius. The empirical Bernstein calculation uses independent
whole-batch means and the unbiased sample variance. Intersecting the two
bounds requires two events, which are explicitly charged.

- Base groups: 8 models x 7 laws x 2 policies x 2 looks = 224.
- Each group: A, B, and 8 epsilon values x 2 coefficient modes = 18 moments.
- Base moment/method events: 224 x 18 x 2 = 8,064.
- Additional labeled H events: 4 contexts x 8 models x 2 policies x 2 looks
  x 2 coefficient modes x 2 methods = 512.
- Total moment events: 8,576. Add 14 coefficient intervals and 32 cell
  intervals to obtain 8,622; delta = .05/8622.

The labeled contexts reuse the already counted A bounds. The global-TV
comparator reuses A/B bounds and supplied analytic density information, so it
needs no additional concentration event. Shared samples, nested looks and
shared label intervals do not require independent events for the union bound.
Including deliberately invalid-envelope rows in the arithmetic is conservative
and does not make those rows theorem-covered. The count supports only the
fixed reported epsilon grid and labeled values, not arbitrary post-hoc epsilon
choices.

## Minor wording clarification for the final report

THEORY.md Section 6 says the corrected expression in the cos(3phi)
counterexample coincides with the negative gain. That equality is exact for
the **un-clipped sharp expression**, or when the clipping constant satisfies
c >= max_phi log cosh(.1 cos(3phi)) = log cosh(.1). If c is smaller, replacing
J by min(J,c) makes the bound strictly lower in general. The sign argument and
validity of the lower bound are unaffected. The final report should qualify
the equality accordingly; no frozen method change is required.

Ordinary floating-point checks and a recovered roundoff assertion are not a
directed-rounding proof. That limitation is already acknowledged, and the
review does not upgrade numerical diagnostics into formal verification.
