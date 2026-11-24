# Exact symmetry, budgeted averaging, and the strong baseline

## Assumptions and notation

Let T be a measurable involution. Inputs in each deployment batch are IID,
their law is T-invariant, and the binary posterior satisfies p(Tx)=p(x).
The model h is fixed before deployment. No invariance of h is required.
Write A(t)=log(1+exp(t)), m(x)=(h(x)+h(Tx))/2, and

J(x) = [A(h(x))+A(h(Tx))]/2 - A(m(x)).

Then J(Tx)=J(x) and J(x)>=0 by convexity. The logistic conditional loss is
L_p(t)=A(t)-pt. The network output here is a logit; averaging probabilities
would require a separately stated argument and changes the numerical policy.

For n=256, a mask S_i in {0,1} selects k=64 inputs. Assume the complete mask
is unchanged when any individual input X_j is replaced by TX_j, keeping all
other inputs and random keys fixed. This stronger individual-coordinate
condition is satisfied by ranking a pointwise T-invariant score with fixed
tie keys. Invariance only under flipping an entire batch is insufficient for
the following coordinate-by-coordinate proof.

## Expected improvement identity

Define D_i=L_p(h(X_i))-L_p(m(X_i)). Replacing X_i by TX_i preserves the joint
batch distribution and the mask. Since p(TX_i)=p(X_i),

[D_i(X_i)+D_i(TX_i)]/2 = J(X_i).

Multiplying by the unchanged S_i and taking expectations therefore gives

E[raw mean loss - allocated mean loss] = E[n^{-1} sum_i S_i J(X_i)] >= 0.

This is an expectation over inputs and random keys, not a guarantee for each
realized labeled batch. It holds for every T-invariant hidden posterior under
the same invariant input law. Full SO(2) invariance, a uniform continuous
angle, a known radial density, and a scalar quotient are unnecessary. In
particular, the half-turn-invariant posterior can depend on folded angle.
The equality conditions on the already fitted model and table; independent
confirmation batches are required for statistical statements about its mean.

Two individually invariant policies S and V satisfy the signed identity

E[loss(V)-loss(S)] = E[n^{-1} sum_i (S_i-V_i) J(X_i)].

There is no unconditional sign for this contrast. Jensen safety relative to
raw predictions does not prove superiority to another invariant policy.
The identity generally fails for confidence or orientation-sensitive rank
selectors; their hidden-posterior contribution need not cancel.

## Why this is a strong baseline

On a fixed batch, choosing the k largest exact J values maximizes the sum
of selected J values. The exact-J mask is itself invariant. It is therefore
an upper envelope for the expected gain of all individually invariant masks
with the same exact quota, including randomized masks. This statement does
not imply that it maximizes realized loss improvement or beats arbitrary
orientation-sensitive policies for a particular posterior.

The exact-J oracle needs both model predictions at every input. Its cost is
512 forward examples per 256-input batch, even if its final average is used
on only 64. It is an evaluation diagnostic, not a 320-example deployable
method. The interpolation method ranks stored approximations to J using
only input coordinates. It actually needs 256 first-view examples and 64
selected partner examples at deployment.

If a score approximates J uniformly within epsilon on a batch, exchanging
the oracle and score-selected sets gives a selected-sum regret at most
2*k*epsilon. Dividing by n gives at most epsilon/2 in this study. The present
experiments measure approximation error and oracle headroom; they do not
claim a certified uniform interpolation-error bound. Approximation error
affects gain, while structural score invariance preserves non-harm.

The grid stores 128 radial anchors times 512 angles for one invariant, or
16 times 16 quotient anchors times 256 angles for two invariants. Every
grid costs exactly 65,536 forward examples. Both views at the table's grid
nodes come from the same full-orbit table. Only one half-orbit Jensen field
is retained. Quotient interpolation is multilinear and angle interpolation
is periodic linear interpolation of centered nodes. Canonicalization of the
first harmonic pair gives bitwise equal scores under T, including signed
zero normalization. Clamping preserves this invariance outside the grid;
it does not establish a uniform approximation guarantee there.

## Cheap verification without a score CDF

An observed selected Jensen gain can be computed by querying both views only
on selected inputs: at most 128 forward examples per verification batch for
one policy. Comparing two masks queries both views on their union, at most
256 examples per batch, and can reuse overlaps. This is distinct from
deployment, which must also predict every unselected input and costs320.
No partner is free: all queried examples are counted.

Independent verification batches give IID bounded scalar observations.
For d=(h(x)-h(Tx))/2 with |d|<=D,

0 <= J(x) <= log(cosh(D/2)) <= D/2.

For completeness, write h=m+d and hT=m-d. Differentiating the Jensen gap in
m shows it decreases as |m| grows; its maximum is at m=0, where the gap is
log(cosh(d/2)). This also follows from
cosh((m+d)/2)cosh((m-d)/2)=[cosh(m)+cosh(d)]/2.
Thus a batch gain is in [0,(k/n)log(cosh(D/2))]. A signed contrast is within
the corresponding symmetric range, and independent batches permit a valid
bounded-mean confidence inequality without rank-probability or CDF work.

For a residual network with |h(x)-x_0|<=C_h, use D=R+C_h when |x_0|<=R.
The ring input support is unbounded; a finite R bound applies only after
restricting to an invariant region. Dropping nonnegative contributions
outside this region gives a valid absolute lower bound. A signed contrast
cannot drop its tail without a separate two-sided tail correction.
Clipping nonnegative J likewise gives an absolute lower bound, but a
clipped signed contrast needs separate control of the omitted difference.

The certification branch states its own fixed clipping, tail corrections,
confidence allocation, and model-versus-batch uncertainty. These are not
inferred from a positive development mean.

## Scope and novelty boundary

Orbit averaging, convexity-based improvement, and Rao-Blackwell reasoning
are established principles. Input-dependent test-time augmentation and
baseline-relative safety also have substantial precedent. This study's
constructive content is the exact-quota invariant allocation identity,
transparent interpolation baseline, queried-example cost accounting, and
fresh controlled comparisons across one- and two-invariant tasks. We do not
claim a first label-free guarantee, a new general identifiability theorem,
or universal superiority of asymmetric selection. See LITERATURE.md.
