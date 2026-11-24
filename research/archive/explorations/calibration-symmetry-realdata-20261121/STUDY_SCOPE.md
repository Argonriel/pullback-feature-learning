# Calibration, imperfect symmetry and a real-data cost study

User authorization on21 November2026: complete all three recommended follow-ups.
Previous studies and frozen archives remain unchanged. Each branch writes to a
new folder, freezes its scientific protocol before confirmation, and preserves
all unfavorable outcomes. This scope is written before confirmation outcomes.

## Objective 1: improve the underlying two-invariant model

Keep the previously frozen two-amplitude task and144-parameter architecture.
Train on256 balanced labeled examples. A bounded four-seed development pilot
compares PB rates2/4/8 and GD rates0.1/0.3/1, each with weight-only L2 strengths
0/0.001/0.01. Evaluate checkpoints100/250/500/1000 on2048 IID labeled validation
examples. Select PB and GD configurations by development validation loss;
fresh per-model checkpoint choice uses its own validation set. The historical
PB4/500 remains an unchanged control.

Fit one positive temperature per predictor on2048 additional IID labeled
calibration examples, separately from training, validation and evaluation.
These additional labels are an explicit cost: this is not a comparison at the
old raw predictor's total label budget. Posterior probabilities are evaluator
information only and never choose training, temperature or selectors.

There are16 fresh training/initialization pairs, six predictor variants
(legacy PB, selected PB, selected GD, each before/after temperature), and
raw/full/strong-histogram/interpolated selectors with exact64-of256 quotas.
Calibration tables retain the previous65,536-forward standalone budgets.
When logits can be reused across temperatures, record shared actual calls
separately from standalone budgets. Report error, log loss, confidence and
calibration separately; positive scalar temperature alone preserves decisions.

## Objective 2: quantify tolerance to imperfect label symmetry

Keep an exactly half-turn-invariant input law. For d=(h(x)-h(Tx))/2 and Jensen
gap J, the exact paired gain identity for an individually invariant mask is
E[M(J-(p(x)-p(Tx))*d/2)]. If a known label-symmetry error bound satisfies
abs(p(x)-p(Tx))<=epsilon, then gain is at least
E[M min(J,c)] - epsilon E[M abs(d)]/2.

Construct simultaneous lower/upper confidence bounds on those two moments,
using both independent-batch empirical Bernstein and top-quota bounded
differences. The resulting affine lower-bound line is valid for every epsilon
in[0,1], not merely a plotted grid. Conditional on the fitted predictor and
policy, choose c by the predeclared calibration-only rule
min(0.5,max(1e-6,quantile95(grid_J))). Use the compact task's deterministic
global half-logit-difference bound D for the second moment; no empirical
maximum substitutes for D. Signed-weight sensitivity uses a range width.

Compare32 predictor variants: legacy PB raw and selected PB temperature for
each of16 seeds, with two symmetric selectors. Fix certificate looks512 and
2048 independent batches. The64 lines x2 moments x2 methods x2 looks share
alpha0.05 (512 elementary events). These confidence claims are separate from
the training-performance bootstrap intervals and real-data descriptive CIs.

Stress epsilon in{0,.025,.05,.1,.2,.4,.8,1}. Use both a fixed cosine odd
perturbation and an adversarial sign(d) perturbation, clipped to preserve
valid posterior probabilities. The latter is model-dependent and must be
described as adversarial sensitivity, not a common fixed population across
models. Confidence selection is an empirical control, not covered by the
invariant-mask guarantee. No theorem for an approximately invariant input
law or the natural real-image distribution is claimed here.

## Objective 3: real images and total computation cost

Use official Fashion-MNIST, with source, license and checksums retained.
Horizontal reflection is a defensible garment-category transformation, but
the original distribution is not assumed exactly reflection-invariant.
Train three fixed small CNNs. Partition official training data, by class,
into40k training/5k validation/5k temperature/8k selector-fit/2k selector-check.
The official10k test set remains untouched until source and policy commits.
Retain all test cases:39 batches of256 plus16 inputs with quota4.

A real-image selector requires a different surrogate from the low-dimensional
grid. Use exactly reflection-even pooled pixel features; fit a simple ridge
regression to unlabeled multiclass Jensen gaps. Compare development-selected
histogram and nearest-neighbor controls using the same calibration information,
confidence selection, raw/full averaging and the full-information oracle.
Count the oracle's two views on every input. Do not call this an unchanged
transfer of the previous interpolator or a universal best real-image method.

Measure complete per-batch inference, including transformations, selector
work and actual network calls, on CPU and the existing5070Ti. Record warmup,
repetitions, GPU synchronization and transfer conventions. Report training,
selector calibration, inference and verification separately. Fewer forward
examples need not give lower wall-clock time. Any paired-reflection finite-pool
analysis has constructed symmetry and must be distinguished from the original
test distribution; report original-distribution results empirically.

## Independent streams and scientific release

Synthetic development uses2091..2098 seed families. Fresh training, initial
state, validation, temperature, selector fit, performance, ties and labels
use2101000 through2108000 plus replication1..16. Robustness certificate
inputs/ties use2121000/2122000, stress inputs/ties2124000/2125000 plus1..16.
Real-data split/training/selection/benchmark streams use2131..2139 families.
Root independent checks use2141 families, or exactly replay declared streams.

Each branch records its complete concrete configuration and hash freeze.
Internal root release is a scientific safeguard within existing user
authorization, not a request for user permission. Source/protocol amendments
before fresh access must be documented; post-outcome diagnostics cannot tune
the frozen experiment. Root reviews proofs, independent reconstructions,
actual cost ledgers, held-out boundaries and the final report before archiving.

No first calibration, approximate-invariance or concentration theorem is
claimed. The research value is the controlled result, explicit cost/benefit
tradeoff and honest failure boundary. Next directions depend on those results.
