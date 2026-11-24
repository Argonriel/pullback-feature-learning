# Independent fresh-results audit

Status: PASS. The audit completed at 2026-11-21 10:57:56 UTC. It did not
modify any frozen method, task, validation, or certification source.

`fresh_results.py` independently reconstructs risk, errors, posteriors,
histogram/grid masks, oracle headroom, and the three primary model-paired
bootstrap comparisons. It uses the frozen data/model adapter only to
regenerate input streams and replay training. It does not call the primary
validation metric or selector functions.

## Checked outcomes

- All 384 policy/model rows matched. Maximum numerical metric discrepancy:
  4.440892098500626e-16.
- All saved masks matched independently reconstructed selections; zero
  changes. Every quota was exact. Saved output/averaging relations matched
  exactly.
- All 240 state, policy, and prediction files matched their committed hashes.
  Method freeze, validation freeze, root release, training, calibration, and
  scoring timestamps were in the required order. Frozen validation source
  hashes were unchanged.
- All 48 training and performance input streams, initializations, tie keys,
  and sampled labels were regenerated. Training and performance input
  hashes were distinct across the 48 models.
- Six 100-step training replays, pairs 1 and 16 in each cohort, reproduced
  the saved intermediate state exactly. This checks deterministic replay,
  not an independent derivation of the training algorithm.
- All 98,304 batch/policy ledger records were unique and valid. Performance
  used exactly 35,389,440 forward examples; calibration used 6,291,456 across
  the two shared calibration pools for 48 models.

## Primary contrast reconstruction

Differences below are interpolated Jensen policy minus the frozen strongest
histogram in expected logistic loss. Negative values favor interpolation.
Intervals use the frozen 20,000 model-paired bootstrap resamples and
98.333333% marginal level for the three primary comparisons.

| Cohort | Mean difference | Lower | Upper | Models favoring interpolation |
|---|---:|---:|---:|---:|
| Rings | -0.00001766325 | -0.00003368948 | -0.00000505749 | 13/16 |
| One-invariant harmonic | -0.00178496907 | -0.00221848886 | -0.00135043491 | 16/16 |
| Two-invariant harmonic | -0.00538082612 | -0.00633723306 | -0.00455379654 | 16/16 |

The bootstrap is a finite-replication approximation. It is separate from
the certification branch's bounded-observation, finite-sample guarantees.
The ring effect is small in absolute terms. These results do not establish
dominance over confidence selection or over arbitrary transformation
families. Confidence can exploit posterior-specific behavior and is not
covered by the invariant-mask selected-Jensen identity.

## Pre-release review

The independent executed harness check used three development models and
all eight policies, verifying actual predictor calls against the ledger.
It confirmed 24 nonoverlapping fresh stream namespaces containing 384 seed
values. A missing failure snapshot was flagged before release; the training
loop was amended to save its failing state/trace/step before rethrowing, with
no seed replacement. The final source was frozen only after that correction.
No fresh training failure was observed.

The additive McDiarmid argument was reviewed independently. With fixed
pointwise scores and tie keys, replacing one input/key pair changes the
selected weighted sum by at most the weight range: the remaining items keep
their relative order and at most one item is exchanged. A difference of two
policies has twice that sensitivity. The proof permits independent input
coordinates even though the quota makes selected items dependent. The
addendum splits the confidence budget across both bound methods and all
prespecified looks before fresh data.

## Interpretation limits

The guarantees require exact half-turn invariance of the input law and
posterior and individual-coordinate invariance of the policy mask. They
concern expected logistic loss, not every realized batch or classification
error. The two-amplitude quotient separates orbits on the phase-coupled
support, not all ambient R4. Verification savings arise from this symmetric
policy restriction; they do not accelerate the previous asymmetric policy's
certificate or demonstrate a serving-time speedup. All arithmetic checks
use ordinary floating point, without directed rounding.
