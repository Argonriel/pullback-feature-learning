# Independent-batch verification results

The frozen primary budget is 512 independent batches of 256 inputs per model. Each policy selects exactly 64. The lower bounds are simultaneous over all 48 models, both absolute-gain statements, the relative comparison, three batch-count looks, and both concentration methods (864 one-sided events; family error at most 0.05). Training and policy calibration were frozen before the independent verification stream.

The reported bound is the maximum of empirical Bernstein and bounded differences, both using the final joint error allocation. The earlier 432-event empirical-Bernstein outputs are retained as diagnostics and are not the primary result. These are bounds on expected logistic-loss reductions under the stated symmetry law. They are not classification-error guarantees. Structurally even selectors are already non-harmful by the theorem; a positive bound quantifies a guaranteed gain.

| Task | Statement at 512 batches | Positive / 16 | Mean lower bound | Model range |
|---|---|---:|---:|---:|
| Gaussian-noisy rings | Candidate absolute gain | 16 | 0.0060797 | [0.0020774, 0.0137405] |
| Gaussian-noisy rings | Strong histogram absolute gain | 16 | 0.0060608 | [0.0020724, 0.0136931] |
| Gaussian-noisy rings | Candidate gain minus strong histogram gain | 0 | -0.1225385 | [-0.1462677, -0.0870609] |
| One-amplitude waveform | Candidate absolute gain | 16 | 0.0471006 | [0.0197167, 0.0726216] |
| One-amplitude waveform | Strong histogram absolute gain | 16 | 0.0452426 | [0.0186541, 0.0703964] |
| One-amplitude waveform | Candidate gain minus strong histogram gain | 0 | -0.1540412 | [-0.1960614, -0.1208733] |
| Two-amplitude waveform | Candidate absolute gain | 16 | 0.0483549 | [0.0339758, 0.0635318] |
| Two-amplitude waveform | Strong histogram absolute gain | 16 | 0.0430409 | [0.0300765, 0.0565792] |
| Two-amplitude waveform | Candidate gain minus strong histogram gain | 0 | -0.1489236 | [-0.1709490, -0.1152092] |

Negative statistical bounds are retained. The structural zero bound is available for each absolute gain separately; it cannot replace a negative relative bound. Absolute observations use the fixed clipped gap (0.25 for rings, 0.5 for both waveform tasks). Relative observations are untruncated on compact tasks; rings use a radius-three region plus a deterministic tail subtraction. Clipping the signed contrast was not used.

## Additive concentration comparison

Both columns below use the same final family allocation and exactly the same saved observations. The bounded-difference calculation adds no prediction calls. The primary budget remains 512; 128 is a predeclared cost-curve look.

| Candidate, at 128 batches | Empirical Bernstein positive / 16 | Bounded differences positive / 16 | Mean EB lower bound | Mean bounded-difference lower bound |
|---|---:|---:|---:|---:|
| Gaussian-noisy rings | 1 | 16 | -0.0052558 | 0.0045392 |
| One-amplitude waveform | 14 | 16 | 0.0237691 | 0.0441457 |
| Two-amplitude waveform | 16 | 16 | 0.0255427 | 0.0452564 |

## Cost and tightness

| Task | Shared actual forwards at 512, mean | Candidate lower bound / 2048 lower bound, mean | Candidate lower bound / empirical full gain, mean |
|---|---:|---:|---:|
| Gaussian-noisy rings | 70,438.4 | 0.860 (16 positive denominators) | 0.636 (16 positive denominators) |
| One-amplitude waveform | 74,953.2 | 0.962 (16 positive denominators) | 0.874 (16 positive denominators) |
| Two-amplitude waveform | 81,200.8 | 0.968 (16 positive denominators) | 0.912 (16 positive denominators) |

The last two columns are descriptive ratios calculated per model only when the denominator is positive. The final column uses the same policy's unclipped gain on the same 2048-batch stream; it does not mean averaging every input or an oracle policy, and is not an independent certified recovery denominator. All counts, unreached first-positive budgets, raw observations, and alternative fixed looks are retained in the accompanying tables.

A standalone absolute certificate at 512 batches requires exactly 65,536 network forwards (both views of 64 selected inputs per batch). The joint run evaluates both views only on the selected-set union and reuses them for both absolute and relative statements. Evaluating both views of all 256 inputs at the same 512 batches would cost 262,144 forwards and produce exactly the same selected-gap statistics. Coordinate inputs cost no network calls. These are offline verification counts, distinct from 320-forward deployment batches and the 65,536-forward calibration of each competing method.

The completed joint 2048-batch run used 14,497,086 actual network forwards across 48 models and 25,165,824 coordinate inputs. Its wall time was 45.13 seconds including invariance audits and bookkeeping. This is a descriptive run duration, not an isolated serving benchmark; no prefix runtime is estimated by dividing it.

The policy class changed from the earlier asymmetric rank construction. Exact symmetry removes the need for rank-CDF and interval-derivative certificates. This does not establish that the same asymmetric certificate was accelerated, or that clipping preserves its tightness.

## Audits and reproduction

The bounded-difference calculation uses the proved fact that replacing one input-key pair changes a selected clipped-weight sum by at most the clipping level. It accounts for within-batch dependence and does not treat selected contributions as IID. Each of the 98,304 complete verification batches has an exact-quota and actual-union-forward ledger. Full and coordinatewise mixed half-turn masks are checked bitwise. Algebraic antithetic identities, deterministic range bounds, seed provenance, source hashes, and policy hashes are saved. No verification label or hidden posterior is queried. The guarantees are mathematical real-arithmetic results; floating-point audits are not interval proofs of every numerical operation.

From a fresh reproduction directory with committed model/policy artifacts and root approval:

```text
python certification/run.py run
python certification/run.py analyze
python certification/final_compare.py fresh
python certification/report.py
```

The source freezes are `FREEZE.json` and `ADDENDUM_FREEZE.json`; the declarations are `PROTOCOL.md` and the superseding `ADDENDUM.md`; the derivation and primary references are in `LITERATURE_AND_PROOF.md` and `ADDENDUM.md`. The old development pilot is preserved under `development_old32`, including the 432-event recalculation and final 864-event comparison. Its historical rerun requires the old study path passed explicitly to `development.py`; fresh confirmation uses only the current portable study dependencies. Outputs are write-once, so rerun in a new output tree instead of overwriting an existing result.
