# Symmetric allocation method branch

The method and histogram choices were frozen at
2026-11-21 10:50:55.540406 UTC, before access to this round's fresh models.
`METHOD_FREEZE.json` contains source and development-commit hashes. Frozen
files must not be edited. `THEORY.md` gives the exact invariance argument;
`LITERATURE.md` distinguishes established principles from this study's scope.

## Development results

The previous 16 ring models and 16 one-invariant harmonic models are now
development data. The two-invariant results use four training-pilot models,
not 16 fresh confirmations. Each method receives 65,536 calibration-forward
examples. Histogram candidates reuse the same 32,768 input/partner pairs.
All choices use selected Jensen gain, without labels or hidden posterior.

| Cohort | Models | Chosen histogram | Histogram gain | Interpolated gain | Nearest gain | Exact-J oracle gain |
|---|---:|---|---:|---:|---:|---:|
| Rings | 16 | 64 x 32 | 0.01037407 | 0.01038969 | 0.01038796 | 0.01039015 |
| One-invariant harmonic | 16 | 64 x 32 | 0.05149816 | 0.05309630 | 0.05305207 | 0.05310292 |
| Two-invariant harmonic | 4 | 16 x 16 x 16 | 0.04853064 | 0.05307864 | 0.05268201 | 0.05309867 |

These are mean selected Jensen gains, in nats per deployment input, not
realized gains on each labeled batch. For individually invariant selectors
they identify expected logistic-risk improvement under the assumptions in
THEORY.md. The oracle is evaluated with both views of every input and costs
2 forward examples per input. It is not a deployable 1.25-cost policy.

Mean modelwise oracle fractions for interpolation are 99.9951%, 99.9885%,
and 99.9635%, respectively. The strongest histogram already reaches
99.8331% on rings, so that cohort has very small remaining headroom. No
development comparison establishes superiority on fresh training draws.

All 180 executed selector checks passed: individual-input half-turn flips
changed no mask or score; each actual deployment check queried 256 first
views and exactly 64 selected partners; outputs matched saved evaluation
outputs within the recorded floating-point tolerance. No settings failed
or were excluded. Every candidate histogram result is retained, including
the two unselected resolutions in each cohort.

## Integration API

```python
grid = baseline.fit_grid(adapter, theta)
folded = baseline.fit_folded(adapter, theta, calibration_x,
    config=chosen_config, prepared=(calibration_h, calibration_hT))
mask = baseline.select(adapter, x, grid, ties)
nearest_mask = baseline.select_nearest(adapter, x, grid, ties)
folded_mask = baseline.select_folded(adapter, x, folded, ties)
out, mask = baseline.deploy(adapter, theta, x, grid, ties)
```

The adapter provides `name`, `dimension`, `coordinates`, `quotient`,
`partner`, and `predict`. The score paths never receive logits or a hidden
posterior. The `prepared` argument lets every histogram resolution share
one paid calibration-pair pool. Its `forward_examples` records the total
pool budget; `additional_forward_examples=0` records the lack of extra
queries during histogram refitting. Stored grid logits cost exactly one
forward example at each grid point; a shifted copy supplies partner nodes.

Fresh integration uses names `rings`, `waveform`, and `two_invariants`.
Development folders retain historical names `rings`, `transfer`, and
`task2d`, with the correspondence recorded in `summarize.py`.

## Files and reproduction

`baseline.py` is the self-contained NumPy selector implementation.
`pilot.py` created the bounded development results. `summarize.py` is a
deterministic aggregator. `development_per_model.csv` contains all model
results; `development_summary.csv` averages models equally.
`development_checks.csv` holds all deployment/invariance checks.
Each `development/<cohort>_<pair>/` contains calibration tables, a policy
hash commitment before evaluation, evaluation inputs/logits/masks, per-batch
results, source-state hashes, and a completion record.

The original development pilot uses the immediately preceding study's
adapters and 32 frozen states. With both studies present as sibling folders,
the original commands are:

```powershell
.venv/Scripts/python.exe tmp/symmetry-frontier-2d-20261121/method/pilot.py --cohort rings
.venv/Scripts/python.exe tmp/symmetry-frontier-2d-20261121/method/pilot.py --cohort transfer
.venv/Scripts/python.exe tmp/symmetry-frontier-2d-20261121/method/pilot.py --cohort task2d --pairs 1,2,3,4 --state-pattern 'C:/Users/liwei/Documents/ChatGPT/277/tmp/symmetry-frontier-2d-20261121/task2d/pilot_results/{pair}_pb4_500.npz'
.venv/Scripts/python.exe tmp/symmetry-frontier-2d-20261121/method/summarize.py
```

Run a full pilot reproduction in a separate copy: it deliberately refuses
to overwrite existing model output folders. The prior study remains
unchanged. Recomputing only the aggregate tables is deterministic but would
replace their files; use a copied branch for that too. Fresh confirmation
uses the current study's portable adapters and the frozen baseline API;
it does not depend on the preceding study.

This branch measures forward examples and calibration timings. It makes
no GPU latency claim and does not silently treat stored predictions or
oracle partner calls as free. Real-arithmetic invariance proofs and ordinary
float64 checks are distinct; no directed-rounding certificate is asserted.
