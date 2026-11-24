# Portable source and data reproduction

`source/` contains a complete snapshot of the reusable implementation in `pullback-feature-learning/experiments/allocation_frontier/`. `SOURCE_SNAPSHOT.json` hashes that portable copy; the primary `SOURCE_FREEZE.json` hashes the outcome-generating code, old development reference, protocol and preflight. `VENDOR_PROVENANCE.json` checks the inherited model/task bytes. Older studies are untouched.

Use Python with NumPy, Matplotlib and threadpoolctl. The original environment is the workspace `.venv/Scripts/python.exe`. Replace `PYTHON`, `ARCHIVED_SOURCE` and `NEW_RESULTS` below with absolute paths. Choose a **new empty** result directory, copy this study's `PROTOCOL.md` there, and execute:

```text
PYTHON ARCHIVED_SOURCE/reproduce.py preflight --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py freeze --out NEW_RESULTS
```

After reviewing that newly generated source freeze and protocol, create `ROOT_APPROVAL.json` in the new output directory with `fresh_authorized: true` and `source_freeze_sha256` equal to the SHA-256 of its new `SOURCE_FREEZE.json`. This is an internal scientific-release record. The original approval cannot be copied because it is bound to the original freeze. Retaining the same streams exactly reproduces the study; new scientific confirmation requires a new frozen protocol and different stream family.

Run the following stages sequentially:

```text
PYTHON ARCHIVED_SOURCE/reproduce.py train --phase development --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py calibrate --phase development --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py score --phase development --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py select --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py train --phase confirmation --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py calibrate --phase confirmation --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py score --phase confirmation --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py analyze --out NEW_RESULTS
```

The selected identity is derived automatically by the frozen sampled-label criterion and committed before confirmation. A baseline win is handled automatically without fitting a duplicate challenger. The runner preserves failures and uses exclusive primary-file creation; never point a new run at a completed archive.

After confirmation is committed, obtain a coordinator-granted exclusive CPU window with other numerical jobs idle, then run the frozen complete-procedure timing audit. Subsequently generate derived cost ledgers and English figures/report:

```text
PYTHON ARCHIVED_SOURCE/reproduce.py timing --out NEW_RESULTS --exclusive-confirmed
PYTHON ARCHIVED_SOURCE/reproduce.py account --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py report --out NEW_RESULTS
```

Timing is optional for reproducing predictive outcomes, but belongs to the complete study. It reuses development pair 1 and adds no new labels. Its one warmup plus three measured paired blocks reproduce baseline/selected training, diagnostic checkpoints and both calibration fits; output checks occur outside the clock. No timing measurement changes selection or schedules.

For independent scoring, load `development/predictions/{pair}.npz` or `confirmation/predictions/{pair}.npz`. Both contain `x`, sampled `y`, and `{procedure}_raw`/`{procedure}_full` unscaled logits; confirmation additionally contains `p`, the known posterior. The corresponding `temperatures/{pair}_{procedure}.json` stores raw/full inverse temperatures. Raw own-temperature uses raw beta; full own-temperature uses full beta; the transfer diagnostic applies raw beta to full logits. Expected NLL is mean(logaddexp(0,h)-p*h); sampled NLL is mean(logaddexp(0,-y*h)); expected error is mean(where(h>=0,1-p,p)).

Development selection uses only the eight-pair mean sampled NLL for `raw_T`, with minimum-training-n tie handling inside 1e-12. Reconstruct it from `development/per_model.csv` and compare with `SELECTION_COMMIT.json`. Confirmation contrasts pair on the `pair` column with bootstrap seed 2419050. `confirmation/contrasts.csv` declares primary versus secondary metrics and the no-test baseline-identity case.

States, initializations, nested training pools, calibration examples, diagnostic checkpoints and rejection metadata are retained under each phase's `states/` and `splits/`. Standalone procedure counts and full recipe-discovery cost are separated in `standalone_cost_ledger.csv` and `COST_TOTALS.json`. Each phase's `MODEL_COMMIT.json`, `FIT_COMMIT.json` and `SCORE_COMPLETE.json` permit independent integrity checks before recomputing results.
