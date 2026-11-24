# Portable reproduction

The archived `source/` directory is a complete copy of the reusable repository implementation at `pullback-feature-learning/experiments/budget_allocation/`. The four inherited model/task/development files remain byte-identical to the previous study, as recorded in `VENDOR_PROVENANCE.json`. The primary freeze covers `runner.py`, `model.py`, `preflight.py`, and `reference/`; later reporting, cost-accounting and timing helpers do not change primary outcomes. `SOURCE_SNAPSHOT.json` hashes the complete archived source directory.

Use Python with NumPy, Matplotlib and threadpoolctl installed (the original workspace environment is `.venv/Scripts/python.exe`). The portable wrapper sets the output directory before loading any stage. No network access is required.

To reproduce the complete experiment, create a **new empty** result directory, copy `PROTOCOL.md` there, and run these commands, substituting absolute paths for `PYTHON`, `ARCHIVED_SOURCE` and `NEW_RESULTS`:

```text
PYTHON ARCHIVED_SOURCE/reproduce.py preflight --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py freeze --out NEW_RESULTS
```

Review the new source freeze/protocol, then create `NEW_RESULTS/ROOT_APPROVAL.json` with `fresh_authorized: true` and `source_freeze_sha256` equal to the SHA-256 of the new `SOURCE_FREEZE.json`. This is an internal scientific review record. The original approval cannot be reused because it is bound to the original freeze. Exact reproduction retains the same seed streams; a new prospective scientific confirmation instead requires a newly frozen protocol and new streams.

After that review, run sequentially:

```text
PYTHON ARCHIVED_SOURCE/reproduce.py train --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py calibrate --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py score --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py analyze --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py accounting --out NEW_RESULTS
PYTHON ARCHIVED_SOURCE/reproduce.py report --out NEW_RESULTS
```

Primary stages use exclusive file creation and verify prior-stage hashes; never point a fresh run at the archived completed results. To regenerate only derived charts/report from an intact completed archive, use the final report command with that archive as the output directory. That command changes only derived report/figure files.

The optional timing audit requires fit and score commits and a coordinator-granted exclusive CPU window. Stop other numerical jobs, then run:

```text
PYTHON ARCHIVED_SOURCE/reproduce.py timing --out NEW_RESULTS --exclusive-confirmed
```

This is a separately recorded development-state latency audit and never changes schedules or fits. Running the report stage afterwards includes its timing summary. All fitting, scoring and timing numerical work uses one BLAS thread.

For independent result reconstruction, load `predictions/{pair}.npz`: it contains `x`, `p`, `y`, and `{procedure}_raw`/`{procedure}_full` unscaled logits. The corresponding `temperatures/{pair}_{procedure}.json` gives `fits.raw.beta` and `fits.full.beta`. Use beta=1 for uncalibrated outputs, raw beta for the full transfer diagnostic, and own beta for calibrated raw/full outputs. Expected NLL is mean(logaddexp(0,h)-p*h); expected error is mean(where(h>=0,1-p,p)). `per_model.csv` has all 720 model/procedure/output records. Bootstrap pairing is the `pair` column; the eight primary definitions and seed are in the frozen runner/protocol.

`training.csv` is a per-procedure standalone ledger; `actual_training_work.csv` records the shared trajectory experiment. `standalone_cost_ledger.csv` combines training, validation, calibration and serving units. `COST_TOTALS.json` includes synthetic rejection proposals and scalar temperature label evaluations. `MODEL_COMMIT.json`, `FIT_COMMIT.json`, and `SCORE_COMPLETE.json` hash the primary artifacts.
