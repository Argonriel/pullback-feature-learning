# PB label allocation and explicit training-work experiment

This executable study preserves the inherited 144-parameter model and sample-size-normalized PB update. It compares nine label allocations/training schedules on 16 paired seeds, with independent temperatures for actual raw and full mean outputs. Read the archived protocol before interpreting results.

The runner defaults to the research output branch at `tmp/budget-distillation-structured-bounds-20261121/allocation` under the workspace. Its output writer uses exclusive creation for primary artifacts; never delete or overwrite an existing study to rerun it.

From the workspace, use `.venv/Scripts/python.exe` to execute:

```text
pullback-feature-learning/experiments/budget_allocation/preflight.py
pullback-feature-learning/experiments/budget_allocation/runner.py freeze
pullback-feature-learning/experiments/budget_allocation/runner.py train
pullback-feature-learning/experiments/budget_allocation/runner.py calibrate
pullback-feature-learning/experiments/budget_allocation/runner.py score
pullback-feature-learning/experiments/budget_allocation/runner.py analyze
pullback-feature-learning/experiments/budget_allocation/accounting.py
pullback-feature-learning/experiments/budget_allocation/report.py
```

Training requires an internally reviewed `ROOT_APPROVAL.json` bound to the SHA-256 of `SOURCE_FREEZE.json`. Each later stage checks source and previous-stage artifact manifests. `FIT_COMMIT.json` is created before test streams are opened. The primary source freeze covers the runner, calibration/model helper, preflight and vendored model/task/development reference; plotting/accounting/timing are derived tools and never modify fits or test outputs.

For an independent reproduction, choose a new empty result directory, copy the original `PROTOCOL.md` and passing `PREFLIGHT.json` there, invoke each runner stage with `--out ABSOLUTE_DIRECTORY`, and create a new internal approval bound to its new freeze. The immutable preflight can alternatively be rerun by importing the runner, setting `runner.OUT` to the new directory and calling `preflight.main()` with one BLAS thread. Retain the original seed policy for exact reproduction; a new scientific confirmation needs a separately frozen protocol and fresh streams.

`timing.py --exclusive-confirmed` is a separate audit, permitted only after fit and scoring commits and a coordinator-granted exclusive CPU window. It runs seven short development-state workload blocks per training size and reports projected full training times. It does not tune or replace the frozen operation-count schedules.

The data law, model and temperature-scaling implementation are copied from the immutable previous calibration study. `reference/core4d.py`, `reference/task.py`, `reference/DEVELOPMENT_STATE.npz`, and `model.py` preserve their original bytes. They are intentionally self-contained to make the scientific source freeze inspectable.
