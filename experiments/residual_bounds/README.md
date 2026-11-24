# Residual-envelope certificate experiment

These are source copies of the prospective residual-envelope study archived at
`tmp/frontier-replication-residuals-20261121/structured` in the shared workspace.
The archive contains the frozen protocol, proof, models, score tables, raw evidence,
results, figures, release records, and standalone replay inputs. Research outputs
and model assets are intentionally kept outside the repository.

The extension uses two odd harmonics, a supplied orthogonal residual envelope,
independent coefficient intervals, and an exact joint endpoint penalty. Optional
independent labels provide a posterior-defect bound only under the stated fixed
eight-cell posterior model. No new model or selector is fitted.

From the completed standalone archive, with NumPy, Matplotlib and threadpoolctl:

```text
python reproduce.py --output ../fresh_residual_replay
```

The output directory must be new. Replay verifies frozen source and asset hashes,
uses the reviewed probability-endpoint recovery for the truth stage, and generates
all statistics, plots and reports. The original runtime source remains unchanged;
`stress_recovery.py` records and guards its roundoff-only correction. `assets.py`
documents the initial historical vendoring operation and is not needed for replay.

Use the archive's `REPORT.md`, `THEORY.md`, `LIMITATIONS.md`, `PROTOCOL.md`,
`STRESS_RECOVERY.md` and `README.md` for the assumptions and detailed file map.
The source-copy manifest in the archive verifies these repository copies.
