# Readout-decoupling novelty audit

This is the curated English archive of a skeptical novelty review completed on 21 November 2026. Decision: retire the current candidate as the main innovation. This does not rule out every future architecture-specific theorem.

## Contents

- `REPORT.md`: decision, closest primary-source comparisons, elementary reductions, and evidence.
- `MATHEMATICAL_APPENDIX.md`: independent derivations, trajectory-control conditions, and explicit counterexamples.
- `EMPIRICAL_APPENDIX.md`: retrospective stored-state diagnostic methods, findings, and limitations.
- `DECISION.json`: structured status and scope.
- `PACKAGE_CHECKS.json`: source/input digest checks and historical reconstruction status.
- `source/verify_linear_mimic.py`: analytical linear example and independent direct-update check.
- `source/audit_archived_states.py`: original-training-input and stored-checkpoint diagnostic driver.
- `linear_mimic/`: all four simulated paths, matched-clock checks, numerical summary, and source digest.
- `empirical_outputs/`: endpoint contrasts, local diagnostic tables, raw numerical arrays, source snapshot, input provenance, and a portable copy of the historical dependencies.
- `manifest.json`: sizes and SHA256 digests for every content file other than the manifest itself.

The detailed literature-search working memo remains in the workspace. Verified comparisons needed for the decision are consolidated in REPORT.md. Original historical archives have not been overwritten.

## Reproduction

Use Python 3.12 with NumPy and threadpoolctl. The recorded environment is Python 3.12.14, NumPy 2.5.5. From this archive directory, choose fresh output paths:

```text
python source/verify_linear_mimic.py --output new-linear-output
python source/audit_archived_states.py --archive empirical_outputs/portable_inputs/archive --repository empirical_outputs/portable_inputs/repository --out new-checkpoint-output
```

Both drivers refuse existing output directories. The second command requires only the bundled historical dependency snapshots; it reconstructs the recorded training inputs and reads saved parameters. The included historical training driver and protocol are provenance inputs only and are not executed. No package installation of the original project is necessary for that audit driver.

Input copies have their own `INPUT_HASHES.json`; the top-level manifest additionally covers the whole package. Reproducing the audit at a new filesystem location changes absolute paths in its run record but should reproduce numerical diagnostics.

The linear simulation is an illustrative alternative explanation, not a fitted model of the original neural-network experiment. Checkpoint comparisons are descriptive retrospective training-set results, not generalization estimates or full-trajectory equivalence tests. Neither the current candidate nor a replacement is claimed to be novel.
