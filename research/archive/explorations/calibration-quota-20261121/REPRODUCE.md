# Reproducing the completed study

The archive is self-contained for confirmation and verification. No network or account access is needed. All source and research documents are English. The original Word agenda is retained as an unmodified source document.

Recorded experiment runtime: Python 3.12.14, NumPy 2.5.5, Matplotlib 3.11.2, threadpoolctl 3.8.0. Experiments restrict numerical libraries to one thread. PDF creation separately uses ReportLab and the Windows Calibri and Georgia fonts. Timing is machine-specific.

## Verify existing results

From the archive root, run:

```text
python verify_archive.py
python audit/math_checks.py
python audit/artifact_checks.py
```

The audits regenerate numerical check JSON files. For strictly immutable archive verification run only verify_archive.py, or copy the archive before running the audits. Existing scientific result files are not overwritten by these audits. File integrity and independently reconstructed numerical results are separate checks.

## Recreate confirmation in a new directory

Create a fresh study layout by copying only source/, legacy/, the study scripts, historical inputs, and freeze/analysis manifests. Omit generated fresh/, calibration/ and results/ directories and generated report tables. In particular, do not copy idea2b/summary.csv, contrasts.csv or latency_summary.csv into the rerun layout: summarize refuses to overwrite them. Never remove results from this delivered archive.

```text
python idea2a/study.py train
python idea2a/study.py calibrate
python idea2a/study.py evaluate
python idea2a/study.py benchmark
python idea2a/analyze.py
python idea2a/contrast_report.py
python idea2b/experiment.py calibrate
python idea2b/experiment.py score
python idea2b/experiment.py summarize
python idea2b/report.py
python ideas34/study.py fit
python ideas34/study.py evaluate
python ideas34/analyze.py
```

Use the delivered freeze manifests and unchanged frozen source; training and calibration commands refuse existing output directories. The supplied ideas34 freeze is already sufficient for fit. To create a wholly new timestamped protocol instead, retain the theory/source and run study.py freeze in a copy without freeze.json before fit. Recreated outputs will have new timestamps, timing values and file hashes even if numerical predictions reproduce.

The supplied history/historical_curves.npz and old_pilot/historical_gain_curves.npz make historical defaults inspectable without prior archives. The original optional 2A history command also expects the older retrospective risks.csv and population_gain_curves.csv; local copies are in inputs/retrospective_rotation/. It is unnecessary to rerun history for confirmation. If rebuilding history in a new layout, pass that path with --retrospective-source and generate a new freeze.

## Report and preservation

REPORT.md is the combined narrative. Run python build_report.py with ReportLab installed to regenerate research_report.pdf. Scientific plots are supplied as PNG and vector SVG/PDF where available.

Do not edit frozen study.py or experiment.py while claiming to reproduce this run. New scientific changes need a new protocol and output directory. The original and amended 2B code, freezes and pre-test calibration outputs are intentionally retained. They document a stronger comparator addition and an independent phase-stream correction before held-out scoring, not outcome-driven selection.
