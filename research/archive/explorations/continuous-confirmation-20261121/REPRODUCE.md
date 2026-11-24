# Reproducing this follow-up study

Use Python 3.12 with NumPy, threadpoolctl, and Matplotlib. The recorded run uses Python 3.12.14, NumPy 2.5.5, Matplotlib 3.11.2, and threadpoolctl 3.8.0. Core model and unchanged comparison-policy source are bundled under source/, legacy/, and reference/. No network or previous archive is required for numerical experiments. Experiments set numerical libraries to one thread.

## Preserve the delivered archive

The frozen sources, protocols, original model states, calibration, outcomes and independent audits belong together. Use a fresh directory for reproduction; copy only source, scripts, protocols, and freeze/analysis manifests required for that stage. Do not copy generated result directories or exclusive-create summary CSV files into a rerun layout. Never erase results from the delivered archive.

The three individual study READMEs provide their exact reproduction stages. Confirmation training creates states used by the continuous and certificate studies. Reusing those states is intentional; the two theory studies are not independent additional training cohorts.

## Independent verification

From the archive root:

```text
python verify_archive.py
python audit/group_math.py
python audit/rectangle_math.py
python audit/confirmation_checks.py
python audit/continuous_checks.py
python audit/continuous_repair_independent.py
python audit/certificate_checks.py
```

verify_archive.py is read-only. The mathematical/numerical audits write their own JSON results; run them in a copy if the archive must remain byte-for-byte immutable. Scientific outcomes are not overwritten. File hashes and independent numerical reconstruction test different things.

build_report.py creates research_report.pdf from REPORT.md using ReportLab and the Windows Calibri/Georgia fonts. The experiment runtime need not contain ReportLab. Scientific charts are independently available in the individual study directories. Temporary PDF page previews are excluded from the delivered manifest.

New fitting decisions or changes to model training require a new protocol and output directory. A new run has different timing values, timestamps and output-file hashes even if its numerical predictions reproduce.
