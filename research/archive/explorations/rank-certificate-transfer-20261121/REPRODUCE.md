# Reproduction and archive verification

The archived outputs are immutable research evidence. Fresh-stage runners refuse to overwrite completed directories. Use a new working copy for reproduction; do not remove or replace the archived original.

## Runtime

The experiments used Python3.12.14, NumPy2.5.5 and threadpoolctl3.8.0 on Windows, with BLAS restricted to one thread per worker. Matplotlib3.11.2 builds the scientific figures. ReportLab in the bundled document runtime renders the report. No GPU or network service is required by this study. Full environment paths and loaded BLAS details are retained in the run records. Small floating-point differences may arise on other BLAS implementations.

## Verify the saved study

Run `python verify_archive.py` at the archive root. It checks each saved byte count and SHA-256 digest without deleting files. Source freezes can also be checked through `validation/study.py` or `certificate/experiment.py`; all main scientific sources remain unchanged after their recorded freezes. `root_audit/freeze_checks.json` records the final cross-freeze audit.

The original fresh release was2026-11-21T10:15:56.872195UTC. It preceded training, policy calibration and scoring. The complete primary policy bank was committed at10:17:30.961775UTC before fresh outcomes. New model seeds differ from all development models. The three prior rings inputs needed for development checks are copied under `development/`; the four harmonic pilot seeds and every recipe are under `transfer/pilot/`.

## Replay in a new run directory

Copy the scientific sources, legacy dependencies, protocols and frozen metadata into a new empty run directory with the same relative layout. Preserve the original seeds and source hashes. Exclude generated fresh states, policies, predictions, certificate-result directories and completion records from that new directory. Retain the approved frozen configuration records; record that the new execution is a reproduction rather than a new confirmation cohort. Then use these stages in order:

1. `python validation/study.py train`
2. `python validation/study.py calibrate`
3. `python validation/study.py score`
4. `python validation/study.py analyze`
5. `python certificate/experiment.py run --task rings --kind known`
6. `python certificate/experiment.py run --task waveform --kind known`
7. `python certificate/experiment.py run --task rings --kind empirical`
8. `python certificate/experiment.py run --task waveform --kind empirical`
9. `python root_audit/certificate.py known`
10. `python root_audit/certificate.py empirical`
11. `python root_audit/audit_validation.py`
12. `python root_audit/audit_certificates.py`

The four enclosure commands can run in parallel after the complete policy commit. Each root confidence pass needs both corresponding task enclosures. Do not retune parameters after seeing a replay or confirmation result. Reproduction uses the same seeds and is not an additional independent replication.

The independent certificate audit is documented in `transfer/INDEPENDENT_CHECK_PROTOCOL.json` and its source. The fixed six-model audit has its own implementation and input streams. The independent bootstrap audit is documented in `validation/RESULTS.md`. These are separate from the original scoring and construction workloads.

## Final artifacts and provenance

`REPORT.md` is the human-readable manuscript, and `research_report.pdf` is its rendered version. `make_figures.py` regenerates the scientific figures; `write_report.py` assembles completed results into the manuscript; `build_report.py` renders it. Review the final page images after regeneration.

`root_audit/CERTIFICATE.md`, `certificate/THEORY.md`, `transfer/THEORY.md` and `VALIDITY_NOTES.md` state the mathematical assumptions and numerical limits. Known-law and empirical-law results have separate95% confidence families. Empirical integrated-bound summaries use the synthetic law descriptively; the actual empirical certificate does not.

`verify_archive.py --build` is for a newly finalized run, after all results and reporting files stop changing. `archive_to_research.py` copies a verified manifest into a destination that does not already exist, then checks every copied digest. It never overwrites a previous study.
