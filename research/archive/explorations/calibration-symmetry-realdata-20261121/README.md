# Calibration, imperfect symmetry, and real-image transfer

Completed on 21 November 2026. All three authorized studies and their independent audits are complete. Read `research_report.pdf` for the six-page report or `REPORT.md` for its source text. Prior study archives are unchanged.

## Main findings

- Independent temperature calibration improves the legacy synthetic model's raw loss from 0.697837 to 0.647410 without changing classification error. The more elaborate selected training plus calibration does not beat that simple baseline in mean raw loss. Fitting temperature before averaging can worsen the final averaged predictions.
- Under an exactly symmetric input law, the 64 primary safe-policy certificate lines remain positive when the uniform paired-posterior gap is at most 0.10. The selection mask must be invariant to transforming each individual input. These assumptions are not established for the real-image task.
- Fashion-MNIST original-image error rises from 9.593% to 20.907% under the ridge reflection selector. Independent direct replay reproduces this failure. Raw inference is faster and more accurate. The RTX 5070 Ti accelerates the CNN, but all limited-view GPU policies are slower than the strongest full-averaging implementation.

## Contents

|Location|Evidence|
|---|---|
|`STUDY_SCOPE.md`, `LITERATURE_REVIEW.md`|Prospective scope and primary-source literature notes|
|`training/`|72-trajectory pilot, 16 fresh paired replications, six predictor variants, fitted temperatures, calibration tables, outputs, deployment ledger, and fixed bootstrap contrasts|
|`robustness/`|Exact defect identity, predeclared concentration bounds, 64 primary certificate lines, 2,560 stress-result rows, physical costs, and source commitments|
|`realdata/`|Official Fashion-MNIST data and license, three CNNs, fixed splits, selectors, original and artificial paired evaluation, 504 latency rows, and CPU/GPU agreement|
|`root_audit/`|Independent gradient checks, arithmetic and stream replays, final-model audit, all-image direct CNN replay, classwise diagnosis, and post-outcome binary input-TV derivation|
|`SOURCE_INTEGRITY.json`|Verification that all five scientific source freezes still match current files and internal release commitments|
|`REPORT_QA.json`|Final report hash, six-page visual inspection and independent report review|
|`archive_manifest.json`|Archive-relative byte counts and SHA-256 hashes; excludes temporary previews and Python caches|

## Reproduction and integrity

Use the branch READMEs and protocols for exact execution stages. Python with NumPy, matplotlib and threadpoolctl runs the synthetic studies; the image branch uses PyTorch with CUDA for the recorded GPU conditions. Saved results remain available for verification without retraining. Report construction requires ReportLab and the fonts specified in `build_report.py`.

`python verify_sources.py` verifies the scientific freezes. `python verify_archive.py` verifies all manifested artifacts. Run research stages only in a fresh separate workspace with new output locations; do not overwrite this completed record. The internal approval records enforce the scientific separation between selecting procedures and evaluating unseen outcomes; they are not user permission requirements for ordinary artifact review.

The report can be regenerated from saved tables with `write_report.py` and `build_report.py`. Rendering can change PDF metadata and therefore its file hash; a regenerated report needs new visual review and an updated manifest. The original final PDF is preserved in this archive.

## Next direction

1. Calibrate the final inference pipeline and compare methods at equal total label budget.
2. Train with and without validated reflection augmentation, then evaluate frozen choices on a new held-out set; retain raw inference as the primary reference.
3. Address input-distribution mismatch explicitly. The post-outcome binary TV calculation is explanatory and cannot certify the multiclass image experiment.

No models or policies were retuned after the fresh test failures. This archive records negative outcomes as completed results, not as successful deployment evidence.
