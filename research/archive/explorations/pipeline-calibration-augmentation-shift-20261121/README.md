# Final-output calibration, learned reflection, and input-law mismatch

Completed 21 November 2026. All three user-authorized follow-ups are complete and independently audited. Start with `research_report.pdf` (six pages) or its readable source `REPORT.md`. Earlier study archives are unchanged.

## Decisions supported by this study

- At the same 4,352-label budget, allocating 2,304 labels to training rather than 256 reduces synthetic raw calibrated loss from 0.641332 to 0.547136 and error from 36.773% to 26.989%. Training costs more; this is not an equal-compute result.
- Fitting temperature to final synthetic outputs improves full and fixed selective pipelines on all 16 paired seeds. Positive temperature changes probability quality, not class decisions. The previously selected validation-heavy allocation performs worse than legacy at this equal label budget.
- Learning horizontal reflection repairs the severe real-image flip failure. Within the augmented CNN arm, full averaging lowers holdout error from 9.437% to 9.157%, while raw augmentation alone has no clear benefit. Separate full-output calibration adds no clear holdout benefit once augmentation is learned.
- The multiclass input-TV theorem is valid under its assumptions, but its global range penalty prevents positive certificates for every tested nonzero input shift. The smallest shift still has positive measured gains. No real-image certificate follows.

## Important data and cost boundaries

The primary image holdout is excluded from this round's fitting, but all its images were present in historical work. It is not a pristine external replication. All six CNNs are trained from scratch, and the already observed official test is secondary. Three paired initializations share one dataset split.

Steady-serving timing includes transformations, model calls, output merging, fitted scalar correction and named transfers. It starts from normalized tensors. Decoding, normalization and loading are excluded and not individually timed. CPU measurements use one thread. Full averaging improves accuracy slightly in the augmented arm but requires two model views and more measured latency.

## Artifact map

|Path|Contents|
|---|---|
|`STUDY_SCOPE.md`, `LITERATURE_REVIEW.md`|Prospective scope, primary comparisons, data boundary and literature context|
|`calibration/`|16 fresh paired replications, 48 models, 144 scalar fits, all 384 risks, fixed masks, fitting-label accounting, physical call records and uncertainty estimates|
|`augmentation/`|Six scratch CNNs, official data/license, exact new partition, paired training, 10k holdout plus descriptive official-test outputs, 378 serving measurements and all classwise outcomes|
|`shift/`|Multiclass theorem, eight fixed historical models, four controlled input laws, simultaneous moment bounds, stress tests, exact TV upper-bound obstruction appendix and full ledgers|
|`root_audit/PREFLIGHT.json`|Independent mean-loss gradient, normalized PB duplication and finite-population multiclass-TV checks|
|`root_audit/CALIBRATION_RESULTS.json`|All calibration risk rows, fits, streams, masks, checkpoint choices, primary intervals and deployment costs independently reconstructed|
|`root_audit/AUGMENTATION_RESULTS.json`|Splits, scalar fits, all image risks and intervals, and complete 120k-forward independent holdout replay|
|`root_audit/SHIFT_RESULTS.json`|All 512 elementary bounds, 2,560 stress rows, mask checks and six certificate-prefix replays|
|`root_audit/CNN_RANGE_BOUNDS.json`|Analytical pixel-cube logit-range bounds evaluated with numerical padding; explicitly not a natural-data certificate or formal machine proof|
|`SOURCE_INTEGRITY.json`, `REPORT_QA.json`|Frozen-source and release-hash verification; final six-page visual and independent review|
|`archive_manifest.json`|Archive-relative SHA-256 checksums and byte counts; Python caches and temporary PDF previews excluded|

## Reproduction

Each branch README gives exact run stages and dependencies. Preserve all completed results and run fresh experiments in a separate destination. The synthetic branches use Python, NumPy, matplotlib and threadpoolctl. The image branch uses the existing CUDA-enabled PyTorch runtime and RTX 5070 Ti. All scientific source dependencies and data licenses are included.

`python verify_sources.py` checks source freezes and internal scientific release commitments. `python verify_archive.py` checks all manifested archive files. Internal release records enforce the historical ordering of fitting and unseen evaluation; they are not new user permission requirements for ordinary artifact review.

`write_report.py` and `build_report.py` regenerate the report from completed tables. The latter uses ReportLab and installed Windows Calibri/Georgia fonts. Regeneration may alter PDF metadata and its hash, so a regenerated PDF requires new visual review and manifest creation. `record_report_qa.py` records the completed review of the delivered report; it is not a replacement for human-style visual inspection.

## Next direction

Prioritize label allocation under equal training compute. Then test whether a single-pass student can retain the augmented two-view gain, with an equal-compute baseline and genuinely untouched external evaluation. For guarantees, tighten transformation-specific bounds in explicit law families where input and posterior defects can be justified. No post-test model, selector, temperature or threshold was retuned in this completed study.
