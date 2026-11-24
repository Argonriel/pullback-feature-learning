# Allocation confirmation, independent replication, and residual-aware guarantees

Start with `research_report.pdf` or its English source `REPORT.md`. This is a new study, completed on 21 November 2026 local time. All three preceding studies remain intact and independently checksum-verified.

## Decisions

- The frozen eight-pair raw-loss selection retains the existing 2,304-training/2,048-calibration recipe. Sixteen independent baseline-only confirmation fits give raw/full expected loss 0.552949/0.532814. Identity contrasts have no improvement tests. The exploratory 3,072-training full-output observation is not promoted or independently confirmed.
- Six fresh paired image fits do not replicate the former external student advantage. On 1,500 independently acquired UT Zappos products, student error is 47.011%, versus 46.778% plain and 44.800% full-compute plain. Both primary seed-based comparisons fail the frozen replication rule. The separate pooled serving criterion passes, with individual reversals retained.
- A two-harmonic input model with an explicit residual envelope gives useful conditional certificates. On the primary third-harmonic law, 16/16 lines are positive when epsilon=0 is supplied; 14/16 remain positive using independently labeled posterior intervals. Residual and within-cell posterior assumptions remain essential. Violated-envelope outputs are invalid sensitivity results, not guarantees.

## Artifact map

| Path | Contents |
|---|---|
| `STUDY_SCOPE.md`, `LITERATURE_REVIEW.md` | Prospective questions, boundaries and primary-source context |
| `allocation/` | Frozen development/confirmation, all 48 final states, 96 scalar fits, 240 risks, complete-fit timing, costs, source and replay instructions |
| `distillation/` | Eighteen CNNs, all checkpoints/targets, historical split, two external populations, predictions, six-seed and crossed-product inference, isolated serving measurements |
| `structured/` | Full theorem, frozen assumptions, parameter and actual-label evidence, 4,288 moment rows, 1,920 bound rows, 5,504 truth rows, negative cases and recovery |
| `root_audit/` | Independent predictions/calibration/inference/selection/bounds/cost reconstruction, two mathematical reviews and historical integrity |
| `REPORT_QA.json`, `COMPLETION.json` | Final PDF page review and completion checks |
| `SOURCE_INTEGRITY.json`, `archive_manifest.json` | Commitment/chronology audit and delivered-file SHA-256/byte manifest |

## Source data and scope

The new [UT Zappos50K official source](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/) is restricted to academic, noncommercial use; retain its official README and Yu/Grauman citations. The private research archive includes every evaluated original image, complete metadata, exact group/color ranking and processed pixels. The 305,323,328-byte original image ZIP contains many unused photos and is excluded from compact delivery, with its digest and retrieval URL preserved; it remains in the working copy. `root_audit/external_selection.py --zip PATH` independently replays from that official ZIP.

This is a three-class footwear stress test, with all ten model logits active. Source and category composition change together; only sandals overlap the previous six-class catalog. Product identities remove repeated colors, not all near designs or brand dependence. Fashion-MNIST and the earlier Myntra photos are historically exposed. No external labels fit or tune these pipelines. The old catalog owner's MIT declaration is preserved as source metadata, not independently audited ownership clearance.

The image inference uses six paired training randomizations on one fixed training dataset. Small-sample Student-t and crossed-bootstrap intervals are approximate; neither establishes variation over new training datasets or resolves brand dependence. Nominal arithmetic matching is explicitly defined and does not equate all hardware work or elapsed time. Actual benchmark windows paused other numerical workloads.

The structured labels support the posterior bound only under constancy inside the eight fixed cells. The density envelope and orthogonality are supplied assumptions. `LIMITATIONS.md` and the sine-posterior counterexample show why arbitrary finite unlabeled observations or cell means cannot remove these restrictions.

## Preserved numerical failure

The original structured stress run stopped on tiny negative endpoint roundoff; the recovery observed a maximum excursion of 2.78e-17. Its frozen source, original start/failure records and empty raw directory are preserved. A separately frozen and reviewed recovery checks excursions <=1e-12, clips only endpoints and replays exactly the same seeds into `stress_recovery/`. Its maximum correction and all affected endpoints are recorded. The failed attempt's 65,536 inputs/131,712 forwards are included as overhead. The successful rerun is not independent extra experimental evidence.

## Verification and reproduction

`python archive_tools.py` verifies the archive without writing. `python verify_sources.py` checks frozen sources, committed data and stage chronology without writing. The audit scripts under `root_audit/` likewise default to read-only; `--record` deliberately writes a new audit record and belongs only in a new working copy. `archive_tools.py --build` creates a release manifest and must not replace the integrity record of a completed archive during ordinary verification.

Branch reproduction instructions define new-output workflows. Never run stages against completed output directories; exclusive-create writes intentionally refuse overwrites. Preserve both original and recovery freezes. Re-running a benchmark produces new measurements and must not silently replace this study's recorded outcomes.

NumPy experiments use the existing workspace CPU Python environment; CNN replay uses its PyTorch 2.7.1/CUDA12.8 environment and the RTX 5070 Ti. The PDF builder uses the bundled document runtime with ReportLab and Windows fonts. Branch records retain versions, seeds, device details and dependency provenance. The independent audits replay all final predictions and key statistics; full training trajectories are not independently repeated, and the proof is not machine-checked with directed rounding.

Reusable implementation is also retained in new repository directories `experiments/allocation_frontier`, `experiments/distillation_replication`, `experiments/residual_bounds` and `experiments/independent_validation_frontier_20261121`. Research results stay in this separate archive. Local source hashes and event records establish internal consistency; they are not an externally timestamped preregistration.

Next priorities are to separate optimizer/calibration allocation mechanisms, establish a target-domain adaptation baseline on untouched product groups before further distillation claims, and justify residual/posterior regularity bounds beyond the current supplied structure.
