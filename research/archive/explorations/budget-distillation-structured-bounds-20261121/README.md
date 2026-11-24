# Training budgets, one-pass distillation, and structured guarantees

Completed 21 November 2026. Start with `research_report.pdf` or its English source `REPORT.md`. This is a new, separately frozen study. The preceding pipeline/calibration/augmentation/shift and earlier calibration/symmetry/real-data archives remain preserved and checksum-verified.

## Main results

- The 2,304-training / 2,048-calibration baseline remains strongest in the frozen synthetic allocation grid. At matched nominal dense arithmetic work, all six primary smaller-training-versus-baseline comparisons favor it, with adjusted loss intervals excluding zero. Raw calibrated expected loss is 0.553049, versus 0.662787 for 256 training examples optimized for 4,332 updates. Validation reallocation shows no clear benefit. Arithmetic equality is explicitly defined; isolated time projections are not equal complete-fit elapsed times.
- One-view distillation fails the frozen preservation criterion. On the historically exposed Fashion-MNIST holdout, student error is 9.813%, versus 9.443% plain augmentation and 8.497% the full-compute plain control. A genuinely new 1,500-photo catalog source reverses part of the ordering: student error is 43.511%, versus 47.467% plain, with substantial seed variation and poor absolute accuracy. No external data tune the method. The serving criterion also fails in the transfer mode; one forward is not invariably faster.
- Under the specified first-harmonic input family and posterior-TV assumption, the new transform-specific bounds certify 16/16 model-policy lines at a=.01; the old global-TV bound certifies 0/16. This holds both for known a and an independently estimated a with conservative uncertainty. Larger defects and family misspecification remain explicit failure cases. No natural-image or unrestricted unlabeled guarantee is claimed.

## Artifact map

| Path | Contents |
|---|---|
| `STUDY_SCOPE.md`, `LITERATURE_REVIEW.md` | Prospective scope, evidence boundaries and primary literature |
| `allocation/` | Nine label/work allocations, 16 paired streams, models, all 720 risks, 288 fits, raw predictions, isolated timing, resource ledgers and portable source |
| `distillation/` | Nine scratch CNNs, soft targets, historical split, 1,500 independent external originals, source metadata/license declaration, exact selection manifest, raw predictions, all comparisons and serving measurements |
| `structured/` | Full sharp-bound proof, source-frozen experiments, parameter intervals, all 640 moment/look rows, 1,024 bounds, 3,072 truth-stress rows, negative results and source-family counterexample |
| `root_audit/` | Independent normalization/algebra checks, all allocation/image metric reconstructions, full CNN replay, external eligibility replay, bound/truth reconstruction, timing-cost checks and historical preservation |
| `SOURCE_INTEGRITY.json`, `REPORT_QA.json` | Source/event commitment verification and completed page-by-page report review |
| `archive_manifest.json` | Relative SHA-256 and byte counts for every delivered artifact |

The branch READMEs/protocols define all methods and exact reproduction commands. Run new experiments into fresh directories. Do not run output-generating stages against these completed results. Scientific release files document internal separation within existing user authorization; they are not external preregistration or a request for new user permission.

## Data and cost limits

Fashion-MNIST has historical exposure. Its holdout is excluded from this round's fitting, not pristine external data. The independent catalog source covers six semantic subsets and uses all ten logits; it is a shifted-domain stress test, not an IID ten-class replication. Owner API metadata declares MIT licensing; ownership clearance is not independently audited. Exact duplicate checks do not remove all near duplicates or product-family dependence. Image-bootstrap intervals condition on the nine fitted networks.

The compact archive includes every evaluated original JPEG, all processed pixels, the complete source taxonomy, retrieval metadata, ZIP digest and all selection/exclusion records. The original 592,614,936-byte source ZIP, containing tens of thousands of unevaluated photos, is preserved in the working copy but omitted from the compact delivery. Rebuilding the evaluated images needs only the archived JPEGs. Replaying all skipped-candidate eligibility independently needs that versioned ZIP; `root_audit/external_selection.py --zip PATH` documents this distinction.

Network and nominal-arithmetic budgets are defined precisely, with unmatched operations, shared physical work, standalone procedures, fitting, serving, warmup and audit overhead separate. Actual serving benchmarks paused all other research numerical workloads. Hardware variability remains, including unfavorable one-view timings. Neither theoretical lower bounds nor bootstrap intervals establish a universal deployment advantage.

## Integrity and reproducibility

`python verify_sources.py` performs 2,056 source/data hash checks and verifies stage ordering. `python archive_tools.py` verifies the delivered archive against its manifest. `python archive_tools.py --build` is only for a new reviewed release; it must not silently replace the integrity record of this delivered result.

Reusable implementation is also retained in the repository under `experiments/budget_allocation`, `experiments/distillation_external`, `experiments/structured_bounds`, and `experiments/independent_validation_20261121`. Research results and reports stay in this separate archive. The PDF builder uses the bundled document runtime; NumPy and CUDA experiments use the existing workspace environments. Exact dependency versions and provenance are retained in branch records.

The next priorities are a fresh confirmation of the promising allocation region, an independently acquired and better-aligned apparel replication with more training seeds, and justified residual-asymmetry bounds beyond an assumed one-harmonic family. Keep the current strong baselines; do not retune against this study's external outcomes.
