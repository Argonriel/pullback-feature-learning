# Symmetric allocation frontier and two-invariant transfer

Completed 21 November 2026. Start with `research_report.pdf` or `REPORT.md`.
This new archive preserves previous studies without modifying them.

## Findings

- All three primary model-paired bootstrap comparisons favor the interpolated grid over the development-selected strong symmetric histogram. Rings has a tiny effect and confidence selection still wins on its original posterior.
- The new harmonic support has two independent invariant amplitudes. Transfer improves loss and error, but the underlying model remains overconfident and far from Bayes performance.
- All 96 absolute policy/model gain bounds are positive at 128 batches. Use the final 864-event confidence family. No signed comparison to the strong histogram is certified positive, even at 2048 batches.

## Files

`STUDY_SCOPE.md` explains the fixed study and pre-fresh seed correction.
`literature/` contains the primary-source review. `method/` contains the
frozen surrogate, strong-control selection, proof and all development results.
`task2d/` contains the frozen law, all pilot recipes and fresh diagnostic note.
`validation/` contains 48 trained states, 144 policy tables, 48 prediction sets,
all 384 metric rows, primary contrasts and actual deployment ledgers.
`certification/` contains frozen proofs, protocols, raw independent-batch data,
actual costs and final confidence bounds. Its `final_*` CSV files and
`FINAL_CERTIFICATE_COMMIT.json` supersede the historical 432-event CSV results.
`root_audit/` contains independent geometry, sensitivity, performance, training,
range, tail and confidence-bound reconstructions. Diagnostic plots are provided
as PNG/SVG or PDF; these diagnostics did not tune the frozen methods.

## Reproduction and preservation

The experiments require Python 3.12, NumPy and threadpoolctl. Figures also use
Matplotlib; the PDF uses ReportLab and Windows Calibri/Georgia fonts. Current
fresh experiments are portable with the included `source/`, `legacy/`,
`reference/`, task, method, choice and protocol dependencies. Historical
development scripts additionally reference the previous study specified in them.
Do not rerun writers into this completed archive: many intentionally refuse
to overwrite existing results. Use a separate working copy for reproduction.

The recorded stage order is task pilot/freeze, method development/freeze,
certificate freeze/addendum, validation smoke/freeze, internal scientific
release, train, calibrate, score, analyze; independent certificate sampling
starts after policy commit. All 48 training runs completed without failure.
Training/calibration/performance and certificate streams are separate. The
source freezes and root audit records document provenance and chronology.

Read-only integrity verification: `python verify_archive.py`. The SHA-256
manifest excludes caches and PDF preview renders, but includes the final PDF,
all study data, source code, freezes, protocols and audit results.

The study uses CPU computation. Forward-example accounting is not a claim
about wall-clock acceleration. The proof assumes an invariant input law and
half-turn-invariant posterior; it does not certify arbitrary real-world shifts.
See the report's limitations, including the extreme-radius rings posterior
helper limitation and the distinction between empirical and certified relative
improvement. The scientific computations use float64, not formal interval proofs.
