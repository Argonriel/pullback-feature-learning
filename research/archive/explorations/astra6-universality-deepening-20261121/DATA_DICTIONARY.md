# Evidence and reproducibility map

All research prose is English. User-facing dates use November 21, 2026 in
America/Los_Angeles; UTC retrieval timestamps can fall on November 22.

| Path | Purpose |
|---|---|
| `REPORT.md` | Completed outcome, exact assumptions, main theorem and final retain/narrow judgment |
| `independent-regularity/REGULARITY_THEOREM.md` | Full nonanalytic universality proof, sharp exception, dynamics quantifiers and local convergence justification |
| `independent-regularity/OLD_ALL_K_AUDIT.md` | Independent verification of every analytic order, static SPD weight realization and first-detection quantifiers |
| `independent-regularity/exact_plateau_results.json` | Exact rational guarded counterexample check, strict bounds, source and theorem hashes |
| `independent-transform/TRANSFORM_AUDIT.md` | Independent exact response, continuous injectivity, fixed two-by-two identification, conditioning and proof review |
| `independent-transform/VALIDATION_REVIEW.md` | Separate implementation, error, controls and counterexample review |
| `independent-literature/PRIORITY_AUDIT.md` | Nine primary theorem-level comparisons and novelty boundary |
| `independent-literature/SEARCH_LEDGER.md` | Bounded query record, reductions tested and limitations |
| `independent-literature/MOMENT_CEILING.md` | Classical support bound with exact leading error and its limited endpoint consequence |
| `independent-literature/moment_constant_check.json` | Exact rational orthogonality and leading-error checks for one through four atoms |
| `independent-literature/SOURCE_HASHES.json`, `EXTRA_SOURCE_HASHES.json` | Retrieved original bytes, URLs, timestamps and hashes |
| `validation-v1/results.json` | Seven new response checks, two quadrature settings, two weight steps, explicit scalar tail bounds and Gaussian transcription checks |
| `source_snapshot/astra6_universality_deepening_20261121/` | Reusable code snapshot, README and preservation/packaging utility |
| `reference_inputs/` | Unchanged copies of prior theorem, algebra, review and dynamics inputs needed to evaluate the new proof |
| `provenance.json` | Source-hash resolution, input hashes and old-archive manifest preservation |
| `manifest.json` | SHA256 and size of every non-cache package content file |
| `ARCHIVE_CHECK.json` | Final archive equality and package/link verification; bookkeeping file excluded from its own manifest |

`validate_nonanalytic.py` uses Python 3.12.14 and NumPy 2.3.5, with no SciPy or
mpmath dependency. All matrix calculations are deterministic. The saved source
SHA256 is resolved to both the current new-code file and its snapshot. The first
and only actual response-validation run is retained as `validation-v1`; earlier
environment failures are listed separately rather than masquerading as runs.

The first damping derivative is evaluated along the exact zero-damping path.
Finite differences perturb one fixed SPD weight for each path. They are not
finite-damping simulations. The scalar integral and matrix path are independently
assembled, although they use the same declared discrepancy kernel. This checks
normalization, signs and weight realizability; proof of injectivity and universal
classification is analytic, not statistical.

The numerical output's moment-error lower bound uses the conservative denominator
`4^m`; the independent exact note improves it to `4^m-1`. Both inequalities are
valid, and neither is represented as sharp. The aggregate pass flag is limited
to its explicit gates, as detailed by the independent review.

Imported prior documents preserve their original text and relative references;
they are not presented as newly authored reports. Their original packages and
hashes remain recorded. Final link checks apply to the newly authored documents
and explicitly exclude imported historical prose.
