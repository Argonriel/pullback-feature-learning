# Reproducibility and evidence map

All calculations are deterministic finite-dimensional checks. There is no statistical inference, training benchmark, external dataset, or reserved cohort.

| Path | Evidence | Main limitation |
|---|---|---|
| `THEOREM.md` | Self-contained finite-blend and autonomous endpoint theorem, exact area rank, initialization corollary | Generic initialization, small targets, finite positive damping |
| `static_loss/THEOREM.md` | Detailed independently derived area/endpoint proofs and limit analysis | Not a global finite-target reachable-set theorem |
| `exceptional/STATIC_THEOREM_REVIEW.md` | Independent adversarial mathematical and numerical review, reviewed hashes | Not a formal proof certificate |
| `exceptional/REPORT.md` | Exact n=2 classification, block invariance, n=3 counterexamples | All connected states in n>=3 remain unclassified |
| `validation_v1/results.json` | One 3x3 actual endpoint derivative; three amplitudes, normalization and curvature checks, step refinement | Initial tiny-increment FD decay flag is false and preserved |
| `endpoint_difference_refinement/results.json` | Larger-increment endpoint differences; rank-vs-step diagnostic margin | Step discrepancy is not a certified ODE error bound |
| `static_loss/validation_v1/results.json` | One 4x4 area-family calculation with k=0 through 4 and scalar-weight control | Tests algebraic ranks, not nonlinear endpoints |
| `exceptional/validation_v1/results.json` | Six exact rational rank cases | Examples do not prove all-state classification |
| `exceptional/derivative_validation_v1/results.json` | Centered difference of nested brackets | Numerical corroboration of the exact derivative implementation |
| `literature/AUDIT.md` | Closest primary theorem comparisons and successful classical reduction | Bounded search, not priority certification |
| `literature/SEARCH_LEDGER.md` | Forty literal queries, failed reductions/retrievals, version correction | Unsuccessful search is not evidence of originality |
| `literature/SOURCES.json` | Eight retrieved primary byte hashes and URLs | HTML is a retrieval representation, not canonical paper bytes |
| `FAILED_ATTEMPTS.md` | Rejected overclaims, initial false check, focused correction | Nothing is silently discarded |

## Coordinates and weights

Symmetric matrices use the Frobenius pairing. Hidden coordinates are raw strict-upper entries, ordered lexicographically (12,13,23 for n=3). The area matrix is skew and uses the same pair order. The normalized fields have diagonal e_i. Positive quadratic weights are full SPD matrices, not just scalar rates or diagonal reweighting.

In the root validation, `C` is the transformed weight, `P=A0^(1/2)`, `weight=P^-1 C P^-1`, and `target_direction=P d`. The three `area_directions` are symmetric right singular-vector directions of the exact area differential in an orthonormal symmetric basis. `weight_directions` transform them to actual B perturbations. `endpoint_jacobian` differentiates strict-upper endpoint entries in those three directions at a fixed target. `scaled_jacobian` divides by epsilon^2. `predicted_scaled_endpoint_jacobian` is the theorem's curvature-area coefficient.

`minimum_sampled_eigenvalue` samples integration states; positivity for the theorem comes from its compact-neighborhood argument. `refinement_error` is a step discrepancy, not a proved absolute error. Endpoint and sensitivity residuals are the norms of the final diagonal residual and its weight derivatives. The finite horizon is 42 with RK4 steps .02 and .01.

## Source and environment

All reusable source is new and lives under `experiments/astra6_blended_geometry_deepening_20261121/`. The archive contains the same files under `source_snapshot/`, excluding bytecode. Each calculation records source SHA-256; checks importing another new source record its dependency hash. No prior code was imported or modified. Python 3.12.14 and NumPy 2.3.5 were already available; exact Fraction arithmetic uses the standard library.

The source README supplies reproduction commands with fresh output directories. `provenance.json` maps source hashes and the old input files. `manifest.json` hashes content files. `ARCHIVE_CHECK.json` verifies local links, source-hash resolution, primary-source snapshots, numerical sanity, and all manifest-listed files of the predecessor archive. The two bookkeeping files do not hash themselves.

The predecessor is `three-candidate-exploration-20261121`. Its manifest was first read with SHA-256 `65482ffc1bc0e590d481e8a43b70d2fe8d9dc9374e87c3b49ee60d2c4ba99214`; the final check verifies it and its listed content remain unchanged. The new report does not overwrite that report, its frozen configurations, or earlier outputs.
