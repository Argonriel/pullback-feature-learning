# Data dictionary and provenance

All numerical calculations are deterministic float64 checks in small matrix spaces. They establish numerical consistency, not statistical predictive performance or publication originality.

| File | Meaning |
|---|---|
| `SCREEN_REGISTER.md` | Questions and exclusion boundaries recorded before this round's new numerical tests |
| `GENERAL_GEOMETRY.md` | Exact projected-form criterion, finite algebraic classification, and counterexamples |
| `STATIC_WEIGHT_THEORY.md` | Static-weight attainability, limit-order qualifications, cubic/quartic coefficients |
| `general_geometry_checks/general_geometry_results.json` | Initial direct geometry identities and counterexamples |
| `general_geometry_checks/source_v1.py` | Exact source of the initial geometry checks |
| `general_geometry_certificates/general_geometry_results.json` | Fresh follow-up with complete finite polarization probes |
| `LITERATURE_AUDIT.md` | Primary-source overlap audit, including the closest 2026 SLD integrability theorem |
| `METRIC_CONTROL.md` | Known entropy-Hessian comparison and original-parameter replay |
| `static_weight_response_results.json` | Exact-path response grid, fixed-target diagonal control, cubic and quartic asymptotics |
| `static_weight_quartic_mixed_results.json` | Separate mixed quartic coefficient check, preserving the first script and results |
| `metric_control_results.json` | Eight finite-flow runs: 2 metrics, 2 positive weights, 2 integration steps; spectral series and log-derivative checks |
| `factor_replay_results.json` | Four original 18-parameter integrations and comparison with the spectral product solver |
| `source_snapshot/` | Exact reusable source, repository instructions, and environment specification |
| `manifest.json` | Hash/byte manifest excluding itself and its verification record |
| `ARCHIVE_CHECK.json` | Source/dependency/data-hash resolution and previous-archive preservation |

The geometry report describes its polynomial probes. Those probes are finite polarization checks for polynomial closure conditions, not random evidence for a general mathematical claim. The initial output was preserved when finite polarization was added to the reusable script.

The static-response grid uses raw target amplitude epsilon in U=E12+E21 and V=E13+E31. Its orthonormal measured coordinates are sqrt(2) epsilon. This normalization matters: choosing epsilon as each orthonormal coordinate instead divides the displayed quartic coefficient by four.

`C_B` and the static-response arrays concern the derivative of the endpoint in damping at lambda=0, followed by a small-target expansion. Finite-damping endpoints in `metric_control_results.json` and `factor_replay_results.json` are different quantities. No finite-lambda uniformity follows simply by comparing these numbers.

The finite-flow control fixes raw amplitude .25, s=1, lambda=.2, and loss weights (1,1)/(1,2). The metric uses every entry of the product matrix; only the objective weights change. `null_contrast_norm` measures the unobserved part of the weight contrast. `max_endpoint_refinement_norm` compares RK4 step .02 with .01 in Frobenius norm. The independent factor replay uses steps .01 and .005 and records factor-balance, symmetry, interpolation, and SPD checks.

Source hashes are stored in results. Imported-source and control-data hashes are separately resolved by packaging. All configured cases are retained. The archive's completed manifest is immutable; any later follow-up must use a fresh directory.
