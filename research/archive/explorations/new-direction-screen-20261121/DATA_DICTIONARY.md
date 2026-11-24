# Data dictionary

All calculations use float64. Results are deterministic algebra checks or integrations in tiny models; none estimates a population-performance effect or a statistical confidence interval. Source hashes recorded in results resolve to a packaged source snapshot or preserved earlier source.

| Artifact | Meaning |
|---|---|
| `SCREEN_REGISTER.md` | Original bounded screen and exclusions, written before the corresponding minimal validations |
| `INVENTORY.md` | Read-only inventory of previous directions and explicit exclusion map |
| `DEPTH_AUDIT.md` | Scalar reductions, original sources, matrix model, endpoint/control interpretation |
| `ENDPOINT_PRIOR_AUDIT.md` | Independent primary-source review of the narrowly surviving endpoint question |
| `ENDPOINT_MATH_REVIEW.md` | Independent response derivation and exact-path coefficient review |
| `depth_checks/depth_results.json` | Depth-2/3/4 parameter and mirror reductions; prefix velocities; non-Hessian defect |
| `depth_endpoint/depth_endpoint_results.json` | Original 16 reduced-product integrations: 4 damping values, 2 weights, 2 integration steps |
| `depth_endpoint_refined/depth_endpoint_refined_results.json` | Same cases with smaller steps; discovery follow-up for numerical precision |
| `depth_commuting/depth_commuting_results.json` | Nontrivial diagonal mirror control with 3 parameters and 2 constraints |
| `endpoint_parameter_validation/endpoint_parameter_results.json` | Initial independent 8-factor-parameter replay, including its larger coarse integration error |
| `endpoint_parameter_validation/source_v1.py` | Exact source of that initial replay |
| `endpoint_parameter_refined/endpoint_parameter_results.json` | Independent replay at steps .0025 and .00125; 3 damping values, 2 weights |
| `endpoint_response_review_results.json` | Independent response coefficient along exact undamped paths; one-form and curl checks |
| `endpoint_response_quadrature_attempts.json` | Replayed 16/32/64/128-node coefficients, including the failed initial precision check and its provenance limit |
| `record_validation.json` | Cross-implementation endpoint comparison and exact poisoning replay, with source provenance |
| `stochastic_pb_results.json` | Exact binomial expectations for all batch sizes 1–1024, mean/variance and first negative mean |
| `stochastic_label_conditioning_results.json` | Exact finite label enumeration conditional on inputs, same mean but changed covariance |
| `topology_results.json` | Static exact one-parameter residual slice, parameter metric and input-fold checks |
| `poisoning_checks.json` | Fixed-gradient rank-one/mixture/target-direction matrix identities |
| `source_snapshot/` | Reusable implementation, core dependency, repository instructions and environment specification |
| `manifest.json` | Per-file byte counts and SHA-256, excluding itself and its verification record |
| `ARCHIVE_CHECK.json` | Manifest verification, result/source correspondence, and prior-archive preservation |

For endpoint records, `unobserved_x12` is the upper off-diagonal entry of the product matrix. `x12_weight9_minus_weight1` compares the two positive weightings at the smallest integration step. `max_constraint_residual` checks the same two unweighted equality constraints regardless of weight. `minimum_path_eigenvalue` or `min_product_eigenvalue` is sampled on accepted steps; it is numerical domain validation, not a global analytic theorem. `max_balance_error` measures W1 W1^T−W2^T W2 in the original factor replay. The zero-damping parameter replay uses the minimum-norm pullback velocity because the 8-by-8 parameter Gram matrix is singular.

For the stochastic lane, the metric is a batch average and damping is held fixed across batch sizes. Independent sampling consumes a separate metric batch; no same-cost superiority is claimed. For topology, only one allowed parameter varies and the other 79 original parameters are frozen. For poisoning, the objective gradient is held fixed and all stated contaminations are precisely specified in the audit; large leverage is not presented as a subtle or realistic attack.

The coefficient files are described in `ENDPOINT_MATH_REVIEW.md`; their finite-time and quadrature precision should not be confused with the endpoint integrator precision. The original full-file hash for the failed quadrature assertion was not captured; its values were replayed with the unchanged coefficient function and this limit is explicitly recorded. All discovered/refined cases are retained, and initial captured source/data records have not been overwritten.
