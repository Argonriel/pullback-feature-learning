# Independent fresh-model certificate checks

Result: **PASS** on the fixed subset of model seeds 1,8,16 from both cohorts. All inputs, grids, model indices, and tolerances were specified before this investigator read fresh model or bound files. An execution-only wrapper waited for each producer provenance marker; it did not change the frozen audit algorithms or choose models based on outcomes.

| Cohort | Seed | Orbit-moment failures | Jet-value failures | CDF violation | Rank-marginal violation | Centered-variable violation |
|---|---:|---:|---:|---:|---:|---:|
| rings | 1 | 0 | 0 | 0 | 0 | 0 |
| rings | 8 | 0 | 0 | 0 | 0 | 0 |
| rings | 16 | 0 | 0 | 0 | 0 | 0 |
| transfer | 1 | 0 | 0 | 0 | 0 | 0 |
| transfer | 8 | 0 | 0 | 0 | 0 | 0 |
| transfer | 16 | 0 | 0 | 0 | 0 | 0 |

The audit evaluated 222 conditional moments using 16,384 angular points per radius, 4,800 analytic point-jet values inside independently chosen polar boxes, 1,542 known-law CDF probes, and 1,548 input/partner marginal and centered-variable probes. The latter include two explicit outside-support points per model; every tested outside fallback has nonnegative centered gain. The actual half-turn is used for each task, including the waveform action that flips only its first harmonic.

For the dense orbit integrations, the independent 256-node-per-interval population quadrature differs from the split-CDF reference by at most 6.6e-08 on the fixed check grid. The largest 16,384-versus4,096 angular subsample difference is 0.000192. These are numerical sensitivity checks, not rigorous integration-error bounds. The waveform CDF comparison uses its independent analytical polynomial primitive; the rings comparison splits radial intervals at both score thresholds and integrates the Rice density with 32-point Gauss-Legendre rules. The local certificate enclosures themselves come from the separate real-arithmetic derivation, not from interpreting these point probes as an exhaustive proof.

The verification used **3,643,896 ordinary model example forwards** and **1,920 analytic point-jet propagations**, separately counted. There are no additional model evaluations in the coordinate-only CDF calculations. Every row and failure count is preserved. Frozen task, training, calibration, and policy sources remain unchanged.

Files are `independent_checks/summary.csv`, `moment_probes.csv`, `jet_probes.csv`, per-model raw CDF/W arrays and summaries, `SOURCE_COMMIT.json`, plus `INDEPENDENT_CHECK_PROTOCOL.json`, `INDEPENDENT_EXECUTION.json`, and `INDEPENDENT_STAGING_WAITS.json`. `CERTIFICATE_REVIEW.md` records the separate source/theorem review and its precise full-orbit-invariant-posterior scope.

From the study root, `python transfer/audit_report.py` regenerates this summary. `python transfer/audit_wait_runner.py` reruns the fixed checks after existing output and execution-marker files have been preserved in a separate run directory. It waits for the fixed producer commits and refuses to overwrite its outputs. These checks supplement the certificate argument but do not convert padded float64 calculations into a machine-checked directed-rounding proof.
