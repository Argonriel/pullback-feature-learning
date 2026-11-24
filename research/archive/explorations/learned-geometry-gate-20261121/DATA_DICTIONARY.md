# Data and reproduction guide

All research text and machine-readable keys are English. The study date is November 21, 2026 in America/Los_Angeles; machine timestamps are UTC, often November 22. All arrays use float64.

## Study records

- `PROTOCOL.md` and `discovery_config.json`: original frozen design, saved before training.
- `COVERAGE_AMENDMENT.md`: pre-evaluation extension of censored rings tangent controls, preserving the original result.
- `REFINEMENT_AMENDMENT.md`: pre-evaluation repair that halves both scalar step constraints.
- `discovery/`: 64 original physical training runs; 12 censored rings tangent trajectories retained.
- `coverage_extension/`: 16 rings tangent reruns with unchanged controller and 60000-update ceiling.
- `refinement/`: 32 fine-resolution reruns with target 0.005, parameter cap 0.05, and 120000-update ceiling.
- `discovery_supplemental/`: coverage-only composite, preserved for provenance; not the final numerical refinement.
- `discovery_refined/`: final supplemental composite, replacing each component trajectory explicitly without changing the source archives. The selected per-row `param_step_cap` and `component_max_steps` identify the controller.
- `discovery_scores/`: separate evaluation of the original frozen design.
- `discovery_refined_scores/`: separate evaluation and decisions for the final supplemental comparison.
- `source_snapshot/`: runnable source tree and configurations, including original core functions and recorded dependencies. `historical_source/` preserves the first assembler version referenced by the coverage-only composite.
- `figures/`: static publication-style summaries; source tables remain authoritative.

The main report's primary numbers come only from `discovery_refined_scores/` at fraction 0.25, lambda 0.1, step size 0.005; the coarse 0.01 result supplies the required sensitivity comparison. This is explicitly supplemental, not a relabeling of the original frozen result.

## Training file schemas

`data.npz` contains keys `task__pair__x`, `task__pair__y`, and `task__pair__theta0`. Inputs have shape (256,2), labels (256,) with values -1/+1, and parameters (80,). `actual_seeds.json` records effective task seeds; rings adds 10000 to the declared base seeds. Data and initialization can be regenerated exactly.

`states.npz` maps state keys such as `s000000` to 80-dimensional parameter vectors. Use `snapshots.json` or `snapshots.csv` to interpret them. For tangent arms these vectors encode theta0+delta; predictions must use z0+J0*delta, never a nonlinear forward pass at the stored vector.

Snapshot fields include task, pair ID, cohort, arm, damping (null for GD), target step size, fraction of initial loss, step count, measured loss, target error, status, state key, elapsed diagnostic training time, cumulative logit/representation/second-coordinate path, backtracks, target searches, maximum Taylor ratio, maximum parameter step, and accumulated scalar clock. Fraction 1 means initialization, 0.75/0.5/0.25 are actual target states, and -1 marks a retained missing-target terminal state. All primary endpoints in the final composite were attained.

`trajectory.csv` contains sparse training-only records and target records. `summaries.csv` contains each physical or selected trajectory's termination status. `provenance.json` contains timestamps, configuration/source/output hashes, numerical environment, and component references. Assembled records also contain explicit overlap checks. Timing includes experimental controller work and must not be used as a fair optimizer benchmark.

## Evaluation schemas

`evaluation_data.npz` saves exact 20000-example moons evaluation draws. Rings evaluation is deterministic quadrature under the known population law; its definition is in `score.py` and does not require sampled evaluation labels.

`metrics.json/csv` extends each snapshot with risk, descriptive accuracy, evaluation status and role. Rings rows also contain Bayes risk, radial/angular excess, quadrature node count, grid-change estimate, decomposition residual and omitted-tail risk bound. The radial/angular components are inherited diagnostics, not new signed eigenmode contributions.

`contrasts.json/csv` contains one pair/step/target/damping record, all available arm risks, an explicit four-arm coverage flag, B (tangent PB minus nonlinear PB), D (nonlinear GD-minus-PB contrast minus tangent GD-minus-PB contrast), F when fixed-metric arms exist, and nonlinear-versus-tangent training-logit RMS difference. F is absent/null in this study because the conditional intervention was not triggered.

`group_means.json/csv` contains paired means, sample SDs and contributing counts. Missing endpoints are never imputed. `gate_decisions.json` evaluates the declared primary screen and distinguishes insufficient coverage from effect/sign/refinement failures. Partial means in the original censored rings records must not be read as four-pair estimates. `audit.json` records exact data/loss checks, independent quadrature comparisons, failures and file hashes.

## Reproduction

Use the recorded Python/NumPy environment and one BLAS thread. The repository experiment README in `source_snapshot/experiments/learned_geometry_gate_20261121/README.md` gives the commands for training each component, assembling explicit components, scoring, checking and plotting. Every command requires a fresh output directory. Production scoring imports no held-out information into training. Reproduction may change diagnostic runtime but should preserve deterministic parameter states and predictive metrics within the recorded numerical tolerances.

The SHA-256 manifest covers archive files excluding itself. It is an integrity check, not a digital signature or proof of external timestamping. Pre-evaluation ordering is evidenced by the archived protocol/amendments, component provenance timestamps and scoring audit timestamps; the study is not externally preregistered.
