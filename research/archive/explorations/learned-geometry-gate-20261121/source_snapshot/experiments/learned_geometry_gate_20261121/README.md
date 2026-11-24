# Full initial-tangent geometry gate

`runner.py` trains the complete initial representation linearization and the
original nonlinear model with GD or the full damped representation pullback.
The optional `nonlinear_fixed_pb` arm combines the nonlinear predictive map and
gradient with the initial full representation metric. It is an intervention,
not an attribution supplied by the four primary arms.

Run `python experiments/learned_geometry_gate_20261121/verify.py` before a study.
Run `python experiments/learned_geometry_gate_20261121/runner.py --config PATH
--output FRESH_DIRECTORY` for a frozen configuration. The output directory must
not already exist. Research protocols, run outputs, and analysis belong in the
separate research archive specified by the repository guidelines.

Required configuration fields are `tasks`, `seeds`, `arms`, `dampings`,
`step_sizes`, and `loss_fractions`. Each seed has `id`, `train_seed`, `init_seed`,
and `test_seed`; `cohort` is optional. The rings task adds 10000 to all three
seeds, and `actual_seeds.json` records those effective seeds. The runner never
generates test data or uses held-out scores. Optional controls are `max_steps` (6000),
`n_train` (256), `param_step_cap` (0.1), `remainder_rtol` (0.05), `armijo` (0.0001),
`target_atol` (1e-10), and `log_every` (100). GD runs only once per step size,
without repeating the same trajectory at each damping.

The step controller normalizes predicted RMS training-logit displacement,
limits parameter displacement, and backtracks for descent and local linearity.
It shortens a target-crossing step by bisection to the declared training loss,
then resumes from the actual resulting state. Timings include these diagnostic
controls and explicit Jacobians for GD and cannot support optimizer cost claims.

Outputs are `config.json`, `provenance.json`, `data.npz`, `states.npz`,
`snapshots.csv`, `snapshots.json`, `actual_seeds.json`, `trajectory.csv`, and
`summaries.csv`. Data keys are
`TASK__SEED_ID__x`, `TASK__SEED_ID__y`, and `TASK__SEED_ID__theta0`. Each snapshot
row identifies its parameter vector via `state_key`. Fractions 1 are initial
states; target rows use the configured fraction; -1 denotes a terminal failure
or sparse trajectory point. Incomplete targets remain missing and must not be
silently treated as matched-progress evidence. All completed prefixes are saved.

For scoring, `predict(theta0, theta, x, linear=True)` evaluates the fixed full
initial-tangent map on any inputs. It does not recompute the predictive Jacobian
at the fitted parameters. The data generator, core source, configuration, and
runner hashes are recorded for reproducibility.

After a finished training pass, run `python
experiments/learned_geometry_gate_20261121/score.py --run RUN_DIRECTORY --output
FRESH_SCORE_DIRECTORY`. Scoring regenerates and audits training inputs, checks
saved training losses and source/artifact hashes, then evaluates every saved
state without selecting parameters. Moons uses 20,000 fresh balanced examples;
rings uses the known polar law with refinement and a separate representative
quadrature check. Both nonlinear and tangent risk include an analytic bound on
the omitted radial tail. Accuracy is descriptive and is not a quadrature
refinement target.

`metrics`, `contrasts`, and `group_means` are saved in JSON and CSV forms;
`gate_decisions.json` applies the frozen primary effect, sign, coverage, and
step-refinement thresholds. `audit.json` records failures and verification.
Missing primary targets are explicitly classified as insufficient coverage.
The scorer requires all compared arms in one run bank; any later supplement
must be merged transparently with its source provenance before scoring it as a
separate amended run. Keep the original run and its evaluation intact.

## Reproduce the November 21 screening sequence

Use a fresh archive root for all commands below. Replace `ARCHIVE` with that
absolute path; never reuse the preserved original output directories. Execute
commands from the repository root. The separately archived protocol and both
amendments explain the chronology and interpretation.

```text
python experiments/learned_geometry_gate_20261121/verify.py
python experiments/learned_geometry_gate_20261121/runner.py --config configs/learned_geometry_gate_20261121.json --output ARCHIVE/discovery
python experiments/learned_geometry_gate_20261121/runner.py --config configs/learned_geometry_gate_coverage_20261121.json --output ARCHIVE/coverage_extension
python experiments/learned_geometry_gate_20261121/runner.py --config configs/learned_geometry_gate_refinement_20261121.json --output ARCHIVE/refinement
python experiments/learned_geometry_gate_20261121/assemble.py --runs ARCHIVE/discovery ARCHIVE/coverage_extension --output ARCHIVE/discovery_supplemental
python experiments/learned_geometry_gate_20261121/assemble.py --runs ARCHIVE/discovery ARCHIVE/coverage_extension ARCHIVE/refinement --output ARCHIVE/discovery_refined
python experiments/learned_geometry_gate_20261121/score.py --run ARCHIVE/discovery --output ARCHIVE/discovery_scores
python experiments/learned_geometry_gate_20261121/score.py --run ARCHIVE/discovery_refined --output ARCHIVE/discovery_refined_scores
python experiments/learned_geometry_gate_20261121/make_figures.py --scores ARCHIVE/discovery_refined_scores --output ARCHIVE/figures
```

The original 6000-update run remains the frozen-protocol result. The coverage
extension reruns all rings tangent trajectories with 60000 updates; it does not
change their directions or controllers. The refinement runs all arms and both
tasks with both the logit-step target and the parameter-step cap halved, using
120000 updates. Both supplements were training-only repairs declared before
scoring, with all histories preserved. They are not retroactive replacements
for the original gate.

Assembly uses whole trajectories from later components for matching run keys,
preserves the component provenance, checks paired training data exactly, and
checks shared states exactly when the controller is unchanged. Refined states
are expected to change because the parameter cap changed. The original and
amended results must be labeled separately. The primary gate did not retain
either task in this discovery study, so no conditional confirmation, fixed-metric
intervention, damping sweep, or signed-spectrum expansion follows these commands.

The figure script exports a PNG, editable SVG, plotted numerical values, and
source hashes. Points show individual paired outcomes and diamonds show means;
the display contains no confidence intervals or significance claims.
