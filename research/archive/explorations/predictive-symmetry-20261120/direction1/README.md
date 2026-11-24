# Actual-loss-matched predictive study

The frozen design tests whether pre-state population directional derivatives forecast the change in symmetry fitting loss, nuisance loss, and the value of a second true-symmetry view after GD and full pullback achieve the same actual training-loss decrease. The main contrasts compare pullback minus GD from an exactly shared model state.

Six exploratory seed pairs precede sixteen independent new pairs. Original rings have initial and PB100 states. Matched synthetic radial/angular tasks share every input and initialization, have identical posterior-value distributions, class probability 1/2, and Bayes difficulty. Their true symmetry groups differ. Angular averaging uses the valid pi-rotation subgroup {I,Rpi}, preserving its informative cos(2phi) label. Its endpoints are pi-subgroup fit and pi-antisymmetric nuisance; fit can retain reflection nuisance because the full D2 group is not averaged.

## Prospective boundary and evidence

All 16-seed updated states were computed with training data only. `new/predictions.csv`, root records and state bank were physically saved and SHA256 committed before `new/score_start.json`. `freeze.json` binds protocol, code and old outcomes before fresh prediction generation. Population evaluation is deterministic refined quadrature of the known data-generating law, not an independent iid sample. The fitting derivative uses the oracle posterior and is a mechanism diagnostic, not a deployable training-only estimator. The independent iid risk supplement is supplied by the root study.

The drop-clock predictor is fully pre-state: Qprime*delta/(-Lprime). The realized-tangent diagnostic uses the actual training-only root or endpoint displacement, so it is outcome-blind but not fully pre-update. The old pooled competitor is fit only on old outcomes, stratified by task/state/method/experiment/horizon/drop; empty strata use flagged same-task/state/method reachable-slope fallback. Zero predicts no change.

## Optimizer-contrast results

Each row includes only pairs where both roots reached the target. Sign counts compare the sign of the predicted pullback-minus-GD difference to the observed sign. MAE is in nats. Each seed appears once within a row; confidence intervals in the CSV bootstrap seeds, never treating repeated states or endpoints as independent.

|Task|State|Drop|Endpoint|Pairs|Clock sign|Clock MAE|Old-pool MAE|Zero MAE|PB lower outcome|
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
|rings|initial|1e-05|fit|16|16/16|2.34e-09|9.12e-07|3.94e-06|16/16|
|rings|initial|1e-05|nuisance|16|16/16|2.08e-09|2.17e-06|5.79e-06|0/16|
|rings|initial|1e-05|view_value|16|16/16|1.02e-09|1.05e-06|2.83e-06|0/16|
|rings|initial|0.02|fit|6|6/6|0.00179|0.00249|0.00794|6/6|
|rings|initial|0.02|nuisance|6|6/6|0.00187|0.00317|0.00993|0/6|
|rings|initial|0.02|view_value|6|6/6|0.000908|0.00155|0.00488|0/6|
|rings|trained|1e-05|fit|16|16/16|1.72e-10|6.19e-07|2.29e-06|16/16|
|rings|trained|1e-05|nuisance|16|16/16|1.22e-10|5.6e-07|2.6e-06|0/16|
|rings|trained|1e-05|view_value|16|16/16|6.31e-11|2.78e-07|1.31e-06|0/16|
|rings|trained|0.02|fit|12|11/12|0.00089|0.00158|0.00371|11/12|
|rings|trained|0.02|nuisance|12|12/12|0.000652|0.00106|0.00453|0/12|
|rings|trained|0.02|view_value|12|12/12|0.000339|0.000527|0.00227|0/12|
|radial|initial|1e-05|fit|16|16/16|1.57e-10|7.25e-08|7.16e-08|14/16|
|radial|initial|1e-05|nuisance|16|16/16|7.01e-10|1.34e-06|1.77e-06|1/16|
|radial|initial|1e-05|view_value|16|16/16|3.45e-10|6.58e-07|8.73e-07|1/16|
|radial|initial|0.02|fit|13|6/13|0.000877|0.000842|0.000828|5/13|
|radial|initial|0.02|nuisance|13|11/13|0.00163|0.00336|0.00348|2/13|
|radial|initial|0.02|view_value|13|11/13|0.000808|0.00166|0.00172|2/13|
|radial|trained|1e-05|fit|16|16/16|3.99e-09|1.8e-06|3.83e-06|16/16|
|radial|trained|1e-05|nuisance|16|16/16|1.19e-08|6.76e-06|7.24e-06|5/16|
|radial|trained|1e-05|view_value|16|16/16|5.94e-09|3.38e-06|3.61e-06|5/16|
|radial|trained|0.02|fit|0|—|—|—|—|—|
|radial|trained|0.02|nuisance|0|—|—|—|—|—|
|radial|trained|0.02|view_value|0|—|—|—|—|—|
|angular|initial|1e-05|fit|16|16/16|2.44e-09|1.05e-06|2.47e-06|16/16|
|angular|initial|1e-05|nuisance|16|16/16|1.88e-09|1.46e-06|3.21e-06|0/16|
|angular|initial|1e-05|view_value|16|16/16|1.88e-09|1.46e-06|3.21e-06|0/16|
|angular|initial|0.02|fit|5|5/5|0.00343|0.00823|0.00629|5/5|
|angular|initial|0.02|nuisance|5|5/5|0.00216|0.000698|0.00605|0/5|
|angular|initial|0.02|view_value|5|5/5|0.00216|0.000698|0.00605|0/5|
|angular|trained|1e-05|fit|16|16/16|8.05e-10|1.84e-06|3.31e-06|15/16|
|angular|trained|1e-05|nuisance|16|16/16|1.27e-09|1.75e-06|1.83e-06|10/16|
|angular|trained|1e-05|view_value|16|16/16|1.27e-09|1.75e-06|1.83e-06|10/16|
|angular|trained|0.02|fit|9|8/9|0.0023|0.00376|0.00297|7/9|
|angular|trained|0.02|nuisance|9|8/9|0.00305|0.00492|0.00458|7/9|
|angular|trained|0.02|view_value|9|8/9|0.00305|0.00492|0.00458|7/9|

## Scope, failures, and numerical checks

A missing pair means unreachable within the prespecified direction-ray search and displacement norm cap4, not mathematically impossible. All failures remain in roots/predictions/outcomes and coverage tables. Initial rings pullback often cannot achieve the substantial drop under this cap. Synthetic radial GD300 is especially flat: its substantial targets cannot support a general transfer-success claim. The predeclared tiny and .005-step targets remain reported without outcome-driven retuning.

Multi-step horizons1,2,4,8 retain the initial-state prediction while recomputing each optimizer direction after each .005 actual loss drop; coverage and every horizon are in `summary.csv` and `contrast_summary.csv`. The horizon8 prediction concerns .04 cumulative loss reduction. These reports include failures and finite-change deterioration.

Local directional derivatives are independently central-difference checked; root residuals and all quadrature convergence diagnostics are retained. `shared_states_check.json` verifies shared ring models with direction2 exactly, and matched synthetic tasks use identical X and initial theta. Shared directions are not independent replications. The angular two-view value equals its pi nuisance by construction; it is intentionally reported as a redundant identity check.

An early old-cohort attempt produced NaNs from floating-point 0*log(0) in entropy before any scoring; its output remains in `old_attempt01_nonfinite_entropy`. Stable entropy fixed it before the frozen experiment. No failed scientific target was removed.

## Reproduction

Requirements: Python, NumPy, threadpoolctl. The frozen model core is local; the portable prior polar evaluator lives in `../legacy/rings-mechanism-20261120/` with workspace fallback. Use a fresh copied output directory, preserving existing outputs. Run `study.py generate --cohort old`, `study.py score --cohort old`, `study.py freeze`, `study.py generate --cohort new`, `study.py score --cohort new`, then `analyze.py`. Code intentionally rejects existing cohort directories and an already frozen protocol. All arrays, input seeds, endpoint hashes, root scales and raw rows are supplied.
