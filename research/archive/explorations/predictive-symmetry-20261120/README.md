# Prospective update predictions and symmetry aware inference

This local research archive contains the completed two-direction extension for 277 final. It retains exploratory cohorts, frozen confirmation cohorts, subsequent independently frozen controls, unsuccessful targets, raw per-seed outcomes, trained states, and an independent audit. Nothing in this archive was uploaded to GitHub.

## Read the results

- [English final manuscript](FINAL_MANUSCRIPT.md) and [formatted PDF](final_insert.pdf).
- [Derivations and scope](THEORY.md).
- [Local forecast and horizon figure](figures/loss_matched_prediction.png), [Fourier and rotation figure](figures/fourier_rotation_results.png), and [calibration curves](figures/calibration_gain_curves.png). Vector PDF versions accompany every PNG.
- [Independent audit](audit/AUDIT.md), with runnable checks in `audit/audit.py` and machine-readable audit outputs in the same directory.
- [Separate-directory reproduction check](audit/REPRODUCTION_CHECK.json): the full main pipelines reproduced nine scientific CSV files byte-for-byte and every saved model array exactly. Timings and timestamps were excluded from byte comparisons.

## Data and frozen boundaries

| Directory | Contents and role |
|---|---|
| `direction1/old` | Six old exploratory seed pairs and the baseline fitting data. |
| `direction1/new` | Sixteen fresh paired seeds; physical prediction commit precedes endpoint scoring. Includes every requested target, cap failure, input array and state. |
| `direction1/contrast_summary.csv` | Every task/state/target/component contrast, prediction error, sign count and paired interval. |
| `direction1/coverage_by_stratum.csv` | Which GD and pullback branches reached each prespecified target. |
| `direction1/second_order_followup` | Separately frozen 16-seed confirmation of a second-order alternative prompted by the main transfer failures. |
| `direction1/balanced_control` | Another 16-seed control with exactly 128/128 training labels in both tasks and correct one-observation marginals. Within-pair dependence is explicit. |
| `iid_results` | Independent 32,768-input/seed check of the prespecified trained-state 0.02-drop total-risk comparison, including unsuccessful cases. |
| `direction2/exploration` | All 48 archived models and all proposed Fourier/Jensen selectors used during exploration. |
| `direction2/fresh_results` | Sixteen new seed pairs, three schedules, raw unlabeled calibration orbits, frozen selectors, population risks, iid labeled risks, calibration and deployment costs. |
| `direction2/bridge_old`, `bridge_new` | Forecasting second-view decisions at PB100. All reachable branches retain the half-turn, including the updated-model comparator. |
| `direction2/late_bridge_source*`, `late_bridge_results*` | Preserved PB500 feasibility attempts and a separately frozen, independent confirmation of the selected training-only feasible drop. |
| `legacy`, `source` | Exact copies of the original ring generator, population integration routine, old model bank and numerical model core used for portable reproduction. |

Direction1 and the main direction2 share the same 16 new original-ring training seeds and PB100 states. They are linked measurements, not independent replications of one another. The synthetic task pairs share X and initialization within each seed. All confidence intervals resample whole seed records; they are pointwise. The population quadrature and iid tests are distinct evidence sources. Fitting-risk derivatives use the known task posterior; the angle selectors do not use labels.

The exact-balance control changes both label-count variation and training-pair dependence. Its outcome cannot identify a causal effect of balancing alone. Two-view selection changes the inference procedure on a fixed network; it does not improve the network's raw single-view predictions.

## Reproduce without overwriting anything

The supplied `reproduce.py` prepares a new directory from frozen source/configuration files and old exploratory inputs, then reruns the confirmations and follow-ups. It refuses an existing destination. It never deletes an archived output or reselects the scientific configuration.

```powershell
& 'C:\Users\liwei\Documents\ChatGPT\277\.venv\Scripts\python.exe' .\reproduce.py --out 'C:\Users\liwei\Documents\ChatGPT\277\tmp\predictive-symmetry-reproduction'
```

Use `--main-only` for the two principal confirmations plus the iid supplement, or `--prepare-only` to inspect the copied source and command list before execution. The wrapper sets `PYTHONPATH` to the archived numerical core, so no Git checkout is needed for the experiments. Scientific dependencies are in `requirements.txt`; `environment.json` records the actual versions used. Reproductions can differ slightly across numerical-library versions and hardware. The independent audits specify numerical tolerances rather than relying only on byte-identical output archives.

For the original exploratory stages or separate individual stages, see each direction's README and protocol. Prediction generation and scoring guard their existing outputs; do not rerun a freeze command in the archived directory. `reproduce.py` reuses the supplied freezes instead of creating new ones. The raw old outcomes remain legitimate inputs to the old-seed baselines and are copied unchanged.

`build_report.py` renders the manuscript with ReportLab and the Windows Calibri/Georgia fonts. ReportLab is only required for PDF rendering, not for experiments. Figures use Matplotlib. The original PDF and rendered visual checks are included. The file manifest records archive-relative paths and SHA-256 checksums of the delivered files; `python verify_archive.py` verifies them without changing the archive.

## Preserved negative results

Long-horizon derivative forecasts lose to the old-seed baseline. Finite-drop transfer has low or zero coverage in some task/state strata. Second-order radial fitting improvement in an iid cohort does not replicate under exact balanced antithetic training, while angular fitting can worsen. PB100 forecasted views do not change any decision. Unweighted Fourier scores and truncated proxies remain in the archive even when weaker. The early entropy numerical failure, failed ray searches, and failed PB500 large-drop pilot are retained separately from confirmatory results.
