# Structured transformation bounds

Standalone inputs and frozen sources for the third branch of the 21 November 2026 follow-up. Start with REPORT.md for results, THEORY.md for the complete derivation and conditional limitations, and PROTOCOL.md for the prospective design. All old completed study artifacts remain unchanged.

The central bound is

    G_P >= L_A - a U_W - epsilon U_Ra,

where W and R_a are nonnegative, transform-specific selected moments measured on the same law P as deployment. If a is estimated independently within the specified density family, use aU and R_aL. The theorem still assumes a valid posterior-TV bound and the specified input-law family. It does not infer either assumption from unlabeled success, and does not replace support-wide ranges with observed maxima.

## Reproduce without overwriting

Use Python with numpy, matplotlib and threadpoolctl. The original environment used the workspace `.venv/Scripts/python.exe`. From this directory, choose a destination that does not yet exist:

    python reproduce.py --output ../structured_replay

This vendors only the frozen source/input files and existing approval manifest into the fresh destination, verifies their hashes, then runs prepare, algebra checks, parameter estimation, same-law certificate, independent stress and descriptive analysis. It does not access any preceding study's paths. `assets.py` is the original historical-copy provenance script and is not needed during replay. Runtime source and model/table hashes must match FREEZE.json; all output writers use exclusive creation. The original ROOT_APPROVAL.json records scientific review of that frozen source, not a claim of user permission for unrelated actions.

Full original run order was assets.py, run.py freeze, internal source review, run.py prepare, run.py checks, run.py parameter, run.py certificate, run.py stress, analyze.py. The original branch's outputs should never be overwritten by repeating those commands in place.

## Files and accounting

- FREEZE.json and ROOT_APPROVAL.json bind runtime sources, protocol, theorem and historical assets before any fresh draw. CONSTANTS.json commits deterministic D and historical c. PROVENANCE.json and HISTORICAL_INPUTS.json preserve upstream hashes and original paths.
- parameter_raw.npz and parameter_intervals.csv retain the independent, model-free parameter samples and their simultaneous intervals. PARAMETER_COMMIT.json binds the sample and interval table.
- certificate/*.npz retain original/partner logits for the selected union, union record indices, full cosine coordinates, packed two-policy masks, raw per-batch A/W/B/R_known/R_unknown, and exact union-forward counts. They are tied to actual P_a by sampler seeds and hashes in certificate_costs.csv. Model indices 1..8 and law indices 0..3 determine filenames.
- moment_bounds.csv includes mean, variance, both radii, deterministic item/batch ranges and final one-sided bound. certificate_curves.csv includes every known/estimated/old bound, propagation endpoint, constant and shift penalty. certificate_ledger.csv is one row per batch. The aggregate costs charge coordinate inputs separately from neural evaluations.
- stress/*.npz retain full original/partner logits, original invariant posterior p0, cosine and packed masks, allowing deterministic reconstruction of every fixed posterior world and outcome row. stress_results.csv and stress_costs.csv retain precision and expense. Deployment replays are in deployment_audits.csv.
- algebra_checks.csv and MISSPECIFICATION.json distinguish numerical algebra audits from the analytic third-harmonic counterexample. CHECKS.json accounts for this diagnostic work. There are no model forwards in these checks.
- CERTIFICATE_COMMIT.json and STRESS_COMMIT.json bind all raw files. RESULTS.json, certificate_summary.csv, stress_summary.csv and certificate_vs_truth.csv are output-only summaries; the latter comparison is a Monte Carlo consistency diagnostic, not proof of confidence coverage.
- structured_frontier.png and its SVG version visualize primary bounds, posterior sensitivity and negative stress outcomes. Ranges across fixed historical models are descriptive and are not new-model confidence intervals.

Selection costs 320 neural example evaluations per 256 deployment inputs. Verification costs two evaluations for every record in the union of both masks. Both policies have quota 64, but dependence within a batch is handled through batch concentration and proved top-quota replacement sensitivity, never an independent-selected-record assumption. The estimated-parameter extension charges additional coordinate-only data. Concurrent wall times are descriptive and are not a fair hardware benchmark.

Independent root audit scripts/results, if attached, live in the sibling root_audit directory and charge additional calls separately. This branch never silently incorporates those audit calls into a claimed deployment budget.
