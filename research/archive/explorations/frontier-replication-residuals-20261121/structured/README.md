# Residual-envelope certificate study

Read REPORT.md for results, THEORY.md for the full proof and explicit limitations, and PROTOCOL.md for the prospective experiment. The old first-harmonic study and all earlier models/results are untouched. The new independent evidence and verification use 243-family seeds.

The density model retains cos(phi) and cos(3phi), with an odd orthogonal residual bounded by a supplied rho. Independent coefficient intervals induce an odd pointwise interval for r. The exact joint worst-case posterior/input penalty occurs at an interval endpoint. Independent finite-cell label audits optionally supply a posterior-TV upper bound, conditional on within-cell posterior constancy. Neither a density residual envelope nor pointwise posterior regularity can be learned without restrictions from these moment/cell averages. Outside-envelope sensitivity outputs are explicitly invalid and are excluded from certificate success counts.

## Standalone replay

With Python, NumPy, Matplotlib and threadpoolctl installed, run from this archived branch:

    python reproduce.py --output ../fresh_residual_replay

The destination must not exist. The script copies every frozen input/source and the scientific approval manifest, then runs prepare, checks, evidence, certificate, stress, analysis, report and completion validation. It does not read the original historical study. `assets.py` documents the one-time vendor step and is not needed for replay. ENVIRONMENT.json records the exact original versions. The original executable was the workspace `.venv/Scripts/python.exe`.

Do not rerun stages into the completed original directory: output creation is exclusive. Runtime source/assumption edits invalidate FREEZE.json. Replays produce identical deterministic scientific values under the same numerical environment, while timestamps, elapsed times and ZIP container hashes can differ. Raw array values and stream hashes are the reproducibility target.

## Evidence and raw verification

- FREEZE.json binds runtime Python, THEORY.md, PROTOCOL.md and all copied models/score tables. ROOT_APPROVAL.json binds that freeze. CONSTANTS.json commits parameter-derived D and historical clipping c before verification.
- PROVENANCE.json and HISTORICAL_INPUTS.json preserve upstream hashes; reference/ and historical/ make the branch independent of earlier paths. Historical model training used labels. This round performs no fitting.
- evidence/parameter_j.npz stores the three harmonic feature values for the independent parameter sample. evidence/labels_j_k.npz stores the eight-cell IDs, actual sampled Bernoulli labels and the separate generator probabilities. The latter are not treated as observed label evidence. Parameter and cell interval tables retain every count, success and endpoint. EVIDENCE_COMMIT.json binds the raw files, coefficients and independently estimated epsilon values.
- certificate/i_j.npz stores both logits and all harmonic features on the selected union, the union indices, full cell IDs, packed masks, per-batch A/B and each known/estimated endpoint-penalty moment. Grid keys e0..e7 follow the frozen epsilon order; label0/label1 follow true cell defects 0/.1. Input/tie seeds and hashes are in certificate_costs.csv; original coordinates regenerate from the frozen sampler.
- moment_bounds.csv stores each mean, sample variance, deterministic item/batch range, both concentration radii and final one-sided bound. certificate_curves.csv stores every grid and labeled risk value, known/estimated ranges and the explicit coverage flag. certificate_ledger.csv charges two forwards per union record in every batch.
- stress_recovery/i_j.npz stores full logits, p0, harmonic features, true imbalance, cell IDs and masks. These reconstruct all fixed posterior worlds and every result row, with probability endpoints clipped as documented in STRESS_RECOVERY.md. stress_results.csv keeps direct IID and same-law orbit-corrected gains with batch standard errors; those comparisons are descriptive, not new certificates. Deployment replays have their own cost ledger.
- algebra_checks.csv and COUNTEREXAMPLES.json retain numerical checks and the analytic density/label limitations. No model is evaluated in these diagnostic cases. STATISTICS_LEDGER.json counts confidence-statistic work separately from model forwards and labels.
- RESULTS.json, certificate_summary.csv, stress_summary.csv and certificate_vs_truth.csv are output-only summaries. residual_report.png is the print-sized primary plot; residual_stress.png displays both covered and explicitly invalid sensitivity outcomes. The SVG counterparts are also retained.
- COMPLETE.json verifies all frozen sources, raw-file commits and expected row counts and records the final branch manifest. Independent root audits are separate in the sibling root_audit directory and have their own additional-call accounting.

The 95% guarantee is simultaneous over the prescribed grid/label points, looks, models, policies and covered laws, conditional on the stated assumptions and fixed historical fitting artifacts. It does not cover a new population of trained models or natural-image transformations. Selected records are dependent within batches; whole-batch concentration and the proved top-quota replacement sensitivity handle that dependence explicitly. Concurrent wall time is not a hardware-equivalence benchmark.

The first truth attempt encountered theoretical-zero probability roundoff. Its failure record and original frozen source are preserved. STRESS_RECOVERY.md explains the separately frozen recovery; stress_recovery/ and STRESS_RECOVERY_COMMIT.json hold its successful raw stage. The replay driver uses this reviewed correction directly. Original failed-attempt overhead is reported separately.
