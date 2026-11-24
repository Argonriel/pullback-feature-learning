# Classified damped flow: boundary and spectral checks

This directory contains new, deterministic, small-matrix validation code for the independent November 21, 2026 global-convergence deepening. It does not modify historical experiments or frozen configurations.

`verify_boundary.py` requires Python 3 and NumPy. Run it with an output directory that does not yet exist:

```powershell
& '<python executable>' '.\verify_boundary.py' --output '<fresh research output directory>'
```

The default run uses one 5-by-5 coupled-center PSD-only target, two fixed SPD weights, RK4 steps 0.05 and 0.025, and horizon 1600. It writes all sampled residuals, invariant errors, finite-time endpoint errors, refinement errors, exact inconsistent-data KKT controls, input matrices, runtime versions, and its source SHA-256. Numerical checks are consistency evidence; the theorem is proved in the separate research archive.

The generated `passed` flag checks predefined algebra, positivity, invariants, step refinement, late error decrease, and exact KKT controls. It does not certify an infinite-time limit, novelty, or a general rate.

Three independent checks are also preserved: `check_quantitative.py` (NumPy, one noncommuting rate example and a Lyapunov counterexample), `exact_controls_check.py` (standard-library rational arithmetic), and `lyapunov_counterexample.cjs` (built-in Node modules, a separately derived witness and three controls). These scripts write JSON beside themselves. Copy a script into a fresh empty research directory before rerunning it to preserve previous outputs.

`package_audit.py --package <package> --repo-code <this-directory> --prepare` prepares the source snapshot and integrity record. Without `--prepare`, it performs a read-only verification. It also checks both required historical archive manifests without modifying them. It does not rerun ODEs or prove theorems.

Research report and derivations are stored in the separate `astra6-global-convergence-deepening-20261121` research package. Source snapshots in that package preserve reproducibility if this live repository later changes.
