# Pilot findings and next-stage budget

Recorded on November 19, 2026 (America/Los_Angeles). The machine used an AMD Ryzen 9 9900X CPU; computation was float64 with one BLAS thread. The reproducible numerical environment is recorded in [environment.json](../reports/pilot/environment.json). No GPU was used.

## Implementation checks

All 10 numerical unit tests passed. The additional full-training-sample verification used a 512-by-80 Jacobian and recorded:

| Check | Measured value |
| --- | ---: |
| Full analytic Jacobian vs. centered finite differences, relative error | 2.27e-10 |
| Independent reverse-mode gradient vs. J-transpose times g, relative error | 5.53e-16 |
| Damped normal-equation relative residual | 1.36e-15 |
| GD spectral velocity relative error | 2.29e-15 |
| Pullback spectral velocity relative error | 4.35e-14 |
| Taylor-remainder ratio when the step is halved | Approximately 4 for both methods |

Both directions had a negative directional loss derivative and decreased the loss at the checked small step. Acceptance tolerances and all measurements are in [verification.json](../reports/verification.json). These checks support the local derivation and numerical implementation; they do not prove global convergence.

## The 200-step run

Both methods started from the exact same arrays and parameter vector. The initial training loss was 0.570782 and initial validation accuracy was 69.53%. The settings were GD learning rate 0.5, pullback learning rate 32, and fixed damping 0.01. These were initial pilot choices, not winners of a parameter search.

| Quantity | GD | Damped pullback |
| --- | ---: | ---: |
| Final training loss | 0.190637 | 0.040684 |
| Final validation loss | 0.268577 | 0.090131 |
| Final validation accuracy | 87.50% | 98.44% |
| Training compute time, 200 updates | 0.0147 s | 0.1274 s |
| Mean update time | 0.0737 ms | 0.6369 ms |
| Diagnostic time | 0.0275 s | 0.0332 s |
| Loop wall time | 0.0423 s | 0.1608 s |
| Training-loss increases above 1e-12 | 0 | 0 |

The pilot produced finite, monotonically decreasing training losses for these settings. The pullback achieved lower loss after 200 iterations, while each of its updates cost more. This is exactly why both iteration and time axes are needed. The GD trajectory stops after 200 steps, so it cannot be extrapolated to the longer pullback time budget. A matched-time superiority claim would be premature.

The two-dimensional point clouds illustrate the learned deformation. They are training-set visualizations with varying panel scales, not a held-out probe evaluation. A trained linear probe and at least three tuned paired seeds are still required to make a feature-quality comparison.

The run used no line search or fallback update, and no configuration adjustment was needed. An identical 200-step run was repeated after adding explicit tolerance metadata to the verification report. Numerical trajectories were retained for a reproducibility check; the final recorded run supplies the table above. This repetition was not a hyperparameter search. Timing is inherently noisy for such short computations and is only a preliminary budget estimate.

## What to do next

1. Predeclare a small equal candidate-run budget, separate learning-rate ranges, pullback damping values, and the validation selection rule. Keep the per-run update budget explicit and also record actual tuning cost.
2. Start with a 2,000-step budget, then use training/validation behavior to decide whether that budget is adequate before formal evaluation. The 200-step pilot does not establish convergence.
3. Freeze the chosen settings and run at least three paired seeds. Include comparable time ranges for both methods, not just equal update counts.
4. Fit linear probes on initial and learned frozen features under one train/validation protocol, then evaluate the reserved test data. Complete the spectral and feature-displacement explanations.

A straight-line extrapolation of the measured update costs gives about 0.15 seconds for 2,000 GD updates and 1.27 seconds for 2,000 pullback updates. For illustration, 10 runs of each at that budget would require about 14 seconds of core updates. This excludes diagnostics, plotting, environment startup, file writes, and any longer training required for a meaningful comparison; it is not a promised end-to-end runtime. Matrix sizes remain fixed, but timing should be remeasured over longer runs before drawing cost conclusions.

The earlier hour-scale compute allowance was conservative: with this explicit small NumPy implementation, a minutes-scale machine window is a more useful starting allowance for the next small experiment batch. The larger time commitment remains experimental design, result checking, mathematical explanation, and writing.

Stages 1-3 are complete. Formal tuning, multi-seed evaluation, linear probes, and any distribution-shift extension have not been executed.
