# Empirical premise audit: readout decoupling

**Verdict: do not promote the replacement direction as an innovation.** The archived 49-fold unused-feature path reduction is real, but near equality of total readout path lengths conceals nonzero predictor differences. Ordinary compensation through an unused output head is present. Neither the path statistics nor the special zero-motion witness establishes an unexplained finite-trajectory decoupling phenomenon.

This retrospective audit used only the 16 original training sets reconstructed from recorded seeds and verified against their original hashes, plus 288 saved model states. All 384 logical training-loss records reproduced exactly. There was no training, validation evaluation, new holdout evaluation, population integration, or selection of a new method. The seven historical input/source files were hashed before and after the audit and remained unchanged. All numbers below are descriptive; 16 model/data pairs, not the individual examples, are the independent replication units.

## 1. The headline path comparison is correct but insufficient

For native full pullback at update 500, comparing unused-coordinate weights 0 and 9:

| Quantity | Archived/recomputed value |
|---|---:|
| Mean unused-coordinate cumulative path, weight 0 | 1.760375 |
| Mean unused-coordinate cumulative path, weight 9 | 0.035850 |
| Ratio of those means | 49.1046 |
| Mean readout cumulative path, weight 0 | 2.760037 |
| Mean readout cumulative path, weight 9 | 2.761083 |
| Relative difference of mean readout paths | +0.03787% |
| Mean endpoint training-logit RMS difference | 0.129409 |
| Mean relative training-logit RMS difference | 4.9653% |
| Range of relative RMS differences over pairs | 1.2482%–11.1305% |
| Mean absolute probability difference | 0.006423 |
| Largest single-example probability difference | 0.405379 |
| Class disagreements across the 16 × 256 training records | 9 / 4,096 |
| Mean cosine between net readout displacements | 0.998419 |

Relative logit difference uses the weight-0 endpoint logit norm as denominator. Probabilities use the model's original sigmoid without fitting a temperature. The 49.1046 value is a ratio of mean paths; the mean of per-pair ratios is 68.3187. These are different statistics.

The predictors are often close in aggregate and retain nearly all training classifications, but are not identical. Total distance traveled is a scalar and cannot establish equality of either paths or endpoints. Even the per-pair relative readout-path differences range from -1.1918% to +0.5605%, so the very small mean also conceals variation across pairs. The largest probability discrepancy shows that some local predictions differ substantially despite similar average path lengths. This audit does not establish how these differences behave off the training inputs.

The archived control matching predicted local readout-motion lengths does not remove the endpoint discrepancy: its update-500 mean relative logit RMS difference is 4.9538%, with probability MAE 0.006399. This control matches length at each arm's own state, not complete predictive progress across arms. In the archived hybrid schedule, which changes to GD after 100 pullback updates, the update-500 relative logit RMS difference is 23.8747% and probability MAE 0.032876. The latter is a secondary archived comparison and is not a new experiment.

## 2. Ordinary unused-head compensation is available and observed locally

The model has two residual blocks. The eight entries of the final block's second output row affect only the unused coordinate. They never affect the readout, at any input or parameter state. In the original 80-parameter ordering these are indices 64 through 71. Thus there is an exact architectural nullspace without the specially constructed symmetry witness.

Write the training Jacobians as A=(A_c,0) for the readout and B=(B_c,H) for the unused coordinate, separating these eight head parameters. For the pullback direction with metric A^T A + gamma B^T B + lambda I, block elimination gives

    u_head = -gamma (gamma H^T H + lambda I)^(-1) H^T B_c u_active,
    B u = lambda (gamma H H^T + lambda I)^(-1) B_c u_active.

The active block consequently pays the ordinary ridge-residual penalty

    gamma B_c^T [I - gamma H(gamma H^T H+lambda I)^(-1)H^T] B_c.

These are standard Schur-complement/ridge identities, not new results. Increasing gamma lets the unused head compensate changes induced by the rest of the model while also altering the active update.

At the 16 saved weight-1 update-500 states, hypothetical directions were computed for weights 0, 1/9, 1 and 9 using the original damping 0.1. No direction was applied and no trajectory was run. For the weight-9 directions:

| Fixed-state diagnostic | Mean over 16 baseline states |
|---|---:|
| Active-body unused velocity energy lying in the unused-head span | 98.3172% |
| Cosine between active-body and unused-head velocity contributions | -0.991411 |
| Total unused velocity norm / active-body contribution norm | 0.097660 |
| Readout-velocity relative change versus weight-0 direction | 6.2773% |
| Readout-velocity cosine versus weight-0 direction | 0.996637 |
| Unused-velocity norm / weight-0 unused-velocity norm | 0.028413 |

The compensation is strong locally. These diagnostics do **not** quantitatively explain the whole 49-fold cumulative path reduction, because directions and Jacobians change along different trajectories. The full endpoint parameter differences also are not confined to the eight trivial head coordinates: those coordinates contain only 3.3462% of their squared norm on average. Thus “only the unused head changed” is false, while “ordinary architectural compensation is available and materially used” is supported.

At those baseline endpoints, both coordinate Jacobians have numerical rank 72, including when the unused-coordinate rank threshold is changed to 1e-10 times its largest singular value. The readout gradient has a mean 20.3817% of its squared norm outside the readout motions obtainable through ker(B), ranging from 0.5050% to 64.3161%. The exact nullspace does not accommodate every task-gradient direction. This is a local numerical rank/geometry statement, not an assertion about nonlinear function equivalence.

## 3. The exact witness does not explain the empirical premise

The proposed witness assumes zero second-coordinate output rows, a second block that ignores its second input, reflection-paired first-block units, and reflection-paired training records. Under those restrictions the unused coordinate is exactly x_2 and never moves. It therefore has zero unused-coordinate cumulative path at every weight. It cannot by itself explain two positive paths whose ratio is 49.1.

The archived generator samples angles independently; it does not construct the required reflection pairs. The model starts with random, untied weights, not the witness's constrained initialization. Near equality of path lengths gives no evidence that training approaches the witness. A quantitative approximate-witness explanation would require controlling distance to the relevant structure and its evolution, not merely displaying an invariant submanifold. The archived states alone do not supply that argument.

Moreover, the old project had already proved q'(gamma)=-||B s_gamma||^2, the exact same-state no-change condition B s_0=0, and their local consequences. Rephrasing these as a coupling derivative would not add a project result.

**Elimination decision:** the strong empirical premise of nearly unchanged finite-trajectory predictors does not follow from the path statistic; the predictors measurably differ, and standard unused-head compensation is visible. The special witness is mismatched to the observed nonzero motion. These checks are sufficient to reject the present justification for a new decoupling mechanism. They neither prove that every possible trajectory question is already solved nor identify a new method or research goal.

## Files and reproduction

The reusable driver is `pullback-feature-learning/experiments/readout_decoupling_novelty_audit/audit_archived_states.py`. An identical snapshot is in `empirical_outputs/audit_archived_states.py` (SHA256 `edbfa90df881008f35fc6c93e8f8c270eb45095e82f7852b2c02f1f08d7f34dc`). Dependencies are Python 3.12, NumPy and threadpoolctl; the recorded run used Python 3.12.14 and NumPy 2.5.5. No installed project package is needed because the supplied core and generator are loaded directly without writing bytecode.

From this report's directory, using the included exact input copies:

```powershell
python -B empirical_outputs/audit_archived_states.py --archive empirical_outputs/portable_inputs/archive --repository empirical_outputs/portable_inputs/repository --out NEW_OUTPUT_DIRECTORY
```

The output directory must not exist. In a consolidated package with renamed input folders, the equivalent arguments are `--archive inputs/rings --repository inputs/repository`. The required archive files are `generator.py`, `stage2_run.py`, `stage2_plan.md`, and `stage2/{states.npz,training_rows.csv,training_record.json}`. The required repository file is `src/pullback_feature_learning/core.py`. All seven exact inputs are included under `empirical_outputs/portable_inputs/`, with `INPUT_HASHES.json`.

The driver verifies all archived state hashes, 16 training-data/initialization hashes, and all 384 recorded training losses. Outputs comprise 480 endpoint pair comparisons, 576 diagnostic rows at 144 saved/initial states, all training inputs and per-item outputs, fixed-state Jacobians/gradients/directions/singular values, descriptive summaries, and the provenance record. No full per-update trajectories were stored in this old archive; the audit cannot retrospectively establish persistence between checkpoints or construct exact matched-progress trajectories without new training. No such reconstruction was attempted.
