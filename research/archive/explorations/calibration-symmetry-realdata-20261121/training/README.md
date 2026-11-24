# Training, held-out calibration, and selective symmetry

This experiment retains the previous two-amplitude label law and 144-parameter residual architecture. It separates training selection, scalar temperature fitting, selector calibration, and performance evaluation into disjoint streams. All 16 fresh replications and all six model variants are retained. Temperature scaling is established prior work, not a new method claim.

The bounded pilot completed all 72 trajectories without failure. The selected recipes were **PB rate 2, lambda 0** and **GD rate 0.3, lambda 0**, each followed by per-model selection among 100, 250, 500, and 1,000 steps using 2,048 independent sampled validation labels. The pilot did not select either nonzero regularization strength. Thus any improvement of the selected PB procedure is evidence about the slower-rate/validation-selection procedure, not evidence that L2 regularization helped. Legacy PB rate 4 at 500 steps remains unchanged.

Each temperature is fitted separately from 2,048 independent sampled labels. Only positive inverse temperatures in [exp(-3),exp(3)] are allowed. Known posterior values are excluded from all fitting and selection. Positive temperature preserves raw and full-half-turn class decisions; the implementation checked identical signs for every fresh performance input.

| Model variant | Raw log loss | Raw classification error | Interpolated log loss | Folded log loss | Full log loss |
|---|---:|---:|---:|---:|---:|
| legacy_raw | 0.697837 | 0.373042 | 0.644565 | 0.649652 | 0.625515 |
| legacy_temp | 0.647410 | 0.373042 | 0.633679 | 0.634943 | 0.628686 |
| pb_raw | 0.654007 | 0.378900 | 0.630366 | 0.631538 | 0.621340 |
| pb_temp | 0.650172 | 0.378900 | 0.634786 | 0.635562 | 0.628933 |
| gd_raw | 0.700543 | 0.491923 | 0.696529 | 0.696573 | 0.694452 |
| gd_temp | 0.692856 | 0.491923 | 0.692671 | 0.692675 | 0.692578 |

Raw prediction costs one model call per input. The interpolation and strong folded policies use exactly 64 extra calls in each 256-input batch, or 1.25 calls/input. Full half-turn averaging costs two calls/input. These are separate cost points. Both allocation methods have the same standalone 65,536-forward calibration budget and use only input coordinates and the fitted model at selection time.

The four primary expected-log-loss differences below use 16 paired model seeds, 20,000 bootstrap draws, and two-sided alpha=0.05/4 per contrast. Negative differences favor the first procedure. This bootstrap summarizes finite-model variation; the separate robustness branch supplies different, distribution-free certificate statements.

| Contrast | Mean difference | Adjusted interval | Seed wins |
|---|---:|---:|---:|
| primary_total | -0.047665 | [-0.061755, -0.034219] | 16/16 |
| primary_preserved_gain | -0.015386 | [-0.017838, -0.013192] | 16/16 |
| training_only | -0.043830 | [-0.057380, -0.030699] | 16/16 |
| legacy_temperature | -0.050427 | [-0.061537, -0.039632] | 16/16 |

The improvements are specifically in log loss. Selected training worsens raw classification error from 0.373042 to 0.378900 (12 of 16 seeds worsen); temperature leaves that error unchanged. Legacy temperature calibration alone has descriptively lower raw log loss, 0.647410, than selected PB plus temperature, 0.650172, while using fewer fitting labels. The combined procedure therefore does not establish that retraining is necessary or that the new model is better in every metric.

Calibration also interacts with averaging. The temperature is fitted to first-view logits, not to final averaged-policy outputs. For selected PB, it lowers raw log loss from 0.654007 to 0.650172 but raises interpolated log loss from 0.630366 to 0.634786 and full-half-turn log loss from 0.621340 to 0.628933. Averaging remains beneficial relative to that calibrated model's own raw predictions, as the prespecified preserved-gain contrast shows; nevertheless, first-view calibration and averaging do not combine monotonically. The confidence plot shows underconfidence after averaging. A separately validated future study could calibrate the final deployed policy's outputs, but no such additional temperature or policy was fitted here.

All secondary contrasts, including selected-PB temperature-only, GD context, the strong histogram comparison, and classification errors, remain in `contrasts.csv`; none is promoted to a primary endpoint after inspection. The selected GD control remains close to random classification (error 0.491923). This is a finite-budget optimization limitation, not a general theorem comparing optimizers. Per-model values and seed-paired differences are retained. The Bayes values remain approximately 0.492907 log loss and 0.226563 classification error; reduced overconfidence is not the same as solving the classification task.

## Data, cost, and limitations

The standalone label budgets differ: legacy raw uses 256 training labels; legacy plus temperature uses 256+2,048 labels; selected raw uses 256+2,048 validation labels; selected plus temperature uses 256+2,048+2,048 labels. These are not claims of improvement at equal total label budget. The validation and temperature data are shared across model families within a replication for paired comparisons, while the two splits remain independent of each other and of performance data. PB and GD receive the same maximum update/search/checkpoint budgets, but their wall-clock costs and optimizer geometry differ.

Raw and temperature selectors reuse base-network calibration logits. Actual shared selector-calibration work is 6,291,456 forwards; the sum of their standalone equivalent budgets is 12,582,912 forwards. The saved grids recompute Jensen values after temperature scaling, so the two variants need not choose identical masks. The complete physical performance replay executes 34,603,008 model-example forwards and checks 98,304 batch-call records. It does not pretend to cache calls that were actually executed. Temperature fitting adds 98,304 model forwards (48 base models × 2,048 points); validation-selection and separate diagnostic forwards are listed in `training.csv`, alongside runtime. Training itself additionally computes gradients and Jacobians.

The input is a bounded synthetic phase-coupled harmonic signal, not general ambient-four-dimensional or real-world data. The true two-dimensional quotient is known, and the performance posterior is evaluator information. The preceding failure of probability calibration motivated this run, so it is a prospective confirmation on new data after retrospective diagnosis, not an unprompted discovery. No model capacity, label law, or frozen calibration policy was changed after fresh performance outcomes.

## Artifacts and reproduction

- `PILOT_FREEZE.json`, `pilot/`, and `TRAINING_CHOICE.json`: all 72 pilot trajectories, four checkpoints each, failures if any, and the deterministic selection rule.
- `FRESH_FREEZE.json`, `ROOT_APPROVAL.json`: source and stream commitment before fresh generation.
- `states/`, `splits/`, `temperatures/`, `MODEL_COMMIT.json`: all fresh model checkpoints, independent labeled splits, scalar fits, and hashes.
- `policies/`, `POLICY_COMMIT.json`, `robustness_inputs.json`: immutable exported selectors, actual/shared costs, and the 32-model robustness contract.
- `predictions/`, `deployment_ledger.csv.gz`, `SCORE_COMPLETE.json`: per-input outputs and masks, exact executed calls, and input/output hashes.
- `per_model.csv`, `reliability_bins.csv`, `summary.csv`, `contrasts.csv`, `paired_contrasts.csv`: all outcomes and comparisons.
- `LITERATURE.md`, `PROTOCOL.md`, `FRESH_PROTOCOL.md`: literature, assumptions, and finite-scope design.

Dependencies are Python, NumPy, matplotlib, and threadpoolctl. This branch is portable and includes byte-identical copies of its original task/model and selector source. From this directory, the execution stages are:

```text
python pilot.py freeze
python pilot.py run
python smoke.py
python study.py freeze
python study.py train
python study.py calibrate
python study.py score
python study.py analyze
python report.py
```

Existing freeze and result paths are protected against overwrite. Replay training/evaluation only in a separate copy with fresh output destinations and a matching explicit approval record; do not remove or rewrite this run's originals. The reporting command alone can regenerate the readable report and figure from completed tables without retraining or fitting any parameter. `adapters.load_variant(pair,variant)` exposes the calibrated prediction function, true half-turn, and explicit scaled logit-difference bound for external verification.
