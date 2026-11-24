# Two-amplitude task: frozen development record

The task has a genuine two-dimensional quotient on its specified support. Both harmonic amplitudes vary independently, and the label posterior depends on their smooth XOR. Total input norm loses label information. The half-turn flips the first harmonic only.

The fixed four-seed pilot selected **PB, rate 4, 500 steps, damping 0.1**. All 12 declared fits completed; no task or architecture change followed the results. The model learns class signal, but its raw probability estimates remain poorly calibrated: mean classification error is well below 0.5 while mean log loss is almost log(2). The longer or faster recipes have lower training loss but higher population log loss. The fresh evaluation must preserve this limitation rather than portray the model as a strong calibrated classifier.

| Recipe | Raw expected log loss | Full half-turn log loss | Raw classification error | Full half-turn classification error | Training loss |
|---|---:|---:|---:|---:|---:|
| PB 8 / 500 | 0.850375 | 0.702016 | 0.373253 | 0.342700 | 0.268722 |
| PB 4 / 500 | 0.692926 | 0.619494 | 0.366167 | 0.337911 | 0.370544 |
| PB 8 / 1000 | 1.086643 | 0.840363 | 0.381172 | 0.348973 | 0.196228 |

The table is development evidence, not an independent confirmation or a statistically selected final-test result. Each recipe uses the same training inputs and initialization within a pilot seed; each seed has an independent 32,768-input evaluation sample. Recipe selection used the mean raw expected log loss only. Full half-turn averaging uses two calls per input and is a distinct-cost reference.

Population Bayes log loss is 0.4929065112 and Bayes classification error is 0.2265626510. Exact prevalence is 0.5. Training has exactly 128 samples per class by class-conditional rejection; this fixed-count stratified sample is not iid in its labels. Calibration and evaluation use iid inputs. The class-conditional distributions are preserved.

## Artifacts and scope

- `TASK_LAW_FREEZE.json` predates all pilot training and specifies the task, recipes, seeds, criterion, and calibration grid.
- `TRAINING_CHOICE.json` records the recipe selected by that criterion and hashes the pilot rows.
- `TASK_FREEZE.json` commits the complete task/training interface before the reserved fresh seeds are opened.
- `pilot_results/rows.csv` and `rows.json` retain all 12 fits; per-seed NPZ files retain inputs, initialization, evaluation inputs/posteriors/labels, model states, and loss traces.
- `TASK_CHECKS.json` records numerical checks of 4,096 orbit points, model derivatives at initialization and a trained state, exact class counts, and Bayes calculations. The unsplit absolute-value quadrature is retained alongside the accurate split-domain Bayes-error check.
- `THEORY.md` gives the quotient argument, known input law, source conventions, and finite logit bounds. On arbitrary ambient R4 data two norms would not be a complete quotient; the claim is restricted to the stated phase-coupled support.
- `LITERATURE.md` records the primary-source review that preceded implementation.
- `core4d.py` is an unchanged copy of the previous 144-parameter residual source (SHA-256 `7f6c227887f5a45606d206a4805e4694a48409f1b345a8a12def21dd7f628f77`). Its old data functions are unused. `task.py` owns every new sampling and posterior function.

The shared calibration grid is 16×16 amplitude anchors and 256 full-orbit midpoint angles, exactly 65,536 model forwards. The iid folded baseline receives 32,768 input/partner pairs at the same cost, with a different sampling design. Deployment and confirmation live in the parent study and are not run by this development script. Reserved independent training/init seeds are 1141001–1141016 and 1142001–1142016.

## Reproduction

Use Python with NumPy, matplotlib, and threadpoolctl. From this directory:

```text
python check_task.py
python report_pilot.py
```

To reproduce training in a **new copy** of this directory, retain the frozen source files and `TASK_LAW_FREEZE.json`, move the existing `pilot_results` and `TRAINING_CHOICE.json` aside, then run `python pilot.py run`. The runner verifies all frozen hashes and refuses to reuse an existing result directory. To create a separate new preregistration instead, `python pilot.py freeze` requires that no task-law freeze already exists. Do not overwrite original results. The original 12 fits took about 17 seconds of model training on one CPU thread; hardware affects runtime. No external archived files are required after this folder is copied.

The public API is `coordinates(r1,r2,phi)`, `quotient(x)` returning an N×2 array, `angle(x)`, `partner(x)`, `transform(x,theta)`, `sample_inputs(seed,n)`, `sample_train(seed,n)`, `sample_test(seed,n)`, `predict(theta,x)`, `train(...)`, and `network_constants(theta)`. `training_config()` reads the frozen selected recipe. Calibration can draw inputs without access to labels or posteriors.
