# Offline two-view distillation and external apparel evaluation

Implementation for the frozen teacher/student/plain full-compute experiment.
Research outputs and protocols belong in the separate research archive.

The saved run directory is self-contained: models, all selected source JPEGs,
processed arrays, Fashion-MNIST source bytes, splits, calibration and predictions.
To replay it, run its own study.py so paths resolve beside the preserved artifacts.
The exact implementation copies here have the same hashes as that run.

Dependencies: Python, NumPy, PyTorch (CUDA for training/scoring); Pillow for
external preparation/rebuild; Matplotlib for reporting. Runtime versions are
recorded in the archived TRAINING_START.json and acquisition log.

Fresh experiments require a new output directory containing these scripts,
the data/ folder, splits.npz and external/ population artifacts. Never rerun
mutation stages into a completed experiment. prepare_external.py builds the
frozen category population from the original public source ZIP and styles.csv.
rebuild_external.py verifies saved selected JPEGs without that large ZIP.

Stage order: study.py freeze, train, score, analyze, benchmark; report.py renders
saved results. Before training, an independently reviewed ROOT_APPROVAL.json
in the parent directory must bind the exact SOURCE_FREEZE.json SHA-256 under
"distillation_freeze_sha256". Benchmark additionally needs TIMING_RELEASE.json
in the run directory and an exclusive numerical-work window. Frozen methods
cannot change in response to held-out outcomes.

All matrices, logit arrays, selected checkpoints and original images remain in
the archive, not in Git. Dense-matrix multiply-add budgets include teacher
training, target precomputation, validation and calibration forward passes.
They do not claim equality of whole-program floating operations or wall time.
