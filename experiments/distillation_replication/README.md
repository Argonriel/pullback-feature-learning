# Independent-source distillation replication

Implementation for six paired seeds on the frozen Fashion-MNIST fitting split,
with an independently acquired, product-grouped UT Zappos50K footwear evaluation.
Research protocols, data, models and reports remain in the separate archive.

The run is self-contained. Use its own scripts so artifact paths resolve beside
its preserved data. Dependencies are Python, NumPy and PyTorch for fitting;
Pillow for population preparation; Matplotlib for output-only report plots.
The inference code computes df=5 Student-t quantiles analytically, without SciPy.

Prepare a fresh output directory with scripts, Fashion-MNIST data/ and splits.npz,
old exposed data in legacy_external/, and official UT Zappos metadata plus image
ZIP in external/. prepare_zappos.py creates a fixed group-aware population;
academic/non-commercial source terms in the official README must accompany it.
Never regenerate populations or fitting stages into a completed run.

Stage order: study.py freeze, train, score, analyze, benchmark. Training requires
parent ROOT_APPROVAL.json binding the exact SOURCE_FREEZE.json SHA256 under
"distillation_freeze_sha256". Benchmark additionally requires TIMING_RELEASE.json
and a coordinated exclusive numerical-work window. report.py reads saved results.

Primary inference is over six paired training randomizations on one fixed split.
Crossed seed/product bootstrap is secondary and reuses identical sampled product
indices across chosen seeds and methods. Full-compute matching includes teacher
training, target generation, validation and calibration network forwards; it
is not a claim of whole-program FLOP or wall-clock equality.

No source code from the earlier experiments/distillation_external implementation
is overwritten. Full raw images, selected originals, models and statistical
outputs belong in the archive rather than Git.
