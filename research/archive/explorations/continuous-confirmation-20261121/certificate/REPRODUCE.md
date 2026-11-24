# Reproduce the certificate study

Run from the study root using a Python environment with NumPy, Matplotlib, and threadpoolctl. The recorded environment was Python 3.12.14, NumPy 2.5.5, Matplotlib 3.11.2, and threadpoolctl 3.8.0 on Windows 11. The scientific computations restrict BLAS to one thread.

The following command copies only the required code and model inputs into a **new, nonexistent directory**, then reproduces both historical and fresh calibration, evaluation, summaries, figures, and resident-table serving timings:

```powershell
python certificate/reproduce.py --destination C:/Users/liwei/Documents/ChatGPT/277/tmp/certificate-reproduction-20261121
```

Using the current workspace environment explicitly, from the workspace root:

```powershell
.venv/Scripts/python.exe tmp/continuous-confirmation-20261121/certificate/reproduce.py --destination C:/Users/liwei/Documents/ChatGPT/277/tmp/certificate-reproduction-20261121
```

The destination must not exist. The launcher refuses to overwrite old outputs; choose a new path for a second run. It preserves the original scientific freeze and creates new selector and scoring commitments. It does not retrain models. Reproducing the fresh model training is handled by the parent study's confirmation package.

To regenerate only tables and figures from the archived raw outcomes, without rerunning scientific scoring:

```powershell
python certificate/analyze.py
```

That analysis script verifies frozen source, table, and raw-result hashes. It also repeats a small outcome-neutral raw/oracle timing reference; numerical scientific summaries remain deterministic, while timing values can vary. The final resident-table benchmark is a separate script and must run in a clean reproduction directory because it refuses to overwrite its prior output:

```powershell
python certificate/serving_benchmark.py
```

No internet, SciPy, external labels, or external model calls are needed. The copied model source is `source/pullback_feature_learning/core.py`; historical states are under `legacy/budgeted-rotation-final/results/models.npz`; fresh PB500 states are `confirmation/states/original_1.npz` through `original_16.npz`. Deployment selection never uses the evaluator-only posterior or obtains both fresh views before allocating the extra-call budget.

The full reproduction performs 5,242,880 interval propagations and scores 16 historical plus 16 fresh models. Fresh-model confirmation results are summarized separately from historical checks. Runtime is ordinarily a few minutes on this machine, with additional time for compressed table I/O. Timings and compression hashes can vary across environments; policy decisions and floating-point statistics should agree to numerical tolerance.
