# Reproduce the local-bound analysis

Dependencies are NumPy, Matplotlib, and threadpoolctl. The numerical run uses one BLAS thread. Starting from the complete study directory, reproduce into a new, nonexistent directory:

```powershell
python bounds/reproduce.py --destination C:/Users/liwei/Documents/ChatGPT/277/tmp/local-bound-reproduction
```

The launcher copies the frozen code, fitted old and fresh fields, and required dependencies, then constructs all bounds and diagnostics without retraining or changing any prediction policy. It refuses to overwrite an existing destination. Fresh model training and field calibration are separately reproducible through the root validation workflow.

In a clean working copy with the frozen code, development inputs, source/reference/legacy dependencies, and fresh fitted fields already present, the component commands are:

```powershell
python bounds/experiment.py old
python bounds/parity_control.py old
python bounds/experiment.py fresh
python bounds/parity_control.py fresh
python bounds/cdf_check.py
python bounds/analyze.py
```

The experiment commands refuse to overwrite old or fresh outputs. Do not rerun the freeze action in an archived study: the original method freeze and its timestamp are part of the provenance. Analysis can regenerate summaries and scientific figures from existing raw outputs.

The local-bound computation costs 262,144 paired interval-jet propagations and 262,656 ordinary point forwards per model. Additional numerical moment diagnostics cost 2,624,512 point forwards per model, with small derivative spot checks recorded separately. Neither diagnostic angular nodes nor confidence samples count as independent trained models. The finite-sample all-posterior certificate belongs to root_audit/certificate.py and is intentionally not refitted or recomputed by this component launcher.

The prior four-model, four-configuration development sweep and all its outputs are preserved under pilot/. The exact configuration is fixed in freeze.json. The global odd-only attribution control has a separate parity_control_freeze.json and requires no additional network forwards. The independent CDF check and numerical differentiation convergence checks do not change the frozen deterministic bound.
