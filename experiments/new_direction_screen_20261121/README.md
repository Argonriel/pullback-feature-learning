# Reproduce the bounded new-direction screen

All scripts require Python and NumPy. `topology.py` also imports the repository's core model. Run from this directory or invoke the scripts by absolute path. Every output must be a fresh file/directory outside the Git repository, in the research archive. Scripts refuse to overwrite result files or experiment directories.

```powershell
python depth_screen.py --output-dir C:/path/to/fresh/depth_checks
python depth_endpoint_probe.py --output-dir C:/path/to/fresh/depth_endpoint
python depth_commuting_control.py --output-dir C:/path/to/fresh/depth_commuting
python endpoint_parameter_audit.py --steps 0.0025 0.00125 --output-dir C:/path/to/fresh/parameter_refined
python endpoint_response_review.py --output C:/path/to/fresh/response.json
python stochastic_pb.py --output C:/path/to/fresh/stochastic.json
python stochastic_label_conditioning.py --output C:/path/to/fresh/conditional_labels.json
python topology.py --output C:/path/to/fresh/topology.json
python poisoning.py --output C:/path/to/fresh/poisoning.json
```

`depth_endpoint_probe.integrate(damping, weight, step)` is the reusable reduced-flow integrator. Its CLI evaluates damping in `[0,.003,.03,.3]`, weights in `[1,9]`, and steps in `[.02,.01]`. The archived numerical refinement calls the same function with steps `[.005,.0025]`, retaining the same cases, stopping rule, and source hash. This is a post-observation precision check, not an independent performance cohort. To regenerate the refined records in an interactive Python session:

```python
import json
from pathlib import Path
from depth_endpoint_probe import integrate

records = [integrate(damping, weight, step)
           for damping in [0.0, 0.003, 0.03, 0.3]
           for weight in [1.0, 9.0]
           for step in [0.005, 0.0025]]
with Path("C:/path/to/fresh/refined_records.json").open("x", encoding="utf-8") as output:
    json.dump(records, output, indent=2)
```

`endpoint_parameter_audit.py` independently integrates the original eight parameter coordinates; it imports no reduced-flow implementation. Its default steps reproduce the coarser check, and `--steps .0025 .00125` reproduces the refined check. `endpoint_response_review.py` instead evaluates an exact matrix-exponential zero-damping path and integrates its first-order response; it does not replay a damped ODE.

`package_archive.py` snapshots source and prepared research records, verifies the explicitly supplied earlier archive manifests, resolves source hashes, checks report links, and writes a new manifest. It is intentionally restricted to the named `new-direction-screen-20261121` archive and refuses to overwrite different content or an existing manifest. Research interpretation and all data remain in the independent archive; this README only documents usage.
