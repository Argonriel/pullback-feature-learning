# Reproduce the endpoint-selection deepening checks

These scripts need Python and NumPy. They implement deterministic checks in tiny matrix spaces, not a neural performance study. Use fresh output paths in the separate research archive. Every script refuses to overwrite its output.

```powershell
python general_geometry_check.py --output-dir C:/path/to/fresh/geometry
python static_weight_response.py --output C:/path/to/fresh/static_response.json
python static_weight_quartic_mixed.py --output C:/path/to/fresh/quartic_mixed.json
python metric_control.py --output C:/path/to/fresh/metric_control.json
python factor_replay.py --control C:/path/to/fresh/metric_control.json --output C:/path/to/fresh/factor_replay.json
```

The first three commands can run independently. The quartic script imports reusable functions from `static_weight_response.py` and records that source hash. `factor_replay.py` needs the completed control data file, whose hash it records; it independently integrates the original 18 factor parameters rather than importing the product-flow implementation.

`general_geometry_check.py` includes finite polarization checks of the two polynomial conditions. The archived initial geometry output predates that addition and retains its own source snapshot. Current output goes to a fresh directory.

`metric_control.py` fixes the three-by-three star-plane example at raw matrix amplitude .25, isotropic scale 1, damping .2, and weights (1,1)/(1,2). It compares the representation-pullback metric with an established entropy-Hessian control, and runs both RK4 steps .02/.01. `factor_replay.py` uses .01/.005. The static-response scripts evaluate the first damping derivative along exact undamped paths; those outputs should not be confused with finite-damping endpoints.

`package_archive.py` copies completed staged records and source into the explicitly named fresh archive, checks source/dependency/data hashes, checks links, and verifies three prior archives against their known manifests. Reports and scientific interpretation belong in the research archive, not this source directory.
