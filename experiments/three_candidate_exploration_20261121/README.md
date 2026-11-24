# Three-candidate exploration checks

These scripts implement bounded mathematical checks, not model-performance
benchmarks. Each writes to a fresh output directory and preserves its settings.
The reports and complete outputs belong in the independent research archive.

- `universality/spectral_delay_check.py`: normalized SLD/RLD mixture, delayed
  static-weight endpoint response, and positive moment-matching rules.
- `nonisotropic/coherent_blend.py`: exact normalized-field curvature, a generic
  rank check, and fixed-weight endpoints from a coherent initial state.
- `nonisotropic/refine_bures_control.py`: refinement of the pure-Bures control.
- `global_convergence/global_target_check.py`: one strictly feasible large target
  with an indefinite unconstrained Euclidean interpolant. It imports the unchanged
  `../endpoint_classification_audit_20261121/verify_finite_damping.py` dependency.
- `discrete/factor_step_cancellation.py` and `discrete/critical_quartic_certificate.py`:
  supplementary fixed-step cancellation checks, not counted as a fourth direction.
- `robustness/near_diagonal_probe.py`: exploratory rejected-lane calculations.

Use `--help` for each script's options. The main checks require Python and NumPy;
high-precision supplementary arithmetic uses the standard-library Decimal type.
No additional package installation is required.
