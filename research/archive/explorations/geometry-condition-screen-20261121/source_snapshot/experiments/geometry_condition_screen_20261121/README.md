# Structural-condition screening

These deterministic calculations screen specific explanations for adaptive
representation geometry. They do not extend the previous empirical gate or
claim a new mechanism. Research reports and outputs belong in the independent
archive, not in the repository.

From the repository root, use the recorded project Python environment and
fresh output paths:

```text
python experiments/geometry_condition_screen_20261121/parity.py --output ARCHIVE/parity_results.json
python experiments/geometry_condition_screen_20261121/curvature.py --output ARCHIVE/curvature_results_v2.json
python experiments/geometry_condition_screen_20261121/curvature_search_full.py --output ARCHIVE/curvature_search_full.json
python experiments/geometry_condition_screen_20261121/noise.py --output ARCHIVE/noise_results_verified.json
python experiments/geometry_condition_screen_20261121/commuting.py --output ARCHIVE/commuting
```

`parity.py` checks the full 80-parameter architecture's inversion parity,
metric blocks, small-residual scaling and two-step feature creation.
`noise.py` enumerates all sixteen conditional-label draws on four inputs,
checks the initial metric derivative, and uses tiny loss-drop flows only to
verify the Taylor coefficient's interpretation. Neither uses the prior study's
reserved confirmation cohort.

`curvature.py` and `commuting.py` hold 78 of the original parameters fixed.
Their exact architectural embeddings are checked against the existing core.
They are restricted counterexamples, not unrestricted network experiments.
The curvature construction searches 120 declared combinations; all attempted
cases are recorded by `curvature_search_full.py`, including non-witnesses.
Multiple ODE resolutions and an exact first integral check the selected case.

All scripts refuse to overwrite their requested outputs. Historical versions
and their matching source hashes are retained in the study archive. The final
reports distinguish exact algebra, numerical witnesses, ordinary prior-work
reductions, and unresolved structural research questions.
