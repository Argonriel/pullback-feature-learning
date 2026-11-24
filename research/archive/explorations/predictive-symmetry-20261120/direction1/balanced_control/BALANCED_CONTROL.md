# Exact finite-label balance control

This sensitivity control uses sixteen new datasets with exactly128 positive and128 negative training labels in each task. Shared inputs and initialization are preserved. Each dataset contains128 independent antithetic pairs, so its256 observations are correlated within pairs. Every observation has the original correct input/posterior marginal. The first/second-order formulas, targets and root cap are unchanged. A separate freeze and physical prediction commit precede endpoint scoring. Original iid-old6 pooled slopes are an out-of-design comparator, explicitly labeled in raw rows.

## Paired optimizer-contrast accuracy

|Task|Drop|Metric|Pairs|First MAE|Second MAE|Old iid MAE (out of design)|Second sign|Second-minus-first benefit95%CI|
|---|---:|---|---:|---:|---:|---:|---:|---|
|radial|1e-05|fit|16|1.8e-10|4.952e-14|1.067e-07|16/16|[7.878e-11,3.011e-10]|
|radial|1e-05|nuisance|16|3.649e-10|1.046e-13|1.186e-06|16/16|[1.663e-10,6.524e-10]|
|radial|1e-05|view_value|16|1.797e-10|5.136e-14|5.717e-07|16/16|[8.344e-11,3.217e-10]|
|radial|0.02|fit|10|0.0003131|0.0003132|0.0004787|7/10|[-0.0001093,8.448e-05]|
|radial|0.02|nuisance|10|0.0008763|0.0004278|0.00295|10/10|[0.0002013,0.0007246]|
|radial|0.02|view_value|10|0.0004293|0.0002078|0.001457|10/10|[9.965e-05,0.0003586]|
|angular|1e-05|fit|16|1.623e-09|1.408e-12|9.953e-07|16/16|[1.042e-09,2.24e-09]|
|angular|1e-05|nuisance|16|1.489e-09|1.635e-12|1.819e-06|16/16|[9.377e-10,2.099e-09]|
|angular|1e-05|view_value|16|1.489e-09|1.635e-12|1.819e-06|16/16|[9.28e-10,2.114e-09]|
|angular|0.02|fit|2|0.003077|0.002544|0.001856|1/2|[0.0004488,0.0006175]|
|angular|0.02|nuisance|2|0.002883|0.002037|0.005302|2/2|[0.0003427,0.001349]|
|angular|0.02|view_value|2|0.002883|0.002037|0.005302|2/2|[0.0003427,0.001349]|

Coverage: 107/128 roots reached target, 21 retained cap failures. Maximum coarse/fine second-derivative relative discrepancies were 1.321e-06 for training loss and 0.0008032 for outcomes. No discrepancy-triggered tuning or outcome-driven refit was performed.

The primary balance-control endpoint is radial initial fitting contrast at.02. This is a small prospective validation, conditional on paired root reachability. Angular fitting retains informative angle structure; its nuisance and view-value endpoints are identical pi-subgroup quantities. The second-order model is population-informed, and cannot be interpreted as a training-only deployable predictor.

Reproduce in an empty copy with the parent frozen study and old6 data: balanced.py freeze; balanced.py generate; balanced.py score; balanced.py summarize. Raw finite differences, states, roots, predictions, outcomes, hashes and bootstrap summaries are supplied.

Only two angular substantial-drop pairs were jointly reachable. Their percentile intervals are descriptive of these two successes and do not establish robust transfer. The exact-balance primary radial fitting comparison shows no reproducible second-order improvement; the iid follow-up improvement must not be generalized across sampling designs.
