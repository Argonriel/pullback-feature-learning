# Independent second-order follow-up

The first confirmation cohort motivated this predeclared second-order alternative. Sixteen wholly new initial radial/angular pairs were evaluated after a new code/protocol freeze and saved prediction commit. No original result, seed or target was replaced. All failures and all competitors remain visible.

## Paired optimizer-contrast accuracy

|Task|Drop|Metric|Pairs|First MAE|Second MAE|Old MAE|Second sign|Second-minus-first benefit95%CI|
|---|---:|---|---:|---:|---:|---:|---:|---|
|radial|1e-05|fit|16|2.021e-10|1.081e-13|8.081e-08|16/16|[9.935e-11,3.189e-10]|
|radial|1e-05|nuisance|16|4.219e-10|1.072e-13|1.455e-06|16/16|[2.421e-10,6.247e-10]|
|radial|1e-05|view_value|16|2.051e-10|5.012e-14|7.152e-07|16/16|[1.202e-10,3.017e-10]|
|radial|0.02|fit|13|0.0004702|0.0002639|0.0007209|11/13|[9.157e-05,0.0003292]|
|radial|0.02|nuisance|13|0.0009381|0.000446|0.002602|12/13|[0.000212,0.0008306]|
|radial|0.02|view_value|13|0.0004664|0.0002192|0.001289|12/13|[0.0001105,0.0004163]|
|angular|1e-05|fit|16|1.184e-09|1.304e-12|9.807e-07|16/16|[6.707e-10,1.883e-09]|
|angular|1e-05|nuisance|16|1.658e-09|1.897e-12|1.743e-06|16/16|[9.127e-10,2.564e-09]|
|angular|1e-05|view_value|16|1.658e-09|1.897e-12|1.743e-06|16/16|[9.283e-10,2.579e-09]|
|angular|0.02|fit|7|0.001667|0.004393|0.007116|6/7|[-0.005156,-0.0007675]|
|angular|0.02|nuisance|7|0.003382|0.005356|0.005967|6/7|[-0.004521,0.0001616]|
|angular|0.02|view_value|7|0.003382|0.005356|0.005967|6/7|[-0.004617,0.0001825]|

Coverage: 116/128 roots reached target, 12 retained cap failures. Maximum coarse/fine second-derivative relative discrepancies were 9.771e-07 for training loss and 0.005338 for outcomes. No discrepancy-triggered tuning or outcome-driven refit was performed.

The primary follow-up endpoint is radial initial fitting contrast at.02. This is a small prospective validation, conditional on paired root reachability. Angular fitting retains informative angle structure; its nuisance and view-value endpoints are identical pi-subgroup quantities. The second-order model is population-informed, and cannot be interpreted as a training-only deployable predictor.

Reproduce in an empty copy with the parent frozen study and old6 data: followup.py freeze; followup.py generate; followup.py score; followup.py summarize. Raw finite differences, states, roots, predictions, outcomes, hashes and bootstrap summaries are supplied.