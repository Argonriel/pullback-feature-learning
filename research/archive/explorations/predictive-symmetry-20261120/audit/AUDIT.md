# Independent audit of predictive updates and rotation selection

The independent checks found no numerical inconsistency in the frozen results. They support the reported local predictions and the measured benefit of rotation selection within the tested settings. They do not establish that a local derivative predicts arbitrary large updates, that every requested loss target is attainable, or that unlabeled selection works without the stated joint-distribution symmetry.

The executable auditor is [audit.py](audit.py), supported by [experiment_checks.py](experiment_checks.py). Full numerical records are in [audit.json](audit.json). [MATHEMATICAL_REVIEW.md](MATHEMATICAL_REVIEW.md) records the independent mathematical scrutiny. Run the auditor with the project Python environment after both investigators' result files exist. It reads their outputs and writes only its own audit files. NumPy and threadpoolctl suffice. The bundled numerical core is preferred for portability.

## Actual-loss-matched predictions

- All 3,456 metric prediction rows were preserved in the outcome table. They represent 1,152 requested endpoint cases: 950 successful endpoints and 202 displacement-cap failures. The failed exploratory entropy run also remains in `old_attempt01_nonfinite_entropy`.
- Recomputing the training loss from every saved successful parameter state and its saved training inputs/labels gave maximum error **9.65e-16 nats** against the requested actual decrease. Every saved single-step endpoint exactly equaled its saved initial state plus its recorded step size times the independently reconstructed direction. All saved state hashes matched.
- The frozen source, model core, predictions, state bank and roots matched their manifests. Recorded prediction-commit timestamps preceded recorded scoring-start timestamps, and the scoring code reads the frozen predictions. This is an internally reproducible audit trail, not an independently hosted preregistration service.
- Independently reconstructed component derivatives for 24 task/stage/method states agreed with stored derivatives to **1.60e-16**. Central finite differences differed by at most **5.55e-11 per unit parameter direction**. The normalized loss-clock formula agreed to **6.87e-20**.
- Twelve representative trained tiny-drop comparisons recomputed both endpoints on a common 192-by-256 grid. They agreed with reported changes to **7.22e-16**, with no component-sign disagreements. The maximum change between common coarse and fine grids was **1.52e-11**. This supports the tiny-drop numerical calculations in the audited sample; it is not an a priori quadrature error bound for every possible model.
- All **420 summary rows** had their MAE, RMSE and mean advantages independently reconstructed from raw outcomes, with zero discrepancy. Seed-level grouping was preserved; the repeated methods, metrics and horizons are not independent training draws.

The radial/angular pair uses identical sampled inputs and initialization in every seed; all 32 paired checks passed. Both laws have class probability 0.5 and the same distribution of posterior values. Independent dense integration gave Bayes log loss **0.38955080927034924** for both and Bayes classification error approximately **0.1638867594**. The angular posterior was invariant to a half-turn to 1.67e-15, while a quarter-turn changed it by as much as 0.90515. Its fitting term therefore preserves informative angle dependence; it is not the radial fitting term. Equal Bayes difficulty does not establish equal approximation difficulty for this network.

The mathematical first-order prediction is delta times the outcome directional derivative divided by the negative training-loss directional derivative, all at the common pre-state. The realized-step tangent additionally uses the training-only solved step or multistep displacement. Its stronger information must remain explicit. The geometric search finds the first detected crossing within its cap; it does not prove that no earlier narrow crossing exists between sampled step sizes. Failures establish failure of that search under its cap, not global impossibility of attaining the target.

## Independent iid confirmation supplement

All 96 requested trained-stage 0.02-drop branches were preserved: **51 succeeded and 45 hit the displacement cap**. All 192 regenerated input/label hashes matched. The independently recalculated iid risk changes and standard errors differed by at most **3.47e-18**, and 21 complete paired method contrasts differed by at most **4.34e-19**. Synthetic paired evaluation inputs also matched in all 16 seeds.

| Task | Complete pairs | Loss-clock contrast MAE | Zero contrast MAE | Correct predicted signs |
|---|---:|---:|---:|---:|
| Original rings | 12/16 | 0.0003392622 | 0.0018091089 | 12/12 |
| Matched radial | 0/16 | Not estimable | Not estimable | Not estimable |
| Matched angular | 9/16 | 0.0018587751 | 0.0057880459 | 9/9 |

The corresponding old-pooled contrast MAEs were 0.0018091089 for rings and 0.0067657836 for angular. The complete pairs are conditional on both methods reaching the requested decrease; they do not describe all 16 seeds. In particular, no trained radial finite-drop paired transfer result is available.

Fifty of the 51 successful iid risk changes were within 1.96 estimated Monte Carlo standard errors of deterministic population integration; the largest standardized difference was 3.0662. These correlated checks do not require every draw to fall within a pointwise 95% interval. The final iid model comparisons use shared inputs within seed, with separately generated labels. Their known-posterior scores are evaluation-only.

## Fourier and label-free rotation selection

All 48 fresh models' 18 calibration policies were independently reconstructed from saved raw orbit logits, using a full FFT and direct softplus Jensen gaps. All selected angle bins agreed. The maximum spectral-power error was **3.55e-15**, maximum selector-curve error **1.78e-15**, and maximum predicted-gain error **8.88e-16**. Representative raw calibration logits reproduced exactly.

Thirty representative population risks were independently integrated using a different angular phase, 768 angles, and 80 radial Gauss nodes per split interval. Their maximum discrepancy was **3.17e-11 nats**. Orbit mean preservation held to **2.66e-15**. Twenty-four independently regenerated iid risks agreed to **1.39e-17**. Every source/protocol/model/selector hash matched its frozen record, and the recorded training, initialization, calibration and iid-test seed ranges were disjoint.

| Training method | Half-turn risk | Exact global-angle risk | Exact radius-dependent risk | Radius-dependent gain over half-turn | Wins over half-turn |
|---|---:|---:|---:|---:|---:|
| GD | 0.10300924 | 0.09894168 | 0.09832661 | 0.00468262 | 15/16 |
| Pullback | 0.10611029 | 0.10277559 | 0.10232546 | 0.00378483 | 16/16 |
| Hybrid | 0.04788443 | 0.04250910 | 0.04217487 | 0.00570956 | 16/16 |

These means were recalculated from raw seed-level risks. An independent paired-seed bootstrap gave 95% intervals for the radius-dependent gain over half-turn of [0.00112630, 0.00957151], [0.00225887, 0.00561926], and [0.00330875, 0.00843941], respectively. These are pointwise intervals from the auditor's own resampling seed, so endpoints can differ slightly from the investigators' bootstrap. The GD loss is retained rather than removed. Global exact selection improved 8/16, 15/16 and 11/16 models; remaining cases tied the half-turn. No fresh result was used by the audited calibration code to choose its angle.

Each calibration used **65,536 network example-forwards** (512 radii times 128 angles). An observation-only wrapper around six representative real deployments counted exactly two calls of 257 examples each. Every universal two-view policy therefore uses two model evaluations per input; the result is improved allocation of the second view's angle at equal inference evaluation count, not a reduction in count relative to the half-turn. Offline orbit calibration, Fourier computation and sorting/binning costs remain additional. All truncations and unweighted proxies are retained; the independent audit does not declare the local quadratic Fourier approximation an exact risk identity.

## Limits of the guarantees

The two-view Fourier multiplier is exact, but its conversion to logistic risk through curvature is a local approximation. Unlabeled Jensen gain identifies expected logistic-risk reduction only under the relevant joint input-label invariance and admissible selection rule. Radius-dependent rotation preserves the radial law used here. General angle-dependent rules and the angular-label task require their own symmetry check.

The auditor verified this boundary with the same unlabeled predictor and input law but two invariant constant-label posteriors: an orientation-sensitive gate had gains **+0.7168904** and **-0.2831096**. Thus unlabeled observations alone cannot identify that gate's risk change. Synthetic numerical controls confirmed the exact normalized finite-root derivative and Fourier identities; one of 12 synthetic ray searches did not bracket a target and remains in the audit record. These controls are mathematical diagnostics, not additional independent research replications.

Direction 1 and direction 2 deliberately share fresh ring seeds and their common initial/warm-up states. They test different questions on related trained models and must not be counted as two independent training replications. The audit checked representative derivatives and population integrations, all saved loss matches and calibration choices, and all reported direction-1 aggregate means. It did not independently rerun every full training trajectory, certify global line-search uniqueness, prove finite quadrature error bounds, or validate transfer to new architectures and datasets.

## Separately frozen second-order follow-up

After the initial results exposed finite-drop limitations, a new protocol tested the second-order loss-clock expansion on a third, disjoint 16-seed synthetic cohort. The source, parent source, old discovery outcomes, protocol, predictions and states matched their frozen hashes; the recorded ordering was freeze, prediction commit, scoring start. All 384 metric rows remained, representing 128 endpoint requests, 116 successes and 12 cap failures. Recomputed actual training drops differed from their targets by at most 1.57e-16.

The auditor independently propagated the first and second directional derivatives analytically through both tanh residual blocks for all 64 normalized directions. This uses no finite-difference curvature estimate. The saved second-order formula was exact, and replacing its finite-difference coefficients with the independently derived analytic coefficients changed no prediction by more than **9.80e-9**. Thus the large angular deterioration below is not explained by numerical noise in the finite differences. All 60 summary means were reconstructed with zero discrepancy.

For the prespecified radial initial fitting-loss contrast at a 0.02 actual drop, 13 complete pairs gave first-order MAE **0.0004702237** and second-order MAE **0.0002639486**. The recorded pointwise paired bootstrap interval for the reduction in MAE was [0.0000915747, 0.0003291929]. On the angular fitting contrast, only seven pairs were complete; MAE increased from **0.0016673196** to **0.0043928632**, with a recorded reduction interval [-0.0051563298, -0.0007675225]. The radial improvement and angular failure must both be retained. This follow-up was motivated by earlier evidence and is not an additional analysis of the original frozen confirmation cohort; its success is restricted to the prespecified radial endpoint and reachable pairs.

## Exact finite-label balance sensitivity

Another separately frozen, disjoint 16-seed cohort enforced exact finite training-label balance by adding an antithetic companion to each of 128 iid latent triples. Independent regeneration reproduced all 32 task-specific training datasets and confirmed exactly 128 positive and 128 negative labels in each. The companions' posterior probabilities summed to one within 8.88e-16, and their labels were exact negatives. Shifting uniform radius rank and uniform angle preserves each observation's input marginal, while complementing its independent uniform label draw preserves the corresponding conditional Bernoulli law. The paired tasks still share inputs and initialization.

There were 107 successful roots among 128 requests and 21 cap failures. The maximum actual-drop error was 2.05e-16. All frozen hashes and the freeze/commit/score ordering passed. Independent analytic curvature checked all 64 directions; using it in place of finite differences changed no forecast by more than 9.44e-9. All 60 summary means reproduced exactly. The iid-old pooled comparator was explicitly marked out of design in every raw row.

The earlier radial fitting improvement did **not** replicate in this sensitivity cohort: 10 complete substantial-drop pairs gave first-order MAE **0.0003130650** and second-order MAE **0.0003132467**. The recorded interval for first-minus-second MAE was [-0.0001093450, 0.0000844788]. The angular substantial-drop comparison had only two complete pairs, which is inadequate for a broad conclusion even if a two-point bootstrap interval excludes zero.

This is a change in training dependence as well as in finite label balance: each training dataset contains 128 independent antithetic pairs, not 256 independent observations. The sensitivity result limits robustness of the curvature correction. It does not isolate label balancing as the cause of the difference from iid training.

## Forecasting a post-update view at PB100

The bridge from the original trained states reused the main ring cohort, with new calibration orbits. Its 32 requested branches yielded 28 successes (12 GD and 16 pullback) and four retained cap failures. All frozen source, parent prediction/state and selector hashes matched. Independent training-loss reconstruction had maximum error 7.29e-17, and the normalized forecast displacement agreed to 2.78e-17.

The auditor independently reconstructed 15 representative selector policies, using the full representation Jacobian for their directional logits; maximum JVP error was 2.22e-16 and maximum gain-forecast-curve error was 3.47e-18. All 12 summary means and all selector-to-population-gain mappings reproduced exactly. **Every stale, prospective, realized-step and post-update-recalibrated selector chose the half-turn**, so every improvement over stale selection was exactly zero. More accurate gain-value predictions did not change the inference decision in this setting.

## Separately frozen PB500 decision follow-up

The late bridge preserved its original 0.02-drop old-seed pilot, then chose 0.005 using old training-loss feasibility only: it was the larger candidate with both methods reaching the target in at least four of six old seeds. The independently checked complete-pair counts were five of six for 0.005 and six of six for 0.001. The selected protocol and separate source copy were frozen before the new 621001–621016 training cohort; the main PB100 source stayed unchanged. This development choice and its failed earlier setting remain visible.

The new cohort had 24 reachable branches, comprising eight GD and 16 pullback branches, plus eight cap failures. Every actual drop was recomputed, with maximum error 2.35e-17. All 16 training draws and initializations were independently regenerated; two complete 500-update pullback states were independently retrained and reproduced exactly. Frozen source, parent and selector hashes matched. Fifteen representative policies were independently rebuilt, with maximum JVP error 5.00e-16, gain-curve error 3.47e-18, and no angle-choice mismatch. All 12 summary means and every selected-gain lookup reproduced exactly.

Independent integration of 15 risks initially differed by up to 4.34e-7 on the auditor's coarser grid. Refining that alternative construction to 384 radial nodes and 1,536 differently phased angles reduced the maximum discrepancy to **1.86e-10**; both coarse and fine diagnostics remain in the audit record. The investigators' stored risks were not changed.

The prospective loss-clock logit forecast changed the angle in only two of eight reachable GD branches and one of 16 pullback branches. Mean improvement over keeping the stale angle was **0.0000135551 nats** for GD and **0.0000100267** for pullback; the recorded pointwise bootstrap intervals were [0, 0.0000401307] and [0, 0.0000300800]. The first-order gain selector made the same choices. These sparse, small improvements do not establish a broadly useful decision advantage from derivative forecasting. They also do not overturn the PB100 null result. Forecast calibration uses the pre-state orbit predictions plus directional-logit computation; post-update recalibration is a stronger comparator that incurs new offline model evaluations.
