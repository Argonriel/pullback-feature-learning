# Independent validation: empirical quality and invariant-posterior sensitivity

The confidence comparison changes on the new harmonic task. Rank-balanced loses to confidence on every fresh rings model, but beats it on 14 of 16 harmonic models. It has a small mean loss advantage over rank-even in both cohorts. The strong folded symmetric gate remains essential: the rings comparison is unresolved, and folded wins on all 16 harmonic models. No selector settings were changed after the freezes.

The primary sample contains 16 independently trained PB models in each of two cohorts, with 65,536 independent IID test inputs per pair. The 16 transfer GD states are underfitted optimizer context, not a third selector cohort. Training data, architecture, observation law and Bayes difficulty differ between the two tasks, so their absolute losses are not an equal-difficulty comparison. The four-dimensional transfer action is R(alpha) on the first pair and R(2alpha) on the second; its exact half-turn flips only the first pair. Its quotient remains scalar.

## Prespecified primary family

Differences are rank-balanced minus the reference; negative favors rank-balanced. Intervals are 98.75% paired-model percentile bootstrap intervals for the four prespecified comparisons. Ordinary 95% intervals are also retained. These model-cluster bootstrap intervals are empirical uncertainty summaries, distinct from the separate finite-sample certificate.

| Cohort | Reference | Mean loss difference | 98.75% interval | Balanced wins |
|---|---|---:|---|---:|
| rings | Confidence + half-turn | +0.0024545 | [+0.0015922, +0.0034503] | 0/16 |
| rings | Rank-even | -0.0001058 | [-0.0001770, -0.0000498] | 15/16 |
| transfer | Confidence + half-turn | -0.0139346 | [-0.0233601, -0.0062219] | 14/16 |
| transfer | Rank-even | -0.0009161 | [-0.0024945, -0.0001931] | 14/16 |

The folded comparisons are descriptive:

- rings: balanced minus folded = +0.0000331, 95% interval [-0.0000294, +0.0001015], 6/16 balanced wins.
- transfer: balanced minus folded = +0.0072755, 95% interval [+0.0056067, +0.0091540], 0/16 balanced wins.

## Loss and classification metrics

Each entry is a mean across 16 trained models. Expected loss uses the evaluator-only analytic posterior. The sampled-label loss and error use a separate Bernoulli-label stream. Logistic loss and classification error are different outcomes; a loss advantage is not automatically an accuracy advantage.

### Original rings: primary PB

| Policy | Expected logistic loss | Sampled-label loss | Expected error | Sampled error |
|---|---:|---:|---:|---:|
| Raw | 0.110814 | 0.110777 | 2.077% | 2.072% |
| Full half-turn | 0.100237 | 0.100274 | 1.370% | 1.369% |
| Confidence + half-turn | 0.098049 | 0.098060 | 1.371% | 1.369% |
| Folded symmetric gate | 0.100471 | 0.100512 | 1.371% | 1.369% |
| Rank-balanced | 0.100504 | 0.100541 | 1.373% | 1.371% |
| Rank-even | 0.100610 | 0.100651 | 1.372% | 1.371% |
| Rank-unbalanced | 0.102757 | 0.102806 | 1.480% | 1.478% |
| Confidence + quotient angle | 0.096677 | 0.096706 | 1.277% | 1.276% |
| Full quotient angle | 0.099175 | 0.099285 | 1.276% | 1.276% |

### Two-harmonic task: primary PB

| Policy | Expected logistic loss | Sampled-label loss | Expected error | Sampled error |
|---|---:|---:|---:|---:|
| Raw | 0.393727 | 0.393084 | 15.660% | 15.612% |
| Full half-turn | 0.335819 | 0.335644 | 13.628% | 13.622% |
| Confidence + half-turn | 0.369954 | 0.369516 | 13.803% | 13.784% |
| Folded symmetric gate | 0.348744 | 0.348574 | 14.289% | 14.282% |
| Rank-balanced | 0.356020 | 0.355545 | 14.766% | 14.738% |
| Rank-even | 0.356936 | 0.356490 | 14.789% | 14.764% |
| Rank-unbalanced | 0.363493 | 0.362958 | 15.062% | 15.033% |
| Confidence + quotient angle | 0.361917 | 0.361702 | 13.287% | 13.285% |
| Full quotient angle | 0.320166 | 0.320322 | 12.887% | 12.905% |

On the harmonic task, the loss advantage over confidence is explicitly **not an accuracy advantage**: rank-balanced has 14.766% expected classification error versus confidence at 13.803%. The folded gate also has lower logistic loss but higher error than confidence. Confidence plus quotient angles has the best mean classification error among the limited policies in this cohort. These are different quality objectives; the primary claims concern logistic loss.

### Transfer GD: optimizer context only

| Policy | Expected logistic loss | Sampled-label loss | Expected error | Sampled error |
|---|---:|---:|---:|---:|
| Raw | 0.667119 | 0.666990 | 39.095% | 39.043% |
| Full half-turn | 0.661693 | 0.661638 | 37.398% | 37.385% |

Both optimizers use 500 updates, but their work per update is different. Rates were selected from the bounded four-seed supervised pilot. These GD/PB results are not a matched-compute optimizer comparison. The task Bayes classification error is 10.9375%, and these GD states remain visibly underfitted.

## Four invariant-posterior sensitivity scenarios

Entries below are mean raw-minus-policy logistic loss; positive means an improvement. The model, coordinates, policy, quota and selected views are held fixed while only the hypothetical invariant posterior changes. Half uses p=1/2; rising uses normalized quotient u; falling uses 1-u; threshold uses 1[u>=1/2]. Rings use u=clip((r-.25)/1.25,0,1); transfer uses u=r-.4. These are four prespecified scenarios, not sharp adversarial extrema or a proof over every invariant posterior. Per-model signs and 95% model-bootstrap intervals are in world_gains.csv.

### rings

| World | Confidence | Rank-balanced | Rank-even | Folded gate | Full half-turn |
|---|---:|---:|---:|---:|---:|
| Half | -0.005245 | +0.010390 | +0.010344 | +0.010484 | +0.010745 |
| Rising | +0.003491 | +0.010375 | +0.010305 | +0.010423 | +0.010673 |
| Falling | -0.013980 | +0.010405 | +0.010382 | +0.010546 | +0.010818 |
| Threshold | +0.013089 | +0.010269 | +0.010162 | +0.010305 | +0.010535 |

### transfer

| World | Confidence | Rank-balanced | Rank-even | Folded gate | Full half-turn |
|---|---:|---:|---:|---:|---:|
| Half | -0.014695 | +0.037677 | +0.036772 | +0.045018 | +0.057492 |
| Rising | +0.005562 | +0.037699 | +0.036801 | +0.044998 | +0.057757 |
| Falling | -0.034953 | +0.037656 | +0.036743 | +0.045039 | +0.057226 |
| Threshold | +0.037263 | +0.037696 | +0.036743 | +0.044864 | +0.057934 |

Confidence has a negative mean gain in the half and falling worlds in both tasks. In falling worlds it harms all 16 models in each cohort. Rank-balanced and folded improve all 16 models in every displayed scenario. The folded gate has an exact half-turn symmetry property; near-flat finite-sample gains are not themselves a proof of exact identification. The new rank certificate is evaluated separately on independent probes.

## Calibration and actual prediction work

| Cohort | Shared fit | Forwards/model | Additional coordinate-only quotient draws/model | Mean fit seconds |
|---|---|---:|---:|---:|
| rings | rank | 66,048 | 131,072 | 0.6820 |
| rings | folded | 66,048 | 0 | 0.0162 |
| rings | angles | 8,192 | 0 | 0.0090 |
| transfer | rank | 66,048 | 131,072 | 0.7188 |
| transfer | folded | 66,048 | 0 | 0.0175 |
| transfer | angles | 8,192 | 0 | 0.0100 |

Across 32 primary models, these independent fits total 4,489,216 network forwards plus 4,194,304 coordinate-only quotient draws. The rank fit is shared by balanced/even/unbalanced policies; folded scores and angles are separate fits. The angle fit has 64 independent quotient clusters, not 8,192 independent observations. Confidence has zero model-specific calibration cost; GD context has no policy fit.

All 81,920 scored batches pass actual call and quota checks. Raw uses [256], limited policies [256,64], and full references [256,256]. The 320 model-policy scoring rows used 29,360,128 forward examples as research evaluation work. There are 2,097,152 unique primary test inputs; transfer GD reuses its PB cohort's inputs. Certificate, model-training and local-bound costs are separate. No new latency or GPU claim is made in this round.

## Independent analysis audit and chronology

The post-run audit reconstructs all 320 outcome rows from saved predictions using an algebraically distinct logistic-loss expression, then recomputes all 136 contrasts with bootstrap multiplicity weights rather than indexed row averages. Maximum outcome discrepancy is 8.88e-16; maximum bootstrap discrepancy is 3.32e-16; all win counts match. All 160 mean/interval summaries also pass. The audit source and outputs are outside the frozen scientific files.

Root independently reconstructed all 320 risks, checked all 81,920 ledgers, reproduced nine complete training trajectories exactly, and replayed 54 full-input deployments. The frozen design and preserved raw files make these checks reviewable. No failed training or evaluation was silently excluded.

- Validation freeze: 2026-11-21T10:14:11.955959+00:00.
- Root approval: 2026-11-21T10:15:56.872195+00:00.
- Training start: 2026-11-21T10:16:16.457297+00:00.
- Training commit: 2026-11-21T10:16:40.205740+00:00.
- Policy commit: 2026-11-21T10:17:30.961775+00:00.
- Scoring start: 2026-11-21T10:18:29.606074+00:00.
- Scoring complete: 2026-11-21T10:19:01.030964+00:00.

## Files and reproduction

Frozen scientific sources are study.py, adapters.py and PROTOCOL.md plus the dependencies listed in freeze.json. States, policy tables, predictions and manifests use relative paths. The results report and audit code are separate from method selection. The complete metrics are per_model.csv, summary.csv, contrasts.csv and world_gains.csv; analysis_audit.json and its CSVs document the independent arithmetic audit.

For a fresh output copy with the same relative source/reference/legacy/transfer dependencies, run the stages below. Fresh training requires an approval record linked to its new freeze hash; stages refuse to overwrite completed outputs. The archived approval documents this completed run.

```text
python validation/study.py freeze
python validation/study.py train
python validation/study.py calibrate
python validation/study.py score
python validation/study.py analyze
python validation/audit_analysis.py
python validation/write_results.py
```
