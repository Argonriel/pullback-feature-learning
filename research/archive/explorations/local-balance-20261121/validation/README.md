# Independent confirmation of richer exact-quota allocation

The richer balanced rank selector improves the previous one-of-four selector substantially, and improves its new even-action control slightly. It still loses to global confidence on all 16 fresh models. This is a result about expected logistic loss; it is not evidence that the balanced selector has higher classification accuracy than its even control.

All 16 reserved PB500 models trained successfully. Every policy was fixed and calibrated without test labels. Each model was evaluated on 65,536 independent IID inputs, giving 1,048,576 test inputs in total. The inference unit is the trained model: intervals use 16 paired model clusters, not a million independent model replications.

| Policy | Mean logistic risk | Deployment forwards/input |
|---|---:|---:|
| Raw | 0.110633 | 1.00 |
| Previous group-balanced | 0.107371 | 1.25 |
| New rank-balanced | 0.101001 | 1.25 |
| New rank-even | 0.101139 | 1.25 |
| New rank-unbalanced | 0.103899 | 1.25 |
| IID-calibrated folded gate | 0.101016 | 1.25 |
| Global confidence, half-turn | 0.098669 | 1.25 |
| Confidence, calibrated radius angles | 0.096782 | 1.25 |
| Full half-turn | 0.100690 | 2.00 |

The primary differences are candidate minus reference; negative is better. New balanced minus new even is -0.0001379, with a 97.5% paired-model bootstrap interval [-0.0002351, -0.0000694], and 16/16 wins. New balanced minus global confidence is +0.0023316, interval [+0.0017376, +0.0030433], and 0/16 wins. These two intervals use the prespecified two-comparison adjustment. Ordinary 95% intervals are also retained in `contrasts.csv`.

The descriptive balanced-minus-old-group contrast is -0.0063699 (95% interval [-0.0081642, -0.0048842], 16 wins). The comparison with the strong folded gate is unresolved: -0.0000152 (95% interval [-0.0000999, +0.0000484], 8 wins). New balanced is worse than full half-turn by 0.0003107 on average, although it uses fewer forwards. Full half-turn is a reference policy, not a universal risk ceiling.

Mean expected classification error is 1.4033% for new balanced, 1.4027% for new even, 1.4084% for confidence, and 1.3032% for confidence plus radius angles. In particular, the small logistic-loss improvement of balanced over even does not translate into a lower mean classification error in this sample. Independently sampled-label risk and error checks are retained separately.

## Calibration and implementation costs

| Calibration | Network forwards/model | Additional coordinate-only inputs | Mean fitting wall time |
|---|---:|---:|---:|
| Previous continuous fields | 66,048 | 0 | 0.0275 s |
| Three new rank policies, shared fit | 66,048 | 131,072 | 0.7944 s |
| IID radius/folded gates, shared fit | 66,048 | 0 | 0.0196 s |
| Radius-specific angles | 8,192 | 0 | 0.0140 s |

The 64 radii used for the angle table are independent radius clusters; their 128 angular evaluations each are not 8,192 independent observations. The new rank policies additionally estimate a coordinate-only score distribution from 131,072 draws. Those draws cost no model forwards, but their computation is included in fitting time. Confidence needs no model-specific calibration. These controls have explicitly different calibration budgets.

All 61,440 scored batches pass actual prediction-call and quota checks. Every limited policy uses [256, 64] forward examples in this scoring implementation. Raw uses [256]; full uses [256, 256]. A separate equivalent full timing path uses one 512-example call and passes output checks. The root also prespecified stronger fused implementations for input-only rank and invariant gates, which can use a single 320-example call. The root's isolated `root_audit/cpu_serving.py` measurements are the definitive CPU efficiency comparison; they avoid treating avoidable sequential calls as a policy limitation.

`latency.csv` and `costs.csv` preserve the original scoring-time measurements and derived break-even calculations. Those timings occurred in the main scoring process while other research work could run. They are not promoted over the later isolated implementation audit. No reduction in forward examples alone is called a wall-time speedup. Mathematical local-bound verification, independent certificates and post-fit diagnostics are separate offline costs.

## Scope and records

The original-ring observation law and half-turn symmetry are known. The true class posterior is used only in evaluation. Three additional frozen posterior-world diagnostics (constant one-half, increasing radial sigmoid, decreasing radial sigmoid) show sensitivity to alternative radial label laws; they are not sharp worst-case extrema or identification of an unknown posterior. The separate bounds and certificate workstreams address those questions.

Selection and validation sources were frozen before root approval at 09:39:19 UTC. Training began at 09:39:24. All policy tables were committed at 09:39:58, before test scoring began at 09:41:38. Scoring ended at 09:42:06. The separate diagnostic specification was frozen at 09:41:46, after scoring had launched but before its author opened fresh tables or outcomes. The scientific policy itself was never changed after its pre-training freeze. No training or deployment failure was dropped.

Key artifacts are `freeze.json`, `ROOT_APPROVAL.json`, `states/`, `policies/`, `policy_commit.json`, `predictions/`, `per_seed.csv`, `summary.csv`, `contrasts.csv`, `deployment_audit.csv.gz`, `per_batch.csv.gz`, `samples.csv`, and `calibration.csv`. Frozen source paths are relative to the study root. The archive vendors the model source, previous gate/angle code and new selector; the completed confirmation has no dependency on the previous study's absolute location. Old-only smoke records document development checks and are not inputs to new-cohort scoring.

To reproduce in a fresh output copy, retain the same relative `source/`, `legacy/`, `reference/`, `selection/`, and `validation/` layout. Generate new freezes and an approval record before training; output stages refuse to overwrite completed artifacts. Commands are:

```text
python validation/study.py freeze
python validation/train.py train
python validation/study.py calibrate
python validation/study.py score
python validation/study.py analyze
python validation/make_figure.py
```

The archived approval is evidence for this completed run. A fresh reproduction must link its own approval to its new freeze hash. Windows-generated manifest paths use Windows separators; verification on another platform should normalize those recorded path separators without modifying the scientific sources or parameter arrays.
