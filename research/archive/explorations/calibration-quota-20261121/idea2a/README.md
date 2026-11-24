# How much calibration does a new model need

On 16 fresh model pairs, calibrating on 64 independent radii (8,192 forward evaluations) recovered 93.9% of pullback's additional gain over its historical default, and 96.2% for the hybrid. GD crossed the proposed 90% target at 16 radii (2,048 calls), but its bootstrap threshold ranged from 2,048 to 8,192. Every policy still uses exactly two network calls per deployed input.

The historical angle did not establish an advantage over the simpler half-turn on this fresh cohort. Pullback's transferred 143.4375° angle had slightly higher mean risk than 180°, with an interval spanning both directions. This contrasts with the motivating retrospective result and prevents treating a shared historical angle as a reliably better free default.

## Prospective separation

The defaults were fitted only from the 16 archived models per method (training 121001–121016, initialization 122001–122016). Each historical model used 512 unlabeled radii, expanded into 128-angle orbits with calibration seed 603000 plus pair index. Averaging the exact unlabeled Jensen gain curves selected GD 180°, pullback 143.4375°, and hybrid 163.125°. No inspected confirmation risk selected these angles.

The motivating pilot was explicitly reproduced on the **already-inspected** 611/612 cohort: pullback's shared angle removed 62.7022% of angular excess loss, versus 54.8726% for the half-turn. These are retrospective numbers, preserved in `history/retrospective_pilot.csv`; they are not the prospective evidence below.

The code, defaults, budgets and shrinkage rule were frozen before training the new cohort with seeds 721001–721016 and initialization seeds 722001–722016. For each pair, we saved initial weights, PB100, GD500, PB500, the PB100→GD400 hybrid, and all training inputs/labels. Idea 2B reuses these model states with independent calibration and test draws; it is not another independent training cohort.

Each new model received 20 independent calibration pools of 512 radii. The same pool was shared across the three methods within a pair. Prefixes of 1, 4, 16, 64 and 512 radii cost 128, 512, 2,048, 8,192 and 65,536 forward evaluations. All 128 angles at a radius form one dependent orbit: the independent calibration units are **radii**, not individual forward evaluations. Every smaller-budget choice uses only its allowed prefix. Twenty pools quantify calibration randomness; they do not create 320 independent trained models.

The primary selector maximizes its own prefix's exact Jensen curve over the same 64 candidate separations. The prespecified secondary selector mixes that curve with the historical mean using 16 equivalent prior radii: `(sum of new radius curves + 16 × historical curve)/(new radius count + 16)`. This fixed strength was not tuned on the fresh outcomes.

All 11,520 decisions, raw orbit logits and calibration hashes were committed before evaluating any fresh population risk or test labels. The main statistical script was also recorded before outcome evaluation. A separately recorded pre-outcome amendment made bootstrap threshold interval endpoints use tested, discrete budgets and cleaned descriptive wording; the earlier exact-hash source is retained.

## Absolute risks first

Mean population log losses in nats per input, averaging calibration repeats within each model and then the 16 model pairs:

| New-model forward budget | Independent radii | GD | Pullback | Hybrid |
|---:|---:|---:|---:|---:|
| 0, historical default | 0 | 0.172338 | 0.105377 | 0.046311 |
| 128 | 1 | 0.171909 | 0.104896 | 0.043430 |
| 512 | 4 | 0.170168 | 0.104584 | 0.042852 |
| 2,048 | 16 | 0.169616 | 0.103847 | 0.041849 |
| 8,192 | 64 | 0.169397 | 0.103442 | 0.041287 |
| 65,536 | 512 | 0.169342 | 0.103316 | 0.041090 |
| Fixed half-turn, no new calibration | 0 | 0.172338 | 0.104865 | 0.044960 |
| Uniform random angle, no new calibration | 0 | 0.181712 | 0.106282 | 0.049455 |

Positive paired improvements below favor the named policy. Intervals resample the 16 paired model clusters 20,000 times; they are pointwise, not multiplicity adjusted.

| Method | Full calibration over historical default | Full calibration over half-turn | Historical default over half-turn |
|---|---:|---:|---:|
| GD | 0.002996 [0.001060, 0.005452] | 0.002996 [0.001047, 0.005415] | 0, identical rule |
| Pullback | 0.002060 [0.001502, 0.002629] | 0.001548 [0.000714, 0.002482] | −0.000512 [−0.001691, 0.000715] |
| Hybrid | 0.005221 [0.003051, 0.007985] | 0.003870 [0.001157, 0.007454] | −0.001351 [−0.002655, −0.000066] |

The pullback default won over the half-turn on 6/16 fresh models and lost on 10; the hybrid default won on 4 and lost on 12. Finite full calibration also has small failures: after averaging its 20 calibration repetitions, it beat/tied/lost to the half-turn on 10/4/2 pullback models. No model or calibration draw was discarded. Across methods, the full reference's mean regret to the posthoc population-best candidate was 0.00000289, 0.00000978 and 0.00001654 nats. This population diagnostic was never available to a selector.

The separate 32,768-input IID sampled-label test per pair confirms the budget ordering. Pullback losses at the historical default, 8,192 calls and 65,536 calls were 0.105129, 0.103309 and 0.103202. Posterior expected losses on the same fresh inputs are reported separately from sampled-label losses.

## Recovery and uncertainty

Recovery divides the cohort mean improvement over the historical default by the cohort mean improvement from the **finite 512-radius no-shrinkage calibration**. It is a ratio of means, not a mean of model-wise ratios, and that reference is not an oracle. The denominator was positive with a positive lower confidence endpoint for all three methods. The full reference is 100% by definition; this does not imply that its population risk is known without uncertainty.

| Method | First tested budget reaching 90% | Recovery there | 95% recovery interval | Bootstrap threshold interval |
|---|---:|---:|---:|---:|
| GD | 2,048 | 90.84% | 76.75–94.62% | 2,048–8,192 |
| Pullback | 8,192 | 93.89% | 90.26–96.56% | 8,192–8,192 |
| Hybrid | 8,192 | 96.23% | 93.10–98.00% | 2,048–8,192 |

The discrete threshold distribution is more informative than its endpoints alone. For pullback, 98.16% of model-cluster bootstrap draws first reached 90% at 8,192 calls and 1.84% needed the full 65,536-call reference. The corresponding probabilities of needing the full reference were 0.22% for GD and 0.065% for the hybrid. No bootstrap denominator was nonpositive in this run. `threshold_summary.csv` and all 120,000 method/rule threshold draws retain every tested-budget probability and unattained category. These are cohort-average targets, not per-model guarantees.

Calibration Monte Carlo variation is reported separately from model-seed uncertainty. For pullback without shrinkage, the mean within-model standard deviation across 20 calibration draws fell from 0.001287 nats at 128 calls, to 0.000294 at 8,192, to 0.0000270 at 65,536. The residual calibration Monte Carlo standard error of the reported 20-draw cohort mean was 0.0000218 at 8,192; the between-model standard error of the absolute risk means was 0.001194. Paired gain uncertainty is smaller than absolute-risk uncertainty and is the relevant quantity for policy comparisons.

Shrinkage reduced downside frequency at very small budgets but did not improve mean risk at any tested nonzero budget in this cohort. For pullback at 128 calls, the fraction of model/calibration runs worse than the historical default fell from 36.9% to 2.8%, while recovered mean gain fell from 23.4% to 10.8%. Its GD 90% threshold moved from 2,048 to 8,192. This is a measured caution-versus-average-gain tradeoff for one fixed prior strength, not a claim that shrinkage can never help.

## Costs and numerical checks

All averaging policies use exactly two deployed network calls. Random angle samples uniformly from the 128-angle group, including identity; it still executes two calls. The historical defaults cost 1,048,576 forward evaluations per training method to construct (16 models × 512 radii × 128 angles), but zero **new-model** calibration forwards. That historical cost is not treated as free overall.

Median standalone new-model calibration times for 128, 512, 2,048, 8,192 and 65,536 calls were approximately 1.05, 1.39, 2.93, 8.66 and 88.99 milliseconds on this CPU model. These include scoring all candidates after the allowed forwards. They are local timings, not accelerator-serving claims. The research run actually generated all full pools—62,914,560 forward example evaluations across 48 models and 20 repetitions—then reused prefixes; the simulated budget policies did not execute separate complete experiments for every budget. Historical reconstruction and timing checks are additional, separately recorded costs.

Population evaluation used the known rings law and refinement from 192 radial × 512 angular nodes to 384 × 1,024 for all 48 models. The stopping tolerance was `2e−7 + 2e−6 × risk scale`; the largest observed change across the full risk/gain vector was 2.82×10⁻⁸. This integration is not IID data. Independently seeded IID test draws, frozen selectors, full raw orbit arrays (about 479 MB compressed), state hashes, curve hashes and costs are preserved for independent reconstruction.

## Reproduction and portability

For a fresh confirmation, copy the study's `source`, `legacy`, and `idea2a` code plus the existing frozen `idea2a/history` and freeze/analysis manifests into a new study folder, excluding generated `idea2a/fresh`. Using the project Python environment with NumPy, threadpoolctl and Matplotlib:

```text
python idea2a/study.py train
python idea2a/study.py calibrate
python idea2a/study.py evaluate
python idea2a/study.py benchmark
python idea2a/analyze.py
python idea2a/contrast_report.py
```

These confirmation commands use only relative in-archive dependencies. Training and calibration stages refuse to overwrite existing generated directories. `history/historical_curves.npz` contains the frozen old-only curves and defaults; `history/retrospective_inputs.npz` preserves all data needed to inspect the retrospective claim.

The optional **rebuild of the original historical stage** uses the bundled old model bank but also reads the earlier rotation study's `risks.csv` and `population_gain_curves.csv` to reproduce the inspected pilot. Its default path refers to the original workspace. In a completely fresh historical rebuild, pass that earlier archived `direction2/fresh_results` directory explicitly through `study.py history --retrospective-source PATH`, then run `freeze`. This dependency is unnecessary for reproducing the prospective confirmation with the supplied historical bank.

Useful next tests would prespecify a guard against changing a cheap baseline, repeat historical-default fitting across several old cohorts to measure transfer uncertainty, or use predefined radial strata so a tiny calibration pool cannot miss an important radius region. A gain-based stopping rule would need independent validation or a simultaneous uncertainty bound across candidate angles, rather than an uncorrected interval after maximizing noisy scores. Those would be new prospective experiments; no setting was optimized on the results above.
