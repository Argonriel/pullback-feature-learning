# Independent confirmation of confidence allocation with calibrated angles

The previous angle-calibration result replicated on new original-ring models and remained useful under a known elliptical coordinate change. Keeping the confidence-selected 64 examples fixed and replacing their half-turn with a calibrated radius-specific angle reduced pullback log loss on 16/16 original-ring models and 13/16 ellipse models. Both primary confidence intervals remained favorable after accounting for the two geometry tests.

This is a prediction-quality result, not a measured speedup. On this small CPU model, the learned policy was slower than efficient half-turn implementations, even though it evaluates fewer examples than applying two views to everyone. The complete cost comparison therefore separates network-evaluation counts, one-time calibration, and actual batch latency.

## Frozen independent experiment

There are 16 new training/initialization pairs per geometry and three unchanged schedules per pair: GD500 at rate 0.5, PB500 at rate 8 and damping 0.1, and PB100 followed by 400 GD updates at rate 2. All 96 endpoints, their initial/PB100 states, and their training data are retained. No training failed and no rate or schedule was retuned.

Original rings use training seeds 821001–821016 and initialization seeds 822001–822016. The ellipse cohort uses 821017–821032 and 822017–822032. Schedules within a pair share data and initialization; the two geometry cohorts use separate draws and are not paired with one another.

Ellipse observations are `x = A z`, with `A = diag(1.5, 0.7)` and latent `z` drawn from the original noisy-ring distribution. The network trains on observed x without whitening. Every selector receives the same known A and may use `z = A^-1 x`. Valid views are `A R_alpha A^-1 x`; ordinary Euclidean rotations of the ellipse are not assumed label-preserving. The half-turn is exactly `−x` in either geometry. The map changes observed geometry and optimization while preserving the latent task's Bayes difficulty. This is not an independent real-world task or a test of discovering an unknown symmetry.

The primary calibration budget is 8,192 forwards: 64 independent radii, each with a 128-angle orbit. The fixed 65,536-forward sensitivity uses 512 radii with the primary pool as a prefix. Sixteen radius quantile bins, the minimum-four-radii rule, the 64 candidate separations, and all fitting/routing functions are imported unchanged from the previous frozen factorial. Orbit nodes are correlated evaluations, not independent calibration samples. Independent radius and phase streams are retained. The strong legacy folded half-turn control fits its unchanged rule on 4,096 IID latent inputs plus antipodes at the same primary forward budget.

All calibration policies were committed before test scoring. We also waited for the other study components' algorithm freezes before generating empirical outcomes. Each pair has 65,536 independent IID test inputs, shared across its models, policies and budgets. Independent Bernoulli labels give a sampled-label check; the known latent-radius posterior is used only to evaluate conditional expected risk. Each limited policy actually selects exactly 64 of each 256 inputs and evaluates only those examples a second time. The confidence masks are identical for half-turn, global-angle and radius-angle variants.

## Primary results

Losses are nats per input. Positive gain means lower risk with the learned radius angle. Primary intervals are 97.5% per geometry, giving a Bonferroni 95% family interval statement for the two geometry comparisons. They resample 16 model-pair clusters, not test batches.

| Pullback, 8,192 calibration forwards | Confidence + half-turn | Confidence + radius angle | Paired gain | 97.5% interval | Wins / losses |
|---|---:|---:|---:|---:|---:|
| Original rings | 0.100848 | 0.098283 | 0.002565 | [0.001398, 0.003833] | 16 / 0 |
| Known ellipse symmetry | 0.097467 | 0.096378 | 0.001089 | [0.000345, 0.002042] | 13 / 3 |

The corresponding ordinary 95% intervals are [0.001520, 0.003656] and [0.000414, 0.001926]. Sampled-label risks confirm the direction: 0.100936 versus 0.098355 on original rings, and 0.097401 versus 0.096303 on ellipses. This measures log loss; no classification-accuracy improvement is claimed.

The original-ring gain of 0.002565 nats is close to the previous cohort's 0.002583, now evaluated on entirely new training, initialization, calibration and test draws. The geometry results support robustness to this particular known transformation. They do not identify a causal geometry effect from the difference between the two mean gains, because the cohorts are independently drawn and finite-network optimization can differ.

## Fixed controls and preserved limitations

All pullback primary-budget policies are shown below. The first five use 1.25 deployment calls per input. Raw and full-view references use different budgets.

| Policy | Original risk | Ellipse risk | Deployment calls/input |
|---|---:|---:|---:|
| Confidence + half-turn | 0.100848 | 0.097467 | 1.25 |
| Confidence + calibrated global angle | 0.098963 | 0.096971 | 1.25 |
| Confidence + calibrated radius angle | 0.098283 | 0.096378 | 1.25 |
| Legacy IID folded half-turn gate | 0.102530 | 0.100977 | 1.25 |
| Joint radius selection and angle | 0.101970 | 0.101014 | 1.25 |
| Single view | 0.112580 | 0.111673 | 1 |
| Full half-turn | 0.102281 | 0.100505 | 2 |
| Full calibrated radius angle | 0.100638 | 0.099462 | 2 |

The learned radius angle improves on the learned global angle by 0.000680 nats on original rings (95% interval [0.000348, 0.001076]) and 0.000593 on ellipses ([0.000303, 0.000887]). A global angle alone has an uncertain ellipse improvement over the half-turn: its 95% interval includes zero.

Confidence plus radius angles beats the legacy folded gate on all 16 seeds in both geometries. The joint radius allocator remains weaker than confidence allocation: joint minus confidence-plus-half-turn risk is +0.001122 on original rings ([0.000045, 0.002256]) and +0.003547 on ellipses ([0.002054, 0.005177]). Thus this confirmation supports adding angle calibration to a strong allocator, not replacing confidence with the tested joint surrogate.

The GD and hybrid controls also have favorable mean radius-angle additions at the primary budget: respectively 0.009518 and 0.003625 nats on original rings, and 0.009054 and 0.001862 on ellipses. All corresponding pointwise 95% intervals exclude zero. They are secondary controls; comparing raw risks across schedules does not match optimization progress or establish a superior training method.

The larger calibration sensitivity preserves the pullback effect, with gains of 0.002275 on original rings and 0.001205 on ellipses. Larger calibration does not uniformly improve the selective policy. The calibrated objective is an unconditional radius-wise Jensen gain, while confidence selects orientations unevenly; the tested rule is not a theorem of optimal confidence-conditioned angle selection. The three primary-budget ellipse losses and all sensitivity losses remain in the tables.

Full-view policies are not universal risk ceilings. A confidence mask can avoid orientation-specific harmful averages and outperform applying the same angle rule everywhere. The better selective risks above therefore do not contradict the symmetry identity for a fully averaged or symmetry-preserving rule.

## Real costs and break-even

Median pullback setup times at the primary cap were 9.75 ms for original rings and 10.08 ms for ellipses. The full 65,536-forward orbit fits took 75.77 and 75.25 ms. The legacy IID gate at 8,192 forwards took 1.48 and 1.65 ms. Forward counts alone do not include the different arithmetic needed to fit the score tables.

The frozen main implementation inherited unnecessary coordinate/bin work for fixed-pi references. We preserved those timings and added a separately frozen **timing-only** audit of stronger, algebraically equivalent baselines: full pi needs just predictions at x and −x, and confidence pi needs its first predictions, top-64 confidence mask, and predictions at the selected −x. All scientific masks, predictions, risks, learned policies and calibration settings remained unchanged.

The timing audit checked 576 reference executions. Masks and actual call counts agreed exactly; maximum prediction discrepancy was 8.11×10⁻¹⁵. It repeated the learned candidate and optimized references in randomized order on the same batches.

| Pullback, 8,192 budget | Learned confidence + radius | Efficient confidence + pi | Efficient full pi |
|---|---:|---:|---:|
| Original median ms per batch of 256 | 0.10752 | 0.08280 | 0.06700 |
| Ellipse median ms per batch of 256 | 0.10866 | 0.07634 | 0.06602 |

The learned policy has **no finite latency break-even against either efficient pi reference on any of the 16 models in either geometry**. Its per-batch time is higher, in addition to its setup cost. Timing savings against the efficient legacy folded gate are uncertain; their intervals include zero, and seven original/eight ellipse models have no finite crossing. Conditional crossing medians are retained rather than hiding those cases.

Computational-count thresholds remain different. Against full pi, calibration plus 1.25-call deployment first uses no more total model-example evaluations after 43 batches, or **11,008 inputs**, at the primary budget. There is no forward-count break-even against confidence pi because both already use 1.25 calls per input. Setup overhead falls below 10% of cumulative deployment forwards after 65,536 inputs and below 1% after 655,360.

Measured runtime setup overhead falls below 10% of cumulative learned-policy deployment time after median 226,560 inputs on original rings and 233,728 on ellipses; the 1% thresholds are approximately 2.26 million and 2.34 million. These are medians of model-specific thresholds, not latency improvements over a baseline. The 65,536-call sensitivity and every seed's finite/non-finite threshold are preserved. See `COST_DEFINITIONS.md`, `results/amortization_by_seed.csv`, and the stronger-reference `strong_timing_*` tables.

This small CPU implementation trades additional routing/setup time for lower log loss. Counts, latency and prediction quality should not be collapsed into one claim. An application-specific value for better predictions was not assumed, and these measurements do not establish accelerator-serving throughput.

## Audit and reproduction

The independent root audit replayed 192 complete test cases across both geometries, all three schedules and both calibration caps. Expected-risk discrepancies were at most 5.6×10⁻¹⁷. All 1,536 deployment audit records, frozen hashes, phase chronology and shared-confidence-mask checks passed. Every limited deployment audit observed calls [256, 64]; full references observed [256, 256]. Conjugated rotations preserved latent radius to floating-point precision.

All dependencies are relative to this archive: the numerical model in `source/`, the unchanged gate/generator in `legacy/`, and the previous frozen fitting/routing implementation in `reference/experiment.py`. No prior workspace path, network account or external dataset is needed.

For a fresh reproduction, copy the code, frozen manifests, protocol, `source`, `legacy`, and `reference` into a new archive-layout folder, omitting generated `confirmation/states`, `confirmation/calibration`, `confirmation/results`, and `training.csv`/`training_commit.json`. Then run:

```text
python confirmation/study.py train
python confirmation/study.py calibrate
python confirmation/study.py score
python confirmation/analyze.py
python confirmation/timing_controls.py run
```

Retain the supplied `freeze.json`, `analysis_freeze.json`, and `timing_controls_freeze.json`; do not refreeze to bypass a mismatch. Generated directories and result files refuse overwriting. To perform a genuinely new design rather than reproduce this one, use a new directory and explicit new freeze.

Primary artifacts include `states/*.npz`, `training_commit.json`, raw calibration orbits and tables, `calibration/policy_commit.json`, independent `results/test_inputs/*.npz`, per-seed and compressed per-batch losses, all deployment/count audits, ordinary and strong-reference timing records, paired contrasts, and figures. All seeds, policies, failures of individual comparisons, and non-finite amortization cases are retained.
