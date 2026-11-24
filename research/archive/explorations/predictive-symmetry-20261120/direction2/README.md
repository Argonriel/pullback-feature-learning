# A label-free choice of the second rotation

The fixed half-turn leaves even angular Fourier modes intact. Choosing a rotation from an independent, unlabeled calibration sample improves the same two-call prediction rule. In 16 fresh training/initialization pairs, the pullback model's half-turn removed 54.9% of angular log loss; an exact Jensen-calibrated global rotation removed 70.7%, and a radius-dependent rotation removed 72.9%. The earlier 16-pair cohort had put the half-turn near 60%; the new cohort is separate and is reported without pooling.

## Frozen experiment

The 48 archived models were used only for exploration. The main protocol was then frozen in `protocol.json`, with its SHA-256 recorded in `freeze.json`, before generating the 48 fresh endpoint models. Training seeds are 611001–611016, initialization seeds 612001–612016, calibration seeds 613001–613016, and fresh IID test seeds 614001–614016. Each pair contains GD (500 updates at 0.5), pullback (500 at 8, damping 0.1), and 100 pullback updates followed by 400 GD updates at 2. All policies, raw calibration predictions, radii, phases and trained weights were saved and hashed in `fresh_results/selectors_frozen.json` before either population-risk integration or IID evaluation.

Calibration uses 512 independent unlabeled radii and 128 equally spaced views at an independently randomized starting phase for each radius. The 64 candidate positive angles are 2πj/128, j=1,…,64. Negative angles have identical population gain. Exact global and radius-specific policies choose the largest estimated Jensen gain; the latter uses eight unlabeled calibration-radius quantile bins. Radius-only choice preserves rotational symmetry. The random policy samples uniformly from all 128 rotations, including the identity with probability 1/128. It uses two calls even when that sampled rotation is the identity. No test labels or test population risks select an angle or radial bin edge.

## Results

Mean population log losses in nats per input, using independent converged polar integration:

| Training | Single view | Half-turn | Random angle | Curvature Fourier K=8 | Exact global | Exact radius-specific |
|---|---:|---:|---:|---:|---:|---:|
| GD | 0.122742 | 0.103009 | 0.106753 | 0.098967 | 0.098942 | 0.098327 |
| Pullback | 0.117658 | 0.106110 | 0.106444 | 0.102843 | 0.102776 | 0.102325 |
| Pullback → GD | 0.067481 | 0.047884 | 0.050465 | 0.042528 | 0.042509 | 0.042175 |

For pullback, the global selector's paired improvement over the half-turn is 0.003335 nats (95% seed-bootstrap interval 0.001906–0.005082), with 15 wins and one exact tie. Radius-specific choice adds 0.000450 nats (0.000289–0.000671), winning on all 16 pullback seeds. Exact global beats or ties the half-turn on every fresh model across all three methods. Radius-specific choice improves on the half-turn on 15/16 GD, 16/16 pullback and 16/16 hybrid models; the remaining GD result is a small loss, retained in the tables.

The 32,768 fresh IID labeled examples per pair independently confirm the direction. Pullback sampled-label losses are 0.106223 for the half-turn, 0.102666 for exact global and 0.102316 for exact radius-specific. The same IID inputs also support lower-variance posterior expected losses, reported separately. Deterministic quadrature and these fresh random observations are distinct checks.

The reported angular percentages divide mean gain by mean angular loss across seeds. They are not averages of per-seed percentages. Intervals use 20,000 paired seed bootstrap draws and are pointwise rather than multiplicity adjusted. The diagnostic population-optimal global candidate is calculated only after policies are committed; it is never deployed. Exact global's mean regret to that diagnostic candidate is 0.00000151 nats for pullback, showing that 512 calibration radii suffice for angle choice in this experiment even though absolute gain estimates have sampling noise.

## What the Fourier approximation did and did not predict

For h(r,θ)=Σ c_k(r)e^{ikθ}, two-view averaging multiplies c_k by (1+e^{ikα})/2. Its squared magnitude is cos²(kα/2). Thus a half-turn cancels odd modes and preserves even modes, while a general rotation trades their attenuation. That identity concerns squared Fourier energy; logistic risk is nonlinear.

Writing μ(r)=Eθh and q(r)=sigmoid(μ)(1−sigmoid(μ)), the second-order risk-gain approximation is

`G_K(α) = 1/2 E_r[q(r) Σ_{k=1}^K P_k(r) sin²(kα/2)]`,

where P_k combines the positive and negative frequency energies. The implementation accounts for the Nyquist frequency once. The stronger exact calibration score is

`G(α) = E[φ(h) − φ((h+h_rotated)/2)]`, with `φ(h)=log(exp(h/2)+exp(−h/2))`.

Under a rotation-invariant joint input-label law, this is exactly the population log-loss decrease. Complete calibration orbits make the signed linear term cancel for every candidate even in the finite calibration set. A generic IID pointwise difference does not have this cancellation and must not be called an exact finite-sample risk improvement. The nonnegative symmetric Jensen gap has the same orbit average.

The K=8 curvature-weighted proxy ranks angles surprisingly well here: its pullback mean loss exceeds exact global by only 0.000067 nats. It is less reliable as a gain forecast, understating pullback gain by 0.002476 nats on average. The unweighted energy selector is weaker, especially for the hybrid model (0.044222 loss versus 0.042509 for exact global); its scores have squared-logit units and are **not** estimates of risk decrease. Fields named `predicted_gain` in the raw exploratory and selector archives contain those raw energy scores for `energy_*` policies; their units must not be compared to nats. K=4, 8, 16 and 64 versions, both weighted and unweighted and both global and radius-specific, are all preserved. No failed or weaker design was removed.

This is evidence that Fourier transfer explains the half-turn's ceiling and can cheaply summarize angle preference. It is not evidence that finite Fourier energy alone determines logistic risk. When all orbit predictions are already available, exact Jensen calibration is both stronger and comparably cheap.

## Costs and audit

Every deployed averaging policy uses exactly two model evaluations per input, independent of how many candidate angles were considered offline. Calibration costs 65,536 forward example evaluations per trained model, shared by all 18 fitted rules. Median full calibration time is recorded in `training_and_calibration.csv` (approximately 0.07 seconds on this small CPU model). Computing FFTs and all exact Jensen curves has additional arithmetic cost but requires no additional network calls. Deployment median times per 256 inputs were 0.0403 ms single-view, 0.0823 ms half-turn, 0.1044 ms exact global, 0.1041 ms radius-specific and 0.1074 ms random. These are local microbenchmarks, not claims about accelerator serving latency.

An independent auditor reconstructed all 48 calibration tables and all 18 policy choices from stored logits, with zero angle mismatches, maximum gain-curve discrepancy 1.78×10⁻¹⁵, and maximum Fourier-energy discrepancy 3.55×10⁻¹⁵. Thirty representative risks on an alternative integration grid agreed within 3.17×10⁻¹¹. Twenty-four IID risks agreed within 1.39×10⁻¹⁷. Actual deployment auditing observed exactly two forward calls of equal input count. Frozen source, model, calibration and protocol hashes all matched. The audit also preserves the one GD radius-specific loss.

## Files and reproduction

- `rotation_selectors.py`: reusable `calibrated_angle_selection(theta, calibration_radii, phases, logit_velocity=...)` interface; it never imports labels or a risk evaluator.
- `experiment.py`: separate archived exploration, freeze, fresh fit, and risk-evaluation stages.
- `analyze.py`: seed bootstrap summaries, deployment timing and publication-ready figures.
- `bridge.py`: separate pre-update loss-clock forecast experiment linked to direction1; its selections are committed before bridge risk evaluation.
- `exploration/`: all archived-cohort attempts, including the weaker proxies.
- `fresh_results/`: model weights, raw calibration predictions, immutable selectors, population and IID risks, costs, paired estimates and figures.

Run the stages with the project's Python environment in a new output directory. The original files are preserved; rerunning `fit` or `explore` refuses to overwrite their output directories. Numerical dependencies are NumPy, threadpoolctl and Matplotlib plus the repository's `pullback_feature_learning` core. The neighboring `legacy` directory supplies frozen original generator and polar-integration code; the script also supports the original workspace layout.

```text
python experiment.py explore --out new_exploration
python experiment.py freeze
python experiment.py fit --out new_fresh_results
python experiment.py evaluate --out new_fresh_results
```

`freeze` intentionally refuses to overwrite an existing freeze. To reproduce an already-frozen run, retain the supplied protocol and run `fit`/`evaluate` to fresh output locations. The analysis script's default locations are the archived run; use a copied study directory for a fully separate reproduction.
