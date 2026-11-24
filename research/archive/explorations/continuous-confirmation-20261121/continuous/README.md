# Continuous conditional balance with an exact 64-view quota

The continuous extension is implemented and tested. A grouped randomized policy allocates exactly 64 second views per 256 continuous iid inputs, without quota repair or a free partner query. Its fitted field reduces numerical hidden-posterior ambiguity dramatically, but remains only approximately conditionally balanced between calibration radii. The available global smoothness guarantee is valid and too loose to certify useful neural-network gains. The resulting allocator loses clearly to strong confidence, radius and folded policies.

## Exact construction and the remaining approximation

Randomly partition each batch into 64 groups of four. A bounded continuous field u selects one member with probability q_i=1/4+(u_i-group_mean(u))/6. Probabilities lie in [0,1/2] and sum to one in every group. Our zero-angular-mean fields have marginal inclusion probability 1/4+u/8. Thus b(r)=E[u(h-m)|r]/8 controls all dependence on an unknown radial posterior, with no quota-repair bias. This exact construction is not a collection of finite-support test points: held-out inputs are continuous iid draws on the plane.

The field uses Fourier orders 1–8 with coefficient l1 norm at most one, fitted on 129 radial anchors and 512 angular nodes, then continuously interpolated in radius. Endpoints and values outside radius 3 are zero. The balanced field maximizes a label-free local objective subject to its sampled moment constraint. The unbalanced field removes that constraint. The even-only field is an exact antipodally symmetric control, and therefore has exact conditional balance everywhere. The deployment table stores coefficients; it does not cache predictions for future continuous inputs.

The derivation and explicit network derivative bounds are in THEORY.md. The supplied proof distinguishes anchor residual, angular quadrature, radial interpolation and floating-point checks. Calibration is deterministic model quadrature, so there is no random calibration-sample error term to hide inside the approximation. Test uncertainty and trained-model variation are separate.

## Sixteen independent neural-network confirmations

|Field|Mean unrestricted ambiguity width|Mean prevalence-half width|Conditional interpretation|
|---|---:|---:|---|
|balanced|3.76e-05|3.76e-05|Numerically integrated continuous moment; not an exact identification theorem|
|unbalanced|0.0128|0.00911|Numerically integrated continuous moment; not an exact identification theorem|
|even|0|0|Exact zero by antipodal symmetry|

The fitted balanced field has mean orientation asymmetry 0.515; it is not merely an even field in disguise. Yet zero anchor constraints do not make it conditionally balanced between anchors: the largest observed |b(r)| was 0.005013 nats. Its small integrated ambiguity reflects the radial mass of these deviations, not a uniform zero-moment claim.
The independent angular refinement discrepancy was at most 9.42e-09 nats, and the largest fine/coarse radial discrepancy in integrated absolute balance was 9.3e-07. These are numerical sensitivity estimates, not formal interval enclosures. Corrected extrema are sharp for each model's piecewise-interpolated moment surrogate and continuous radial density, while their relationship to the exact neural moment remains numerical. They are model-specific adversarial worlds, not one shared alternative-label task.
The analytic uniform conditional-imbalance bound averaged 2.108 nats; its integrated counterpart averaged 0.937. These dwarf the measured residuals. All 16 one-sided finite-sample non-harm certificates were negative, ranging from -4.290 to -0.609. The bound is mathematically valid and practically vacuous. No empirical near-balance observation is presented as a useful rigorous certificate.

## Original-ring deployment risk

|Policy|Expected risk|Model calls/input|Resident-array median ms/batch|
|---|---:|---:|---:|
|group_balanced|0.109036|1.25|0.2776|
|group_unbalanced|0.110160|1.25|0.2716|
|group_even|0.109100|1.25|0.2625|
|group_uniform|0.109829|1.25|0.0842|
|group_confidence|0.102097|1.25|0.0649|
|confidence|0.100739|1.25|0.0834|
|radius|0.103236|1.25|0.0918|
|folded|0.102494|1.25|0.1132|
|raw|0.112327|1|0.0380|
|full|0.102248|2|0.0783|

Compared with global confidence, balanced-group risk was higher by 0.008297 nats, with a paired 95% seed-bootstrap interval [0.006416,0.010332]. It also lost to global folded and radius rules on every seed. Even group-wise confidence was substantially better, so the practical gap cannot be attributed only to the one-per-group restriction.
The balanced rule modestly improved on its even-only grouped control by 0.00006372 nats [0.00001181,0.00012062], and on uniform grouped allocation by 0.00079306 [0.00054366,0.00107117]. These are narrower comparisons within the constrained family. They do not establish that conditional balance is a competitive allocator. Every policy and hidden posterior remains reported.
All intervals resample the 16 whole fresh model/seed records, not the 256 batches per seed. They are pointwise. The exact-64 grouped construction and the global top-64 baselines have different feasible allocation classes. Full two-view and raw outputs are distinct-budget references, not universal performance ceilings.

## Exact continuous existence example

A separate analytic example uses h(r,phi)=1+2r cos(phi), R uniform on [0.5,1.5], and u=-cos(3phi). Harmonic orthogonality gives E[u(h-m)|r]=0 for every radius, despite orientation-sensitive selection. The exact same grouped quota rule has gain 0.05116851 under every radial posterior, versus 0.04965731 for uniform-quarter allocation. An independent implementation checked the construction, with residual at floating-point scale. This proves continuous existence; it does not make the fitted neural fields exact. See ../audit/group_mathematics.json and the corresponding derivation.

## Preserved diagnostic failure and corrected extrema

The first diagnostic evaluator mishandled threshold plateaus. It used strict threshold worlds but complementary integration on one side, and failed to randomize labels fractionally on positive-mass median atoms. Sixty of 96 purported prevalence-half worlds missed the audit tolerance; the largest prevalence error was 0.006724, and the largest integral/deployment posterior-mass discrepancy was 0.061132. The original files and failing audit remain unchanged and must not be used to claim attained fixed-prevalence extrema.
A separately frozen evaluation-only repair detects structural parity-zero moments and positive-mass threshold atoms. It assigns fractional posterior probability on the median plateau and uses the same serialized rule in integration and replay. Corrected continuous prevalence differs from one half by at most 1.45e-15; 62 half-prevalence worlds require fractional atoms. Independent 24-point split quadrature checked all 192 corrected worlds, with prevalence error at most 1.61e-15 and gain disagreement at most 2.43e-17. All original masks and original-ring risks reproduce exactly. No field, model, coefficient, budget, seed, policy or primary comparison was changed.
Use diagnostic_repair/intervals.csv, continuous_gains.csv, per_seed.csv, summary.csv and contrasts.csv for final hidden-posterior claims. The original fresh diagnostics are retained as the failed attempt. The correction spec, code freeze, prepared-world commit and immutable-mask checks are supplied.

## Costs, data and reproducibility

Development used 16 archived old PB500 models. The frozen confirmation used 16 new original-ring PB500 models, train821001..821016 and init822001..822016. These are shared with the other current confirmation tasks and do not constitute additional independent model replications. Each seed has 65,536 continuous iid test inputs, common independent partitions/ties and categorical draws, and sampled labels for the original-ring task.
The three learned fields share 66,048 offline model-example forwards per model. The unchanged legacy radius/folded gates separately use 33,024 iid inputs and antipodes, also 66,048 forwards. Confidence and uniform controls require no learned calibration. Primary numerical conditional-moment diagnostics additionally use 2,624,512 forwards per model, explicitly outside the deployment budget. Standalone paired-view finite-sample certification requires 131,072 forwards per model; this implementation reuses the full/raw reference evaluation outputs, rather than adding hidden deployment queries.
Every limited deployed policy actually makes calls [256,64]; all 160 recorded policy/model call audits passed, including raw/full reference shapes. All arrays are loaded into memory before the separate resident-array latency benchmark. Local CPU timing includes allocation and interpolation overhead and is not a serving-throughput claim. In this small-network prototype, the learned grouped field was slower than the full-two-view reference despite using fewer model forwards; no latency advantage was observed. The evaluation repair repeated 13,631,488 model-example forwards across the 16 models; that correction work is separate from calibration and deployment costs.
Dependencies are Python, NumPy, threadpoolctl and Matplotlib for the plot. Source/core, the unchanged legacy gate and old models are in the surrounding archive. New model inputs are ../confirmation/states/original_{seed}.npz. No original external research folder, network or account access is required.
In an empty archive-layout copy, retain frozen sources/protocols and input model states, then run `python run.py calibrate`, `python run.py score`, `python run.py summarize`; `python audit_worlds.py` reproduces the original diagnostic failure. For the corrected diagnostic stage retain its freeze, run `python diagnostic_repair/repair.py prepare` then `python diagnostic_repair/repair.py replay`, and generate reporting with `python report.py`. Existing scientific outputs are refused. The original old-only pilot can be reproduced with `python run.py oldpilot` in an empty old/ directory; supplied freezes preserve the original pre-confirmation boundary.
Raw files include all coefficient tables and offline logits, input/model hashes, continuous conditional curves and bounds, sample and deployment audits, per-seed outcomes, compressed per-batch outcomes, all corrected posterior arrays, old/new diagnostic risks, confidence intervals and the independently audited correction.
