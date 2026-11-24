# Prospective real-data comparison and total-cost measurement

The official Fashion-MNIST test set remains locked until model/policy commit,
source freeze, and root TEST_APPROVAL.json. All10,000 test images are retained:
39 batches of256 plus one batch of16, selecting64 and4 respectively. A fixed
permutation2136000 sets batch membership for all three models; tie keys use
2135000+seed. Training splits and model fitting are fixed in TRAINING_PROTOCOL.

## Multiclass method and calibration

Horizontal reflection is a defensible clothing-category transformation, but
the original test distribution is not assumed reflection-invariant. This
experiment adapts the prior selector idea to high-dimensional images; it is
not a direct transfer of the low-dimensional interpolation table.

For temperature-scaled ten-class logits z,zT, use
J=[logsumexp(z)+logsumexp(zT)]/2-logsumexp((z+zT)/2).
Feature extraction uses4x4 average pools of the pixelwise symmetric image
(x+Tx)/2 and absolute antisymmetric image|x-Tx|/2, yielding98 coordinates.
These are exactly flip-even and require no CNN evaluation. The target is J,
never a label or supervised loss. All pointwise scores are frozen before test.

Fit ridge coefficients with alpha in[1e-4,.01,1] on standardized features.
The exact k-nearest-neighbor control uses the same98 features and k in[1,5,25].
The histogram control uses mean symmetric brightness and mean absolute
antisymmetry, with8x8,16x16,32x32 quantile cells. Empty cells use the pool mean.
All three controls share8,000 fitting image pairs (16,000 CNN forwards).
Choose each hyperparameter by mean selected J on2,000 independent selector
validation images, including their final208-image partial batch; this costs
4,000 additional CNN forwards shared by all candidates. Exact ties prefer
earlier listed settings. Pick the stronger of the selected histogram and
kNN by this same development criterion, separately per model; retain both.
Ridge is the candidate regardless of its validation result. No test outcome
changes the policy or target. Selector-validation ties use2134000+seed.

## Outcomes and symmetry boundary

Evaluate raw, full_fused, full_sequential, confidence, ridge, histogram, kNN,
and the exact-J oracle. Raw costs1 forward/input; each deployable limited
policy costs1.25; both full implementations and the oracle cost2. Confidence
uses the smallest maximum softmax probability and requires first logits.
Input-only policies select before prediction and fuse original plus selected
partner images into a single CNN invocation. Oracle queries both views for
every input before allocation and remains diagnostic only.

The main empirical outcomes are cross-entropy and classification error on
the untouched original test images, using official labels. Evaluate both
CUDA and one-thread CPU on the complete test set to quantify numerical or
selection differences. The same calibrated model is used for every policy.

A separate flip-paired finite-pool evaluation averages the original and
reflected versions of each labeled test image. The even masks must be
unchanged by individual image flips. On this deliberately pair-closed law,
the multiclass Jensen identity is exact in real arithmetic. This is an
artificially symmetrized evaluation law, not a guarantee for the original
Fashion-MNIST distribution. Flipped images retain the original category
label by the stated semantic transformation assumption. No unlabeled
original-distribution guarantee is inferred.

Primary empirical contrast is ridge minus the validation-chosen stronger
control, averaged over the three fixed trained models. Use5,000 paired
bootstrap resamples of40 original-image batches (clustered for quota
dependence), weighted by batch size; keep both orientations and all three
models together. The final16-image batch is retained. Bootstrap intervals
are descriptive approximations; only three initialization/order replications
on one shared dataset are available. Report each seed and both controls.
The test set and its flipped copy are not20,000 independent observations.

## End-to-end cost and timing

Freeze benchmarks before test. For every seed and policy, use the same four
256-image batches drawn from selector-validation data, never tuned by test
quality. CPU uses one thread. CUDA uses FP32, TF32 off, no AMP, eager mode,
resident model and selector tables. No torch.compile or architecture search.

Benchmark CPU, CUDA-resident, and CUDA-transfer conditions. In transfer,
start with CPU FP32 image tensors, transfer images and only needed tie keys,
and return logits to CPU. Resident runs include feature computation, score
lookup/distance calculations, sorting, indexing, transforms, model inference,
and merging; device synchronization brackets every timed block. Model and
selector loading are separate recorded setup costs, excluded from steady
serving. No precomputed test scores, partner logits, or CNN features are
available to deployment. Tie keys are supplied inputs; their generation is
separate from timing. Full references do not transfer unused tie keys.

Use3 warmup blocks and7 recorded blocks, each containing4 batches, with
independent randomized policy order seed2137000+model. Report per-model
median milliseconds/batch and all raw repetitions. Compare limited policies
to the faster median of full_fused and full_sequential for that model/mode.
No repeated measurements are treated as independent model replications.
Record actual CNN input counts for every quality and timing path; calibration,
training, temperature fitting, selector fitting, model loading, serving, and
verification/audit costs remain separate. Quality is evaluated on both CPU
and CUDA rather than borrowing GPU risks for CPU cost rows.

Predeclared practical target: a limited policy must beat the stronger full
latency reference by at least10%, while its original-test mean cross-entropy
is within0.01 nats of full averaging. Failure is retained. Any amortization
calculation explicitly lists which setup costs it includes. GPU necessity
and speed advantages are empirical outcomes, not assumptions.

All failures and unselected configurations are retained. Source and policy
hashes precede official-test decoding. No post-test tuning or unbounded
additional model/selector search is authorized within this protocol.
