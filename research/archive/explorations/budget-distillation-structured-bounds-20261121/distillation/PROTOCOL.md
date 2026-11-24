# Two-view teacher to one-view student: frozen protocol

## Question and scope

Can a one-view student retain the improvement of an augmented CNN's two-view
logit average, against both ordinary augmentation and a full-training-compute
control? This is a bounded, three-pair experiment, not a hyperparameter search.
All models use the unchanged 426,122-parameter CNN. No historical weights are
reused. Negative results and failed runs are retained.

The original Fashion-MNIST split is copied unchanged from the completed
augmentation experiment: 40,000 training, 5,000 checkpoint validation, 5,000
temperature fitting, 10,000 within-round holdout. Every Fashion-MNIST image has
been historically exposed, and the official test has been observed. The new
holdout results therefore assess this frozen follow-up, not pristine-data
replication. We do not re-score the historical official test.

## Genuinely external population, fixed without outcomes

The independently acquired public source is Param Aggarwal's Fashion Product
Images (Small), Kaggle version 1 (2019), sourced from catalog product photos.
The owner's API declares MIT licensing. The saved owner metadata, source ZIP
digest, retrieval URL, original selected JPEGs, metadata, processed arrays,
selection code and manifest make the exact evaluated population reproducible.
The large source ZIP remains locally preserved but may be excluded from the
final compact archive because all selected original bytes are included there.

Before any model scoring, select 250 distinct images for each exact articleType:
Tshirts -> Fashion-MNIST 0 (T-shirt/top), Trousers -> 1, Dresses -> 3, Sandals -> 5,
Shirts -> 6, Handbags -> 8 (Bag). These are strict semantic subsets, not evidence
on the four missing labels or the full target-domain frequency distribution.
Do not map ambiguous Jackets to Coat, Sweaters to Pullover, or general shoes to
Sneaker/Ankle boot. All ten output logits remain; predictions in absent classes
count as errors. Equal-class mean equals macro average over these six classes.

Rank product IDs by SHA256('external-population-2321000:' + ID), within category;
take first 250 after excluding missing/corrupt images and exact raw/pixel
duplicates. Compare all selected processed uint8 pixels against all 70,000
Fashion-MNIST images; exact overlap is zero. Exact duplicate removal does not
rule out near duplicates or different views of the same product. No filtering
by background, garment color, human model, prediction or difficulty is allowed.
The source includes people wearing garments and differs substantially in pose,
background, catalog conventions and category boundaries from Fashion-MNIST.

Preprocessing is fixed: Pillow L grayscale, invert, aspect-preserving LANCZOS
resize to fit inside 24x24, centered on black 28x28, uint8 / 255. No cropping,
thresholding, adaptation, external calibration, or selection of preprocessing
from external outcomes. The contact sheet is a provenance inspection only.
Population, mapping and preprocessing were committed before outcomes; labels
were used only for the fixed mapping and stratification, never model fitting.

## Paired recipes and teacher targets

Pairs 1,2,3 use initialization 2322000+pair, training-order 2323000+pair, flip
coins 2324000+pair. Teacher/plain, student and compute control have identical
initial weights and identical order/augmentation streams for shared prefixes.
FP32 deterministic CUDA, Adam lr .001, weight decay .0001, batch 256, random
horizontal flip probability .5 replacing the input. Teacher and student train
12 epochs (480,000 forward/backward examples, 1,884 optimizer updates each).

The teacher is exactly the plain-augmentation baseline network: evaluate it
as one-view B and as two-view T, where T logits are (h(x)+h(flip(x)))/2. This
reuse avoids training a redundant identical model. Teacher checkpoint selection
uses raw validation CE at epochs 3,6,9,12, ties toward earlier checkpoints.
Once selected, precompute both views on all 40,000 training images exactly once
(80,000 teacher forward examples); target q=softmax(mean_logits/2). Each
student input is one randomly selected view and the invariant target is the
same for both. Student loss is .5 CE(y,z) + .5 * 2^2 KL(q || softmax(z/2)).
Temperature 2 and weight .5 are fixed without tuning. Student checkpoint
selection also uses raw validation CE at epochs 3,6,9,12.

## Equal updates and full compute control

B is the equal-student-update comparator; it does not pay teacher training.
C is a separately initialized plain-augmentation CNN trained from the same
paired initialization, with its training-example budget calculated from the
entire distilled pipeline's network work: teacher training + 4 teacher
validation passes + teacher raw/full calibration queries + two-view target
precomputation + student training + 4 student validation passes + student
calibration. C pays its own four validation passes and one calibration pass.
Its four selection checkpoints are at quarter-budget boundaries (the final
one includes the partial final epoch); ties favor earlier checkpoints.

The precise matching unit is dense convolution/linear multiply-add FLOPs,
counting multiplication and addition separately. A forward example costs
F=2*(28*28*16*9 + 14*14*32*16*9 + 7*7*64*32*9 + 3136*128 + 128*10).
A training example costs G=3F - 2*(28*28*16*9), since the first convolution does
not require an input gradient. This counts forward, weight-gradient and
activation-gradient matrix work. Distillation network budget is
960000 G + 135000 F; control training examples are
floor((960000 G + 110000 F)/G), plus its own 25000 F. The residual is less than
one training example. No target precompute, fitting data forward or teacher
training is treated as free. Both pipelines use the same 50,000 unique fitting
labels. Labels can be reused more often in C; that is the intended comparator.

This is not exact equality of whole-program operations or wall time. Pooling,
nonlinearities, Adam, CE/KL/softmax, 80-step scalar calibration fits, data
indexing, transfer and storage are explicitly recorded by call/example counts,
fit counts and measured elapsed components. Teacher has two final calibration
fits and student one; C has one. Scalar fit timings and all training timings
are descriptive, not exclusive throughput measurements. We separately report
sunk-teacher marginal and amortized K-student costs (K=1,10,100), without
substituting them for the full-cost primary comparison.

## Final calibration, commitment and outcomes

Positive inverse temperatures are fit only on the fixed 5,000 calibration
labels, independently for B raw logits, T averaged logits, S raw logits and C
raw logits. Convex CE minimization uses 80 bisection steps in [.25,4], retaining
endpoint status. No external label enters any fit. Temperature scaling cannot
change argmax. Commit selected weights, every checkpoint, targets, raw
calibration predictions, scalar fits and pairing audit before scoring either
holdout or external data. Preserve per-example raw predictions and labels.

For each population report CE/error, each seed, macro/classwise errors and
flip agreement for T/B. Four contrasts S-B, S-C, S-T and T-B, each CE and error,
on each of two populations form 16 primary intervals. Use 10,000 paired
stratified image-bootstrap replicates (seed 2328000), sharing image draws across
all fixed fitted models and averaging over the three pairs before resampling.
Intervals use Bonferroni alpha=.05/16 (nominal 99.6875% individual coverage).
They are conditional image-bootstrap intervals, not finite-sample guarantees,
and do not capture training-dataset or product-family uncertainty.

Predeclared preservation success requires, on BOTH populations: (a) teacher
full improves on B at the point estimate in CE and error; (b) S retains at
least half that point-estimate gain for each metric; (c) adjusted upper bounds
for S-T and S-C are below .00325 CE and .0014 absolute error. These noninferiority
margins approximate half the old augmentation gain (.006484 CE and .0028
absolute error). They are fixed before outcomes. Also report
each condition separately; a failure is not repaired by changing margins.
Serving success additionally requires median S latency no more than 1.10 times
B latency in each of the three device/transfer modes (pooling pair medians).

## Serving and release

After quality commitment, benchmark calibrated B,S,C raw, T fused and T
sequential on the same first 1024 validation inputs, batch 256, one CPU thread,
CUDA-resident and CPU-to-CUDA-return modes. Three warm-up and seven recorded
blocks of four batches; randomize methods within block using 2329000+pair plus
mode offset. Synchronize CUDA timers. Include flips, concat, merge and final
beta; exclude loading, decoding/normalization, training and fitting. Compare S
against the faster measured full implementation and retain CPU/GPU agreement.
Run only after the parent grants an exclusive timing window. No timing claim
is inferred solely from call counts.

SOURCE_FREEZE.json hashes protocol, code, split, population and provenance.
Training/scoring require parent scientific release in ROOT_APPROVAL.json.
No new outcomes are read before release. No hyperparameters change after it.

## Primary references

- Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network (2015):
  https://arxiv.org/abs/1503.02531 (soft targets and temperature-scaled loss).
- Shanmugam et al., Better Aggregation in Test-Time Augmentation (ICCV 2021):
  https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html
  (TTA aggregation is a distinct prediction pipeline; no universal guarantee).
- Dataset owner/card: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
- Owner metadata: https://www.kaggle.com/api/v1/datasets/view/paramaggarwal/fashion-product-images-small
