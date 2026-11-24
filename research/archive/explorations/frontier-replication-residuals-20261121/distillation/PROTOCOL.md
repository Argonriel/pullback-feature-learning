# Independent-source replication of one-view distillation

This is a new, frozen six-pair experiment. Earlier studies, their sources,
models, negative preservation results and timing failures remain immutable.
The primary question is whether the previously favorable external error
ordering of the one-view student replicates on a genuinely separate source.
No hyperparameter or preprocessing search is conducted.

## New source and product population

UT Zappos50K was collected by Aron Yu and Kristen Grauman from Zappos.com,
independently of the earlier Myntra catalog. Official source files were acquired
from https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ . The site
identifies academic, non-commercial use only; its exact terms and citations are
preserved in index.html and readme.txt. This is a restricted research dataset,
not an unrestricted commercial-use license. Original selected JPEGs, complete
metadata, official documentation, ZIP digests, download URLs and selection code
are retained. The complete original ZIP remains local; the portable archive may
omit it while retaining every selected original byte and metadata needed to
rebuild the evaluated population.

Choose this source for isolated, centered footwear pictures on white backgrounds,
compatible category labels and explicit product groups, before observing model
outputs. It narrows the scientific population to footwear, rather than claiming
a replication of all six previous apparel labels or all ten Fashion-MNIST labels.
Mapping is fixed: Category Sandals -> 5; Shoes with SubCategory Sneakers and
Athletic Shoes -> 7; Boots with SubCategory Ankle -> 9. Other boots and shoe
categories are excluded on taxonomy grounds. All ten model logits remain active;
predicting an absent class counts as an error. Equal class counts make pooled
error equal the mean risk over these three selected categories.

The source CID is ProductID-ColorID. Exclude ProductIDs whose eligible variants
map to multiple target labels; preserve their complete list. There are seven
such products. This leaves 2,607 sandal, 6,008 sneaker/athletic and 3,365 ankle
boot product groups. Rank groups within each class by
SHA256('external-products-2421000:' + label + ':' + ProductID), and variants
within product by SHA256('external-colors-2421001:' + CID). Select one valid
color variant per group and the first 500 groups per class. A metadata-only
fallback is a common minimum size if fewer than 500 groups are eligible, but
fewer than 400/class would require a new source review before outcomes. This
fallback is not used. Missing/corrupt or exact-duplicate images would be skipped
in hash order; no chosen candidate required such an exclusion.

The population is 1,500 images from 1,500 unique products and 435 brands. All
processed pixels were checked against all 70,000 Fashion-MNIST originals and
the 1,500 earlier Myntra images; exact overlap is zero. Selected raw and processed
pixels have no exact internal duplicates. Product grouping removes repeat colors
of the same identifier, not dependence among brands or related designs; near
duplicates across different product identifiers are not ruled out. No prediction,
background, brightness, difficulty or visual-quality filter is allowed.

Preprocessing is unchanged: Pillow L grayscale; invert; aspect-preserving
LANCZOS resize to fit 24x24; center on black 28x28; uint8 divided by 255 at model
entry. No foreground crop, threshold, adaptive resize, external calibration or
domain adaptation. A contact sheet documents provenance, not sample selection.
The population manifest and hashes precede any new model outcomes.

## Source training and paired procedures

Reuse exactly the historical split: 40,000 Fashion-MNIST training, 5,000
checkpoint-validation, 5,000 calibration, 10,000 within-round held-out images.
All Fashion-MNIST images have previously been exposed. The old 1,500-image Myntra
population is now exposed secondary evidence. Neither is presented as new data.
No official-test predictions are recomputed. Each procedure uses the same 50,000
distinct source fitting/validation/calibration labels; no external fitting labels.

Train six NEW paired seeds (pairs 1 through 6) and 18 CNNs. Initialization is
2422000+pair, image order 2423000+pair, and augmentation coins 2424000+pair.
No historical weights are reused. B/T, S and C start with identical weights
within each pair and have identical order/flip streams over shared prefixes.

Keep the 426,122-parameter CNN, FP32 deterministic CUDA, Adam lr .001,
weight decay .0001 and batch 256. Training uses random horizontal flips with
probability .5 replacing the input. The plain augmentation teacher B and
student S each receive 12 epochs: 480,000 training examples and 1,884 updates.
The teacher is also evaluated as two-view T with averaged logits. This reuses
the same network; there is no redundant independently trained plain baseline.

B/T and S select lowest raw validation CE among epochs 3,6,9,12, breaking ties
toward earlier checkpoints. For the selected teacher, precompute original and
reflected logits for all 40,000 training examples: 80,000 forward examples.
q = softmax((h + h_flip)/2/2). The student receives one randomly flipped input,
with loss .5 CE(y,z) + .5 * 2^2 KL(q || softmax(z/2)). Temperature 2 and weight
.5 remain unchanged. No outcome-dependent model replacement, early stop or
additional sweep is allowed. Partial checkpoints and failures are retained.

## Full fitting-compute comparison and final calibration

C is a plain augmented CNN trained for 997,270 examples (3,914 updates), with
four checkpoint opportunities at quarter-budget boundaries and ties toward
earlier checkpoints. It uses the same paired initialization and stream prefix.
The budget matches the full teacher-plus-student network matrix work, including
all teacher training, validation, target generation and calibration forwards,
student training, validation and calibration. All observed scalar fitting and
inclusive elapsed fitting time are retained separately.

The exact matching unit is dense convolution/linear multiply-add FLOPs, two
operations per multiply-add. F = 4,643,840 per forward; G = 13,705,728 per training
example, equal to 3F minus the first convolution's unneeded input-gradient work.
The full distilled pipeline costs 960000 G + 135000 F = 13,784,417,280,000.
C costs 997270 G + 25000 F = 13,784,407,362,560. The gap is less than one training
example. This is not equality of whole-program operations or wall time: pooling,
nonlinearities, Adam, softmax/KL, scalar fitting, indexing, transfer and storage
are separately counted/timed. Teacher-sunk and amortized costs can be shown only
as secondary scenarios; the full-cost comparison is primary.

Fit independent positive inverse temperatures to the final B raw, T averaged,
S raw and C raw logits using only the fixed 5,000 source calibration labels.
Use 80 derivative-bisection steps within [.25,4], preserving endpoint status.
Distillation temperature 2 is separate from this final fitted scalar. Positive
scaling cannot change class decisions. Commit all selected weights, retained
checkpoints, targets, calibration logits, fits and pairing hashes before scoring
any population. Save per-example predictions, labels, image IDs and group IDs.

## Primary inference and explicit replication criteria

The two primary contrasts are S-B and S-C classification error on NEW Zappos.
For each, compute the error difference separately in each of the six paired
training seeds. Report the mean, seed SD, standard error and paired Student-t
interval with 5 degrees of freedom. Two-sided 97.5% intervals use Bonferroni
correction over these two comparisons (nominal 95% family coverage).
This inference addresses random initialization/order/augmentation on one fixed
training dataset and one fixed evaluated population. It assumes approximately
normal paired-seed differences; with only six pairs this is a fragile approximation,
not a finite-sample guarantee or uncertainty over new training datasets.

Predeclared primary replication SUCCESS requires BOTH primary comparisons to
satisfy all three conditions: mean error improvement at least .01 absolute
(one percentage point), adjusted paired-seed interval upper bound below zero,
and a strict improvement in at least five of six paired seeds. Otherwise the
primary criterion FAILS. The one-point threshold is a fixed practical target,
smaller than the prior external gains, not a value chosen from new outcomes.
No secondary result can reverse failure of this rule.

As a separately labeled sensitivity analysis, perform 20,000 crossed bootstrap
replicates with seed 2428000 on Zappos: sample six training-pair indices with
replacement, and sample 500 product groups with replacement within each of the
three classes. Each bootstrap replicate uses ONE COMMON multiset of product
indices across ALL chosen seed pairs and methods. Because the design retains
one image per product, group resampling equals image resampling in this selected
population, but color variants are not treated as independent observations.
Use 97.5% intervals for the two primary error contrasts and ordinary 95% for
other contrasts. These are approximate secondary seed-and-product intervals,
with unresolved brand/design dependence and only six fitted randomizations.

CE for S-B/S-C, S-T and T-B CE/error, classwise results and flip agreement are
secondary. All comparisons on the old source holdout and exposed Myntra set
are secondary, with ordinary 95% paired-seed t intervals clearly distinguished
from primary inference. Show all seed outcomes and strict win counts. Do not
pool the old three seeds into the new six-pair confirmatory analysis. No tuning
uses external outcomes, source holdout outcomes or secondary-source outcomes.

## Serving and execution boundary

Serving is secondary and its criterion is separate from primary quality.
Keep prior batch 256 and three modes: one-thread CPU, CUDA-resident, and
CPU-to-CUDA transfer including output return. For ALL six pairs benchmark B/S/C
raw, T fused and T sequential, with three warm-up and seven recorded blocks
of four fixed source-validation batches. Randomize method order per block with
2429000+pair plus mode offset and synchronize CUDA boundaries. Include flips,
concatenation, logit merging and beta multiplication; exclude loading and input
normalization. Full's reference is the faster measured fused/sequential median
per pair, retaining all reversals and anomalies. No selective rerun is allowed.

The retained serving SUCCESS rule is median S latency / median B latency <=1.10
in every mode, using pooled per-pair medians. Report both ratio-of-medians and
median-of-pair-ratios for full/S to avoid aggregation ambiguity. No general speed
claim follows from one forward versus two. Benchmark only within a root-granted
exclusive numerical-work window. Fitting times remain descriptive, not isolated.

SOURCE_FREEZE.json binds this protocol, code, both source populations, all
selected raw JPEGs, metadata and Fashion-MNIST split/source bytes. Root release
must bind its exact hash before training or scoring. A later model commitment
must precede scoring. The old branch and repository implementation remain intact;
new code is placed under experiments/distillation_replication.

## Primary-source context

- Yu and Grauman, Fine-Grained Visual Comparisons with Local Learning, CVPR2014:
  https://openaccess.thecvf.com/content_cvpr_2014/html/Yu_Fine-Grained_Visual_Comparisons_2014_CVPR_paper.html
- Official data, grouping and terms:
  https://vision.cs.utexas.edu/projects/finegrained/utzap50k/readme.txt
- Hinton, Vinyals and Dean, distillation: https://arxiv.org/abs/1503.02531
- Shanmugam et al., TTA aggregation: https://arxiv.org/abs/2011.11156
- Bouthillier et al., sources of benchmark variance:
  https://arxiv.org/abs/2103.03098
