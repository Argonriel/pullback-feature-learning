# Paired augmentation and final-pipeline calibration

This bounded follow-up uses the official Fashion-MNIST corpus and the unchanged
426,122-parameter CNN from the previous study. It is not a pristine-data
replication: every official training image has appeared in a previous study,
and the official test results were previously observed. No previous weights are
reused. The new 10,000-image holdout is excluded from all fitting in this round.

## Data and labels

From the previous 40,000 training images, draw 1,000 per class without replacement
with seed 2161000. This is the within-round holdout. The remaining 30,000 images
plus the previous 8,000 selector-fit and 2,000 selector-validation images form
the new training set. The previous 5,000 validation and 5,000 temperature sets
are unchanged. The four sets partition the official 60,000 images, with exactly
4,000 / 500 / 500 / 1,000 images per class. All 10,000 holdout and 10,000 official
test images are evaluated; no remainder is excluded.

Each arm uses the same 50,000 unique fitting/selection/calibration labels:
40,000 training, 5,000 checkpoint selection, and 5,000 calibration. Fitting two
temperature parameters reuses the same calibration labels, not extra labels.
Unscaled reference predictions use only the first 45,000 labels. The official
test is a secondary descriptive replication and never selects any setting.

## Paired fixed training

There are three paired seeds. Initialization is 2162000 + pair, order is
2163000 + pair, and augmentation is 2164000 + pair. Each pair has identical
initial weights and each epoch has identical image order in both arms. The
augmented arm independently flips each training image along its width with
probability one half; this replaces the input and does not double calls.
The unaugmented arm uses the original image. Architecture, Adam learning rate
0.001, weight decay 0.0001, batch size 256, 12 epochs, FP32 and deterministic
runtime match the previous study. Checkpoints are retained at epochs 3, 6, 9,
and 12. Each arm selects its own lowest raw validation cross entropy, breaking
ties toward the earlier epoch. There is no hyperparameter search or replacement
of failed models. Failures and partial checkpoints are retained.

## Positive calibration and evaluation

For each selected model, query both views of the same 5,000 calibration images
once (10,000 forward examples). Let h be the original logits and
m = (h + h_flip) / 2. Fit inverse temperatures beta_raw and beta_full separately
by convex cross-entropy minimization on h and m, respectively, within [0.25, 4],
using 80 derivative-bisection steps. Five pipelines are frozen before outcomes:
raw h; raw_Traw beta_raw h; full m; full_Traw beta_raw m; and full_Tfull
beta_full m. Final-pipeline calibration does not change argmax predictions.
Two shared view queries per image construct all five evaluation outputs;
standalone deployment requires one forward for raw and two for full.

Primary comparisons on the within-round holdout are (1) augmented raw_Traw
minus unaugmented raw_Traw, and (2) augmented full_Tfull minus augmented
raw_Traw, each for cross entropy and classification error. The four intervals
use 10,000 paired image-bootstrap resamples, seed 2168000, with the same image
indices across all six models, and nominal 95% family coverage via four
Bonferroni intervals. These are ordinary bootstrap intervals, not finite-sample
certificates. Models are fixed in this bootstrap; three initialization/order
replications sharing data do not identify population variation over training
sets. Per-seed results and ordinary 95% secondary intervals are retained.

Predeclared diagnostics include every class's original / reflected / full
error, flip prediction agreement, reflected cross entropy, raw-to-full
cross-entropy change under both calibration choices, calibration endpoint
status, and expected loss on the artificial equal original/reflected image
pool. This constructed pool has exact flip symmetry but is not the original
image law. No symmetry guarantee is claimed on the original distribution.

## Timing and complete cost accounting

After quality is committed, measure only calibrated raw and full inference on
the unchanged validation inputs. Full is implemented both fused (2B images in
one CNN call) and sequentially (two B-image calls), and the faster measured
full implementation is the reference. Batch size is 256. For each model and
mode use three warm-up blocks and seven recorded blocks, each containing four
fixed validation batches. Modes are one CPU thread, GPU-resident data, and
CPU-to-GPU transfer including output return. Within each block randomize the
three methods with seed 2169000 + model index + mode offset. Synchronize CUDA
before and after each timed region. Include width flipping, concatenation,
both model calls as applicable, logit merging, and multiplication by the
appropriate fitted beta. Input decoding, normalization, model loading,
training, and calibration are outside the serving timer and separately counted.
No timing claim follows merely from forward-example counts. Benchmarks run
without other agents' numerical workloads. Check CPU/GPU prediction agreement
on the timed validation inputs, recording rather than hiding floating error.

## Freeze and access control

Source and this protocol are hashed in SOURCE_FREEZE.json. A split audit uses
only official training labels before release. Training and holdout scoring
require the parent's ROOT_APPROVAL.json in the study root, with authorization
checked before image decoding. A MODEL_COMMIT.json records selected weights,
all retained checkpoints, both calibration fits, and the split before quality
scoring starts. All sources and data files are copied into this branch; the
prior study remains unchanged. No method or fit is revised from holdout or
official-test results.
