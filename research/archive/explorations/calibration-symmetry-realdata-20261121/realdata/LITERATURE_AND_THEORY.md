# Real-data provenance and scope

Sources were checked on21 November2026 before official-test access.
Fashion-MNIST contains60,000 training and10,000 test images, each28x28
grayscale with one of ten clothing categories. The official Zalando
repository publishes the download files, checksums, and MIT license; these
are retained in `data/`. No external model weights are used.
[Official repository](https://github.com/zalandoresearch/fashion-mnist),
[Xiao, Rasul, Vollgraf,2017 dataset paper](https://arxiv.org/abs/1708.07747).

Horizontal reflection is a semantic assumption about these broad clothing
categories. It is not asserted to preserve the original empirical image
distribution. This study therefore separates ordinary test performance from
an explicitly reflected-pair evaluation. Official dataset documentation
does not establish the stronger distributional-invariance assumption.

Learning input-specific test-time transformations already has precedent in
[Kim et al., NeurIPS2020](https://arxiv.org/abs/2010.11422), and learned
aggregation in [Shanmugam et al., ICCV2021](https://arxiv.org/abs/2011.11156).
Our98-dimensional pooled-pixel ridge surrogate is a new implementation
adaptation for this experiment, not the preceding low-dimensional grid
interpolation method and not a general novelty claim.

The timing design follows the need for warmup and synchronization described
in the [official PyTorch benchmark tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).
The installed runtime is recorded separately; the experiment uses installed
PyTorch2.7.1+cu128, eager FP32 inference, TF32 disabled, and an RTX5070Ti.
CPU and GPU quality are both evaluated rather than assuming identical masks.

## Multiclass identity

For a logit vector z, A(z)=log(sum(exp(z))) and the loss for label y is
A(z)-z_y. Put zT=h(Tx), m=(z+zT)/2 and
J=[A(z)+A(zT)]/2-A(m). Convexity gives J>=0. If the label is retained by T,
the average of the two directional loss reductions is exactly J because
the label-dependent linear terms cancel. This is the multiclass version of
the same Jensen argument, not a first group-averaging guarantee.

The pixel features are identical under T: the symmetric image and absolute
antisymmetric image have the same entries after exchanging x and Tx.
Pooling these entries and applying a frozen pointwise score preserves that
identity. Top-quarter allocation with unchanged tie keys therefore keeps
the entire mask unchanged after any subset of individual input flips.

For fixed original-image batches, independently randomize each input's
orientation with probability1/2. The mask is unchanged, so expected raw
minus allocated loss over these orientations equals batch-mean selected J.
The reported paired score computes that conditional expectation by pairing
each original image with its reflection and retaining its category label.
Batch membership and finite test images are fixed. This is not a statement
about new IID batches from the original Fashion-MNIST distribution.

The exact-J selector maximizes selected J at the fixed quota, but it queries
all partners and costs two CNN forward examples per input. Its upper-envelope
interpretation concerns this paired symmetric objective; it does not imply
minimum realized loss on the original orientations. Confidence selection
need not have an invariant mask and is evaluated separately.

Floating-point logits from different batch shapes may differ slightly.
The numerical paired diagnostic reconstructs a partner from paid raw and
full outputs, with explicit tolerances and separate true-partner auditing.
No formal interval-arithmetic machine certificate is asserted.

## Statistical and computational boundaries

The three models share one training split and differ only in initialization
and training order. Test uncertainty is summarized with paired original-batch
bootstrap clusters, retaining the final16-image batch and averaging the
three fixed models. Reflections are never counted as additional independent
images. These are approximate empirical comparisons, not population-wide
guarantees over arbitrary training sets or distributions.

The benchmark counts model input examples and times all selector, transform,
indexing, inference, merge, and required transfer operations. It starts from
normalized FP32 tensors, with model and tables resident. Dataset download,
decoding, normalization, loading, training, and calibration are separate
setup components; steady-serving latency is not a disk-to-prediction timer.
The full reference is the faster measured implementation of fused or
sequential two-view inference. A smaller example count does not imply a
latency improvement, and use of a GPU for training does not establish that
the GPU is necessary for serving.
