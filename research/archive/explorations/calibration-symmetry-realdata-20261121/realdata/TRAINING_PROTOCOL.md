# Fixed real-data training protocol

Fashion-MNIST is downloaded from its official Zalando repository, with all
four compressed files verified against official MD5 values and retained
with the MIT license. Downloading and hashing test files does not decode,
inspect, or use test arrays. Test decoding is locked behind root approval
of a committed model and selector configuration.

The official 60,000 training images are stratified by label into 40,000
training, 5,000 validation, 5,000 positive-temperature calibration, 8,000
selector-fitting, and 2,000 selector-validation examples. The split is fixed
with seed 2131000 and shared by all three model initializations. Thus the
three models are initialization/training-order replications on the same
dataset, not independent datasets. All official 10,000 test images are held
out from every choice.

The fixed model has three 3x3 convolutions with channels16,32,64, ReLU, two
2x2 max pools, and a128-unit dense head. There is no batch normalization,
dropout, or training augmentation. Inputs use the fixed scaling pixel/255.
Train12 epochs using Adam, batch256, learning rate0.001 and weight decay
0.0001. Evaluate validation loss only after epochs3,6,9,12; choose its
minimum, breaking ties by earlier epoch. Retain all checkpoints and traces.
There is no architecture, optimizer, or learning-rate search.

Initializations use2132001..3 and training-order streams2133001..3. FP32,
deterministic operations, one CPU thread, TF32 disabled, and no automatic
mixed precision are fixed. GPU training is authorized and its actual wall
time, forward/backward example counts, and validation calls are recorded.

Fit a single positive temperature on the independent5,000-example labeled
temperature set by convex inverse-temperature minimization on[0.25,4].
The deployment logits are divided by this fixed temperature. It does not
change class argmax. Save raw/calibrated validation diagnostics separately;
selector fitting uses the temperature-scaled frozen logits.

Failure states, trace, and exceptions are saved; no failed initialization is
replaced. This protocol and training sources are frozen before training.
The selector and timing protocol will be fixed before any official test
access, and all selected models and policies will be hash-committed before
root authorizes test evaluation.
