# Independent real-data negative-transfer check

The negative transfer on the original Fashion-MNIST test distribution was reproduced. This audit found no image/label ordering error, preprocessing-axis error, wrong flip dimension, checkpoint mismatch, or materially inconsistent saved CNN output. It ran only after the serving benchmarks completed and did not train, refit, or tune anything.

The four compressed IDX files match the MD5 values published by the [official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist#Get-the-Data). An independent IDX decoder checked magic values, byte lengths, 28×28 shape, and all 10,000 test labels, with 1,000 examples per class. The frozen permutation (seed 2136000) exactly matches all six CPU/GPU prediction archives, including all image-label pairings. Input normalization is uint8-to-float32 division by 255. The independently computed width-index reversal equals `torch.flip(x,[-1])` exactly and is an exact involution; rows, channels, and labels are unchanged.

An independently instantiated CNN with the saved state dictionaries replayed all original and partner inputs for all three seeds. The first 256 original examples per seed also replayed the exact separate-view deployment path. The complete audit added **61,536 model-example forwards in 126 invocations** (60,000 for full paired replay and 1,536 for separate-view checks); these are evaluation-only audit costs, not part of deployment or the timing benchmark. Direct logits are retained under `root_audit/realdata_transfer_v2/`. The original saved partner diagnostic was reconstructed as twice the averaged logit minus the first logit; this reconstruction can differ slightly in floating point and batch-shape kernel arithmetic, so it was checked against actual partner forwards rather than silently treated as a new exact forward result. The first audit attempt stopped before any CNN forwards because an exact NumPy-versus-CUDA normalization assertion was too strict; its source and failure record are preserved. The corrected check records the normalization difference and allows one float32 epsilon, while the two torch width-reversal implementations must still agree exactly.

| Model seed | Original loss / error | Flipped-only loss / error | Full-average loss / error |
|---|---:|---:|---:|
| 1 | 0.261968 / 9.56% | 2.640250 / 29.15% | 0.568933 / 20.32% |
| 2 | 0.253530 / 9.29% | 2.903917 / 33.28% | 0.633897 / 22.97% |
| 3 | 0.268139 / 9.93% | 2.602538 / 30.95% | 0.578828 / 20.43% |

`realdata_transfer_v2/classwise.csv` contains all 90 seed/view/class rows; `confusions.csv` contains every 10×10 confusion table, including zero cells. `pixel_asymmetry.csv` gives class-conditioned horizontal brightness-centroid means and class-mean-image flip differences as descriptive input-distribution diagnostics. They can support the plausibility of orientation imbalance but are not a proof identifying the causal source of failure or a population symmetry-defect bound.

The full-average Jensen identity holds for the equal-weight mixture of an original image and its flipped partner carrying the same class label. That paired construction changes the evaluated input distribution. A semantic intuition that horizontal reflection preserves clothing class does not establish that the original test distribution is reflection invariant. Therefore positive paired-set gain and harmful original-set transfer are compatible. The paired diagnostic is useful algebraic evidence, not a substitute for reporting the negative original-distribution outcome.

All source hashes, numerical discrepancies, ordering hashes, explicit extra query counts, and aggregate metrics are in `REALDATA_TRANSFER_CHECK.json`. This audit rules out the implementation mistakes it actually checks; it does not prove that no other modeling limitation exists.

## Where the negative transfer concentrates

Across the three independently replayed models, the footwear classes show the largest deterioration.

| Class | Original error | Flipped-only error | Full-average error |
|---|---:|---:|---:|
| Sandal | 2.27% | 55.27% | 18.60% |
| Sneaker | 4.00% | 59.13% | 10.90% |
| Ankle boot | 3.93% | 100.00% | 95.80% |

The original test images also have large positive mean horizontal brightness centroids for sandals (0.165), sneakers (0.160), and ankle boots (0.242), on a normalized column coordinate from −1 to 1. Ankle boots have the largest class-mean image/flip discrepancy. This alignment is consistent with learned sensitivity to the dataset’s directional presentation; it does not independently prove a causal mechanism or a population invariance-defect bound. The classifier was trained without horizontal-flip augmentation under the frozen recipe. These findings support retaining the original-test failure, rather than substituting the favorable artificial pair-closed result.
