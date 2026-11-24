# Pre-test interpretation clarification

The canonical practical timing threshold is the frozen configuration value
`practical_speedup=1.10`: the full-reference median divided by the limited
policy median must be at least1.10. This means limited latency is at most
full latency divided by1.10, or9.09% lower. It does not mean the stricter
criterion of at most0.9 times the full latency. The separate loss tolerance
remains0.01 nats. No frozen source, policy, model, or threshold is changed.

The pair-closed evaluation reuses already paid raw and fused-full logits.
Its reconstructed partner, `2*full_fused-raw`, equals the partner in real
arithmetic; FP32 batch-kernel rounding can introduce small discrepancies
against an independently executed partner batch. The exact Jensen identity
is a mathematical statement. Saved numerical identity checks have an
explicit tolerance, and independent true-partner checks are reported as
audits, not as free deployment calls or a formal floating-point proof.

The serving benchmark starts from normalized FP32 tensors. Dataset download,
IDX decoding, normalization, model loading, training, and selector fitting
are outside the steady-serving timer and must not be silently folded into
an end-to-end disk-to-prediction claim. GPU-transfer mode includes host-to-
device tensors, needed tie keys, all selector/model work, and output return.
