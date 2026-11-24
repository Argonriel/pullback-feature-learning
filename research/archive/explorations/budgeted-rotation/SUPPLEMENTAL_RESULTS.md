# A first-view confidence baseline

The previous comparison showed that coordinates could help allocate second views. A classifier also gives a confidence signal in its first pass. We tested the fixed rule that spends the batch's 64 extra evaluations on the 64 smallest absolute first-view logits. We specified this comparison after seeing the main study and before evaluating a second draw of 32,768 independent inputs for each of the 16 original training pairs. All 48 checkpoints and all calibration tables were reused without fitting.

| Checkpoint | Single view | Random 25% | Radius 25% | Folded angle + radius 25% | Confidence 25% | Two views for all |
|---|---:|---:|---:|---:|---:|---:|
| Pullback (primary) | 0.110281 | 0.107927 | 0.101640 | 0.101127 | **0.098598** | 0.100865 |
| GD | 0.177578 | 0.170448 | 0.164380 | 0.156208 | **0.147296** | 0.149061 |
| Pullback followed by GD | 0.061167 | 0.056817 | 0.045177 | 0.044311 | 0.044876 | **0.043765** |

Entries are mean posterior log losses in nats per input. Each limited policy uses 1.25 forward evaluations per input. Random reports the expected loss for a uniformly chosen subset of exactly 64 per batch. For the confidence rule we reran selection on the entire swapped batch and averaged the two orientation-specific losses. Across all checkpoints, checks on the first batch in both orientations verified that deployment evaluated the network on 256 first views and exactly 64 second views. We checked the quota on every batch.

On full pullback, confidence achieved lower risk than the folded gate for **16 of 16 training pairs**. The mean difference (folded minus confidence) was **0.002529 nats**, with a pointwise 95% paired percentile bootstrap interval of **[0.001978, 0.003166]**. GD also favored confidence in all 16 pairs, by 0.008912 [0.005302, 0.012861]. The hybrid favored the folded gate in 12 of 16 pairs, with folded minus confidence at −0.000565 [−0.001097, −0.000077]. Intervals resample entire training pairs, with the same 16 pairs reused across models.

Confidence also beat averaging every input on mean risk for pullback and GD. Under full pullback, reversing all first views changed the confidence selection decision on 27.6% of examples. This lets the rule retain a confident single prediction in one orientation and average the same two logits when the other orientation receives the first pass. The symmetry argument for the folded gate uses the same selection decision in both orientations. These measurements support a benefit from conditioning on the first prediction for some optimizers on these rings. The confidence rule uses no calibration queries.

[Evaluation design](SUPPLEMENTAL_DESIGN.md) · [Per-checkpoint rows](supplemental_results/models.csv) · [Per-batch rows](supplemental_results/batches.csv) · [Exact estimates and archive hashes](supplemental_results/summary.json)
