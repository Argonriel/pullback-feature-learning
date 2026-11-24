# What happened after the switch?

The same network was copied at pullback update 100 or 350. One copy continued with pullback at rate 8, and the other took GD updates at rate 2. Both copies used the same training data. At every checkpoint we recalibrated the region-based rule on the same 8,192 unlabeled inputs. Each pair used the same 32,768 held-out inputs, model-call budget, and independent tie order throughout its trajectory. We first examined 16 training and evaluation pairs and then repeated the complete design on 16 previously unseen pairs.

The gap below is the region-based rule's loss minus the least-confidence rule's loss. Positive values favor confidence. Each mean and median in the table is expressed in **thousandths of a nat per input**. Statistical intervals in the data files resample whole pairs.

## From the fork at update 100

| GD updates since fork | Discovery mean | Discovery median | Replication mean | Replication median | Region-based wins (discovery / replication) |
|---:|---:|---:|---:|---:|---:|
| 0 | 20.905 | 20.042 | 21.022 | 21.231 | 0/16 · 0/16 |
| 1 | 9.990 | 9.969 | 4.019 | 8.979 | 3/16 · 2/16 |
| 2 | −34.132 | −12.340 | 5.120 | 9.676 | 9/16 · 3/16 |
| 25 | 8.481 | 6.319 | 9.311 | 6.114 | 1/16 · 1/16 |
| 200 | 1.208 | 0.421 | 2.112 | 1.064 | 3/16 · 5/16 |
| 250 | 0.349 | 0.103 | −7.797 | 0.273 | 5/16 · 7/16 |
| 300 | −0.090 | −0.063 | −0.437 | 0.192 | 9/16 · 6/16 |
| 350 | −0.382 | −0.402 | −1.191 | −0.035 | 12/16 · 8/16 |
| 400 | −0.565 | −0.350 | −0.864 | −0.163 | 12/16 · 12/16 |

At the 500-update endpoint, the region-based rule won in 24 of 32 switched models; confidence won in 31 of 32 age-matched models that continued with pullback. The discovery and replication mean gaps for the switched arm had separate pointwise 95% bootstrap intervals of [−1.097, −0.077] and [−2.826, 0.551] thousandths of a nat. The relative endpoint ranking by seed count replicated. The discovery group's specific sign crossing between 250 and 300 GD updates did not replicate. In the second group one model, pair 17, had a gap of −0.140 nats after 250 GD updates; nine of the 16 still favored confidence. The late median crossed between 300 and 350 GD updates in this group.

The exact gap equals the orientation-specific benefit of using confidence minus the region-based rule's advantage when both orientations are treated symmetrically. At 200 GD updates, the orientation-specific benefit was larger in both groups. At 400 GD updates, the region-based rule's advantage was larger in both groups. In the second group these two means changed from 0.005317 and 0.003205 to 0.004447 and 0.005312 nats, respectively. Their relative sizes explain which rule wins. Their individual trajectories varied across seeds.

## The early disturbance and the later fork

After the first two GD updates following an update-100 fork, the original group's mean single-view loss rose from 0.339 to 0.536 nats. Twelve of the 16 models incurred higher risk. In the independent group, mean risk rose from 0.338 to 0.433 even as the median fell from 0.339 to 0.313. Only four models incurred higher risk, and one of them, pair 26, rose from 0.336 to 1.476. A few large effects can therefore change the direction of the mean in this regime. At 25 updates the confidence rule again won for 15 of 16 models in each group.

The update-350 fork reproduced the sensitivity to individual trajectories. At 150 subsequent GD updates, the mean was −0.02132 nats in discovery and +0.00044 nats in replication. The medians were +0.00008 and +0.00004 nats. Confidence won for nine original and eight replication models. These data give no sharp, shared cutoff after the later fork.

## Data and verification

The [discovery trajectories](trajectory_results/by_seed.csv), [independent replication trajectories](trajectory_validation_results/by_seed.csv), [cohort summaries with extremes](trajectory_validation_results/cohort_comparison.csv), and [replication design](TRAJECTORY_REPLICATION_DESIGN.md) record the evidence. The [endpoint check](trajectory_validation_results/endpoint_joins.csv) compared ten metrics across 128 independently computed trajectory/endpoint rows. The maximum difference was 1.12 × 10⁻¹⁶ nats or less.
