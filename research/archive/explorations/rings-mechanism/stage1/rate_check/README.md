# Sensitivity of the population components to the GD rate

After the primary decomposition, we applied the previously selected rings rates to the same eight training draws and initializations. Pure GD used a rate of 0.5. The hybrid shared the exact original pullback prefix and used a GD rate of 2 for its final 400 steps. The preceding timing study selected those two rates on separate rings tuning pairs at a 100 ms budget. This follow-up measures fixed-step checkpoints without further tuning.

| Checkpoint and rate | Mean radial excess | Mean angular excess | Mean population loss |
| --- | ---: | ---: | ---: |
| GD 100, inherited 0.125 | 0.66267 | 0.01323 | 0.70420 |
| GD 100, selected 0.5 | 0.64753 | 0.00441 | 0.68024 |
| Pullback 100 | 0.29698 | 0.01191 | 0.33718 |
| GD 500, inherited 0.125 | 0.63564 | 0.00498 | 0.66891 |
| GD 500, selected 0.5 | 0.07691 | 0.03803 | 0.14323 |
| Pullback 500 | 0.06721 | 0.01599 | 0.11149 |
| Hybrid 500, inherited GD rate 0.125 | 0.05885 | 0.00928 | 0.09643 |
| Hybrid 500, selected GD rate 2 | 0.00527 | 0.03237 | 0.06593 |

At 100 steps full pullback beat GD with the higher rate on population loss in all eight pairs. Its 0.35056-nat radial advantage outweighed GD's 0.00749-nat angular advantage. By step 500, raising GD's rate reduced the mean radial term by 0.55873 nats while increasing the angular term by 0.03305. The selected-rate GD arm had lower total loss than inherited-rate GD in every pair and lower total loss than full pullback in four. A low angular penalty under the slow inherited GD setting therefore gave a misleading impression of overall progress.

Increasing the hybrid's GD-phase rate from 0.125 to 2 reduced its mean radial penalty by 0.05358 and raised its angular penalty by 0.02309 nats. With rate 2, angular variation accounted for about 86% of its mean excess loss at step 500, while radial fit was close to the population oracle. Total population loss decreased on four of eight pairs compared with its inherited-rate counterpart. This observation concerns the average of the 8 pairs and the specified endpoint.

All 24 new states were finite and passed polar grid refinement. The eight source training draws and eight pullback-prefix hashes matched the original data and states. The maximum grid-refinement difference was 1.39e-7 nats; the maximum decomposition residual was 1.39e-16 nats. [All states](states.csv) · [Paired differences](paired.csv) · [Integration audit](audit.json) · [Design](../../rate_check_plan.md) · [Driver](../../rate_check.py)
