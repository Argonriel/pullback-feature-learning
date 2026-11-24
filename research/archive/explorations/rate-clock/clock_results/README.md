# Matching elapsed GD time

All three GD runs for a seed start from the same update-100 pullback checkpoint. We extended each run until its nominal gradient-flow time, learning rate times post-switch updates, reached 1,600. We report binary log loss in nats per example. Each model had the same budget of 64 second views per 256 inputs. We retained 32 paired seeds from the preceding switch-time study.

## The small rate catches up on routing preference

| Comparison | Rate 0.5 | Rate 1 | Rate 2 |
|---|---:|---:|---:|
| Region rule wins at 400 GD updates | 3/32 | 14/32 | 24/32 |
| GD updates at nominal time 800 | 1,600 | 800 | 400 |
| Region rule wins at nominal time 800 | 25/32 | 25/32 | 24/32 |
| Region minus confidence loss at nominal time 800 | −0.000554 | −0.000555 | −0.000715 |
| Region rule wins at nominal time 1,600 | 32/32 | 32/32 | 31/32 |
| Region minus confidence loss at nominal time 1,600 | −0.001899 | −0.002077 | −0.003737 |

Let D denote region loss minus confidence loss. At 400 updates the mean difference in D between rates 2 and 0.5 was −0.003498 nats (95% paired bootstrap interval −0.005263 to −0.002096). Giving rate 0.5 four times as many updates while holding the rate-2 endpoint fixed changed the difference to −0.000161 (−0.000953 to +0.000507). The absolute mean contrast shrank by 95.4%. This percentage describes the observed contrast and is not an estimate of a causal fraction. The same routing rule won in 29 of the 32 seed-matched pairs at the two extreme rates. At nominal time 800 the 16 original seeds favored the region rule in 13, 13 and 12 cases across increasing rates; the second cohort did so in 12 cases at each rate.

The matching aligns the aggregate routing preferences closely. The training trajectories still differ. At nominal time 800 mean training losses were 0.01312, 0.01524 and 0.02134. The rate-2 minus rate-0.5 contrast was +0.00823 nats (+0.00309 to +0.01710). At that time the mean held-out single-view losses were 0.06094, 0.06215 and 0.06612, with a wider interval for the extreme-rate difference (−0.00095 to +0.01401). Nominal time measures neither distance in parameter space nor the number of gradient evaluations. These data do not identify the infinite-time behavior.

## Data and checks

The [per-seed results](by_seed.csv) retain raw, full, region and confidence risks; both components of the region–confidence difference; quotas; and the training loss. The [time series summaries](clock_summary.csv), [matched-time differences](clock_contrasts.csv) and [400-update versus time-800 comparisons](recovery_contrasts.csv) report the two cohorts separately and together. Each bootstrap resamples full seed pairs 20,000 times; intervals are pointwise.

The [training archive](training_paths.npz) preserves the exact float64 parameters and training loss after every GD update, including the shared starting state. Keys use the seed index and ten times the learning rate. For instance, `weights_p1_r5[1600]` is seed 1 after 1,600 updates at rate 0.5. The [anchor check](archive_anchors.csv) reproduces 320 existing policy measurements from the preceding study to a maximum absolute discrepancy of 4.86 × 10⁻¹⁷. Each endpoint evaluation verifies the quota and risk decomposition. The [metadata](metadata.json) records model, code hashes and sampling design.
