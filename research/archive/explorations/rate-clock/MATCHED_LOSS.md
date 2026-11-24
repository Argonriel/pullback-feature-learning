# Do the routing rules agree at the same training loss?

All three learning rates started at the same 100-update pullback checkpoint for each of 32 seeds. We compared the region and first-view confidence rules when different GD trajectories reached an equal **training** cross-entropy. Each rule then used 64 second views per batch of 256 on independent evaluation inputs. Positive D means the region rule has greater held-out loss and thus favors confidence.

We bracketed each loss target between successive GD iterates and bisected that segment of parameter space, matching training loss to within 2.0 × 10⁻¹⁴ nats. We evaluated both the interpolated model and the first actual GD iterate to cross the threshold. Each comparison uses the same seeds across all three rates. Analyses span the 32 seeds studied previously, including the second 16-seed cohort from the switch-time study.

## The preselected common loss level

At training loss 0.05, the rules had nearly the same relative performance across rates, while update counts differed.

| GD rate | Median updates to cross | Seeds favoring regions | Mean D (nats) |
|---:|---:|---:|---:|
| 0.5 | 148.5 | 0/32 | +0.002689 |
| 1 | 99 | 0/32 | +0.002732 |
| 2 | 88.5 | 1/32 | +0.002755 |

The paired mean D at rate 2 minus D at rate 0.5 was +0.000066 nats (pointwise 95% bootstrap interval −0.000393 to +0.000486). Comparing the actual crossing iterates gave −0.000243 (−0.001184 to +0.000493). Requiring the next ten units of nominal time to remain below the threshold also produced a small interpolated difference, −0.000118 (−0.000642 to +0.000379). The largest actual-iterate overshoot at 0.05 was 0.0030 nats at rate 2, compared with 0.0004 at rate 0.5.

We also checked common training losses of 0.20, 0.10, 0.075, 0.04, 0.03 and 0.02. The pointwise 95% intervals for the first-crossing difference between rates 2 and 0.5 included zero at every fixed loss. All 32 seeds contributed to the higher targets through 0.05. The common sample fell to 31 at 0.04 and 0.03 and 27 at 0.02. Lower thresholds describe the seeds that reached them by nominal time 1,600. Overlapping performance estimates do not establish equality of the underlying trajectories.

## Matching the original 400-update endpoint

For each seed we also took the training cross-entropy attained by GD at rate 2 and update 400. These 32 target losses ranged from 0.00213 to 0.14970 nats. Each smaller rate reached its own seed's target. It took median 1,147 GD updates at rate 0.5 and 605.5 at rate 1. With each faster model kept at its original 400th update, the numbers of seeds favoring the region rule were **18/32, 18/32, and 24/32** at rates 0.5, 1, and 2. At a fixed 400 updates these counts had been 3, 14, and 24. The paired mean difference in D between rates 2 and 0.5 was −0.001329 nats (−0.003245 to −0.000218). The original and second cohort means both pointed toward the region rule at rate 2. The second cohort's interval extended just above zero.

## The same target may be visited more than once

A follow-up sensitivity check asked whether a chosen visit to the target mattered. For 12/32 high-rate trajectories the first visit to their 400th-update loss occurred earlier than step 400. In one seed the first visit occurred at step 43. The same loss therefore admits a meaningful choice of checkpoint for the high rate itself.

* Selecting the **first** downward crossing at all three rates gave 18/32, 18/32, and 20/32 region wins. The pooled mean D difference between rates 2 and 0.5 was −0.000451 nats (−0.000855 to −0.000108).
* Requiring **every** crossing to remain below its target for ten more units of nominal time gave 17/31 region wins at each rate. The mean D difference was −0.000294 nats (−0.000604 to +0.000030).

The observed differences between rates at 400 updates became much smaller after matching training progress. A modest residual depends on the target and on the particular visit used. Neither a single routing gap nor one scalar training loss identifies all model predictions, and these finite trajectories leave their long-run limits open.

[Prespecified matching rules](MATCH_DESIGN.md) · [Coverage by seed and rate](match_results/coverage.csv) · [Matched results](match_results/by_rate.csv) · [Paired contrasts](match_results/contrasts.csv) · [Symmetric first-visit sensitivity](match_results/reference_first_all_contrasts.csv) · [High-rate revisits](match_results/reference_high_rate_revisits.csv) · [Independent numerical checks](match_results/audit.json)
