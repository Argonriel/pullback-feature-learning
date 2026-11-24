# Does the smaller learning rate just need more steps?

Yes for most of the previously observed change in the preferred routing rule. With 400 GD updates at every rate, the region rule beat first-view confidence in 3 of 32 models at rate 0.5, compared with 24 of 32 at rate 2. Giving the smaller rate four times as many updates changed its count to 25 of 32. This revises the interpretation of the earlier fixed-update comparison. The smaller rate needs more gradient evaluations to reach that point.

[Figure (PDF)](rate_clock.pdf) · [Figure preview](rate_clock.png) · [Updated final manuscript section](final_section.tex) · [Related work and primary sources](related_work.md)

## Three comparisons from identical starting weights

All 32 seeds start their GD runs from the same seed-specific state after 100 pullback updates. Each model gets a second view on exactly 64 of 256 inputs. We compare selecting the least confident first predictions with selecting regions where a half-turn has a large estimated benefit. All three rates for a seed share the same data and initialization, including 8,192 unlabeled calibration inputs and 32,768 held-out test inputs. These follow-up contrasts reuse both cohorts from the earlier switch-time study.

| How the models are compared | Rate 0.5 | Rate 1 | Rate 2 |
|---|---:|---:|---:|
| Region wins after 400 GD updates at each rate | 3/32 | 14/32 | 24/32 |
| Region wins at nominal time 800 (1,600 / 800 / 400 updates) | 25/32 | 25/32 | 24/32 |
| Region wins at nominal time 1,600 (3,200 / 1,600 / 800 updates) | 32/32 | 32/32 | 31/32 |
| Region wins at the first crossing of the seed's rate-2 step-400 training loss | 18/32 | 18/32 | 20/32 |

Nominal time means GD learning rate multiplied by GD updates since the switch. Positive D = region loss minus confidence loss favors confidence. At 400 updates, the mean paired high-rate minus low-rate difference in D was −0.003498 nats. At nominal time 800 it was −0.000161, with a paired 95% bootstrap interval of [−0.000953, 0.000507]. In this sample the absolute mean contrast shrank by 95.4%. The same rule won in 29/32 matched seeds at the extreme rates. At this time the two seed cohorts separately showed the same catch-up pattern. The percentage describes a change in one contrast. Winner counts summarize directions, and mean loss differences also reflect the sizes of the individual effects. [Full matched-time results](clock_results/README.md)

## If training loss is matched instead

All 32 seeds reached the prespecified training loss of 0.05 at every rate. Confidence won for 32, 32 and 31 seeds, respectively, when the training loss was matched within the crossing update. The rate-2 minus rate-0.5 mean difference in D was +0.000066 nats (pointwise 95% interval −0.000393 to +0.000486). Using the first actual GD iterate below target gave similar results.

For a later comparison we used each seed's own rate-2 training loss at GD update 400. Twelve of those 32 models had previously reached the same loss. The rate-2 count was 24/32 at the fixed endpoint and 20/32 at the first crossing. Keeping every path below the target for the next ten nominal time units gave 17/31 region wins at each rate. First crossings use interpolation between successive real GD states to match training loss. Actual-iterate and sustained-crossing results are archived separately. [Loss-matching design](MATCH_DESIGN.md) · [Threshold results](match_results/by_rate.csv) · [Symmetric first-crossing sensitivity](match_results/reference_first_all_by_rate.csv)

## Where the runs differ

At nominal time 800, mean training losses were 0.01312, 0.01524 and 0.02134 for rates 0.5, 1 and 2. In a separate draw of 8,192 inputs per seed, the extreme rates differed on 1.03% of single-view classifications on average, and their mean absolute difference in predicted class probability was 0.0156. These differences coexist with close agreement on which routing rule wins. [Held-out prediction comparison](matched_time_predictions.csv)

The curvature calculation gives an additional check at the switch. On a local quadratic, GD multiplies a perturbation along a positive-curvature eigenvector by one minus the step size times its eigenvalue. A magnitude greater than one corresponds to a step-size–curvature product above 2. In 31 of 32 starting states rate 2 exceeded this local scale along the steepest positive-curvature direction, compared with zero at rate 0.5. During the first 25 actual GD updates the training loss rose at least once in 31/32 runs at rate 2 and 2/32 at rate 0.5. The calculation characterizes local sensitivity at the switch. It does not establish a persistent stability regime or a cause of the final routing preferences. [Curvature and early loss measurements](curvature_results/by_seed.csv)

## Contribution and reproduction

Earlier research has already scaled GD steps inversely with learning rate, studied departures from continuous-time training and used predictive uncertainty to allocate test-time augmentation. This course experiment applies those controls to a specific downstream decision under a known rotation symmetry, where the two competing policies can be evaluated against the same distribution. Its finding concerns the learning rates, horizons and model family tested here. [Literature review](related_work.md) · [BibTeX entries](references.bib)

Use the project Python environment with the adjacent local studies `budgeted-rotation` and `switch-time`. In a fresh working copy run `clock_run.py`, `clock_summary.py`, `matched_loss.py`, `reference_first_hit.py`, `local_curvature.py`, `audit_paths.py`, `audit_loss_matches.py` and `plot_results.py`. Training, matching and curvature drivers require unused results directories. The [time-control design](DESIGN.md), [per-update losses and parameters](clock_results/training_paths.npz), [training path audit](independent_audit.json) and [loss-match audit](match_results/audit.json) accompany the results.
