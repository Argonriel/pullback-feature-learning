# When does changing the optimizer change which inputs deserve a second view?

An unrotated prediction and a prediction after a half-turn can each be confident and still disagree. We compared two ways of spending 64 extra predictions per batch of 256 rings inputs. The confidence rule picks the 64 least confident first predictions. The region rule picks the 64 inputs whose radius and angle predict the greatest benefit from averaging the two views. Its regions pair opposite orientations, and their scores use only unlabeled calibration inputs. The outcome is binary log loss on held-out inputs, averaged over both starting orientations.

[Final manuscript addendum](final_addendum.tex) · [Four-panel figure (PDF)](switch_time.pdf) · [Figure preview](switch_time.png) · [Original two-view study and proof](../budgeted-rotation/README.md)

## The endpoint replicated across independent seeds

We trained the same model to update 500, switching from pullback (rate 8, damping 0.1) to GD (rate 2) after 0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 475 or 500 pullback updates. Each of the 32 seeds had the same training, calibration and test draws for all 13 training schedules, producing 416 trained endpoints. Each state had 8,192 unlabeled calibration examples and 32,768 evaluation examples. The first 16 seeds repeat the earlier comparison. The other 16 were held out when choosing the grid.

| At update 500 | Original 16 | Independent 16 |
|---|---:|---:|
| All pullback, seeds favoring confidence | 16 | 15 |
| Switch at 100, seeds favoring the region rule | 12 | 12 |
| Paired switch shifts preference toward the region rule | 16 | 13 |

Write D = region loss minus confidence loss, so a positive number favors confidence. In the new cohort, D averaged +0.001929 nats under all pullback and −0.000864 after switching at 100. The paired decrease from switching was 0.002793 nats (95% seed bootstrap interval 0.001224–0.004922). The new cohort's interval for the region rule's advantage on the switched arm alone includes zero. This makes the within-seed change in preference the more precise finding. Neither cohort's fixed-horizon curve supports a single monotone cutoff in the number of pullback updates. Intermediate schedules occasionally have extreme losses. At switch 200, one held-out seed had D = −0.413 nats and single-view loss 2.89. We retained all seeds and report win counts alongside mean losses.

[All 13 endpoint comparisons](endpoint_curve.csv) · [Paired change with the schedule](paired_anchor_contrasts.csv) · [Simultaneous bands across each curve](paired_audit_bands.csv)

## What happens after the original switch

Starting at the exact same update-100 state, we either continued pullback or switched to GD. These counts refer to the switched branch. At the instant of the fork all 32 favored confidence.

| GD updates since the switch | Total updates | Original seeds favoring the region rule | New seeds favoring the region rule |
|---:|---:|---:|---:|
| 200 | 300 | 3/16 | 5/16 |
| 250 | 350 | 5/16 | 7/16 |
| 300 | 400 | 9/16 | 6/16 |
| 350 | 450 | 12/16 | 8/16 |
| 400 | 500 | 12/16 | 12/16 |

The original mean D crossed from positive at +250 GD updates to negative at +300. This bracket failed the independent replication. At +250 the new cohort's mean was already negative, while its median was positive. One validation model had D = −0.140 at that age. There were also temporary reversals during the first few updates. The data support a shift toward the region rule late in this schedule, with substantial differences among models. A separate fork at 350 likewise had an irregular early response. Continuing pullback favored confidence on average at every observed matched age in both groups.

[Trajectory analysis](TRAJECTORIES.md) · [Matched trajectories in both cohorts](trajectory_validation_results/cohort_comparison.csv) · [Independent replication design](TRAJECTORY_REPLICATION_DESIGN.md)

## Why the rules exchange places

The region rule can spend more of the budget on pairs where combining opposite predictions reduces loss substantially. Call this extra gain over the confidence rule's symmetric gain **H**. The confidence rule has another opportunity. Depending on which orientation is seen first, it may keep a good prediction while correcting a worse one. Call this signed contribution **A**. The exact difference satisfies **D = A − H**. Both are measured in nats per example.

| Checkpoint | Original A | Original H | New A | New H |
|---|---:|---:|---:|---:|
| All pullback at 500 | 0.00438 | 0.00185 | 0.00342 | 0.00149 |
| Switch at 100, then 400 GD | 0.00152 | 0.00209 | 0.00445 | 0.00531 |

The orientation benefit is positive at both endpoints in both cohorts. With all pullback it exceeds the region rule's extra gain. After 400 GD updates the ordering reverses. In the original 16, much of the change came from a smaller orientation benefit. In the new 16, both quantities rose and the region rule's extra gain rose more. The measured explanation is their relative size. The experiments leave open which internal parameter movements give rise to that balance.

## The post-switch learning rate matters

We held the branch point at update 100 and the next 400 training updates fixed. At GD rates 0.5, 1 and 2, the region rule won respectively **3/32, 14/32 and 24/32** seeds. Increasing the rate from 0.5 to 2 shifted D toward the region rule in 31 of 32 matched seeds. Both cohorts' paired 95% intervals for this rate change exclude zero. The runs with 0.5 and 1 both favored confidence on average in each cohort. The validation rate comparison reuses the endpoint-validation seeds and was added after inspecting the original rate comparison. These comparisons use update count as the training budget.

[Learning-rate results by cohort, rate, and age](rate_summary.csv) · [Paired learning-rate contrasts](paired_rate_contrasts.csv)

## Reproducibility

The [design](DESIGN.md) describes cohort selection and bootstrap inference. The [independent audit](independent_audit.json) covers all 416 endpoints, 53,248 endpoint batches, 1,728 matched-age records and 384 learning-rate records. It also reconstructs 12 selected checkpoints using independently written prediction and routing formulas, including both cohorts' extreme cases.

The scripts use the installed project environment and the adjacent `budgeted-rotation` study for the training and routing functions. In a fresh working copy with unused output destinations, run `run_endpoints.py`, `trajectory.py` and `rate_control.py` for pairs 1–16, then rerun each with `--first-pair 17 --last-pair 32` and the corresponding `--out` destination (`validation_results`, `trajectory_validation_results` or `rate_validation_results`). Run `summarize_endpoints.py`, `trajectory_comparison.py`, `audit_switch.py` and `figures.py` to regenerate summaries, checks and plots.
