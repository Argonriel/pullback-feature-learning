# A 100-step pullback warm-up followed by gradient descent

At step 500, the warm-up schedule had higher validation accuracy and lower validation loss than plain GD in all eight pairs. Plain GD reached almost the same mean validation loss when allowed the approximate compute budget of the warm-up schedule. By step 2,000, schedules that finished with GD had lower mean validation loss than schedules that finished with pullback.

| First 100 steps | Remaining steps | Validation loss, step 500 | Validation accuracy, step 500 | Validation loss, step 2,000 | Validation accuracy, step 2,000 | Pullback solves by step 2,000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GD | GD | 0.26558 | 88.77% | 0.04268 | 98.68% | 0 |
| GD | Pullback | 0.09191 | 98.63% | 0.08230 | 97.95% | 1,900 |
| Pullback | GD | 0.07816 | 97.90% | 0.04156 | 98.78% | 100 |
| Pullback | Pullback | 0.10021 | 98.10% | 0.09841 | 97.56% | 2,000 |

All values are paired-seed means. Accuracy uses the original fixed readout and weights from the stated step. Schedules with the same first optimizer inherit exactly the same weights at the switch. At step 100, validation accuracy was 96.48% for pullback and 84.38% for GD.

The warm-up saved 1,900 pullback solves relative to full pullback by step 2,000, while lowering validation loss in eight of eight pairs. Against plain GD at step 2,000, the warm-up improved mean validation loss by 0.00112, with lower loss in only three pairs. Relative to full pullback, the warm-up used more ordinary GD steps and fewer full Jacobians and linear solves.

## Comparing approximate computational budgets

The original formal run measured about 8.4 GD updates per pullback update on its single-threaded CPU. That prior measurement assigns the 500-step hybrid approximately 1,240 GD updates, and the 2,000-step hybrid approximately 2,740. This comparison uses measured validation outcomes at these fixed GD steps. The resource conversion is a proxy from the earlier study, and this experiment did not directly time-match these runs.

| Hybrid steps | Approximate equivalent GD steps | Hybrid validation loss | GD validation loss | Pairs with lower hybrid loss | Hybrid accuracy | GD accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 1,240 | 0.07816 | 0.07907 | 4/8 | 97.90% | 97.46% |
| 2,000 | 2,740 | 0.04156 | 0.04028 | 3/8 | 98.78% | 98.68% |

The advantage at 500 equal updates supports a warm-up effect on this task. The paired results at comparable estimated cost provide little evidence for a practical speedup over plain GD. The switch used the previously observed early-training timescale. These exploratory comparisons use eight fresh validation draws from one noise setting and one architecture. A choice of switch time based on these results would need assessment on additional independently generated data.

## Reproduction and records

Use the project's Python environment with the package installed from the specified checkout. Run `python run_switch.py --repository PATH_TO_CHECKOUT --output NEW_DIRECTORY`. The plan was recorded before execution. [Plan](PLAN.md)

Seed-level checkpoints include counts of each update type, parameter hashes, and training and validation measures. Group summaries report descriptive sample standard deviations. Paired differences use hybrid minus comparator and report the number of seed pairs favoring the hybrid. [Checkpoints](results/checkpoints.csv) · [Means and SDs](results/means.csv) · [Paired differences](results/hybrid_paired_differences.csv) · [Seeds and software](results/metadata.json)

