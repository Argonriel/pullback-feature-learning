# The cost of updating the pullback metric

Eight paired two-moons runs per method used the fixed development settings. Each method used one thread on the same CPU. The timing round ran sequentially after the other training jobs finished. The table reports seed means. Each per-seed time endpoint was the last whole update inside the 100 ms core-training budget.

| Method | Updates in 100 ms | Validation loss at 100 ms | Validation accuracy at 100 ms | Time for 2,000 updates | Accuracy at 2,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GD | 1,308 | 0.0732 | 97.51% | 152 ms | 98.68% |
| Full pullback | 164 | 0.2045 | 98.10% | 1,237 ms | 97.56% |
| 100 pullback updates, then GD | 597 | 0.0694 | 98.10% | 207 ms | 98.78% |
| Refresh every 10 steps | 382 | 7.3740 | 62.06% | 516 ms | 83.54% |
| Refresh every 50 steps | 839 | 3.4843 | 83.11% | 234 ms | 81.84% |

The early switch preserved most of the speed of GD and reduced the mean time to 2,000 steps by a factor of about six relative to full pullback. In the 100 ms comparison its validation loss was lower than GD's on four of eight pairs. Its small mean advantage on this task varied across pairs.

Periodic refresh saved time relative to full pullback, but both refresh intervals substantially worsened the validation results with the frozen learning rate and damping. For example, at 500 updates the mean losses were 6.306 and 4.010 under the two intervals, against 0.100 for full pullback. The experiment did not search lower learning rates or alternate damping for the periodic variants.

## Numerical implementation

Each periodic method keeps the training gradient current and factors the Jacobian on a refresh. Its right singular vectors and squared singular values give the eigensystem of the Gram matrix. Between refreshes the code applies the cached damped metric solve to the new gradient. A first implementation tried an eigendecomposition of the formed Gram matrix. Its absolute numerical positivity check aborted the first timing attempt. No endpoint data from that pass entered the comparison. All 40 scheduled trajectories were then run from scratch using the Jacobian SVD. The fresh solve agreed with the original pullback to a relative error of 2.19e-13 at the independent check state. All final trajectories remained finite. Cost measurements include computing and applying the chosen factorization.

The two-moons transfer and direction experiments independently reproduced the pure and switched baselines at the common fixed checkpoints. The audit checks 104 overlapping seed-level comparisons, all 160 timed or fixed endpoints, every time bracket and the counts of metric builds. Timing is a local measurement from one run per method and seed.

## Files

- [Method and endpoint design](PLAN.md)
- [Training and timing code](experiment.py), run as `python experiment.py FRESH_OUTPUT_DIR` with the project installed.
- [Seed-level measures](run01/paired_endpoints.csv), [summary](run01/summary.json), and [environment and order](run01/run.json)
- [Audit and figure source](analyze.py), run as `python analyze.py` to reaggregate the archived round alongside its sibling experiment folders.
- [Figure](run01/validation_cost.png)
