# Rings after tuning at equal training time

On eight fresh noisy-rings pairs, the 100-step pullback warm-up and plain GD yielded similar mean validation loss in 0.10 seconds of core training on this CPU. Warm-up had lower loss on four pairs, and GD had lower loss on four. The positive average accuracy difference for warm-up was driven by a large gain on one pair, while GD had higher accuracy on five of eight.

| Method | Selected GD-phase rate | Mean whole updates | Validation loss, mean ± SD | Validation accuracy, mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| GD | 0.5 | 1,324.1 | 0.06764 ± 0.04367 | 97.67% ± 1.50 pp |
| Pullback for 100 updates, then GD | 2.0 | 599.7 | 0.06708 ± 0.03785 | 97.93% ± 1.29 pp |

The paired warm-up minus GD differences were −0.00056 in validation loss (sample SD 0.05453) and +0.26 percentage points in accuracy (sample SD 1.78 pp). Each seed contributes the mean of its three timing repetitions. Sample SD measures variation across eight pairs. This comparison gives little evidence of a practical advantage for either schedule under the measured conditions. The initial rings experiment used equal step counts and rates inherited from two moons. This follow-up changes the time allocation and tunes the GD-phase rates together, so their individual contributions to the change in ranking are unresolved.

## Tuning results

Both schedules used the same five candidates on four other pairs, with one timing measurement for each method/rate/pair. The criterion was mean validation logistic loss. The pullback rate, damping and switch point were fixed. Every trajectory completed with finite results.

| GD-phase rate | GD validation loss | Warm-up validation loss |
| ---: | ---: | ---: |
| 0.03125 | 0.66454 | 0.18617 |
| 0.125 | 0.21861 | 0.10653 |
| 0.5 | 0.05366 | 0.05036 |
| 2 | 0.05678 | 0.04236 |
| 8 | 0.13606 | 0.11890 |

The [selection record](results/selection.json) was written after all 40 tuning runs and before generating the eight new evaluation pairs. Each fresh pair used three randomly ordered, interleaved repetitions per method. Every selected warm-up run completed all 100 pullback updates. Each retained update ended within the budget, and its successor crossed it. Across all 48 evaluation runs, the earliest retained time was 99.8319 ms and the latest crossing time was 100.1893 ms.

## Files and reproduction

Run `python run_budget.py --output FRESH_DIRECTORY` using the installed project package. The included generator function is copied verbatim from the original rings experiment. Wall-clock endpoints can vary between executions or machines. The [protocol](PLAN.md) defines data seeds, timing boundaries and selection. Files include [all tuning runs](results/tuning_runs.csv), [the full grid](results/tuning_grid.csv), [all evaluation repetitions and their time brackets](results/evaluation_runs.csv), [within-seed means](results/evaluation_seed_means.csv), [paired differences](results/evaluation_paired.csv), [summary statistics](results/evaluation_summary.json) and [software fingerprints](results/metadata.json).
