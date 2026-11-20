# Supplementary 100,000-update diagnostic

All six trajectories completed exactly 100,000 updates under the [bounded long-training protocol](long_training_protocol.md), committed before execution in [7f753a3](https://github.com/Argonriel/pullback-feature-learning/commit/7f753a33acf26edbe679ea057bf098ed2044b218). Extending these fixed settings reduced training loss substantially but did not improve mean development validation performance relative to the 2,000-update checkpoint. Neither method met the declared criterion for proposing still longer training.

This is a post-formal exploratory diagnostic on the original three paired development seeds. Their validation sets were already used for tuning. No test or formal-stage data were generated or evaluated, no optimizer setting was retuned, and no intermediate checkpoint was selected to replace a formal result. The [original formal comparison](formal_findings.md) remains unchanged.

## Observed checkpoints

The table reports arithmetic means across the three development seeds. Both methods share training data, validation data, and initialization within each pair. GD uses learning rate 0.125; pullback uses learning rate 8 and damping 0.1. These settings were selected for 2,000 updates, not optimized for the longer horizon.

| Updates | Method | Training loss | Validation loss | Validation accuracy |
| ---: | --- | ---: | ---: | ---: |
| 2,000 | GD | 0.016793042 | 0.049445960 | 98.307292% |
| 2,000 | Pullback | 0.016506184 | 0.072279237 | 98.177083% |
| 10,000 | GD | 0.001620621 | 0.082649760 | 98.307292% |
| 10,000 | Pullback | 0.003233407 | 0.084931862 | 98.177083% |
| 20,000 | GD | 0.000522732 | 0.094408474 | 98.307292% |
| 20,000 | Pullback | 0.001610695 | 0.094531512 | 98.046875% |
| 50,000 | GD | 0.000133307 | 0.109002938 | 98.307292% |
| 50,000 | Pullback | 0.000642654 | 0.109301878 | 98.046875% |
| 100,000 | GD | 0.000054193 | 0.120321729 | 98.307292% |
| 100,000 | Pullback | 0.000321023 | 0.120368026 | 98.046875% |

Full-precision values, per-seed results, descriptive sample standard deviations, and the other predeclared checkpoints are in the [archived results](../reports/long-training-v1/README.md). The displayed precision supports reproduction and does not imply measurement certainty. There are only three seeds, and sample SD is not a confidence interval.

At 100,000 updates, mean validation losses are nearly equal: 0.12032 for GD and 0.12037 for pullback, with sample SDs 0.02059 and 0.04142, respectively. The mean paired loss difference, pullback minus GD, is only 0.00004630. Pullback has lower validation loss in dev2 and higher loss in dev1 and dev3. Its validation accuracy is lower in dev1, equal in dev2, and higher in dev3; the mean paired accuracy difference is -0.2604 percentage points. These results do not support a blanket claim that GD wins every seed or a statistically established difference in terminal validation loss.

Pullback does retain an early advantage per update: at 500 updates its mean training loss is 0.06806 versus GD's 0.19608, and mean validation accuracy is 98.44% versus 89.84%. By 2,000 updates, GD has lower mean validation loss. The later narrowing of the validation-loss gap occurs while both methods' mean validation losses worsen, rather than through a late improvement in pullback's validation performance. Both methods achieve 100% training accuracy at the recorded checkpoints from 5,000 updates onward.

## Does the second half justify further training?

The declared continuation screen requires at least a 5% decrease in both mean training and mean validation loss from 50,000 to 100,000 updates, together with a validation-loss decrease in at least two of the three seeds.

| Method | Training-loss decrease | Validation-loss change | Seeds with lower validation loss | Criterion met |
| --- | ---: | ---: | ---: | --- |
| GD | 59.35% | 10.38% increase | 0 / 3 | No |
| Pullback | 50.05% | 10.12% increase | 0 / 3 | No |

Continued fitting therefore did not translate into better validation performance in the second half of this experiment. The evidence does not justify automatically extending the same settings beyond 100,000 updates to improve generalization. It also does not prove that training has mathematically converged, that no later improvement is possible, or that different settings would behave the same way. The 5% threshold is a practical review convention, not a statistical test.

## Measured costs

The entire batch took **219.66 seconds**, about 3 minutes 40 seconds. Core training consumed 187.1054 seconds, online diagnostics 28.4676 seconds, warmup 0.0133 seconds, and offline evaluation 0.0649 seconds. The remainder includes other recording, serialization, and plotting work.

At 100,000 updates, mean core training time per trajectory was **6.8418 seconds for GD** and **55.5266 seconds for pullback**, a ratio of approximately **8.12**. This is a cost measurement for this CPU implementation with one BLAS thread, not a universal complexity or hardware claim. The allocation matches update counts, not elapsed time; core timing excludes diagnostic work.

## Interpretation and next step

The original concern was that pullback might simply need more updates. Under the already selected settings and these development seeds, extending the budget fiftyfold does not produce improved validation performance or a late mean-accuracy advantage. The long horizon does not establish optimally tuned long-run performance: the settings were selected at 2,000 updates, and the pullback damping still lies at the upper boundary of the original grid.

The next planned work remains stage 6: frozen-feature linear probes, feature geometry, and spectral interpretation using the archived states. Fixed-readout accuracy alone does not determine feature quality. This analysis should explain the observed early motion and later behavior without presuming that a probe must favor pullback.

Reproducibility evidence is preserved in the [results archive](../reports/long-training-v1/README.md) and [independent artifact audit](../reports/long_training_audit.json). The diagnostic retains the configuration, schedule, source hashes and protocol revision, paired data and initialization fingerprints, saved parameters, complete sampled curves, and measured costs. No run beyond 100,000 updates was executed in this batch.
