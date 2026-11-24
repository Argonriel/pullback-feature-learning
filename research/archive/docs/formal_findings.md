# Stage 5: fresh paired test evaluation

Stage 5 is complete. The methods, seeds, and two evaluation budgets were frozen in [8e24690](https://github.com/Argonriel/pullback-feature-learning/commit/8e24690f4f2af0e2c20a28701a2c0ea591cef4e7); the [formal execution protocol](formal_protocol.md), runner, and tests were committed in [6d675f1](https://github.com/Argonriel/pullback-feature-learning/commit/6d675f114cd15407348d1faa16b71c9723683ba7) before executing the formal batch. No optimizer settings or stopping rules were changed after test evaluation.

## Execution and costs

All six scheduled trajectories completed. Each pair used identical training data and initialization. The three pairs used fresh, predeclared data and initialization seeds. Training used only training inputs and labels; validation and test arrays were generated and evaluated after all six training trajectories had finished.

The recorded batch took **19.14 seconds** including training, online diagnostics, offline evaluation, serialization, and plotting. Core updates consumed 12.0013 seconds; online diagnostics consumed 4.3656 seconds; offline metrics and spectra consumed 0.3842 seconds. The remaining time includes warmup, recording overhead, serialization, and plotting. Computation used float64 on the local CPU with one BLAS thread.

GD used learning rate 0.125. Pullback used learning rate 8 and damping 0.1. The latter remains a boundary selection from the declared finite tuning grid, not a proven globally best damping.

## Separate endpoint results

The entries below are mean plus/minus sample standard deviation across three seeds. Accuracy variability is in percentage points (pp), not a confidence interval.

| Endpoint | Method | Test accuracy | Test logistic loss | Mean updates | Mean core training time |
| --- | --- | ---: | ---: | ---: | ---: |
| Exactly 2,000 updates | GD | 98.87% +/- 0.20 pp | 0.03479 +/- 0.00265 | 2,000 | 0.14589 s |
| Exactly 2,000 updates | Pullback | 98.22% +/- 0.33 pp | 0.07985 +/- 0.00817 | 2,000 | 1.22202 s |
| Last whole update at/before 2 s | GD | 98.82% +/- 0.25 pp | 0.06222 +/- 0.01991 | 27,195.7 | 1.99995 s |
| Last whole update at/before 2 s | Pullback | 98.22% +/- 0.30 pp | 0.08264 +/- 0.00922 | 3,280.3 | 1.99963 s |

For the fixed 2,000-update endpoint, GD has higher test accuracy and lower test loss in each of the three pairs. The mean paired accuracy difference, pullback minus GD, is **-0.65 pp**. The measured pullback core time is about **8.4 times** GD's at this update count. That ratio describes this explicit implementation and hardware, not all implementations of either method.

At the two-second endpoint, GD again has higher test accuracy in each pair and lower mean test loss. GD has lower test loss in two of the three pairs; in the first pair, pullback has slightly lower test loss (0.07894 versus 0.08154). The mean paired accuracy difference is **-0.60 pp**. Each retained time state falls at or before the budget, and its immediately following update crosses the budget. No parameter interpolation or performance extrapolation was used. The two-second comparison uses the same iteration-tuned settings, so it is a cost diagnostic rather than independently time-tuned optimization.

## Reading the learning curves

Pullback makes faster early progress per update in these trajectories, but this advantage does not survive the computation-time comparison. By update 2,000, GD reaches lower mean training loss and better fixed-readout test metrics for the frozen settings.

Additional GD updates within the two-second budget further reduce its training loss, while its mean test loss rises from 0.03479 to 0.06222. More optimization is therefore not automatically better held-out performance in this run. The endpoint rule remains unchanged: the test curves are reported, not used to choose a better stopping time.

These observations are consistent with a distinction between local feature-motion efficiency, update cost, and generalization. They do not, by themselves, identify which spectral directions or feature geometry caused the difference.

## Limits and the next analysis stage

The result is a small controlled comparison on one noisy two-moons distribution and one 80-parameter residual architecture. There are only three paired seeds. No significance claim, architecture-wide superiority claim, or distribution-shift conclusion follows. The finite tuning ranges and boundary damping selection remain limitations.

The reported classifier is the fixed readout used for training. Its performance is not the same as frozen-feature linear-probe performance. Stage 6 should use the archived initial, intermediate, iteration-endpoint, and time-endpoint representations to assess feature quality under one training/validation probe protocol, then relate those results to the already recorded spectra and cumulative feature displacements. No network retuning is performed in response to these test results.

## Reproducibility evidence

- [Raw formal results](../reports/formal-v1/README.md): all checkpoints, data, per-seed endpoint metrics, spectra, timing brackets, and paired summary values.
- [Formal test log](../reports/formal_test_results.txt): all 38 checks pass, including exact time equality, both endpoint orders, sparse logging, failed-prefix retention, and paired summaries.
- [Independent artifact audit](../reports/formal_audit.json): source/config pairing checks, metric recomputation, endpoint brackets, and deterministic replay of the six 2,000-update states.

The time-endpoint update count can vary on a repeated timing run. Its saved state and measured bracket are the audit target; the fixed iteration state is independently reproducible under the recorded numerical environment.
