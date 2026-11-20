# Experimental protocol

## Scope and status

This document specifies the original repository setup, mathematical and numerical correctness checks, and short paired CPU pilot. That pilot does not select final hyperparameters or establish a performance ranking. Stage 4 has since been completed under a separate [predeclared tuning protocol](tuning_protocol.md); its [results and interpretation](tuning_findings.md) freeze settings for the pending formal study. The research question is whether a damped representation pullback produces more useful feature learning than ordinary parameter gradient descent, and whether any improvement survives a comparison by computational cost.

The core project uses one synthetic data distribution, one 80-parameter network, and two update methods. The [mathematical specification](mathematics.md) defines the model, objective, and updates. The [literature review](literature.md) documents their relationship to prior work.

## Reproduce this stage

From the repository root, install the project into the active Python environment, run numerical verification, and then run the pilot:

```sh
python -m pip install -e .
python -m pullback_feature_learning.verify --output runs/verification.json
python -m pullback_feature_learning.pilot --output runs/pilot
```

Proceed to the pilot only if verification exits successfully and reports `"passed": true`. The pilot command does not implicitly rerun verification. Both commands read [configs/pilot.json](../configs/pilot.json) by default; use `--config path/to/config.json` to select another explicitly saved configuration. Run `python -m unittest discover -s tests -v` for the additional regression checks.

The verification command refuses to overwrite an existing output file, and the pilot requires a fresh output directory. For a repeat run, supply new output paths to preserve previous evidence. The pilot creates per-step CSV records, JSON checkpoints, a resolved configuration, environment and data records, plots, and a generated results README. The reserved test split is specified by size and seed but is neither generated nor evaluated by this command.

## Pilot specification

| Item | Specification |
| --- | --- |
| Input and representation dimension | 2 |
| Network | Two residual blocks; hidden width 8; residual scale 0.5; tanh activation |
| Trainable parameters | 80; no trainable readout |
| Fixed readout | $w=(1,0)^\top$ |
| Loss | Mean binary logistic loss, with labels in $\{-1,1\}$ |
| Representation metric | Sum of squared Euclidean changes over training examples |
| Dataset | Balanced, noisy two moons with independently sampled angles |
| Split sizes | 256 training, 256 validation, 2,000 reserved test examples |
| Gaussian noise standard deviation | 0.15 per input coordinate |
| Split seeds | Training 1701; validation 1702; test 1703 |
| Parameter initialization seed | 23 |
| Input preprocessing | Raw inputs; no fitted or hand-selected normalization |
| Initialization | Independent $A$ entries from $N(0,1/2)$; independent $B$ entries from $N(0,0.01/8)$; zero biases |
| Device and precision | CPU, float64 |
| Numerical implementation | NumPy; analytic full Jacobian; independent reverse-mode gradient implementation |
| Thread budget | One BLAS thread, enforced with threadpoolctl |
| Warm-up | Five disposable updates per method, followed by an exact reset |
| Update count | 200 per method |
| GD learning rate | 0.5, provisional pilot value |
| Pullback learning rate | 32, provisional pilot value |
| Pullback damping | 0.01, fixed throughout the pilot run |
| Saved model checkpoints | Before training, after 100 updates, after 200 updates |
| Test-set use | Reserved; no test evaluation during this stage |

The initialization table gives **variances** inside the normal distributions: the corresponding standard deviations are $1/\sqrt2$ and $0.1/\sqrt8$. The residual branch starts small but has a nonzero $B$ so that the parameter blocks can participate in the initial dynamics.

The pilot learning rates are separately chosen starting values, not validation-selected final settings. If a stability problem requires an adjustment, retain the original configuration and failed-run evidence and record the reason for the new configuration. The selected values and run artifacts are the authoritative record of what was executed.

## Data generation and pairing

For each split, sample an equal number of examples from each moon. Draw an independent angle $t\sim\operatorname{Uniform}[0,\pi]$ for each example. Before noise, the first moon is $(\cos t,\sin t)$ with label $-1$, and the second is $(1-\cos t,0.5-\sin t)$ with label $+1$. Add independent $N(0,0.15^2)$ noise to each coordinate and shuffle with the split's random generator.

This continuous-angle construction uses independently sampled examples, rather than slicing a single deterministic evenly spaced curve into separate datasets. Training, validation, and test generators use distinct seeds. The two methods receive exact copies of the same arrays and initial parameter vector. Separate initialization and data seeds make it possible to vary either source of randomness deliberately later.

Input coordinates are left unchanged to preserve the explicitly chosen representation geometry and fixed-readout convention. A later linear-probe evaluation will fit its own standardization only on the training features; that is a separate evaluation procedure.

## Implementation and correctness gate

Both methods use full-batch updates and the same stable logistic loss. GD computes its parameter gradient with the independent reverse-mode implementation, without constructing a full Jacobian as part of its required update. The pullback method constructs the full representation Jacobian and solves the damped $80\times80$ system. It does not explicitly invert the matrix.

Before accepting the pilot, validate the following in float64:

- Representation and parameter dimensions, deterministic data generation, and paired initial states.
- Agreement between reverse-mode gradients and $J^\top g$.
- Agreement between analytic Jacobian directional products and centered finite differences.
- The damped linear-system residual and the two SVD-based feature-velocity predictions.
- Negative directional loss derivative at a nonstationary state, and loss decrease for a sufficiently small numerical step.
- First-order feature-motion agreement over decreasing step sizes, with expected error scaling before the floating-point floor.

Store actual errors and tolerances in the validation report. A short run that produces plausible-looking loss curves is insufficient to pass this gate.

## Pilot execution and recorded artifacts

1. Save the resolved configuration, software versions, platform information, thread settings, and initial parameters.
2. Run the correctness checks and stop if they fail.
3. Warm up numerical routines on disposable copies of the initial state. Warm-up must not consume measured training updates or alter the saved initial conditions.
4. Run both methods for 200 updates from identical initial conditions.
5. Record training loss, training accuracy, validation loss, validation accuracy, update count, elapsed core training time, and cumulative representation displacement. Do not evaluate the reserved test data.
6. Save initial, middle, and final parameter checkpoints and sufficient seed/configuration information to regenerate the input arrays.
7. Produce pilot plots, measured timing summaries, and a short interpretation of correctness, stability, and the next experimental budget.

Use the fixed training inputs to define cumulative displacement:

$$
D_T=\sum_{t=0}^{T-1}\left(\frac1n\sum_{i=1}^n\|z_i(\theta_{t+1})-z_i(\theta_t)\|_2^2\right)^{1/2}.
$$

This is RMS **per-example vector displacement**, not RMS over the $2n$ individual coordinates, and is not just the distance between the initial and final representations. Recording it from the first pilot avoids needing to reconstruct missing trajectories later.

## Timing boundaries

The required update time includes the forward computation, the gradient computations needed by the method, Jacobian construction where required, Gram-matrix construction, the damped linear solve where required, and the parameter update. The GD baseline must not be charged for a diagnostic Jacobian it does not need to train.

Additional loss/accuracy evaluations, feature-displacement diagnostics, SVD computations, and checkpoint collection in memory are counted as diagnostic time. Training-loop wall time and warm-up time are also recorded in `summary.json`; plotting time is reported in the generated README. File serialization, imports, and general setup are excluded from these training/diagnostic measurements. The recorded training-loop wall time therefore does not represent the full command's elapsed time. For these very small matrices, a single run is a rough timing measurement, not a precise hardware benchmark.

Use the same device, numeric precision, thread budget, and timing convention for both methods. Report mean or median measured step time alongside total update time, and use the pilot measurements to estimate a later training budget. A GPU is considered only if measured needs justify it.

## Exit criteria for this stage

The pilot stage is complete when the numerical checks pass, both methods produce finite trajectories from identical initial states, the logged behavior is interpretable, and the measured timing supports a concrete next-stage budget. A decreasing training loss is a useful stability check, but it does not imply convergence or improved feature quality.

If an update diverges, produces nonfinite values, or fails the local verification checks, record the failure and correct the underlying implementation or pilot configuration before proceeding. There is no requirement that the pullback outperform GD in a 200-step trial.

## Subsequent stages, not executed by this pilot

### Validation-only tuning

Predefine an equal tuning budget for the two methods and specify what is equalized, such as candidate-run count and per-run update budget; also report their actual tuning time. Search learning rates separately and choose a positive damping value for each pullback configuration, fixed within its run. Choose configurations using validation data, never the test set. The short pilot does not substitute for this process.

Fix training budgets, evaluation checkpoints, stopping rules, and the planned iteration/time comparisons before the formal runs. Do not assume that equal iteration count means equal computational effort. Final reports should state the tuning-budget convention rather than merely calling the comparison fair.

### Formal paired experiments

After selecting and freezing the configuration, run at least three paired random seeds. Each pair shares data splits and initialization across methods, and the seed-generation rule is declared in advance. Record per-seed outcomes as well as mean and variability. Evaluate the reserved test set under the frozen protocol; do not use those results to revise hyperparameters.

Report training loss and test accuracy against both iteration count and elapsed training time. Compare at explicitly matched budgets within the measured range; do not extrapolate one method's curve beyond observed training time.

### Representation analysis

At initial, intermediate, and final checkpoints, inspect Jacobian singular values and gradient alignment, with the numerical rank convention reported. Plot initial and learned point clouds. Fit regularized linear probes using training features; fit normalization only on training features and select regularization on validation features. Report held-out probe loss and accuracy and compare learned features with initial features. This probe is separate from the fixed readout used in network training.

Compare progress at matched cumulative feature displacement as a secondary diagnostic. Interpret point-cloud movement and spectral changes together with predictive performance; movement alone is not evidence of useful feature learning.

### Final report and optional extension

Combine the mathematical derivation, reproducible experimental setup, performance/cost comparison, and representation diagnostics. Improvements and limitations are both valid findings. A nuisance-feature distribution-shift study is optional and begins only after the core is complete. Shifted test data are never used for tuning.
