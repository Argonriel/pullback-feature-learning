# Reproduction and provenance

The archive is a preserved research run, not a directory whose outputs should be overwritten. Work in a new copy or a new run directory. Every original failure, method freeze and corrected development run remains available.

## Environments

Main mathematical and PB experiments: Python 3.12.14, NumPy 2.5.5, threadpoolctl 3.8.0; figures use Matplotlib 3.11.2. All main CPU numerical jobs limit BLAS to one thread. The unchanged 80-parameter model implementation is source/pullback_feature_learning/core.py.

The GPU extension uses a separate Python 3.12 environment: torch 2.7.1+cu128, NumPy 2.2.6 and threadpoolctl 3.6.0. Complete installed versions are in gpu/environment_requirements.txt. Installation sources and actual CUDA verification are in gpu/setup_record.json. The existing main environment was not replaced. The local environment itself is excluded from the research archive; model weights, calibration tables and outputs are included.

The PDF is generated with build_report.py and the bundled ReportLab runtime. Rendering checks use Poppler. Figures are also saved as standalone PNG and PDF files.

## Source boundaries and data ordering

selection/FREEZE.json freezes the corrected action family, all three candidate policies, the old-only tau choice, theorem and development records. The full initial development is kept in selection/development; the corrected complete rerun is selection/development_v2. The portable 64-model old bank is selection/development/models.npz. These models are development data only.

bounds/freeze.json freezes the local interval-derivative configuration before fresh generation. The parity-only attribution control has its own freeze. root_audit/certificate_freeze.json fixes the independent confidence sample and root_audit/attribution_freeze.json fixes its descriptive controls. root_audit/cpu_serving_freeze.json fixes an equivalent fused CPU deployment comparison.

validation/freeze.json binds the final method and model sources. validation/ROOT_APPROVAL.json records the verified source hashes and that the fresh-state directory did not exist before generation was released. Training, calibration and test scoring have separate commits. All 16 fresh PB models and all candidate/baseline outcomes must be retained.

gpu/freeze.json fixes the genuine MLP architecture sweep before training. The fixed-index GPU implementation and transfer-timing correction have separately versioned freezes and preserve the initial implementation. They change serving mechanics, not learned models or masks. GPU timing requires a recorded approval after the main CPU jobs finish.

## Main stages

For a new run, preserve old input banks and frozen reference sources, then run the following stages in order using new output directories. Scripts intentionally refuse to replace existing output directories or immutable commits.

1. Reproduce old-only selection development with selection/develop.py, or retain the supplied frozen choice. Consult selection/PROTOCOL.md for its versioned input bank and output names.
2. Freeze validation using `python validation/study.py freeze`; a root approval record must bind that freeze before `python validation/train.py train`.
3. Run `python validation/study.py calibrate`, then `score` and `analyze`. This saves all model tables before test outcomes and records actual call sizes for every batch.
4. Run `python bounds/experiment.py fresh` against the saved continuous-field tables, then the separate parity control and analysis scripts described in bounds/README.md.
5. Run `python root_audit/certificate.py run`, then `python root_audit/attribution.py`. The primary confidence sample is separate from fitting and risk comparison. Each model uses 262144 independent inputs and 524288 charged verification forwards.
6. Run the frozen selection diagnostic script against the saved new rank tables; it uses the true top-64 marginal and reports approximation errors separately.
7. Run `python root_audit/cpu_serving.py` after compute-heavy jobs finish. Its equivalent input-only paths use one combined320-example call, with full-output and mask replays.
8. GPU stages are `python gpu/benchmark.py train`, `calibrate` and `score` after its freeze. The separately frozen optimized timing runner performs the fair original/optimized comparison after the timing approval; consult gpu/README.md for the final runner and addendum.

The exact command names and final branch-specific checks are documented in each component README. Hardware timing will vary; repeat measurements only in a new directory and retain the original records.

## Interpretation limits

The mathematical enclosure theorem is a real-arithmetic result. The recorded implementation uses ordinary float64 arithmetic with padding and independent numerical checks, not a formally directed-rounded proof. Confidence refers to expected risk under the specified symmetry assumptions. New rank-policy diagnostics distinguish its exact even-frequency control from approximate learned balance, and numerical integration from analytic bounds.

The GPU MLPs use 1024 training observations and ordinary Adam, whereas the main PB study uses 256 training observations and a different architecture and optimizer. Their risks cannot be used as a controlled causal comparison of training algorithms. The GPU branch tests deployment quality and costs on real networks at three fixed sizes.
