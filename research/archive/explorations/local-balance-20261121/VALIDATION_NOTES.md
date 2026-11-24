# Preserved issues, corrections and interpretation

## Calibration wrap-node defect found before fresh generation

Independent review found that the first global-rank calibration placed some angular quadrature points exactly on sawtooth jumps. Floating-coordinate evaluation could then disagree with the modular FFT sum by approximately 0.00128 in the objective and 0.00141 in the moment. This was material. The complete first old64-by-three-width development, source variants and selected choice were preserved.

Before generating fresh PB models, the action phases were shifted by a fixed quarter grid cell and the entire old64-by-three-width development was repeated on the same inputs. The choice remained tau=1. Direct coordinate/FFT discrepancies fell to floating-point roundoff. The corrected algorithm, theorem, choice and evaluation protocol were then frozen. This was a numerical defect correction, not fresh-outcome tuning.

## Serving controls

The first GPU implementation gathered selected inputs with Boolean indexing. Because CUDA Boolean indexing can require dynamic nonzero extraction and synchronization, it was not sufficient for a definitive efficiency conclusion. A separately frozen fixed64-index implementation uses exact integer priorities for finite FP32 confidence values and independent tie ranks. All selected sets and outputs were checked against the initial paths, including synthetic duplicate, signed-zero, subnormal and extreme finite values. The original implementation and its freeze remain.

The first optimized timing runner also transferred tie keys for raw/full methods that did not use them. A separately versioned runner removes these transfers. This correction was frozen before any latency measurement. The final comparison includes combined512-example full inference, combined320-example input-only inference, actual routing, transfers where required, and both original/optimized implementations in common randomized blocks.

The main 80-parameter CPU experiment received the same equivalent fused-serving control. Its 224 full-input replays gave identical selected sets and zero maximum output difference. Definitive CPU serving measurements were taken after the main compute-heavy jobs finished, independently of the incidental timings collected during the main risk study.

## Diagnostic chronology and numerical precision

The scientific selector and primary validation protocol were frozen before fresh model generation. The evaluation-only conditional-moment diagnostic specification was frozen at09:41:46 UTC, eight seconds after primary scoring began at09:41:38. Its author had not opened fresh policy tables or seen test outcomes. The source was committed before that author accessed the tables. The report does not describe this ancillary diagnostic as frozen before primary scoring.

Initial rank diagnostics showed finite radial/angular quadrature sensitivity. A single uniform all16-model refinement was recorded after those results, solely to assess numerical precision with unchanged policies. Both levels and their differences are retained. Numerical moment widths remain estimates, separate from the deliberately conservative analytic bound.

## Other checks and costs

Coarse finite-difference derivative checks were inaccurate at some outer radii. The interval formulas did not change; a recorded step-size convergence audit separated truncation error from roundoff. Every old/fresh interval enclosure and final derivative diagnostic passed. The analytic theorem remains a real-arithmetic result, with ordinary padded float64 implementation rather than directed rounding.

A root-only summary print initially failed to serialize a NumPy integer into JSON. Casting scalar values fixed the presentation; no experiment, saved observation or scientific result changed.

The initial shell network probe for GPU packages failed DNS. Authorized network execution subsequently installed the official CUDA runtime into an isolated project environment and a real synchronized GPU matrix multiplication succeeded. The pre-install readiness note is historical; setup_record.json and the actual benchmark commits record the completed installation.

All mathematical verification, extra numerical refinement, and independent replay forwards are evaluation-only costs. They are not free deployment predictions and are not counted as independent trained-model replications.

The hardware summary originally attaches CUDA-scored quality to CPU timings of the same weights. A subsequent independent direct CPU replay evaluated full_fused, folded_fused and confidence_radial on all 16384 saved inputs for all nine models. All 27 rows and 1728 batch call records passed; no selected mask or class decision changed versus CUDA, maximum logit difference was 7.63e-6 and maximum expected-risk difference 1.77e-9. The audit costs 663552 additional forward examples. Original quality tables remain CUDA-scored; the separate direct CPU results are in root_audit/cpu_quality_checks.csv.
