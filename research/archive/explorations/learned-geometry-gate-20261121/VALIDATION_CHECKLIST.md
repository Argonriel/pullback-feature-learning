# Validation checklist and limits

## Passed implementation and numerical checks

- [x] Read repository AGENTS.md; inspected docs/reports/runs junction targets and preserved them.
- [x] Reviewed proposal, formal results, representation/probe reports, rings analyses, and both retired novelty claims.
- [x] Original-source literature checks completed with assumptions and overlap recorded.
- [x] Froze design and discovery configuration before training.
- [x] Documented both design repairs before generating or inspecting new predictive scores; original records preserved.
- [x] Full initial tangent map freezes both coordinates on training and evaluation inputs.
- [x] Initial nonlinear/tangent directions agree within numerical tolerance.
- [x] Full representation metric includes the unread coordinate.
- [x] Seventy-one-step tangent/kernel prediction check: maximum error 3.55e-15.
- [x] Nonlinear fixed-metric implementation uses the current nonlinear gradient; implementation tested although conditional study not triggered.
- [x] Duplicated-sample summed-metric scaling verified with doubled damping and scalar rate.
- [x] Seven meaningful new implementation check groups pass (`implementation_checks.json`).
- [x] All ten existing core numerical tests pass (`core_numerics_checks.txt`).
- [x] Original/coverage audit independently checks 314 saved states, with maximum loss discrepancy 2.22e-16.
- [x] All 46 original/coverage shared initial/target parameter states match exactly.
- [x] Independently replayed 10568 initial/coverage updates; twelve target parameter states match bit-for-bit.
- [x] Refined audit independently checks 128 additional saved states (maximum loss error 1.11e-16) and all 256 final composite states.
- [x] Independently replayed 15472 refined updates; six target states match bit-for-bit.
- [x] Refined controller halves both positive scalar step bounds; actual rings tangent maximum predicted step halves.
- [x] Data/initialization regeneration and training source/config/output hashes verified.
- [x] All 64 selected final trajectories attain all three declared training-loss targets.
- [x] Final scorer recomputes 256 training losses exactly; all evaluation rows complete without failures.
- [x] Rings quadrature uses inherited smooth-risk/component refinement tolerances and rigorous nonlinear/tangent risk-tail bounds.
- [x] Four independent Simpson/shifted-angle primary checks pass; largest risk discrepancy 1.0531e-11 nats.
- [x] Independent decision review recomputes primary means, SDs, sign counts, and predictor discrepancies.
- [x] Final figures checked visually; figures show paired values and means, with no confidence/significance claims.
- [x] Toy counterexample analytical/numerical checks pass and are explicitly classified as elementary, not innovative.
- [x] Research data/reports archived separately; reusable source/configurations retained in repository; hash manifest generated.

## Preserved negative or untriggered outcomes

- [x] Original 6000-step rings gate records twelve censored trajectories rather than inventing missing comparisons.
- [x] Original nominal-step halving limitation is documented and repaired in a separate run.
- [x] Supplemental B and D gates fail for both tasks; no success definition is changed after scoring.
- [x] Near-threshold moons B is reported openly; 0.01 is a screening floor, not a statistical theorem.
- [x] No claim of predictor equivalence, zero nonlinear effect, or complete clock-only explanation is made.
- [x] Fixed-M0 intervention, damping expansion, confirmation cohort and signed spectral expansion were not triggered.
- [x] No adaptive-metric causal attribution, deployed early predictor, or independent mechanism validation is claimed.
- [x] No computational-efficiency conclusion is drawn from controller-instrumented runtimes.
- [x] Joint-asymmetry, readout-decoupling and earlier whole-update derivative results are not repackaged as new findings.
- [x] No professor or other person was contacted.

## Remaining research requirements

- [ ] A nonroutine architecture/data condition predicts sustained non-clock geometry feedback: not established.
- [ ] Such a condition survives nearest-literature reductions and elementary counterexamples: not established.
- [ ] An early predictor beats initial-geometry/progress baselines on independent seeds and noise levels: not tested.
- [ ] A mechanism benefit survives adaptive-versus-frozen geometry interventions and offsets computation: not tested.

These unchecked items are precisely why the report does not claim an innovation. The authorized first-gate study, provenance records, audits, and stop decision are complete.
