# Calibration cost, selective views, and identifiable gains

This study executes ideas 2A and 2B and checks ideas 3 and 4 from the supplied Research Agenda for Optimization and Second Views (21 November 2026). It does not continue the optimizer-horizon or pre-update-angle studies. All prior outputs remain unchanged.

## Confirmatory experiments

Idea 2A measures the new-model calibration budget needed to improve on a historically transferred angle. Budgets are 0, 128, 512, 2,048, 8,192 and 65,536 forward example evaluations; each radius contributes 128 related views. Independent calibration repetitions are nested within models, not counted as extra trained-model replications. Historical defaults and scientific settings are frozen before the new training cohort's outcomes. Absolute paired risk differences are primary. Recovery ratios and a 90% target are meaningful only if the full-calibration advantage over the historical default is positive and sufficiently resolved. The finite full-calibration policy is a benchmark, not the population oracle.

Idea 2B compares selection of inputs and selection of angles at exactly 64 extra evaluations per batch of 256. The primary calibration budget is independently prespecified as 8,192, with 65,536 a secondary budget; neither is chosen using the new 2A test outcomes. Identical input batches and ties are used across policies. Strong first-view-confidence and folded-region half-turn baselines remain in the comparison. Fixed-mask and fixed-angle contrasts separate the two decisions. Every deployment policy receives only first-view logits, coordinates and its already-fitted calibration object before selecting the second calls.

The two experiments deliberately share 16 fresh training/initialization pairs (721001-721016 and 722001-722016) and three fixed schedules. They use separate calibration/test streams. Sharing models connects the questions and does not provide two independent replications. All scientific intervals cluster by the training pair; all winners, losers, ties and failed attempts are retained. Selector and prediction manifests precede held-out scoring.

## Mathematical and numerical checks

Idea 3 checks the sharp antipodal bound under a known bound on label mismatch, and tests the specified approximate-symmetry mixture. The mathematical bound is distinct from estimating it on finite calibration data. Exact finite-support certificates, continuous estimated gates and diagnostic policies that already query both views must be labeled separately. Optional unused budget and a forced exact quota are different decision constraints.

Idea 4 checks identification of selective log-loss gain under unknown radial posteriors, including the known-class-prevalence case and sharp bounds. A finite-grid policy construction tests conditional signed-logit balance with exact per-batch quota. Balance in expectation is distinct from per-realization balance. Radius-bin balance does not imply pointwise-radius balance for arbitrary continuous posteriors. These checks use frozen old models and explicit counterexamples; they are not presented as fresh training replications.

## Deliverables and provenance

Scripts, raw per-seed/per-calibration/per-policy rows, states, manifests, failed settings, numerical audits, figures and an English report will be archived together. The root audit independently reconstructs central computations and deployment call counts. A Chinese plain-language report will distinguish demonstrated findings, limits, and recommended future directions. No study artifact is uploaded to GitHub.
