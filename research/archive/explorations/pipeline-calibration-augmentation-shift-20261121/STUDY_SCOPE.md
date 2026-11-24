# Final-pipeline calibration, learned reflection, and input-law mismatch

Prospective scope, 21 November 2026. The user authorized all three recommended follow-ups. This record is written before new confirmation outcomes. Prior archives, model artifacts and frozen protocols remain unchanged. Branches write new source commitments before execution, and all negative outcomes are retained.

## 1. Equal-label-budget final-pipeline calibration

Keep the previous two-amplitude binary task and 144-parameter architecture. Use 16 fresh paired initializations/training streams. Three procedures each consume 4,352 unique labeled fitting examples: legacy training on 256 plus calibration on 4,096; selected training on 256 plus checkpoint validation on 2,048 plus calibration on 2,048; additional training on 2,304 plus calibration on 2,048. Raw uncalibrated arms are diagnostics with their smaller actual label use, not equal-budget calibrated comparisons.

Freeze legacy PB rate 4 / 500 steps and selected PB rate 2 / checkpoints 100, 250, 500, 1,000 from prior work. No new hyperparameter search. For the enlarged training set, scale both the unnormalized pullback metric damping and step rate by n/256; independently verify that replicating each datum leaves the update invariant. This is a documented normalization change, not a claim of an optimizer-independent data benefit.

Compare raw prediction, full half-turn averaging, and the frozen even interpolated exact 64-of-256 allocation. Fit one positive scalar correction to each completed pipeline's outputs using independent sampled labels. Compare that own-output correction against a scalar fitted to raw outputs. The selector is fixed before scalar fitting; do not rebuild its score after calibration. All fitting uses sampled labels, and the known synthetic posterior is evaluation information only.

Primary contrasts: own-output versus raw-output temperature for selected-PB full and interpolated pipelines; enlarged-training raw-temperature versus legacy raw-temperature; selected-training raw-temperature versus legacy raw-temperature. Use 16 paired seed clusters and four-comparison adjusted approximate bootstrap intervals. Record label use, actual/shared forward costs and all failures. Fresh seed families 2151 through 2159 are reserved for this branch.

## 2. Train reflection, then evaluate it on held-out images

Use the official Fashion-MNIST files already verified in the prior archive. The previous official test set has been observed and is only a secondary descriptive replication. It cannot be described as new or pristine.

Create a new within-round 10,000-image holdout, stratified and chosen by fixed seed 2161000 from the old 40,000-image training set. New training uses the remaining 30,000 plus the former 8,000 selector-fit and 2,000 selector-validation inputs. Retain the old disjoint 5,000 validation and 5,000 calibration sets. These are new roles for existing images, not a newly collected external dataset. No holdout input enters any newly trained model, checkpoint choice or calibration fit. Six models are trained from scratch; no prior fitted model is reused.

Compare three paired initialization/order streams with and without independent probability-0.5 horizontal reflection during training. Keep the 426,122-parameter CNN, 12 epochs, Adam, batch size and validation checkpoint rule unchanged. Each arm uses 50,000 distinct fitting/validation/calibration labels. Fit positive temperatures to raw and final two-view averaged logits using the same independent 5,000 calibration labels. Preserve raw, raw-calibrated, full, full with raw-fitted temperature, and full with own-output temperature.

Four primary contrasts are augmentation minus no augmentation under raw-calibrated inference (loss and error), and own-calibrated full minus raw-calibrated inference within the augmented arm (loss and error). Image-bootstrap resamples preserve pairing across all six fixed models, with a four-comparison correction; initialization replications share a split and are not independent datasets. Per-class error and reflected-input diagnostics are predeclared. No tuning after the new holdout is scored.

Measure raw and full inference on one-thread CPU and synchronized GPU resident/transfer modes, using validation inputs. Record training, fitting and serving costs separately. Hold other experimental numerical work during serving benchmarks. Fresh seed families 2161 through 2169 are reserved.

## 3. Extend the guarantee to input-law mismatch and multiclass outputs

For logits z, transformed logits zT, d=(z-zT)/2 and the log-sum-exp Jensen gap J, independently derive the coordinatewise invariant-mask identity with posterior correction (p-pT) dot d / 2. Use span(d)=max(d)-min(d), which ignores irrelevant common logit offsets. Posterior TV bounded by epsilon contributes epsilon E[M span(d)]/2. If input TV(P,T#P) is at most eta and span(d) is globally bounded by D, conditional coordinatewise symmetrization contributes D eta.

The resulting candidate certificate is L_A-epsilon U_B/2-D eta, with moments verified on the same input law P. Do not import moments from a different input distribution. Mask invariance must hold when each single input is transformed with the other records and tie keys fixed. Exact quota alone is insufficient.

Use eight prespecified prior legacy models and two frozen symmetric selectors, without retraining or selection based on new results. Controlled phase densities 1+a cos(phi), for a in {0,0.01,0.1,0.6}, have known input TV 2|a|/pi. The smaller nonzero shift was chosen before any new outcomes to resolve the near-symmetry regime. Use 2,048 batches as the primary certificate look and 512 as an additional fixed look; 512 elementary bound events share alpha 0.05. Freeze complete verification and stress rules before drawing new confirmation inputs. Include numerical identities for multiclass logits and arbitrary common offsets. A separate known-density correction is diagnostic and does not replace the conservative TV guarantee. Natural-image defect bounds remain unestablished; a valid but vacuous certificate is a result and will not trigger retuning. Fresh seed families 2171 through 2179 are reserved.

## Independent verification, literature and archive

Root audits use 2181-family diagnostic streams or exact replays of declared streams. Internal release files are scientific separation checks within the user's existing authorization. Source hashes, sample IDs, models, fitted corrections, uncertainty rules, cost ledgers and final report are retained. Previous successful and failed results are preserved.

Calibration, test-time augmentation and approximate-invariance bias have established literature. This study makes no first-method or first-general-theorem claim. Its purpose is controlled evidence about final-output calibration, learning transformation robustness and the conditions under which extra inference is safe and useful.
