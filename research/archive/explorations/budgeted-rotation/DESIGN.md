# Which inputs deserve a second view?

This design precedes the new training or scores. We connect the angular component of the rings loss to allocation of extra test-time computation. We do not tune the model, gate resolution, or quota using evaluation scores.

## Models and draws

Train 16 new paired concentric-ring draws. Training and initialization seeds are 121000+i and 122000+i for i=1,...,16. Use the unchanged 256-example generator and original network. Evaluate at step 500 for GD (rate 0.5), full pullback (rate 8, damping 0.1), and 100 pullback steps then GD (rate 2). These settings came from earlier studies. Each model and both gates for that model are frozen before generating its evaluation inputs.

Each pair receives 8,192 separate unlabeled calibration inputs (seed 123000+i) and 32,768 fresh independent inputs in batches of 256 for evaluation (124000+i). Generate inputs from the balanced-in-expectation mixture of Gaussian-noisy rings, with independent latent class draws and independent angles. The gate receives coordinates only. The known population posterior is used only by the evaluator (and separately to locate the Bayes radius for an explicitly informed comparator).

## Gates and primary question

Every model predicts once on every test input. A selected input gets one extra prediction after a half-turn; average the two logits. For gate calibration compute the two-view softplus Jensen gap for each calibration input. Use 16 radius-quantile bins fitted on calibration inputs. One gate ranks inputs by the mean calibration gap of the radius bin. The main gate intersects those bins with eight equal-width angular bins covering [0,π), folding antipodal orientations together. A vacant bin falls back to its radius-bin estimate. On each batch take exactly the 64 highest scores, breaking ties with a seeded random key independent of angles and labels. Both gates use cheap coordinates at deployment. Total calibration cost is 16,384 network example-forwards per model; both gates reuse these evaluations.

Compare the original predictor, two views for every input, uniformly random allocation of 64 second views (expected risk), radius-only allocation, radius-plus-folded-angle allocation, and a competing rule that prioritizes radii nearest the true decision boundary (knows the oracle radius). All limited gates get exactly the same inference budget. Full and raw are different-cost reference points.

Primary outcome: paired mean logistic-risk reduction of the folded-angle gate relative to random allocation on full pullback at 500 updates. Secondary outcomes: the same comparisons for GD and the hybrid, the radius-only and boundary comparisons, the share of the full two-view gain, and the absolute change from raw predictions. Use paired seed-level summaries with pointwise bootstrap intervals; loss is in nats. Score using the known conditional label probability, averaging both possible antipodal base orientations to reduce sampling noise. This is a fresh-sample estimate of population risk. The two perspectives are used for offline evaluation of all policies, not charged to the restricted deployment rule.

The primary gate ranks constant-cost second views by a calibration estimate of the benefit conditional on inexpensive invariants. It is approximate. The theoretical optimum uses the true conditional gain. The loss proof assumes the joint law of observations and labels is invariant to the half-turn and a gate that respects the same symmetry. Record whether the learned policies help across fresh trained models, including any failures. Verify actual batched inference evaluates only the selected second views. Report a small, separately timed CPU benchmark and distinguish pass counts from elapsed time.
