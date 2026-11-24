# Which inputs deserve a second view?

This final-project extension connects an error in the learned rings classifier to a prediction-time decision. Rotating a rings input by 180 degrees preserves its class distribution, but the fitted network can assign different scores to the two orientations. Given a budget of 64 extra evaluations for a batch of 256 inputs, which examples benefit most from averaging the two scores?

[Two-page final insert](final_insert.pdf) · [LaTeX manuscript section](final_section.tex) · [Mathematical derivation](proof.md) · [Prior work](related_work.md) · [BibTeX](references.bib)

## What the experiment found

Each row summarizes 16 new paired training seeds at update 500. Each model was tested on 32,768 fresh inputs. Entries are mean binary log losses in nats per input. The last column divides the selective policy's improvement over a single view by the improvement from averaging both views on every input.

| Training method | Single view | Selective second view on 25% | Two views on all inputs | Gain retained |
|---|---:|---:|---:|---:|
| GD | 0.17744 | 0.15630 | 0.14918 | 74.8% |
| Pullback | 0.11038 | 0.10123 | 0.10097 | 97.2% |
| 100 pullback updates, then 400 GD updates | 0.06127 | 0.04460 | 0.04405 | 96.8% |

The selective rule ranks inputs by the estimated gain in their radius and angle region. Opposite orientations belong to the same region. Its estimates use 8,192 unlabeled calibration inputs per model. On pullback the rule retained 97.2% of the full second-view gain (paired 95% bootstrap interval 95.5–98.4%). Choosing a random quarter would retain 25% in expectation. The selective rule beat random allocation in all 16 pullback seeds. The full second view corrected about 60% of the pullback model's angular log loss on an independent polar integration grid.

[Study design](DESIGN.md) · [Main estimates](results/summary.json) · [Allocation results figure](results/allocation_results.png) · [Example map of predicted and measured gains](results/routing_map.png) · [Angular loss calculation](results/angular_ceiling.json)

## What a simpler rule revealed

A second, independent evaluation used the same 48 trained models and selected the least confident quarter by their first-view scores. Confidence achieved lower loss than the region-based rule on all 16 pullback models (0.09860 vs. 0.10113) and all 16 GD models. The region-based rule won on 12 of 16 hybrid models (0.04431 vs. 0.04488 on average). These losses are from the second evaluation sample.

The proof identifies two contributions to a selective policy's gain. One measures disagreement between opposite predictions and is always nonnegative. The other depends on which orientation gets the first pass and can have either sign. The second term vanishes for the region-based rule. On pullback it added 0.00438 nats for confidence routing, explaining how averaging only the least confident quarter could even outperform averaging every example.

[Confidence comparison](SUPPLEMENTAL_RESULTS.md) · [Signed contributions by training method](supplemental_results/confidence_terms_summary.csv)

## Costs and scope

Both selective policies use 1.25 network evaluations per input at deployment, compared with 2 for universal averaging. The region-based rule also used 16,384 network evaluations per model for offline calibration. On this small CPU model, its median time per batch of 256 was 0.109 ms versus 0.069 ms for universal averaging. The experiment demonstrates a saving in model evaluations. Actual latency includes the cost of ranking inputs.

The contribution to this course project is the conditional gain formula for this setting, its use with an exact batch budget, and the empirical explanation for why confidence can win. Related work already establishes group averaging, log-loss diversity, and input-dependent selection of test-time augmentations. The guarantee assumes an antipodally invariant joint input-label law and applies to mean logits under log loss. The true conditional gains give an optimal ranking among policies that use the same invariant regions. Calibration estimates can introduce ranking errors.

## Reproducing the main comparison

Use the source checkout's Python environment with `pullback_feature_learning` installed. From this directory run the following commands, choosing a new results directory to preserve the archived run.

```powershell
python run.py --out fresh_results
python analyze.py --results fresh_results
```

The [model weights](results/models.npz), [fitted calibration tables](results/calibrated_gates.npz), [training and calibration record](results/training_and_calibration.csv), and [independent numerical audit](results/independent_audit.json) accompany the estimates.
