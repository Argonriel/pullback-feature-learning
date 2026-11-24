# Calibration cost and amortization

The measured policy uses one first prediction for each input and exactly 64 additional predictions per batch of 256: 320 model-example evaluations per batch, or 1.25 per input. The full half-turn reference uses 512 per batch. Calibration is additional setup work on each trained model.

For calibration budget C, comparison against full half-turn saves 192 evaluations per deployed batch. The total forward count first becomes no larger after `ceil(C/192)` batches: 43 batches (11,008 inputs) at C=8,192, or 342 batches (87,552 inputs) at C=65,536. This statement compares computational counts, not risk or time.

Confidence plus a half-turn already uses 320 evaluations per batch and requires no model-specific calibration. Learned angles therefore have **no forward-count break-even** against that comparator: their cumulative count always includes the extra C. Their possible benefit is lower risk at the same deployment quota. The relative setup overhead nevertheless decreases as deployment grows. At the 8,192-call budget, calibration is at most 10% of cumulative deployment forwards after 65,536 inputs, and at most 1% after 655,360 inputs. These targets refer to setup cost divided by deployment cost, not its share of the combined total.

Time is evaluated separately. If setup costs are T and T0, and measured per-batch deployment times are L and L0, then the candidate's total time is T+K L after K batches. When L<L0, a nonnegative time break-even is `ceil(max(T−T0,0)/(L0−L))`. If the candidate is slower at deployment and has no setup advantage, no finite break-even exists. Reporting only the models with a finite crossing would be misleading; the tables retain all 16 models and the number with no finite crossing. A median crossing among finite models is explicitly conditional on that subset.

The 10% and 1% timing-overhead targets use `ceil(T/(f L))` batches, where f is 0.1 or 0.01. Timing includes coordinate conversion, routing, rotation and actual model calls. These local measurements concern this small CPU implementation. They do not establish GPU serving throughput, application-specific monetary value, or a risk-versus-computation exchange rate.

All policies know the same geometric map A. In the ellipse cohort, observed inputs are A times latent ring inputs; the valid augmentation is A R A^-1. This changes the network's observed geometry and optimization problem while preserving the latent task's Bayes difficulty. It is not a second independent real-world task.
