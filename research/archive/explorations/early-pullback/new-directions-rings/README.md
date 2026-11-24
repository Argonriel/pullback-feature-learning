# Transfer of a 100-step pullback warm-up to noisy concentric rings

The same 100-step pullback warm-up used on two moons gave gradient descent a head start on a radial classification task. With the two-moons hyperparameters, the primary 500-update comparison favored warm-up over plain GD on validation loss and accuracy in every pair. At 2,000 updates the mean advantage persisted, with more variation across pairs.

| Method | Step 100 loss | Step 100 accuracy | Step 500 loss | Step 500 accuracy | Step 2,000 loss | Step 2,000 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GD | 0.71145 | 50.15% | 0.67210 | 61.18% | 0.14862 | 95.80% |
| Pullback | 0.34402 | 97.17% | 0.11940 | 97.22% | 0.10196 | 97.07% |
| 100 pullback steps followed by GD | 0.34402 | 97.17% | 0.10280 | 98.39% | 0.06558 | 98.05% |

Values are means over eight independent training/validation/initialization draws. Accuracy uses the original fixed classifier. The common first 100 steps in the two pullback-first schedules have identical weights. Each split has 256 balanced examples at noise 0.15. [Generator, seeds and protocol](PLAN.md)

At 500 steps the paired warm-up minus GD loss difference was −0.56930 (sample SD 0.05818), and the accuracy difference was +37.21 percentage points (sample SD 5.40). By 2,000 steps these differences were −0.08304 (sample SD 0.10353) and +2.25 percentage points (sample SD 2.78). Six pairs had lower terminal validation loss with warm-up, and two had lower terminal loss with GD. At the terminal accuracy comparison six pairs favored warm-up and two tied.

The warm-up used 100 full Jacobians and linear solves, compared with 2,000 for full pullback at the terminal endpoint. It had lower terminal validation loss than full pullback in all eight pairs (mean paired difference −0.03638, sample SD 0.02613). Pullback and GD steps have different runtimes. This comparison measures outcomes after the same number of updates. The methods used rates selected on two moons. Whether these rankings persist after tuning on rings or with equal elapsed training time remains open.

The [first-pair feature diagram](FIGURE_CAPTION.md) shows the same 256 validation points before training and at the two early checkpoints, with the original fixed decision boundary.

Run `python run_rings.py --repository PATH_TO_INSTALLED_CHECKOUT --output NEW_DIRECTORY` in the project's Python environment. Run `python plot_rings.py` to reproduce the figure after verifying the original data hashes, checkpoint hashes and accuracies. [All checkpoints](results/checkpoints.csv) · [Means and SDs](results/means.csv) · [Paired differences](results/paired.csv) · [Seeds](results/seeds.csv) · [Software and hashes](results/metadata.json)

