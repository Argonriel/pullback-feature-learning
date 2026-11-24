# Literature-informed continuation

Primary sources were checked on21 November2026. This continuation tests
established techniques and a specialized symmetry-risk calculation; it is not
an exhaustive novelty review and makes no first-calibration or first-robust-
augmentation claim.

1. Guo, Pleiss, Sun and Weinberger, *On Calibration of Modern Neural Networks*,
   ICML2017: [published paper](https://proceedings.mlr.press/v70/guo17a.html).
   Positive temperature scaling fits a scalar to held-out labeled predictions.
   We use independent calibration labels and keep test outputs out of fitting.
   The scalar preserves unaugmented class decisions, so better log loss cannot
   be described as a temperature-caused accuracy increase. Our original model
   has144 parameters, and the effect is measured rather than assumed from the
   paper's deeper-network experiments.

2. Chen, Dobriban and Lee, *A Group-Theoretic Framework for Data Augmentation*,
   JMLR2020: [published paper](https://jmlr.org/papers/v21/20-163.html).
   Section6 already studies approximate invariance and a bias/variance tradeoff
   controlled by distribution discrepancy. Our calculation keeps the input
   law exactly invariant and quantifies a bounded posterior defect in logistic
   loss. It is a narrower, explicit budget for the current frozen selector,
   not a new general approximate-invariance principle.

3. Xiao, Rasul and Vollgraf, *Fashion-MNIST: a Novel Image Dataset for
   Benchmarking Machine Learning Algorithms*,2017:
   [author paper](https://arxiv.org/abs/1708.07747),
   [official source](https://github.com/zalandoresearch/fashion-mnist).
   The official benchmark has60,000 training and10,000 test garment images
   from10 classes. We retain the source files, official checksums and license.
   Horizontal reflection is our stated semantic assumption, not a theorem
   supplied by the dataset. Test decoding is delayed until model/policy commit.

4. Maurer and Pontil, *Empirical Bernstein Bounds and Sample Variance
   Penalization*, COLT2009: [author paper](https://arxiv.org/abs/0907.3740).
   McDiarmid, *On the Method of Bounded Differences*,1989:
   [publisher record](https://doi.org/10.1017/CBO9781107359949.008).
   These established inequalities bound two moments after accounting for
   exact-quota selection dependence. Simultaneous bounds then yield an affine
   line valid for all supplied posterior-error budgets. A grid of plotted
   epsilon values is not treated as additional independent experiments.

5. Jhawar and Wang, *When Is Test-Time Adaptation Identifiable from Unlabeled
   Evidence?*, September2026 preprint:
   [version1](https://arxiv.org/html/2609.11235v1).
   The earlier study already identified overlap with generic indistinguishable-
   world obstructions. The present epsilon assumption requires label or domain
   evidence; it cannot be inferred universally from unlabeled consistency.
   Our fixed-model augmentation is distinguished from online adaptation.

The real-data component measures every major cost category and compares
against both fused and sequential full averaging. Equal forward-example
budgets are not equal wall-clock times. The high-dimensional image surrogate
uses a different fitting representation from the synthetic interpolator;
transfer means a tested allocation principle, not identical implementation.

Branch-specific protocols and proofs give exact choices, confidence units,
assumptions and source locations. Empirical real-test comparisons, deliberately
pair-closed finite-image identities, synthetic posterior-error guarantees and
runtime measurements are reported as different evidence types.
