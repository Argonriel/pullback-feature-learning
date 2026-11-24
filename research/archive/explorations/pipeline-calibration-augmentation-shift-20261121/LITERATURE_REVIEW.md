# Primary-source context for the new study

Reviewed 21 November 2026. These references motivate the prospective protocol; outcomes do not determine which claims are primary.

1. Guo, Pleiss, Sun and Weinberger, *On Calibration of Modern Neural Networks*, ICML 2017. [Official proceedings](https://proceedings.mlr.press/v70/guo17a.html). Positive scalar temperature is an established calibration tool. This study tests where to fit that correction in a completed prediction pipeline, while charging additional labels. It does not claim a new calibration method.

2. Hekler, Brinker and Buettner, *Test Time Augmentation Meets Post-hoc Calibration: Uncertainty Quantification under Real-World Conditions*, AAAI 2023, 37(12):14856-14864, DOI 10.1609/aaai.v37i12.26735. [Official proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/26735). The publisher's indexed abstract explicitly studies combining augmentation with calibration, including limited labels and domain changes. The live publisher page blocked automated opening; no uninspected detailed formula is attributed to it. This prior work rules out claiming the generic combination as novel.

3. Shanmugam, Blalock, Balakrishnan and Guttag, *Better Aggregation in Test-Time Augmentation*. [Author preprint](https://arxiv.org/abs/2011.11156). Their abstract reports that aggregation can turn correct predictions into incorrect ones even when average accuracy improves. Our original-input and transformed-input diagnostics preserve that distinction; our experiment does not implement or compare their learned aggregator.

4. Chen, Dobriban and Lee, *A Group-Theoretic Framework for Data Augmentation*, JMLR 21(245):1-71, 2020. [Official article](https://jmlr.org/papers/v21/20-163.html). Orbit averaging and approximate invariance are established theoretical context. The new controlled study specializes an input-total-variation penalty to fixed exact-quota policies with coordinatewise invariant masks; no first approximate-invariance result is claimed.

5. Xiao, Rasul and Vollgraf, *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. [Author preprint](https://arxiv.org/abs/1708.07747); [official data repository](https://github.com/zalandoresearch/fashion-mnist). Source files and license were previously checked and are preserved. A new within-round holdout from previously used images is explicitly not a pristine external dataset. Horizontal reflection preserving category semantics is an experimental assumption, not a theorem supplied by the dataset.

The final-output fit, equal-label-budget comparison and choice to train augmentation are prospective decisions motivated jointly by this literature and the previous archived failures. Generalization beyond these tasks requires separate evidence.
