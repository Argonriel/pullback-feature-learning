# Literature and scope

This project studies a damped pullback of the **Euclidean metric on finite, sample-indexed representations**. It compares that update with parameter gradient descent in a small residual network. The papers below inform the geometry, diagnostics, and interpretation. The project is a controlled numerical investigation, not a new natural-gradient method or a reproduction of any cited paper's experiments.

Primary sources and bibliographic metadata were checked on November 19, 2026. BibTeX entries are in [references.bib](references.bib).

## How the implementation relates to existing methods

Let `Z = R_X(theta)` stack both coordinates of every training representation, let `J = D R_X(theta)`, and let `g = grad_Z Phi`. The implemented direction solves

```text
(J.T @ J + lambda * I) @ u = -J.T @ g,    lambda > 0.
```

Its defining local problem is to match the desired feature velocity `-g` while penalizing the parameter velocity. The following distinctions follow from this project's objective and metric:

| Object | Meaning in this project | Distinction |
| --- | --- | --- |
| `J.T @ J` | Pullback of the Euclidean representation metric | Uses both representation coordinates, including the coordinate absent from the fixed readout. |
| `J @ J.T` | Tangent kernel of the representation map on a fixed input set | Generally changes during finite-network training; it is not assumed equal to an infinite-width limiting kernel. |
| `J.T @ H_Z @ J` | Generalized Gauss-Newton matrix for the chosen representation/loss decomposition | `H_Z = Hessian_Z Phi` is not the identity for logistic loss, so this is not the implemented matrix. |
| Conditional Bernoulli Fisher | Curvature of predictive distributions induced by the fixed logistic readout | With an empirical input distribution, it coincides with this logistic GGN, not with the Euclidean representation pullback. |
| Wasserstein pullback | Geometry of a parameterized family of distributions | Not implemented: there is no optimal-transport problem, dual potential, or KWNG estimator in the core experiment. |

The representation metric uses a **sum** over samples, whereas the logistic loss uses a **mean**. The gradient therefore contains `1/n`; `J.T @ J` does not. Replacing it by `J.T @ J / n` without adjusting the convention changes the method and damping scale. Although the fixed readout is `(1, 0)`, dropping all second-coordinate Jacobian rows would also change the metric.

Positive damping makes the local linear system positive definite. It does not establish global convergence, guarantee descent for arbitrary finite learning rates, or imply that the method improves generalization. The Euclidean parameter penalty also depends on parameter coordinates, so full reparameterization invariance must not be claimed.

## Core readings

### 1. Martens (2020): geometry, curvature, and damping

**James Martens. New Insights and Perspectives on the Natural Gradient Method. Journal of Machine Learning Research 21(146):1-76, 2020.** [Publisher page](https://jmlr.org/papers/v21/17-678.html) · [Paper](https://www.jmlr.org/papers/volume21/17-678/17-678.pdf)

Sections 5-10 explain Fisher geometry, its relationship to generalized Gauss-Newton curvature, and why damping matters for practical updates. Section 12 qualifies invariance claims for finite steps. This motivates treating the metric, objective, damping, and discretization as separate choices. In our experiment, the borrowed idea is optimization through a pullback geometry; the actual matrix is the explicitly chosen representation Gram matrix. Its resemblance to a natural-gradient formula is not enough to identify it with the Fisher.

**Project consequence:** keep the method name precise, solve the damped system directly, check local predictions numerically, and measure computational cost as well as iteration count.

### 2. Arbel et al. (2020): distribution geometry is a distinct choice

**Michael Arbel, Arthur Gretton, Wuchen Li, and Guido Montufar. Kernelized Wasserstein Natural Gradient. ICLR, 2020.** [Conference record](https://openreview.net/forum?id=Hklz71rYvS) · [Author manuscript](https://arxiv.org/abs/1910.09652) · [Paper](https://arxiv.org/pdf/1910.09652)

This work approximates Wasserstein natural gradients using a dual formulation and kernel methods, addressing the cost of distribution-space optimization. It is relevant because it makes the choice of geometry explicit and connects it to an implementable parameter update. Its distribution-level geometry and estimator differ from matching the motion of a fixed collection of labeled representations.

**Project consequence:** use this as motivation for comparing geometries and accounting for their computation. Do not describe the implemented endpoint metric as Wasserstein or claim to reproduce KWNG. The author manuscript provides a primary-source copy of the ICLR paper.

### 3. Jacot et al. (2018): tangent kernels explain local motion

**Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural Tangent Kernel: Convergence and Generalization in Neural Networks. Advances in Neural Information Processing Systems 31, 2018.** [Proceedings and paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html)

The paper expresses parameter-gradient dynamics through a kernel in function space and analyzes an infinite-width limit in which that kernel remains fixed. In this finite project, the analogous local identity is `dZ/dt = -J @ J.T @ g`. Its eigendirections explain why parameter GD weights feature-motion directions unequally.

**Project consequence:** inspect singular values and gradient projections at several training checkpoints. The local identity is applicable here; the cited infinite-width convergence and constant-kernel conclusions are not automatically applicable to an 80-parameter residual network.

### 4. Chizat et al. (2019): linearization has a regime of validity

**Lenaic Chizat, Edouard Oyallon, and Francis Bach. On Lazy Training in Differentiable Programming. Advances in Neural Information Processing Systems 32, 2019.** [Proceedings and paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html)

This paper connects lazy behavior to model scaling and studies when nonlinear training stays close to its linearization. It cautions against interpreting all successful optimization as substantial feature learning.

**Project consequence:** first validate the one-step linear approximation with decreasing step sizes, then examine learned representations over training. Small local approximation error does not show that the full trajectory is lazy, and a large feature displacement does not establish that the displacement is useful. Initialization scale and residual scaling must be recorded because they influence the operating regime.

### 5. Alain and Bengio (2016): evaluate frozen features separately

**Guillaume Alain and Yoshua Bengio. Understanding Intermediate Layers Using Linear Classifier Probes. arXiv:1610.01644, 2016; revised 2018.** [Author manuscript](https://arxiv.org/abs/1610.01644)

The authors fit independent linear classifiers to hidden representations to study the information accessible to a simple classifier. The probe is a diagnostic and does not train the representation network.

**Project consequence:** compare initial and learned frozen representations with the same probe protocol. Fit normalization on training features, select regularization on validation features, and evaluate held-out loss and accuracy after selection. These split and normalization rules are this project's evaluation controls, not a claim that the paper establishes a universal probing protocol. The fixed readout's accuracy and a newly fitted probe's accuracy answer different questions and should be reported separately in the full study.

## Optional extension reading

### 6. Chen et al. (2023): useful features and distribution shift

**Yongqiang Chen, Wei Huang, Kaiwen Zhou, Yatao Bian, Bo Han, and James Cheng. Understanding and Improving Feature Learning for Out-of-Distribution Generalization. Advances in Neural Information Processing Systems 36, 2023.** [Proceedings and paper](https://papers.neurips.cc/paper_files/paper/2023/hash/d73d5645ddbb9ada6c862116435574f6-Abstract-Conference.html)

The paper studies how invariant and spurious features can be learned at different rates, and proposes Feature Augmented Training. It motivates examining which features an optimizer learns rather than relying solely on in-distribution prediction accuracy.

**Project consequence:** a future nuisance-feature experiment could test whether optimization geometry changes sensitivity to spurious correlations. This is optional, requires its own controlled protocol, and is not evaluated by the initial CPU pilot. The present method is not FeAT and the core two-moons experiment supports no OOD claim.

## Questions the experiments can answer

1. Does the implemented update agree with its local linear and spectral predictions?
2. After validation-based tuning, does it reduce the objective more effectively per iteration, per unit of training time, or per unit of feature displacement?
3. Do frozen learned features improve held-out linear-probe performance relative to initial features and the GD comparison?
4. Are observed differences associated with the Jacobian spectrum and the task gradient's projections, and where does the local explanation cease to be adequate?

A short pilot answers implementation and feasibility questions only. Comparative performance conclusions require the later paired-seed study and its predeclared tuning and evaluation rules.
