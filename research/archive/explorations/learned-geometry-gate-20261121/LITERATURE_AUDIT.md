# Learning-geometry feedback: primary-source novelty audit

Date: 2026-11-21 (America/Los_Angeles). Prepared before interpreting the new gate experiments. This is a screening audit, not a claim of literature exhaustiveness or novelty. Existing historical reports were not modified.

## Decision

Retain the learning-geometry-feedback question conditionally; reject any current claim of established novelty. The four-arm nonlinear/tangent-by-GD/PB gate is necessary. If its residual survives learning-rate and progress controls, the fixed-initial-metric intervention is necessary before attributing anything to metric learning. A theorem's failure to apply automatically to this architecture does not establish a new contribution.

## Three closest primary sources

### Chizat, Oyallon and Bach (2019)

[On Lazy Training in Differentiable Programming, original paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf), Sections 1.1-1.2 and 2.2, Theorem 2.2.

Their baseline is the full model linearization h(w0)+Dh(w0)(w-w0). For suitably scaled smooth models with h(w0)=0, finite-horizon nonlinear and linearized gradient-flow predictions become close as scaling increases. The paper already distinguishes kernel behavior from feature learning and reports cases where nonlinear learning generalizes better. Consequently, a nonlinear model beating its initial tangent model is not itself new. Their theorem does not directly establish this finite-width, finite-step, logistic-loss representation-pullback system's behavior. Our gate implements their relevant baseline concept rather than replacing it with a two-dimensional feature probe or a single-step Taylor check.

### Arbel, Menegaux and Wolinski (2023)

[Rethinking Gauss-Newton for learning over-parameterized models, original paper](https://papers.neurips.cc/paper_files/paper/2023/file/6a14c7f9fb3f42645cfa6bd5aa446819-Paper-Conference.pdf), Sections 3.1-3.2, Proposition 3, Section 5.

Their generalized GN uses an output Jacobian and positive operator H_w. The principal setting is overparameterized one-hidden-layer regression with a smooth, strongly convex output loss. Proposition 3 requires an invertible initial hidden-feature Gram matrix, smooth activation, bounded positive H_w, and a sufficiently optimized initial linear layer. Their analyzed damping scales with the smallest NTK eigenvalue. Section 5 already demonstrates GN feature learning, initialization/step-size dependence, hidden useful features, and a convergence-speed/generalization tradeoff. Our two-dimensional representation Jacobian includes an unread-out coordinate; it is not their scalar-output Jacobian. Logistic loss and the small residual network also preclude automatic application of their convergence theorem. These distinctions define scope, not novelty.

### Yoshida and Nitanda (2025)

[How Does Preconditioning Guide Feature Learning in Deep Neural Networks?, version-one primary text](https://arxiv.org/html/2509.25637v1), Section 3.1, Assumptions 1.P/1.Q, Theorems 1-2; Sections 3.2-3.3.

Their first-layer update is neuron-wise W1^(t+1)=W1^t-eta P^t grad_W1 L, with an input-space matrix P^t. Assumption 1.P explicitly requires P^t=Phi_t(X_train). The other-layer preconditioner may depend on hidden states, labels and Gram history. Theorem 1 additionally requires initialization independent of P^0 with the specified P^0-isotropy. Training/test dependence through preconditioned Gram and cross-Gram histories is already their main framework. Their covariance-power and teacher-alignment examples substantially threaten generic spectral-alignment claims. A label-dependent, all-parameter, generally cross-layer-coupled inverse representation metric is not automatically covered by these assumptions. Merely replacing their input-dependent metric by ours is not a demonstrated contribution.

## Original reductions for this gate

Let J0 be the full initial representation Jacobian on training inputs, let S select the trained readout, and write A0=SJ0. With P0=(J0^T J0+lambda I)^(-1), full-tangent PB is exactly GD in transformed coordinates with feature matrix A0 P0^(1/2). Its prediction kernel is A0 P0 A0^T; its test cross-kernel is equally fixed. Any generalization difference here can arise entirely from ordinary fixed-feature preconditioning. This is an algebraic reduction, not a new result.

Nonlinear training with fixed P0 is GD under a fixed parameter reparameterization. It preserves nonlinear prediction learning while removing adaptive metric feedback.

A common positive scalar change in the update field changes the continuous-time clock. Speed, parameter displacement, representation displacement, or Jacobian condition-number differences alone cannot establish a different predictive mechanism. Finite-step numerical changes require separate control.

For J^T J v_k=mu_k v_k, the local decomposition

    -dR/dt = sum_k [(v_k^T grad R)(v_k^T grad L)/(mu_k+lambda)]

is an identity. It supplies signed diagnostic contributions, not a new prospective predictor. The trained v_k depend on labels, so training noise cannot be averaged away conditionally on them without further justification. The already completed whole-update population derivative forecast must remain prior work within this project.

## Gate interpretation and stopping rules

At each predeclared shared attained training-loss level, define lower-is-better risk R and

    D = (R_N,GD - R_N,PB) - (R_L,GD - R_L,PB).

A favorable D indicates that PB's comparative benefit depends on nonlinearity. It does not identify adaptive metric feedback. Nonlinear PB beating linear PB alone is weaker: GD might benefit equally from nonlinearity.

Report unreachable loss thresholds as failures/censoring, not silently excluded comparisons. Use smaller-step sensitivity checks, shared initialization/data, and explicit summed-versus-averaged damping conventions. If meaningful residuals survive, compare nonlinear adaptive PB against nonlinear fixed-P0 PB. No favorable residual means reject the current explanatory claim and stop its expansion. A metric effect in either direction can justify a diagnostic project; it does not yet provide an advance predictor or an innovation claim.

## Conditional candidate: early label-replica geometry sensitivity

This is a hypothesis for screening, not an established contribution:

> Within predeclared noisy-rings task families, early label-replica variability of the learned metric predicts when adaptive PB's later advantage over frozen-metric PB becomes negative, beyond what initial geometry and training progress predict.

At identical inputs, independently resample labels from the known conditional law, use the same initialization, and measure early metric variability before the prediction horizon. Normalize away scalar metric/clock changes. Freeze the statistic, threshold, checkpoint and later risk endpoint using development seeds; evaluate prospectively on independent seeds and prespecified noise levels. Compare against initial-spectrum and training-loss-only predictors. The known synthetic label mechanism is part of the available-information declaration; this is not automatically deployable for arbitrary real data.

A possible causal follow-up is reciprocal metric exchange between label replicas, with unchanged recipient prediction states and normalized instantaneous training-loss speed. A donor metric can mismatch the recipient state, however, so this intervention is not automatically a clean isolation of noise reuse. Any claim must address this mismatch and distinguish metric response from generic prediction instability.

Reject the candidate if it reduces to generic predictor instability, if its value vanishes after frozen-metric controls, or if its sign cannot replicate prospectively. A standard stability bound, independent-sample debiasing construction, or noise-correlation identity alone does not rescue novelty.

## Bounded additional novelty-threat check

The follow-up search targeted adaptive preconditioning with label noise, second-order feature learning, and independent-sample noise control. Three additional primary works were checked. This bounded search does not prove absence of closer work.

### Zhang et al. (ICML 2025): learned preconditioning and independent batches

[On The Concurrence of Layer-wise Preconditioning Methods and Provable Feature Learning, proceedings record](https://proceedings.mlr.press/v267/zhang25bh.html); [primary version-one text inspected](https://arxiv.org/html/2502.01763v1), Section 3 equations (4)-(5), Section 3.1 Theorem 3.6, Section 3.2 Theorem 3.9.

The stylized KFAC update already includes learned left geometry: in linear representation learning P_G=F^T F, alongside input covariance on the right. Equation (5)'s description explicitly assumes independently sampled batches for G and F updates; Section 3.1 introduces a fresh batch at the current weights. Theorem 3.6 is a local subspace-contraction result with Gaussian/subgaussian inputs, additive label noise, near-correct representation, sufficiently large batch size, no G damping, and an exact least-squares F update. The nonlinear single-index analysis uses one feature-learning step and input-covariance preconditioning. Thus learned geometry plus independent-sample noise control is already an established combination. Repeated-label, all-parameter nonlinear representation feedback is not automatically covered, but independent replica construction alone cannot be our innovation.

### Han et al. (ICML 2025): signal-first and noise-later feature learning

[On the Role of Label Noise in the Feature Learning Process, proceedings record](https://proceedings.mlr.press/v267/han25c.html); [primary version-one text inspected](https://arxiv.org/html/2505.18909v1), Sections 3-4, Condition 4.1, Theorems 4.1-4.2.

The model is a two-layer ReLU CNN trained by GD on logistic loss. Signal/noise patches and class-conditional label flips permit a decomposition of learned weights. Under constant-order n*SNR^2 and flip probabilities, high dimension, sufficient width/sample size, and small initialization/step size, the analysis distinguishes initial signal-dominated fitting from later noisy-sample memorization with degraded test performance. The mechanism is not adaptive PB. Nevertheless, signal-first/noise-later behavior, a population contribution sign reversal, or a benefit from early stopping cannot alone establish novelty here.

### Ishikawa and Karakida (ICLR 2024): stable second-order feature learning

[On the Parameterization of Second-Order Optimization Effective Towards the Infinite Width, original paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/7f63032459a9b644f91373e71b456457-Paper-Conference.pdf), Sections 3-4.

The paper studies parameterization, learning-rate and damping scales for KFAC/Shampoo, using one-step infinite-width analysis to obtain stable feature-learning scales. Its curvature factors depend on current representations/backpropagation quantities. Therefore adaptive second-order geometry enabling feature learning, or damping affecting whether learning remains stable/lazy, is not new. The inspected results do not supply the proposed early replica statistic or a repeated-noise reuse criterion for our finite residual network; this statement is limited to the inspected scope.

## Revised admissibility standard after additional search

The candidate must deliver a specific conditional prediction for *adaptive-versus-frozen metric* risk differences that survives initial tangent, training progress, clock, and generic instability baselines. It must then explain why that prediction is not a special case of learned KFAC geometry with sample splitting or standard signal/noise learning dynamics. Until then the proper status is an exploratory research question. No source reviewed licenses an unconditional statement that preconditioning fits label noise faster or always harms generalization.

## Audit checklist

- [x] Read repository AGENTS.md before research.
- [x] Used original paper/proceedings/arXiv sources for substantive claims.
- [x] Recorded exact theorem/assumption locations and scope mismatches.
- [x] Distinguished output GN from full representation pullback.
- [x] Reduced frozen tangent PB to fixed transformed features.
- [x] Identified clock equivalence and finite-step caveat.
- [x] Kept population derivative decomposition classified as diagnostic.
- [x] Flagged fitted-Jacobian dependence on training labels.
- [x] Checked targeted additional learned-preconditioning/noise-learning threats.
- [x] Retained falsification and stopping conditions.
- [x] Made no positive novelty claim and did not contact anyone.
- [ ] A specific new mechanism has been proved and independently validated: not achieved by this audit.
