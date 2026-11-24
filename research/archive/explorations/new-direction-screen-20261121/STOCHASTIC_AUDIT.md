# Stochastic pullback screen: no research direction retained

Date: 2026-11-21. This screen checks finite-batch effects that do not occur in a deterministic full-population update. It does not reopen the retired deterministic geometry, generic covariance/noise identity, or independent-label-replica routes. Original papers were inspected before the exact counterexample calculations. No neural training was run.

## Decision

**Reject the stochastic coupling/cross-fit lane as a proposed innovation.** Same-batch inverse-metric/gradient coupling is real, but direct prior work already identifies it, distinguishes it from inverse-estimation bias, and removes it with independent batches. The sharper rare-example example below exhibits a genuine batch-size sign reversal, yet its singleton mechanism is exactly ordinary sample reweighting. A further heteroskedastic-label proposal reduces to conditional expectation: at a fixed state, zero-mean label noise cannot alter the mean update when the current metric is label-independent and the gradient is affine in labels.

These are useful exclusion results and implementation cautions, not a new learned-representation theorem. They do not provide grounds for a larger experiment.

## Original-source novelty audit

| Primary source | Inspected result and consequence |
|---|---|
| [Benzing, OPT2021, *Kronecker-Factored Second-Order Optimizers Perform First-Order Descent on Neurons*](https://www.opt-ml.org/papers/2021/paper41.pdf), section 2.1 and appendix E.5 | E.5 explicitly distinguishes nonlinear inversion bias from the additional correlation caused by estimating Fisher and gradient on the same minibatch. It proposes independent minibatches and compares them in Figures 9–10, retuning the learning rate for each damping. The generic coupling/cross-fit claim is directly occupied. Section 2 derives the neuron-output least-squares update, close to the representation-metric setting. |
| [Nayak et al., 2026, *Correcting Stochastic Update Bias in Preconditioned Language Model Optimizers*](https://arxiv.org/html/2605.20756v1), sections 4–6 and appendix D.2 | Separates coupling and inverse-preconditioner biases, uses cross-fitted preconditioning, and proposes delta-method/jackknife corrections for AdamW, Sophia, and Shampoo. Theorem 1's matrix correction requires perturbations diagonal in a fixed population-aligned basis; its empirical-eigenbasis implementation is expressly an approximation. The paper does not establish unconditional improvement when correction increases variance. Neither generic cross-fitting nor scalar inverse-bias correction is available as a new contribution. |
| [Frangella et al., *SketchySGD*](https://arxiv.org/html/2211.08597v3), section 1.1 | Its update already uses independently sampled gradient and curvature batches and a damped inverse. The theory includes smooth strongly convex objectives and least squares. This is a direct prior example of the proposed independent-curvature strategy, not a proof about all nonlinear PB dynamics. |
| [Chung et al., 2017, *Stochastic Newton and Quasi-Newton Methods for Large Linear Least-squares Problems*](https://arxiv.org/pdf/1702.07367), sections 2–3 | For inconsistent full-column-rank least squares, stochastic Newton can converge to a different problem, while the analyzed stochastic quasi-Newton construction uses a preconditioner independent of the current sampling matrix. This already occupies the broad story that inversion inside sampling changes the effective target. Its pseudoinverse theorem is not being presented as a theorem about the damped logistic example below. |
| [Fang, Demmel, and Lavaei, 2026, *Why is Normalization Preferred?*](https://arxiv.org/html/2602.13413v1), sections 4–5 | Studies stochastic preconditioner/gradient dependence under finite-p-moment heavy-tailed noise. Under its assumptions, including a bounded preconditioner condition number, normalization has convergence guarantees whereas clipping can fail in the worst case. This makes a generic dependence/robustness narrative crowded; it does not specifically establish the two-point leverage calculation here. |
| [Gupta, Koren, and Singer, 2018, *Shampoo*](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf), introduction and algorithms | Builds structured tensor preconditioners from accumulated gradient statistics and analyzes online convex optimization. It is a required baseline for any structured inverse-root proposal. Shampoo's label-dependent gradient-statistic metric is not identical to the label-independent Jacobian pullback metric screened here. |
| [Martens and Grosse, 2015, *K-FAC*](https://proceedings.mlr.press/v37/martens15.pdf), section 1.2; [Kunstner, Balles, and Hennig, 2019, *Limitations of the Empirical Fisher Approximation*](https://papers.neurips.cc/paper_files/paper/2019/file/46a558d97954d0692411c861cf78ef79-Paper.pdf), introduction | K-FAC defines Fisher expectation over model-generated targets; the empirical Fisher formed from observed-label gradients is a distinct object. Kunstner et al. document its curvature mismatch and adverse gradient-field distortions. Substituting empirical Fisher to obtain label-dependent drift would change the method and enter existing prior work. |

The noncommuting matrix-noise restriction in Nayak et al. is a real scope boundary, but simply replacing its diagonal approximation by a full matrix Taylor/Fréchet expansion is a standard mathematical extension. No separate research claim is retained from that boundary.

## Exact finite-batch sign reversal

Use a scalar linear logistic prediction and two-coordinate representation

\[
f_\theta(x)=\theta x,\qquad z_\theta(x)=(\theta x,0),\qquad
\ell(y,f)=\log(1+e^{-yf}).
\]

The input/label distribution is `(x,y)=(1,+1)` with probability 0.99 and `(200,-1)` with probability 0.01. At `theta=0`, the population loss gradient is `+0.505`. The metric is the minibatch **average** of `x^2`, with fixed damping `lambda=0.1`. A summed-metric implementation is equivalent only after setting `lambda_sum=m*lambda` and `eta_sum=m*eta`; the code verifies this equivalence so batch size does not silently change effective damping.

For `k` rare observations in a batch of size `m`, define

\[
n_k=\frac{m-k-200k}{m},\qquad
d_k=\frac{m-k+200^2k}{m}+0.1,
\qquad u_k=\frac{n_k}{2d_k}.
\]

Here `theta <- theta + eta*u` defines the update sign. The count `k` has the exact binomial law. For independent metric and gradient batches of the same size, the mean direction is `0.5 E[n] E[1/d]`; its second moment is `0.25 E[n^2] E[1/d^2]`. These quantities are computed by finite summation, not Monte Carlo.

| Batch size | Same-batch mean direction | Independent-metric mean direction | Same-batch direction variance | Independent direction variance |
|---:|---:|---:|---:|---:|
| 1 | +0.449975000 | -0.454500126 | 0.002068016 | 81.814111454 |
| 8 | +0.419242928 | -0.423631365 | 0.014885547 | 9.546787488 |
| 128 | +0.124574447 | -0.127652264 | 0.041559526 | 0.220284991 |
| 512 | +0.001742427 | -0.004273915 | 0.001202225 | 0.002155802 |
| 1024 | -0.001088553 | -0.001430343 | 0.000007408 | 0.000011119 |

At batch size 1 the same-batch direction has population-loss directional derivative `+0.227237375`; it is an ascent direction in mean. The independent-metric counterpart has derivative `-0.229522564`. The full-population-metric direction is `-0.001259069`. Complete enumeration of all batch sizes 1 through 1024 first finds a negative same-batch mean direction at **613**. This is a numerical sign threshold for this specified example, not a universal phase boundary or a proved monotonicity theorem.

The mean independent-metric direction being favorable does not imply a favorable finite-step expected loss: its variance at batch size 1 is much larger. The comparison uses a separate metric batch and therefore does not claim equal data/computation cost.

### Why this does not survive as innovation

For singleton batches, the mean same-batch update is exactly minus the gradient of

\[
\widetilde L(\theta)
=\mathbb E\left[\frac{\ell(Y,\theta X)}{X^2+\lambda}\right].
\]

Thus high-leverage examples are downweighted by a fixed input-dependent factor. The apparent rare-example discovery is sample reweighting; no adaptive representation, feature learning, or new information-flow mechanism is required. The singleton norm bound is equally elementary:

\[
|u|\leq \frac{|x|}{x^2+\lambda}\leq\frac{1}{2\sqrt\lambda}.
\]

This bound is consistent with severe target distortion: bounded individual updates do not ensure population descent. Independence removes self-inclusion but can expose large gradient/denominator mismatches. These observations do not constitute a new optimizer or theorem.

## Two analytic exclusions and a second exact check

**No inverse-expectation divergence from a positively damped PSD metric.** If `M >= 0` and fixed `lambda > 0`, then `||(M+lambda I)^-1|| <= 1/lambda`. Consequently `E||g|| < infinity` implies `E||(M+lambda I)^-1 g|| < infinity`, regardless of dependence. A proposed Wishart-style nonexistence phase at positive fixed damping fails this elementary bound. Undamped inverse singularities or a gradient without a first moment are different assumptions.

**No fresh label-variance drift at a fixed state for the current label-independent PB metric.** For square loss or canonical cross-entropy, write the minibatch gradient as `a(theta,X_B)-C(theta,X_B)Y_B`. Conditioning on the fixed state and current inputs gives

\[
\mathbb E[M(\theta,X_B)^{-1}g_B\mid\theta,X_B]
=M(\theta,X_B)^{-1}
\left(a-C\mathbb E[Y_B\mid\theta,X_B]\right).
\]

Changing only fresh zero-mean label noise therefore changes update second moments, not the conditional mean. This statement requires the conditional mean to remain unchanged and the metric to be independent of current labels given the state and inputs. It does not claim invariance of future parameter distributions, reused-label finite-dataset dynamics, or label-dependent empirical-Fisher methods.

The second script checks this in a nonlinear two-parameter representation held at `theta=(0.3,-0.2)`:

\[
z_\theta(x)=(\theta_1x_1+\theta_1\theta_2x_2,\theta_2x_2),
\]

using squared loss on `z[0]`, two input types, batch size 2, and damping 0.2. All 16 joint input/sign events are enumerated for each of no label noise, homoskedastic noise, and heteroskedastic noise. The largest difference in mean update is `5.55e-17`; the heteroskedastic change in update covariance has Frobenius norm `0.211279593`. This verifies the conditional-expectation reduction and supplies no new variance/replica claim.

## Reproduction and retained artifacts

Repository scripts:

- `experiments/new_direction_screen_20261121/stochastic_pb.py`
- `experiments/new_direction_screen_20261121/stochastic_label_conditioning.py`

Each accepts `--output <new-json-path>` and refuses to overwrite an existing output. Run from the repository using the workspace virtual-environment Python. Both runs completed with status `passed`. Results include source SHA-256 hashes, full data settings, moment checks, and the complete batch-size grid where applicable.

Research artifacts staged beside this audit:

- `stochastic_pb_results.json`
- `stochastic_label_conditioning_results.json`

No route in this lane is promoted. A future proposal would need a mechanism beyond these exact reductions and beyond the directly matching stochastic-preconditioning literature, followed by its own preregistered minimal falsification test.
