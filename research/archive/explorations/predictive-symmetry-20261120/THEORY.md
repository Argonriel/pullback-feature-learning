# Predicting loss matched updates and selecting a second view

## A common training loss clock

Let L(theta) be the mean training logistic loss and let v_m be a descent direction at a fixed state theta, for m in {GD, pullback}. Define q_m = -DL(theta)[v_m] > 0. For a desired actual training improvement delta, select the first descending root s_m(delta) of

\[
L(\theta+s_m v_m)=L(\theta)-\delta.
\]

Only training data determine this scalar. The local implicit function theorem gives s_m(0)=0 and s'_m(0)=1/q_m. Thus, for any differentiable population functional Q,

\[
Q(\theta+s_m v_m)-Q(\theta)
=\delta\,\frac{DQ(\theta)[v_m]}{q_m}+O(\delta^2).
\]

The difference between these two predicted changes is a same-state, equal-progress comparison. It is not a statement that one optimizer always generalizes better. The leading response must be frozen before evaluating the new model on the assessment distribution. Estimating DQ on a separate prediction sample and evaluating changes on an independent assessment sample avoids reusing the same integration noise.

There are two different approximations. The drop-clock predictor uses delta/q_m. The realized-step tangent uses s_m DQ[v_m]; it can be computed after the training-only line search but before held-out scoring. Its scalar already contains nonlinear training information, so it is a stronger diagnostic baseline, not an entirely pre-update prediction.

For a fixed ray and three-times continuously differentiable losses locally (as for the finite tanh/softplus models here), the expansion can be continued:

\[
\Delta Q=\delta\frac{DQ[v]}q+
\frac{\delta^2}{2}\left(\frac{D^2Q[v,v]}{q^2}
+\frac{DQ[v]D^2L[v,v]}{q^3}\right)+O(\delta^3).
\]

This expression is a derivation, not an evaluated second-order method in the reported experiment. At a finite target the exact derivative is -DQ(theta+s v)[v]/DL(theta+s v)[v]. A root may cease to exist before an arbitrarily large target; finite targets and failures must be reported rather than silently rescaled. Recomputing a direction between steps defines another path. Extending the initial slope over that path is an empirical forecast, with no general long-horizon guarantee.

## Separating task fit from nuisance variation

Write B(h)=log(1+exp(h)), so conditional binary cross-entropy is B(h)-p h. For radial labels p=p(r), let mu(r) be the angular mean of h(r,phi). Population risk has the exact decomposition

\[
R=R_{Bayes}+E_r\{B(\mu)-p\mu-H(p)\}
+E_{r,\phi}\{B(h)-B(\mu)\}.
\]

The last two terms are radial fitting excess and angular nuisance excess. If w(x)=Dh(theta,x)[v], then

\[
D R_{fit}[v]=E_r[(\sigma(\mu)-p)\bar w],\qquad
D R_{nuis}[v]=E[\sigma(h)w-\sigma(\mu)\bar w].
\]

The second expression requires no labels or posterior. A symmetry projection P_G, defined by averaging over the full group with its uniform or Haar probability measure, generalizes this decomposition to a joint-distribution-preserving group: fit is the risk of P_G h and nuisance is E[B(h)-B(P_G h)]. The two-view averaging operator below is generally not an idempotent projection. If angle itself carries label signal, averaging over all angles destroys signal. Only the actual task symmetry may define P_G and permissible extra views. Population label balance and Bayes difficulty can be matched by giving the two tasks the same distribution of posterior probabilities; this does not imply equal finite-network approximation or optimization difficulty.

## What a fixed rotation removes

On a circle write h(r,phi)=sum_k c_k(r) exp(ik phi), with c_-k the conjugate of c_k. Two-view logit averaging at rotation alpha gives

\[
P_\alpha h=\tfrac12(h(r,\phi)+h(r,\phi+\alpha)),\qquad
c_k\longmapsto e^{ik\alpha/2}\cos(k\alpha/2)c_k.
\]

Therefore each nonzero mode's squared amplitude is multiplied by cos^2(k alpha/2). A half-turn eliminates odd modes and preserves every even mode. A quarter-turn eliminates modes k=2 modulo 4 but preserves multiples of 4. No single nontrivial rotation eliminates all modes. Parseval makes this an exact logit-variance identity; it is not an exact logistic-risk identity.

When angular fluctuations u=h-mu are small, the angular risk is approximately B''(mu) E[u^2]/2, so weighting spectral power by radial curvature gives a local proxy. Large fluctuations, finite truncation, and nonlinear interactions can invalidate the proxy's ranking. Its comparison with an exact Jensen-based selector is an essential control.

## Unlabeled calibration of actual expected gain

If the joint input-label distribution is invariant under x -> T_alpha x, the population gain is

\[
G_\alpha=R(h)-R(P_\alpha h)
=E\left[\frac{B(h(X))+B(h(T_\alpha X))}{2}
-B\!\left(\frac{h(X)+h(T_\alpha X)}{2}\right)\right]\ge0.
\]

The label terms cancel in expectation. The expression inside brackets is a label-free Jensen gap. A complete angular orbit with a grid-aligned rotation also cancels the label contribution exactly on that discrete orbit for radial labels. This does not say the gain is nonnegative on every individual labeled example, nor does it establish cancellation on a finite iid sample without orbit averaging.

A radius-dependent alpha(r), selected before deployment, preserves the radial law and uniform conditional angle. It satisfies the same identity. Arbitrary angle-dependent choices do not inherit that guarantee. For an angular-label task, transformations must preserve its posterior as well as its input distribution.

For a fixed permissible alpha, its pre-update directional derivative is

\[
DG_\alpha[v]=E\left[\frac{\sigma(h)w+\sigma(h_\alpha)w_\alpha}{2}
-\sigma(P_\alpha h)P_\alpha w\right].
\]

The loss-clock forecast G_alpha + delta DG_alpha[v]/q can select a candidate view using pre-state calibration predictions. The associated selector must be frozen before post-update risk is scored. Selecting an angle from the actual updated model on independent unlabeled calibration data is a stronger comparator with extra offline cost.

Every universal two-view policy deploys exactly two network evaluations per input. An offline orbit grid can score many candidate angles by reusing predictions, but every grid point still incurs a network evaluation during calibration. Derivative evaluation and training-direction construction add costs beyond ordinary forward evaluations. The training model is unchanged by inference averaging: any improvement from view selection is a change in the inference procedure, not evidence that the underlying fitted model's single-view generalization improved.

## Relation to established results

The distinction between optimizer direction and step magnitude follows the experimental motivation of [Agarwal et al. (2020)](https://arxiv.org/abs/2002.11803). The present normalization matches actual training improvement rather than parameter norm. Convex-risk benefits of distribution-preserving averaging are established by [Lyle et al. (2020)](https://arxiv.org/abs/2005.00178). Exact log-loss diversity decompositions are developed by [Wood et al. (2023)](https://jmlr.org/papers/v24/23-0041.html). The elementary Fourier transfer and the specialized derivative/calibration experiment here are presented as a mechanism and a controlled empirical extension, not a claim to have invented symmetry averaging or ensemble diversity.
