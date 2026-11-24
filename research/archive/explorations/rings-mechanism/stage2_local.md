# Charging for an unused feature

The classifier reads only the first output \(z_1\). Let \(A\) and \(B\) be the training-set Jacobians of \(z_1\) and \(z_2\), respectively. With \(g=\nabla_{z_1}L_{\rm train}\), define

\[
b=A^\top g,\quad H_t=A^\top A+t B^\top B+\lambda I,\quad s_t=H_t^{-1}b,\quad
\Delta\theta=-\eta s_t,\quad t\ge0,\quad\lambda>0 .
\]

At weight zero the output-space metric is a semimetric. The positive damping still makes the parameter-space system invertible. Giving \(z_2\) a weight changes the penalty for feature motion, while leaving the model, its readout, labels, and objective fixed.

## Fixed rate

At a fixed state, the predicted first-order loss decrease per unit learning rate is \(q(t)=b^\top s_t\). Differentiating \(H_t s_t=b\) gives \(s'_t=-H_t^{-1}B^\top B s_t\). Consequently

\[
q'(t)=-\|Bs_t\|^2,\qquad
q''(t)=2(B^\top B s_t)^\top H_t^{-1}(B^\top B s_t)\ge 0.
\]

The first identity proves that increasing the unused-feature weight costs predicted immediate training improvement. The second proves that, at equal learning rate, the squared predicted \(z_2\) motion is nonincreasing as well. If \(Bs_0=0\), \(H_t s_0=b\) for every \(t\), giving identical updates. Otherwise \(Bs_t\ne0\) at every finite \(t\) (if it vanished, \(H_0s_t=b\) would imply \(s_t=s_0\)). The fixed-rate loss-decrease ordering is then strict.

## Matching first-coordinate motion

For fixed RMS logit displacement \(r>0\) choose \(\eta_t=\sqrt{n}r/\|A s_t\|\). If \(b\ne0\), then \(g^\top A s_t=q(t)>0\) and this denominator is positive. The new predicted decrease becomes
\[
\sqrt{n}r\,\frac{q(t)}{\|A s_t\|}
=\sqrt{n}r\,\|g\|\cos(g,A s_t).
\]
This quantity depends on the logit direction alone at the common state and need not decrease with \(t\). For instance, let \(A=\operatorname{diag}(1,2)\), \(B=\operatorname{diag}(0,2)\), \(g=(1,1)^\top\), \(\lambda=1\). Then \(A s_t=(1/2,\,4/(5+4t))^\top\). The alignment increases until \(t=3/4\), where it reaches one, and then decreases. Finite parameter updates also introduce nonlinear feature motion.

## Original paired rings states

Eight seeds supply six states each: initialization, GD and pullback at steps 100 and 500, and the warm-up schedule at step 500. At each state all four weighted directions use the same parameters and training data. The matched rule uses the predicted training-logit length of the \(t=1,\eta=8\) update at that state. Each row below first averages six states per seed and then averages eight seeds.

| Scalar rate or normalization | \(t\) | Predicted loss decrease | Observed loss decrease | Predicted \(z_1\) RMS | Predicted \(z_2\) RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rate 8 | 0 | 0.0018564 | 0.0019179 | 0.0059392 | 0.0066270 |
| Rate 8 | 1/9 | 0.0018343 | 0.0018970 | 0.0058960 | 0.0011711 |
| Rate 8 | 1 | 0.0018069 | 0.0018738 | 0.0058460 | 0.0005064 |
| Rate 8 | 9 | 0.0017667 | 0.0018394 | 0.0057774 | 0.0002033 |
| Matched \(z_1\) | 0 | 0.0018199 | 0.0018784 | 0.0058460 | 0.0064784 |
| Matched \(z_1\) | 1/9 | 0.0018150 | 0.0018761 | 0.0058460 | 0.0011536 |
| Matched \(z_1\) | 1 | 0.0018069 | 0.0018738 | 0.0058460 | 0.0005064 |
| Matched \(z_1\) | 9 | 0.0017922 | 0.0018674 | 0.0058460 | 0.0002077 |

At rate 8 the predicted loss decrease and predicted \(z_2\) motion decreased monotonically at all 48 states. The median within-state predicted \(z_2\) motion ratio \(t=9\) relative to \(t=0\) was 0.0294. Actual finite-step loss decrease preserved the ordering at 46/48 states. With predicted logit movement matched, the predicted loss-decrease ordering held at 42/48 states, and the actual loss-decrease ordering held at 33/48. All 384 hypothetical branches reduced training loss at their corresponding states. Counts describe repeated comparisons on eight data/initialization pairs. They measure local training consequences; the trajectory experiment measures the accumulated population-risk consequences separately.

The maximum error in the finite-difference derivative, scaled by \(|q'|+q/(1+t)\), was \(1.26\times10^{-8}\). The matched RMS targets agreed within relative error \(2.23\times10^{-16}\). At \(t=1\) the regrouped Gram solve agreed with the original solver within relative parameter-direction error \(2.97\times10^{-12}\). The script checks all starting-state hashes and training losses. [Design](stage2_local_plan.md), [branch and motion data](stage2_local/branches.csv), [derivative checks](stage2_local/derivatives.csv), [paired seed and state summaries](stage2_local/group_means.csv), [ordering counts](stage2_local/orderings.csv), [audit](stage2_local/checks.json).
