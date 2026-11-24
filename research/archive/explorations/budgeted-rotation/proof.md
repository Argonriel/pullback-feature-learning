# Routing a second view under a prediction budget

Let a fixed binary classifier give a logit \(h(x)\). For label \(y\in\{0,1\}\) its logistic loss is \(\ell(y,a)=B(a)-ya\), where \(B(a)=\log(1+e^a)\). Write \(T x=-x\) and \(P h(x)=(h(x)+h(Tx))/2\). Assume the joint law of \((X,Y)\) is invariant under \((X,Y)\mapsto(TX,Y)\). The noisy rings with independent uniform angles and isotropic noise satisfy this condition.

## The value of a second view

Define the unlabeled score

\[
d(x)=\tfrac12 B(h(x))+\tfrac12 B(h(Tx))-B(P h(x)).
\]

The quantity \(d\) is nonnegative by convexity, constant on each antipodal pair, and strictly positive when the two finite logits differ. On an individual labeled input the change in loss can have either sign. Average the loss change over both orientations with the same label to obtain exactly \(d(x)\): the two linear terms in the label cancel. Consequently, for any invariant observable \(C\) satisfying \(C(Tx)=C(x)\), the expected decrease conditional on \(C\) is

\[
\mathbb E[\ell(Y,h(X))-\ell(Y,P h(X))\mid C]
=\mathbb E[d(X)\mid C].
\]

This identity also holds for a randomized gate whose decision uses only \(C\) and randomness independent of the held-out orientations and labels. The same proof works for a batch gate: condition on all the cells in the independent batch and the tie-breaking randomness, then exchange the two orientations of one observation at a time. The gate's choice stays fixed in each exchange.

The score weights disagreement by uncertainty. Writing \(a=h(x)\), \(b=h(-x)\), \(m=(a+b)/2\) and \(p=\operatorname{sigmoid}(m)\) yields

\[
d(x)=\tfrac12\log\!\left[1+4p(1-p)\sinh^2\big((a-b)/4\big)\right].
\]

At fixed logit disagreement, the gain is largest near \(m=0\). The original softplus expression is convenient for numerical computation. The empirical policy spends two model evaluations per calibration observation to estimate cell averages. At prediction time the cell uses the radius and, where specified, the angle modulo \(\pi\), both computed from the input before any second model call.

## Optimal allocation among cell-based gates

Suppose an independent batch has \(n\) inputs and exactly \(k\) extra model calls are available. Let \(\mu(c)=\mathbb E[d(X)\mid C(X)=c]\). Conditional on the cells, the expected reduction in batch-average loss for subset \(S\), \(|S|=k\), is

\[
\frac1n\sum_{i\in S}\mu(C_i).
\]

If an excluded item has a higher cell score than an included one, exchange them to increase the sum. Thus choosing the \(k\) highest true cell scores maximizes expected gain given these observables and equal second-view costs. Random tie-breaking independent of orientation gives exactly \(k\) calls and the same expected gain at an exact tie. A cell with \(m\) tied items competing for \(j\) remaining slots gives each a selection probability \(j/m\) under uniform tie-breaking. Calibration estimates \(\mu\); finite estimation can misrank cells. More detailed cells can increase the optimum obtainable with true scores and can also increase estimation error.

For example, two equal-mass radii could have zero and positive antipodal disagreement. A radius gate can concentrate its budget on the second radius. Conversely, if \(\mu(c)\) is constant across the available cells, a targeted rule has the same expected gain as random assignment. Computing \(d(x)\) for every deployment item before choosing \(S\) would already incur all the extra calls; that rule only gives an offline comparison.

If every occupied cell estimate is within \(\varepsilon\) of its true score, then for each batch the expected gain of the estimated ranking is within \(2k\varepsilon/n\) of the true cell-based optimum. Indeed, replace true scores by estimates in the optimal set (cost at most \(k\varepsilon\)), switch to the highest-estimate set (no decrease in their sum), then replace its estimates by true scores (at most \(k\varepsilon\) again). This deterministic bound describes what calibration accuracy would suffice. The experiments do not assume a known numerical value of \(\varepsilon\).

## What changes in the previous rings loss decomposition

Under the rings law, for each radius \(r\) the angle is uniform and the class posterior depends only on \(r\). The prior decomposition measures radial fit using \(\bar h(r)=\mathbb E_\phi h(r,\phi)\). For a gate \(q\) that agrees at antipodal inputs, the gated logit \(h_q=h+q(P h-h)\) has the same angular mean at each radius: \(q(P h-h)\) is antisymmetric under \(\phi\mapsto\phi+\pi\), so its integral vanishes. This applies conditional on other batch items and independent random priorities.

Therefore the change in expected loss from a binary gate comes entirely from the angular contribution, including when the gate distinguishes different directions modulo \(\pi\). Using the second view everywhere yields an exact partition of the original angular penalty into \(\mathbb E d\) and the remaining angular penalty of \(P h\). The latter can be substantial. For instance \(h(r,\phi)=\cos(2\phi)\) is angularly varying and already antipodally symmetric, so every antipodal gain is zero.

The guarantee covers expected logistic loss under the stated symmetry. Accuracy, other populations, and gating on an orientation-sensitive first logit require separate evaluation. The per-batch count gives the number of model calls at inference. Sorting inputs, collecting calibration predictions, and hardware batching affect wall time separately.

## A diagnostic for first-view confidence routing

The confidence supplement selects the 64 smallest absolute first-view logits in each batch of 256. Let \(m_+,m_-\in\{0,1\}\) be the indicators from routing the original batch and its sign-reversed counterpart with identical tie priorities. For a particular item let \(\ell_a,\ell_b\) be conditional logistic risks of the two first logits for the same class probability \(p(r)\). Let \(\ell_m\) be the risk of the mean logit. The two-orientation average gain of the gate is

\[
\tfrac12[m_+(\ell_a-\ell_m)+m_-(\ell_b-\ell_m)]
=\tfrac{m_++m_-}{2}d
  +\tfrac{m_+-m_-}{4}(\ell_a-\ell_b).
\]

The first term is nonnegative by Jensen's inequality. The second term can have either sign. It is positive when the gate preferentially changes the worse of the two first predictions, and it vanishes for an antipodally invariant gate. Confidence routing beats using both views everywhere if its expected signed term exceeds the Jensen gain omitted by its budget.

| Model | Symmetric term | Signed orientation term | Confidence gain over raw | Full gain over raw | Pairs with confidence better than full |
| --- | ---: | ---: | ---: | ---: | ---: |
| GD | 0.01152 | 0.01876 | 0.03028 | 0.02852 | 14/16 |
| Pullback | 0.00730 | 0.00438 | 0.01168 | 0.00942 | 16/16 |
| Hybrid | 0.01477 | 0.00152 | 0.01629 | 0.01740 | 1/16 |

The entries are mean reductions in expected logistic loss on the frozen supplement draws. For pullback, \(0.00438\) nats from orientation-selective routing exceeded the \(0.00211\) nats of omitted Jensen gain, yielding a net \(0.00227\)-nat improvement over always using the second view. The model's own first-view confidence is available at inference. Its signed gain here is an empirical measurement using the known posterior for post-hoc scoring; the theoretical symmetry guarantee only covers the Jensen term. [Per-model arithmetic](supplemental_results/confidence_terms_models.csv) · [Grouped means](supplemental_results/confidence_terms_summary.csv)
