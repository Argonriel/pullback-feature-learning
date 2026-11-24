# Pre-fresh additive bounded-difference certificate

The original empirical-Bernstein source, protocol, freeze, and outputs are preserved. This addendum was designed and independently reviewed before fresh models or verification data existed. It adds a concentration method with no additional model calls and leaves the primary budget at 512 batches.

## Pointwise-score replacement lemma

Fix all calibration, model parameters, and a deterministic tie rule. Each item's score and weight depend only on that item's input and independent auxiliary key. Let F be the sum of weights of the top q items in a batch of B, with weights in [0,c]. Replace one input-key pair while holding every other pair fixed. The other B-1 items retain their relative score order.

- If the replaced item is selected before and after replacement, the selected other items are unchanged. Only its weight changes, by at most c.
- If it moves into or out of the selected set, it exchanges places with exactly one unchanged item. The sum changes by at most c.
- If it is unselected both times, nothing changes.

Thus the deterministic bounded difference is c. The score need not equal, or be monotone in, the weight. A score depending on the empirical batch distribution, batch normalization at inference, or recomputed batch quantiles would invalidate this particular argument. The frozen interpolation and histogram selectors are pointwise functions, so they satisfy it. Fixed coordinate-index tie fallback also preserves the relative order of unchanged items.

Across n independent batches, the statistic mean(U_c)=sum(F)/(nB) is a function of nB independent input-key pairs. Each coordinate has sensitivity c/(nB). By the independent bounded-differences inequality,

    P(mean(U_c) - E U_c >= t) <= exp(-2 n B t^2/c^2).

Therefore the one-sided lower bound is

    mean(U_c) - c sqrt(log(1/delta)/(2 n B)).

This does not assume that selected input contributions are IID. It applies a concentration theorem to the full dependent selection functional of independent underlying inputs. The result can also be derived directly: expose the nB independent pairs in a Doob martingale, apply Hoeffding's lemma to each conditional range c/(nB), multiply the moment bounds, and optimize the exponential parameter.

For the paired comparison, each policy's selected sum has replacement sensitivity Jmax. Their difference has sensitivity at most 2 Jmax by the triangle inequality. Apply the same formula with c replaced by 2 Jmax to the signed truncated batch mean, then subtract the same deterministic rings tail allowance. The signed comparison is never naively clipped.

The primary foundational reference is [McDiarmid, *On the method of bounded differences*, Surveys in Combinatorics (1989), pp. 148–188](https://doi.org/10.1017/CBO9781107359949.008). The author's later [*Concentration* survey](https://www.math.cmu.edu/~ctsourak/T797003/Readings/concentration/mcdiamird-conc-survey.pdf) develops the same independent-coordinate method. The replacement lemma above is the task-specific step needed to use it here.

## Final simultaneous family and interpretation

The final family has 48 models times three claims times three looks times two concentration methods = 864 events. Each method uses delta=.05/864. Both empirical Bernstein and bounded differences are recalculated from the same saved batch observations. Their maximum is also a valid lower bound under this joint family allocation. There is no additional search over clipping levels, budgets, or policies on fresh data. The previously generated 432-event empirical-Bernstein rows remain a historical diagnostic, not the final primary certificate.

The addendum has no extra training, calibration, coordinate, or prediction cost. It changes only the concentration calculation. All original 128/512/2048 observations remain, and the fixed primary report is still 512. A favorable 128-batch result may be reported as a cost-curve finding, not as a retrospective change to the primary budget.

`final_compare.py` saves each empirical-Bernstein lower bound, each bounded-difference lower bound, their simultaneous maximum, all penalties, actual shared forwards, and first-positive thresholds. Its old-only development stage uses the already saved old 32-model batch observations; it does not access any fresh states or outcomes. The empirical-Bernstein original source is not modified.
