# Joint quota: what is optimized and what is not

Write h(x) for the fixed network's first-view logit and B(h)=log(1+exp(h)). At radius r, let orientations be uniform and labels depend only on r. For a rotation alpha, define the conditional benefit of averaging two logits:

    g(r,alpha) = E_phi[ B(h(x)) - B((h(x)+h(R_alpha x))/2) | r ].

The label-linear term cancels because both orientations have the same conditional marginal. Equivalently,

    g(r,alpha) = E_phi[ (B(h(x))+B(h(R_alpha x)))/2
                              - B((h(x)+h(R_alpha x))/2) | r ] >= 0.

A full finite angular orbit makes the two empirical marginals exactly equal for every candidate grid rotation. Thus the calibration table uses unlabeled Jensen gaps and incurs no extra candidate-specific model calls after the orbit logits are computed. The128 orbit points are correlated quadrature nodes conditional on a radius. They do not turn64 calibration radii into 8192 independent draws.

For a batch with radii r1,...,r256, suppose the mask and angles depend only on those radii and independent tie keys. Conditional expected total risk reduction is the sum over selected inputs of g(ri,alpha_i). With exact g and a common finite angle candidate set, the best policy within this radius-measurable class independently selects each input's maximizing angle and chooses the 64 largest maxima. Sorting estimated gains gives the experiment's joint policy. This is an optimization of a calibrated surrogate; noisy estimates, coarse radius bins and model variation prevent a universal held-out improvement guarantee.

The fixed-mask factorial isolates the angle choice: the radius-reference mask always ranks the half-turn gain, even when a different angle is deployed. Random and confidence masks likewise stay identical across all four angle rules. The angle-aware radius variants then change only the priority score for a fixed learned/default angle. The final joint arm uses the learned radius angle and its corresponding maximized gain score.

The strong folded half-turn gate is a different measurable class. Its coordinates (r, phi modulo pi) identify an antipodal pair, preserving the conditional half-turn symmetry. Its expected pairwise improvement is exactly the pair's Jensen gap, so it can exploit orientation detail unavailable to a radius-only gate. Its guarantee applies to its half-turn pair and is not transferred to arbitrary angles. Even an oracle radius joint policy is not guaranteed to dominate this finer folded gate.

A first-view confidence mask depends on the orientation-specific logit. Its selection need not preserve the relevant transformation marginals, so one cannot replace its actual risk change by an unlabeled Jensen score or promise monotonic improvement. It can nevertheless outperform the invariant rules. The experiment evaluates confidence directly using conditional label probabilities and independently sampled labels, without imposing a symmetry guarantee.

The original IID-calibrated folded gate is retained unchanged at equal model-forward budget. At 8192 forwards it sees 4096 independent inputs and their antipodes. The orbit method sees 64 independent radii and 128 angular quadrature points per radius. These are different sampling designs with the same forward cap, not equivalent independent-sample sizes. The separate orbit-folded gate reveals how much the calibration design itself changes the established baseline.

Every deployed budgeted policy first evaluates all 256 inputs, chooses its 64 entries using available first-view logits and coordinates, then evaluates only 64 rotated inputs. It never queries candidate second views before allocation. Full-two-view references cost 2 calls per input and use a different deployment budget; they are not universal risk ceilings or competitors under the 1.25-call constraint. A full-radius reference is only a reference for that particular calibrated angle rule, not a global optimum over unrestricted inference algorithms.
