# Why an unrestricted natural-input TV upper bound is unavailable

Explanatory mathematical appendix, written after the controlled experiment. It changes no frozen assumption, selector, sample budget, or empirical result. Root independently reviewed the construction and quantifiers. This is an elementary indistinguishability argument, not a novelty claim.

Fix T(x)=-x on [-1,1] and the invariant null P0, uniform on this interval. Partition the magnitude |x| into m equal bins. For every sign vector s in {-1,+1}^m, define P_s by drawing the magnitude uniformly on [0,1] and assigning the sign s_j whenever the magnitude falls in bin j. Equivalently, P_s has twice the P0 density on its selected half of each reflected bin pair and zero on the other half.

Each P_s and its reflection have disjoint supports up to boundary sets of probability zero, so TV(P_s,T#P_s)=1. In contrast, TV(P0,T#P0)=0. Mix the alternatives by drawing each s_j independently and uniformly once, before drawing the whole IID sample from that fixed P_s.

For n observations, the magnitude-bin indices have the same law under this mixture and under P0. Conditional on all indices being distinct, the observed signs under the mixture are independent uniform signs, exactly as under P0. The within-bin magnitudes agree as well. A coupling therefore gives

    TV(P0^n, average_s P_s^n) <= Pr(a repeated magnitude bin)
                              <= n(n-1)/(2m).

Now let U(X_1,...,X_n) in [0,1] be an upper confidence bound that is valid uniformly over all input laws:

    for every P, Pr_P{ TV(P,T#P) <= U } >= 1-alpha.

Additional independent algorithmic randomization can be included without changing the argument. Since every alternative has TV=1, uniform coverage implies Pr_{P_s}(U=1)>=1-alpha for every s and hence also for their mixture. The coupling inequality yields

    Pr_{P0}(U=1) >= 1-alpha-n(n-1)/(2m).

The null P0 is the same for every m. Letting m tend to infinity proves Pr_{P0}(U=1)>=1-alpha. Thus a universally valid upper bound must be vacuous with at least its nominal coverage probability even at this particular exactly invariant continuous null. The difficulty is unrestricted fine-scale input-law complexity, not insufficient classifier training.

A fixed known finite alphabet, a specified parametric family, or explicit regularity/complexity restrictions can escape this obstruction. The theorem does not rule out useful TV estimation under such assumptions. In our controlled experiment the density is specified, so eta=2|a|/pi is known analytically and no estimation claim is needed.

This argument also explains why a classifier trained to distinguish original from transformed observations cannot supply the required upper bound without additional assumptions. A discovered distinguishing set certifies a lower bound on TV. Failure to find one does not exclude an unseen high-complexity difference. Naive empirical TV of disjoint continuous point clouds is typically one even under the invariant null.

Posterior defect is a separate identifiability issue: two joint laws can share exactly the same input marginal and have different conditional labels. No quantity based solely on those unlabeled inputs can distinguish their posterior symmetry defects. Label evidence or substantive assumptions are required. Neither this construction nor the controlled certificate asserts that label semantics alone establishes input-law symmetry.

For related primary-source context rather than attribution of this exact construction, see [Ben-David et al., AISTATS 2010](https://proceedings.mlr.press/v9/david10a.html) on indistinguishable domain-adaptation tasks and [Jhawar and Wang, 2026 preprint](https://arxiv.org/html/2609.11235v1) on evidence-limited adaptation selection. The fixed-model augmentation experiment here does not update the model using the verification stream.
