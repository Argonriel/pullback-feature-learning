# Independent review of the all-posterior confidence certificate

The centered-gain identity, deterministic range, pointwise local-bound extraction, truncation, and sample-size accounting are correct. The field coefficient bound gives q(x),q(-x) in [1/8,3/8]. For U=softplus(a)-softplus(m)-(a-m)/2 and V=softplus(a')-softplus(m)-(a'-m)/2, each lies in [-D/2,D/2], while U+V is twice a nonnegative Jensen gap. Therefore the antithetic W lies in [-D/16,3D/16]. Subtracting B(R)/2 yields the stated range length D/4+Bmax/2.

For any measurable radial posterior, the unknown term after centering at one-half is bounded by E[B(R)]/2. This is a simultaneous deterministic statement over all hidden posteriors, so it does not require an additional multiple-testing correction over posterior functions. Outside radius 3, the field is zero and the omitted pair-averaged contribution is nonnegative. One antithetic pair is one independent observation, consistent with n=262,144.

I independently checked Theorem 4 of Maurer and Pontil (2009), *Empirical Bernstein Bounds and Sample Variance Penalization*, at the primary source https://www.cs.mcgill.ca/~colt2009/papers/012.pdf. Its pairwise variance equals NumPy's unbiased variance with ddof=1. The constants sqrt(2*s2*log(2/delta)/n) and 7*range*log(2/delta)/(3*(n-1)) match. Applying the theorem to (upper-Z)/range gives the needed lower confidence bound. Setting delta=0.05/16 and taking a union bound gives family confidence at least 95%, conditional on the frozen models and local bounds; independence between trained models is not required for that union bound.

The root's pointwise signed interval construction agrees with the independently derived interpolation remainder, including its sign. Its maximum absolute endpoint is bounded by the stored uniform bound. The finite-sample certificate requires uniform conditional orientation, independent validation inputs, and the real-arithmetic validity of B. It does not require a known radial density. The ordinary floating-point implementation remains numerically audited rather than formally directed-rounded.

No changes to the reviewed source were made by this reviewer.
