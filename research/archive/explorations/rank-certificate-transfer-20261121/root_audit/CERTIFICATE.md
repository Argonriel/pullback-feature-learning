# Independent all-posterior finite-sample certificate

The trained state and deployed rank table are fixed before all verification inputs. The companion batch contains255 independent inputs; its independent action keys preserve the conditional-uniform score law. Scores are continuous because the frozen width is positive. Consequently the true inclusion probability is Q(F(S)), Q=BetaCDF(192,64), averaged over the four possible interpolated actions. Coordinate CDF brackets are evaluator-only and do not alter masks.

Let h'=h(Tx), m=(h+h')/2, d=(h−h')/2, A=softplus(h)−softplus(m), and b(r)=E[q(X)d(X)|R=r]. For any measurable invariant class posterior p(r) in[0,1], the expected improvement in logistic loss is E[qA]−E[pb]. Define e=A−d/2 and f=A(Tx)+d/2, and W=(q(x)e+q(Tx)f)/2. Invariance of the input marginal underT yields E[W]=E[qA]−E[b]/2. Hence improvement>=E[W]−E[|b|]/2. No prevalence restriction or fitted posterior enters this statement.

The local numerical method encloses b(r) for every radius. B(r) is the maximum absolute endpoint. The known-law score CDF brackets enclose the true marginal deterministically in real arithmetic. For empirical-law verification, a separate independent radius sample supplies a uniform DKW band with failure probability delta; convolution with Uniform(0,width) preserves that uniform band. The local method propagates this band without treating fitted deployment calibration as an independent verification sample.

For a q interval, choose its lower endpoint when e>=0 and its upper endpoint when e<0; do the same forf. This givesW_lower<=W, so Z=W_lower−B(R)/2 has expectation no larger than any allowed posterior's gain. Outside the calibrated radius interval the deployed fallback is exactly half-turn-even: b=0 and W=q*JensenGap>=0. Setting W_lower=0 there is conservative.

Write g(z)=softplus(z)−z/2=log(2cosh(z/2)). It is convex and1/2-Lipschitz. Thus e=g(h)−g(m) and f=g(h')−g(m) satisfy |e|,|f|<=|d|/2 and e+f>=0. At most one is negative. Multiplying each by any weight in[0,1] gives −D/4<=W_lower<=D/2 when |d|<=D. Consequently Z lies in[−D/4−Bmax/2,D/2]. For rings restricted toR<=3, D=3+C_h; for the harmonic task D=1+C_h because the first input coordinate is bounded by1. Here C_h is the sum of half the absolute output-row weights of the residual blocks. This is deliberately a conservative global range; the variance term uses the independent observations.

For n=262144 iid observations ofZ, use the one-sided empirical-Bernstein lower bound

mean(Z)−sqrt(2*s2*log(2/delta)/n)−7*range*log(2/delta)/(3*(n−1)),

where s2 is the unbiased sample variance. This follows by applying Theorem4 ofMaurer andPontil (2009) to1 minus the affinely rescaled variable. The known-law family uses delta=.05/32 for the32 independent-model certificates; independence across models is not needed for the union bound. The empirical-law family, if executed, separately uses delta=.05/64 for32 DKW events and32 conditional empirical-Bernstein events. These are two separately labelled95% families, not a joint95% claim across both families.

Reference: Maurer andPontil, *Empirical Bernstein Bounds and Sample Variance Penalization*, COLT2009, Theorem4: https://www.cs.mcgill.ca/~colt2009/papers/012.pdf .

The theorem is a real-arithmetic statement. The implementation uses padded float64 interval calculations and independent numerical checks, without directed-rounding verification. A positive computed certificate is conditional on those enclosures being numerically valid and the stated iid/invariance/model assumptions. This is not a formal machine-checked proof, a per-batch guarantee, an accuracy guarantee, or a claim for arbitrary non-invariant label shifts.
