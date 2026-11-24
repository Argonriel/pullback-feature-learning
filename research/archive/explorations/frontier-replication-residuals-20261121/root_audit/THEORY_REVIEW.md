# Independent mathematical review before new outcomes

Root derived and checked the following independently of the experiment runner. The numerical preflight records 300 finite input populations, 64 exhaustive three-record batches per population, 1,001-point interval checks, and 10,000 posterior-rectangle checks. Numerical agreement supports implementation; the argument below supplies the algebra, not a formal machine proof.

## Density-interval propagation

Write B=S/2, |t|<=B, and f_e(r)=r t-[e+(1-e)|r|]B for e in [0,1]. This is concave and piecewise linear in r. Therefore its minimum on [l,u] is min(f_e(l),f_e(u)); equivalently the required nonnegative penalty is the maximum of the two endpoint penalties. It is nonnegative because |r|B-r t>=0 and e(1-|r|)B>=0. If |r|<=rbar throughout the interval and S<=D, the penalty is at most D[e/2+(1-e/2)rbar]. This bound is support-wide, conditional on independently estimated parameters, rather than an observed maximum.

An odd feature vector with symmetric coefficient uncertainty and an even residual radius gives l(Tx)=-u(x). Clipping to [-1,1] preserves that identity. Since S is even and t is odd, exchanging the two endpoints leaves the maximum penalty unchanged. Thus the resulting weight is even, as required for the coordinate-wise orbit identity and top-quota deployment proof.

For retained cosine harmonics, the equation theta_j=2 E_P cos(j phi) additionally requires the residual's orthogonality to each retained harmonic under the invariant reference measure. Oddness and a small residual bound alone do not establish that identity. Orthogonality, reference density, independence of radii and angle, and the residual envelope must remain explicit assumptions.

## Posterior interval propagation

For binary probabilities in paired cells, a simultaneous confidence rectangle p_j in [L_j,U_j] gives |p_j-p_Tj| <= max(U_j-L_Tj,U_Tj-L_j). Conditional on cell memberships and under within-cell posterior constancy, the observed labels in each cell are IID Bernoulli with that cell's probability. Hoeffding intervals conditional on the random cell counts are therefore valid; a zero-count cell must receive [0,1]. Union bounds do not require independence between the interval events.

This certifies a support-wide posterior defect only within the stated constant-cell model. Cell averages cannot certify arbitrary pointwise posteriors. A concrete counterexample is p(phi)=1/2+(1/2)sin(3phi). Under T(phi)=phi+pi its posterior TV is |sin(3phi)|, with supremum one. Under all the study's cosine-only angular laws, the mean of sin(3phi) on each right/left half-circle is zero: the sine term is odd about each half-circle center, while its density is even. Radii are independent. Consequently every one of the eight radius-by-orientation cells has mean label probability 1/2, and the paired-cell mean defect is zero, even though the pointwise defect reaches one. More labels within these same cell summaries cannot remove that structural limitation.

## Simultaneous concentration

For an invariant top-q score and nonnegative item weight bounded by L, replacing one record/key can change the selected sum by at most L. All other scores and their mutual ordering are fixed; at most one selected record is exchanged. The average across n batches of N records has bounded differences L/(nN). McDiarmid thus gives radius L sqrt(log(1/delta)/(2nN)). A separate empirical Bernstein event on the n independent batch means can be intersected with this bound only if both events are included in the family allocation.

Estimated coefficient intervals and labeled posterior bounds are independent of verification samples. Conditioning on those estimates makes each reported weight and its deterministic range fixed. Integrating the conditional concentration statement and adding parameter/label failures justifies simultaneous coverage. An intentionally violated residual-envelope row lies outside that coverage statement, even if included in the union-bound arithmetic.
