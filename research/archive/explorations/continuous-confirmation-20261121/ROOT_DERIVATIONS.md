# Independent derivations for the new methods

## Continuous input, exact quota, and conditional balance

Partition an IID batch of 256 observations into 64 groups of four independently of their values (fixed consecutive groups are also valid for IID ordering). A random permutation may make the implementation exchangeable. Let u(x) be a fixed, calibrated function in [0,1]. Within each group select exactly one item with probability

q_i = 1/4 + (u_i - group_mean(u))/3.

The centered term is between -3/4 and +3/4, so each q_i is in [0,1/2] and the four probabilities sum to one. Draws across groups give exactly 64 selected inputs. No second-view output enters this decision if u uses only the input, first output, and fixed offline tables.

For an IID companion distribution, the marginal inclusion probability given X_i=x is

q(x) = 1/4 + (u(x)-E u(X))/4.

Write d(x)=(h(x)-h(-x))/2. Under a uniform conditional orientation law, E[d|R=r]=0. Consequently the deployed conditional signed imbalance is exactly

b(r)=E[q(X)d(X)|R=r]=(1/4) E[u(X)d(X)|R=r].

This is a continuous-population statement. There is no quota-repair error term: the grouping enforces the quota directly. Randomness produces exact quota per realization but the balance statement concerns expectation over group companions and the within-group draw. An orientation-sensitive u can have E[ud|r]=0 without being antipodally symmetric. This condition must be proved or bounded; fitting one radius-bin moment is not enough.

If c(x)=A(h(x))-A((h(x)+h(-x))/2), unlabeled C for this deployment is E[c]/4 + Cov(u,c)/4. For a radial posterior, total gain is C-E[p(R)b(R)]. The approximation and calibration analysis belongs to E[ud|r], attenuated by 1/4. If u=0 outside a known radial cutoff, the additional imbalance is also zero there; the uniform 1/4 base allocation remains symmetric.

The grouping restricts marginal probabilities and allocation compared with selecting the global best 64. Its value is a clear balance/quota guarantee, not presumed empirical superiority. Group companions must remain IID from the stated population; sorting by radius first changes the marginal calculation and needs a different proof.

## A tighter rectangle certificate

For F(a,b)=J(a,b)-epsilon|a-b|/4, the rectangle minimum need not equal minimum(J) minus epsilon times maximum(|a-b|)/4. The latter is a valid but potentially loose relaxation. Exact joint minimization can improve certification without additional model evaluations.

For 0<epsilon<1, any non-diagonal stationary point in the rectangle interior satisfies a=-b and sigmoid(a)=(1+epsilon)/2 after ordering a>b. Along the common translation of a and b, the second derivative of J there is sigmoid'(a)-1/4<0. Thus this point cannot be a local minimum. The diagonal has F=0. A global minimum is attained on an edge or the diagonal. For epsilon=0, the closest-point Jensen rule applies; epsilon=1 has no finite non-diagonal stationary solution.

On an edge a=A and b<=A, with c=exp(A/2) and t=exp(b/2), the zero derivative condition is

epsilon*c*t^3 + (epsilon+2)*t^2 + c*(epsilon-2)*t + epsilon = 0.

All positive real roots in the edge interval, all endpoints, and diagonal intersections must be considered. The other ordered edge follows by symmetry. A finite candidate formula is mathematically exact, but numerical root finding still needs robust filtering and conservative tolerance. Random dense-grid tests check implementation; they do not turn ordinary floating-point computation into formal interval arithmetic.
