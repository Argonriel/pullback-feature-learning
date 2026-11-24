# Local prediction identities and scope

Let L(theta) be empirical logistic training loss and Q(theta) one of the population outcomes. At a fixed shared state theta choose d_GD=-grad L or d_PB=-(J^T J+0.1 I)^(-1)grad L. Write s=-grad L^T d>0, q1=grad Q^T d. On the first local descending branch, solve L(theta+alpha(delta)d)=L(theta)-delta. The inverse-function theorem gives alpha'(0)=1/s and

    Q(theta+alpha(delta)d)-Q(theta) = (q1/s) delta + O(delta^2).

Thus the pre-state prediction for the PB-minus-GD outcome contrast at equal actual training-loss drop is delta*(q1_PB/s_PB-q1_GD/s_GD). This is a local mathematical consequence, not a universal finite-step theorem. Tiny-drop success validates the implementation and normalization; the substantial-drop and frozen-horizon comparisons test usefulness beyond that identity.

With twice differentiable L,Q and fixed d, let l2=d^T Hess(L)d and q2=d^T Hess(Q)d. The second-order term is

    (delta^2/2) * [q2/s^2 + q1*l2/s^3].

Small s amplifies both clock displacement and curvature error. A target outside the bounded ray search, or across a turning point, need not be reachable on the selected branch. The study records failures instead of inventing an endpoint. Its geometric search identifies the first sampled crossing under a parameter-displacement cap; it does not establish global first-root uniqueness or global infeasibility.

For multiple reoptimized steps, the exact continuous loss-clock relation is dQ/d(delta)=grad Q(theta(delta))^T d(theta(delta))/[-grad L(theta(delta))^T d(theta(delta))]. Keeping the initial ratio fixed assumes this ratio remains stable; the experiment measures its failure as the path evolves. The discrete version also includes finite-step curvature. At eight steps the old-seed pooled competitor decisively outperforms the initial clock prediction on the original-ring PB100 states, even when the directional sign often remains correct.

## Outcome derivatives

Let h_theta(x) be the first representation coordinate, p(x)=P(Y=1|X=x), phi(h)=log(1+exp(h)), and v(x)=Dh_theta(x)[d]. For an actual task-invariant averaging operator P, define

    Fit(theta) = E[phi(P h)-p P h] - BayesRisk,
    Nuisance(theta) = E[phi(h)-p h] - E[phi(P h)-p P h].

Their directional derivatives are

    DFit[d] = E[(sigmoid(P h)-p) P v],
    DNuisance[d] = E[(sigmoid(h)-p)v] - DFit[d].

For radial labels and original rings, P is uniform rotation at fixed radius. Then these are precisely radial fitting regret and angular nuisance. For angular labels proportional to cos(2phi), P=(I+R_pi)/2 is a valid subgroup projection, so the names are pi-subgroup fitting regret and pi-antisymmetric nuisance. The fitting term retains informative angle structure and potentially remaining reflection nuisance. Full rotational averaging is invalid for that task.

If G is an independent uniform rotation for radial tasks or the fixed pi partner for the angular task, the extra-view value is

    V(theta)=Risk(h)-E_G Risk((h+h o G)/2),
    DV[d]=E[(sigmoid(h)-p)v]
           - E_G[(sigmoid((h+h o G)/2)-p)(v+v o G)/2].

For the angular pi-partner construction V equals the pi nuisance identically. It is a useful identity check but not an independent third endpoint. The radial value integrates independent two-view pairs over the full angular quadrature, rather than selecting an angle after seeing its test risk.

Fit derivatives here use the known posterior. Nuisance and two-view Jensen differences cancel the invariant posterior contribution. This experiment evaluates population-informed mechanism predictions; it does not show that the fit derivative can be estimated accurately from a limited training sample.

## Exact matched-task difficulty

For the synthetic pair, u is uniform on [0,1], phi uniform on [0,2pi], and radius=.3+1.2u. Both tasks use exactly the same inputs and initial parameters. Their posterior logits are 3cos(2piu) and 3cos(2phi). The phases 2piu and 2phi modulo2pi have the same uniform distribution. Therefore the full posterior-value distribution, balance, Bayes logistic loss and Bayes accuracy coincide. Numerical integration gives class probability0.5, Bayes loss0.3895508093, and Bayes accuracy0.8361132405. The distinction is where label information resides, not difficulty or input distribution.
