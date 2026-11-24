# Density envelopes and labeled evidence for a conditional orbit certificate

All losses are cross-entropies in nats. This is a specialization of orbit averaging, finite-dimensional moment estimation and bounded-variable concentration, not a claim to invent density confidence bands, augmentation theory or label identifiability. It extends the preceding fixed first-harmonic experiment by explicitly retaining a third harmonic, allowing a supplied residual envelope, propagating coefficient uncertainty, and optionally replacing an assumed posterior-TV value by independent labels under a stated finite-cell posterior model.

## 1. Fixed deployment conditions and the exact orbit identity

Condition on a fixed predictor, fixed pointwise even selection scores, fixed clipping c>0 and all historical fitting. Let T be a measurable involution. Deployment and verification batches contain N=256 IID inputs from P with independent tie keys, and select exactly q=64. When one coordinate x_i is replaced by Tx_i with every other record/key fixed, its selected indicator M_i must stay unchanged. Our fixed-score top-quota policies have the stronger property that the entire mask stays unchanged. Exact quota or whole-batch symmetry alone is insufficient.

Let mu=(P+T#P)/2 and r=d(P-T#P)/d(P+T#P). Then dP/dmu=1+r, |r|<=1 and r(Tx)=-r(x), almost everywhere. Define logits z,zT, m=(z+zT)/2, d=(z-zT)/2 and F=log-sum-exp, with

    J=(F(z)+F(zT))/2-F(m),     K=(F(z)-F(zT))/2,
    S=max d-min d,            b=(max d+min d)/2,     t=K-b.

J>=0 and |t|<=S/2 because K lies between min d and max d. J,S are even; t is odd. These quantities ignore arbitrary separate common offsets on z and zT. Assume the true posterior p is defined on the transform-closed union of supports, with TV(p(x),p(Tx))<=epsilon, 0<=epsilon<=1, and a deterministic support-wide S<=D. In numerical work D comes from frozen network parameters, never a sample maximum.

Conditional on the other batch records and keys, write f=M[ell(z,p)-ell(m,p)] and fT for its single-coordinate transformed value. Direct algebra gives

    (f+fT)/2=M[J-(p-pT) dot d/2],
    (f-fT)/2=M[K-(p+pT) dot d/2].

The even quantity C=(f+fT)/2+r(f-fT)/2 obeys E_P f=E_mu C=E_P C. Every other batch coordinate stays under P throughout. Thus we neither change deployment law nor apply a product of batch importance weights. Averaging over coordinates remains valid with dependent top-quota masks.

The sharp posterior objective is Q=[(1+r)p-(1-r)pT] dot d/2. Decompose p=v+u and pT=v+w into common mass 1-delta and disjoint residual masses delta=TV(p,pT). For r>=0,

    Q <= r(1-delta)max d +(1+r)delta max d/2 -(1-r)delta min d/2
       = r b + [r+(1-r)delta] S/2
      <= r b + [r+(1-r)epsilon] S/2.

Equality is attained by p concentrated on a maximum-d class and pT mixing masses 1-epsilon and epsilon on maximum/minimum classes. For r<0 interchange p,pT and replace d by -d. Consequently

    G_P >= E_P M[min(J,c)-h_epsilon(r)],
    h_epsilon(r)=[|r|+(1-|r|)epsilon]S/2-r t.                 (1)

The clipped Jensen term is a downward replacement; no adverse tail is omitted.

## 2. A simultaneous interval for r and the exact joint endpoint penalty

Suppose independent evidence gives a pointwise interval I(x)=[l(x),u(x)] containing the true r(x), simultaneously over the support, and satisfying I(Tx)=[-u(x),-l(x)]. Since

    h_epsilon(r)=epsilon S/2+(1-epsilon)|r|S/2-r t

is convex in r for epsilon in [0,1], its maximum on an interval occurs at an endpoint. Define

    H_epsilon(I)=max(h_epsilon(l),h_epsilon(u)).

This is the exact joint endpoint worst case. It is not the sum of independently worst input and posterior penalties, which can use different endpoints and be more conservative. It is even under T because S is even, t is odd, and the interval reverses. It is nonnegative, since |t|<=S/2. Equation (1) implies

    G_P >= A-E_P[M H_epsilon(I)],   A=E_P[M min(J,c)].        (2)

If |l|,|u|<=rbar<=1 everywhere, then

    0 <= H_epsilon(I) <= D[epsilon/2+(1-epsilon/2)rbar].     (3)

Indeed h_epsilon(r)<=epsilon S/2+(1-epsilon/2)|r|S. This conditional support-wide range is essential: uncertainty about r does not force the concentration range to be D if independent evidence bounds all possible |r| more tightly. We do not select a range using the verification maximum.

## 3. Two retained odd harmonics plus a supplied residual envelope

Let mu have independent uniform radii r1,r2 in [.4,1.4] and uniform angle phi in [0,2pi). Coordinates are (r1 cos phi,r1 sin phi,r2 cos2phi,r2 sin2phi), and T adds pi. Assume P has density 1+r relative to mu, with

    r(phi)=theta1 cos phi+theta3 cos3phi+e(phi),
    e(phi+pi)=-e(phi),  |e(phi)|<=rho,
    E_mu[e cos phi]=E_mu[e cos3phi]=0,  |r(phi)|<=1.          (4)

The reference mu, angular form and residual assumptions are structural. Orthogonality makes theta_j=2E_P cos(j phi), j=1,3; the invariant constant term vanishes against odd cosines and their squared uniform means are 1/2. A supplied residual need not be estimated. In this experiment it is zero or a fifth harmonic with a known amplitude envelope. More generally one can define theta as the projection coefficients and explicitly assume the remaining projection residual has sup norm at most rho.

From m independent inputs, set theta_hat_j=2 mean cos(j phi). Since 2cos(j phi) lies in [-2,2], Hoeffding gives a two-sided interval theta_hat_j +/- q_j, q_j=sqrt(8 log(2/delta)/m), with failure at most delta per coefficient. On both coefficient events,

    b_hat=theta_hat1 cos phi+theta_hat3 cos3phi,
    w=q1|cos phi|+q3|cos3phi|+rho,
    l=clip(b_hat-w,-1,1),  u=clip(b_hat+w,-1,1)

contain the true r simultaneously. Clipping each endpoint makes a nonempty interval for every sample, even on failure events, and preserves I(Tx)=[-u,-l]. Use the independent-data constant

    rbar=min(1,|theta_hat1|+q1+|theta_hat3|+q3+rho).

For the known-coefficient comparison, use true theta and q1=q3=0 while retaining rho. Known coefficients do not make an unknown residual vanish. All A and H moments are still measured on the actual P of each law, not on mu or a reused symmetric-law stream. An upper bound for the older TV comparator is eta<=min(1,2(|theta1|+|theta3|)/pi+rho); an empirical discriminator is not involved.

The coefficient estimator remains correctly centered when an omitted fifth harmonic is large, but the r interval is wrong if its supplied rho is too small. That law is explicitly outside theorem coverage. More data about the first two coefficients cannot fix the missing residual bound.

## 4. What independent labels can prove under a fixed posterior model

The optional binary label audit assumes p(x) is exactly constant within each of eight frozen cells: two bins for each radius (cut at .9), crossed with the two angular half-circles. A canonical half-circle assigns the vertical axis using the sign of the second coordinate so T exchanges the two cells of every radial pair. There are four orbit pairs. This within-cell constancy is the posterior regularity assumption; neither the input family nor the label sample establishes it for arbitrary p.

Draw independent IID (X,Y) under the actual P, Y in {0,1}. Conditional on cell counts, the labels in cell k are independent Bernoulli(p_k). For n_k>0, use p_hat_k +/- sqrt(log(2/delta)/(2n_k)), clipped to [0,1]. For n_k=0, use [0,1], with no claim based on a missing cell. These eight two-sided intervals each have unconditional failure at most delta by conditioning on counts. We never plug in an expected cell count as the observed sample size.

If all intervals [L_k,U_k] cover, then the binary pointwise posterior-TV bound is

    epsilon_U=min(1,max over four pairs {U_plus-L_minus,
                                       U_minus-L_plus,0}).  (5)

Thus (2) with epsilon_U has independent labeled support, conditional on within-cell constancy. H is monotone in epsilon because h is, so any larger epsilon_U remains conservative. Independently sampled labels and coefficients make H, its epsilon and its deterministic conditional range fixed before verification.

Every cell has reference mass 1/8. Our label-audit laws satisfy |r|<=.10 (covered_third) or .075 (covered_residual), giving actual cell probability at least .1125 or .115625. These lower bounds explain why zero counts are unlikely but are not needed for coverage; the [0,1] rule handles them rigorously. New sampled Bernoulli labels, not old posterior values presented as evidence, supply the audit. Truth probabilities are retained separately as generator information.

Finite-cell labels alone only bound cell averages when within-cell constancy is absent. They cannot certify the pointwise posterior-TV assumption for arbitrary labels. For example p=(1+sin3phi)/2 has opposite partners with pointwise defect reaching one, but each half-circle's mean is 1/2 under every cosine-only input law here: the sine term is odd about either half-circle's center, whereas its density is even. Independent radii do not change this fact. An audit of the eight cell averages can therefore appear invariant while pointwise symmetry fails. More angular bins do not remove this issue without a justified regularity bound.

## 5. Finite-sample and simultaneous confidence

For an even nonnegative item weight V in [0,L], Y=N^-1 sum_i M_i V_i lies in [0,L/4]. Across n independent batches use the sample variance s^2 with denominator n-1. The one-sided empirical Bernstein radius is

    sqrt(2 s^2 log(2/delta)/n)+7(L/4)log(2/delta)/(3(n-1)).

For a lower bound apply it to L/4-Y. Separately, replacing one independent input/key record in fixed-score top-q selection changes the selected sum by at most L: all other scores keep their ordering, so only one selected item changes. McDiarmid therefore gives L sqrt(log(1/delta)/(2nN)). Pay separately for both concentration events, then intersect them and [0,L/4]. Apply a lower bound to A with range c and an upper bound to H with range (3). The older comparator uses B=M S/2 with range D/2.

There are 8 models x7 laws x2 policies x2 looks=224 base groups. Each has A, B and 16 H moments (8 epsilon-grid values x2 coefficient modes), for 224x18x2 methods=8064 events. Four independent label-audit contexts add 4x8 models x2 policies x2 looks x2 coefficient modes x2 methods=512 events. Fourteen coefficient CIs and 32 cell CIs bring the total to 8622. Set delta=.05/8622. This counts even the outside-envelope sensitivity rows conservatively, but the risk theorem applies only to rows satisfying its input and posterior assumptions.

Conditional on the independent coefficient and label samples, every verification statistic and its support-wide range is fixed. Thus its failure probability is at most delta for every conditioning value, even if a coefficient/label CI fails. Integrate and union bound the 8576 moment events, 14 coefficient events and 32 cell events. With probability at least .95, every covered, prespecified grid or labeled certificate obeys

    G_P >= L_A-U_H.

Shared policies, moments, nested looks and common label-audit intervals across models do not require independence among these events. Unlike the previous separated linear-in-epsilon bound, this exact endpoint implementation has epsilon-dependent statistics and pays explicitly for each reported grid value and each independent labeled epsilon. We do not assert unsampled-epsilon uniformity from this finite grid. The old comparator reuses L_A and U_B and needs no new event.

## 6. Explicit covered and uncovered examples

The earlier density (1+.6cos3phi)/(2pi), h=.2cos3phi, invariant p=1 and input-independent quarter quota has gain (1/4)[E_uniform log cosh(.1cos3phi)-.03]<-.006875. The old first-harmonic projection pretends r=0 and is falsely positive. Including cos3 captures this r exactly; its correct lower expression coincides with the negative gain in this example. This is protection against a false positive, not evidence that the method should certify harmful averaging.

For residual density (1+.6cos5phi)/(2pi) and h=.2cos5phi, the same argument holds. Taking rho=.6 covers it; falsely supplying rho=0 omits it. The first and third coefficient means both vanish. The optional stress row with rho=0 is labeled invalid and excluded from all success counts. No unrestricted unlabeled estimate of a sup-norm residual envelope is claimed. The preceding TV non-estimability construction still applies without bounded complexity or regularity.

## 7. Literature and contribution scope

- [Chen, Dobriban and Lee, JMLR 2020](https://jmlr.org/papers/volume21/20-163/20-163.pdf) develops group averaging and approximate-invariance analysis. Our fixed-model mask and density-envelope specialization is narrower.
- [Maurer and Pontil, COLT 2009](https://arxiv.org/pdf/0907.3740), Theorems 1 and 4, supply Hoeffding and empirical Bernstein formulas. The top-quota replacement argument supplies the bounded-differences specialization, using [McDiarmid, 1989](https://doi.org/10.1017/CBO9781107359949.008).
- [Gine and Nickl, Annals of Statistics 2010](https://arxiv.org/pdf/1002.4801) constructs honest/adaptive density bands on specified regularity classes and explains why unrestricted adaptation is not automatic. Our elementary two-coefficient interval with a supplied remainder is not their adaptive procedure or a substitute for its assumptions.
- [Kerkyacharian, Nickl and Picard, PTRF 2012](https://arxiv.org/abs/1102.2450) develops nonasymptotic density bands on compact homogeneous manifolds using projection estimators under smoothness assumptions. It is prior evidence that structured density inference is possible, not justification for assuming our remainder bound on natural data.
- [Ben-David et al., AISTATS 2010](https://proceedings.mlr.press/v9/david10a.html) gives indistinguishable adaptation tasks under insufficient assumptions. Our finite-cell caution and explicit harmonic examples are derived here, not attributed as their theorems.

The contribution of this round is the proved interval propagation and conditional range, an explicitly counted finite-sample experiment including a labeled finite-cell extension, and the retained covered/uncovered failures. It is not a universal natural-data safety theorem or formal directed-rounding verification.
