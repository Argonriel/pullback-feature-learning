# A transform-specific, density-aware certificate

This note specializes orbit averaging and standard concentration to one fixed-model, fixed-quota deployment rule. The sharp posterior optimization below is derived here for this project; no claim of literature-wide novelty is made. Loss is cross-entropy in nats. A loss certificate is not a classification-error certificate.

## 1. Conditions that must be supplied

Condition on all historical model parameters, score tables and a positive clipping threshold c. Let T be a measurable involution on a transform-closed domain. Deployment batches consist of N IID inputs from P, with independent auxiliary tie keys. There are exactly q selected records. For every coordinate i, its mask M_i, and in this experiment the whole selected mask, must be unchanged when x_i alone becomes Tx_i, holding every other input and every key fixed. A fixed pointwise T-even score followed by top-q selection has this property. Whole-batch invariance by itself is insufficient. The predictor and scores cannot be fitted on verification samples.

The true posterior p(x) is specified on support(P) union support(T#P). Assume TV(p(x),p(Tx)) <= epsilon everywhere on that union, with 0 <= epsilon <= 1. It is substantive label information, not inferred from model confidence or unlabeled input symmetry. We also assume a deterministic, support-wide logit span bound D derived from frozen parameters, never an empirical sample maximum.

The structural input assumption is knowledge of the orbit density imbalance r. Put mu=(P+T#P)/2 and define r=d(P-T#P)/d(P+T#P), so dP/dmu=1+r. Then |r|<=1 and r(Tx)=-r(x), mu-almost everywhere. This Radon-Nikodym formulation also covers non-Euclidean supports; no ambient Lebesgue density is necessary. Knowing r is an assumption. In our example it follows from an explicitly specified parametric law; it cannot be estimated without restrictions over all natural input distributions.

## 2. Exact identity, with other batch inputs left under P

For class logits z=z(x), zT=z(Tx), set m=(z+zT)/2, d=(z-zT)/2, F(z)=log sum exp(z),

    J=(F(z)+F(zT))/2-F(m),     K=(F(z)-F(zT))/2,
    S=max_k d_k-min_k d_k,     b=(max_k d_k+min_k d_k)/2,
    t=K-b.

We have J>=0, S<=D and |t|<=S/2, because K lies between min d and max d. All quantities are invariant to arbitrary separate common offsets added to the logits at x and Tx, except the offset terms K and b, whose difference t is invariant. J and S are even; K,b,t are odd under T.

For a fixed coordinate, condition on the other records and keys, define f=M[ell(z,p)-ell(m,p)], and let fT denote its value after that coordinate transforms. Since M is invariant,

    (f+fT)/2 = M[J-(p-pT) dot d/2],
    (f-fT)/2 = M[K-(p+pT) dot d/2].

The density identity yields

    E_P f = E_mu[(f+fT)/2+r(f-fT)/2]
          = E_P[(f+fT)/2+r(f-fT)/2].

The second equality holds because the bracket is even in this coordinate. The other coordinates retain their original P law throughout, so there is no product of batch importance weights and no change to the deployment distribution. Averaging coordinates is legitimate even though the top-q masks within a batch are dependent.

## 3. Sharp posterior optimization

For fixed r,d and TV(p,pT)<=epsilon, the posterior-dependent objective to upper-bound is

    Q = [(1+r)p-(1-r)pT] dot d / 2.

Let delta=TV(p,pT), and write p=v+u, pT=v+w, with common component v of mass 1-delta and residuals u,w of mass delta. For r>=0,

    Q = r v dot d + (1+r)u dot d/2 - (1-r)w dot d/2
      <= r(1-delta) max d + (1+r)delta max d/2
          - (1-r)delta min d/2
       = r b + [r+(1-r)delta] S/2
      <= r b + [r+(1-r)epsilon] S/2.

For r<0, interchange p,pT and replace d by -d. Therefore, for all r,

    sup Q = r b + [|r|+(1-|r|)epsilon] S/2.

Equality is attainable: when r>=0, take p concentrated on a maximum-d class and pT=(1-epsilon)p+epsilon times a minimum-d class. The negative-r case follows by the interchange. S=0 and |r|=1 are included. Thus the coefficient is sharp given only the posterior-TV constraint; replacing it by |r|+epsilon loses the shared simplex constraint.

The exact identity gives the population theorem

    G_P >= E_P M[ min(J,c) + r t
                    - (|r|+(1-|r|)epsilon) S/2 ].                 (1)

Here and below E_P includes the IID batch and keys and averages its coordinates. This is a lower bound on expected raw-minus-averaged loss of this fixed deployment procedure. The right side is label-free conditional on the stated posterior constraint and supplied r. It is not an unconditional label-free guarantee. Clipping J downward only weakens it. There is no truncation of W or S and no unaccounted tail remainder.

## 4. Known parametric tilt, and an estimated parameter

Let r1,r2 be independent uniform on [.4,1.4] and phi have density (1+a cos phi)/(2 pi), independent of the radii, with a in [0,1]. Input coordinates are (r1 cos phi,r1 sin phi,r2 cos 2phi,r2 sin 2phi). T adds pi. Then r=a u for u=cos phi, and TV(P_a,T#P_a)=2a/pi. Define nonnegative even weights

    W=|u| S/2-u t,           B=S/2,
    R_a=(1-a|u|) S/2.

Their support-wide ranges are W in [0,D], B,R_a in [0,D/2]. Equation (1) becomes

    G_P >= A-a E_P[M W]-epsilon E_P[M R_a],
    A=E_P[M min(J,c)].                                         (2)

This replaces the old global D*TV(P,T#P) penalty with a selected, transform-specific moment. Although these even moments happen to have the same populations across this tilt family, this experiment samples and measures every moment from its actual P_a. No a=0 verification sample is recycled to a shifted law.

If a is unknown but the exact family and a in [0,1] are assumed, E_P cos phi=a/2. With m independent coordinate-only observations, let ahat=2 mean cos phi and rho=sqrt(8 log(2/delta)/m). A two-sided Hoeffding interval is aL=clip(ahat-rho,0,1), aU=clip(ahat+rho,0,1). It covers a with probability at least 1-delta. Conditional on this independent sample, R_aL is a fixed nonnegative even function. On the coverage event, aW<=aU W and R_a<=R_aL pointwise. Hence

    G_P >= A-aU E_P[M W]-epsilon E_P[M R_aL].                   (3)

This deliberately conservative separate-endpoint propagation remains valid even when the combined linear coefficient in a has either sign. It avoids silently plugging an uncertain ahat into the density formula. The known-a and estimated-a results use the same independently generated verification P_a, with the extra parameter-estimation inputs and zero model evaluations charged separately. Independence is useful for conditional concentration; parameter and verification samples are distinct.

## 5. Finite-sample bound and simultaneous accounting

For each nonnegative item weight H in [0,L], the batch statistic Y=N^-1 sum M_i H_i lies in [0,qL/N]=[0,L/4]. Its n independent batch values have unbiased sample variance s^2 (denominator n-1). Maurer-Pontil's one-sided empirical Bernstein radius is

    rad_EB=sqrt(2 s^2 log(2/delta)/n)
              +7(L/4) log(2/delta)/(3(n-1)).

For a lower bound apply the inequality to -Y, equivalently L/4-Y. For an upper bound use it directly. Separately, replacing one independent input/key record changes a top-q selected sum by at most L: the other scores retain their order, so at most one selected item is exchanged, or one selected item's weight changes. Thus McDiarmid gives radius

    rad_McD=L sqrt(log(1/delta)/(2 n N)).

Use the tighter bound after paying for both events, intersected with [0,L/4]. Take a lower bound L_A, upper bounds U_W,U_B,U_Ra,U_RaL, with respective item ranges c,D,D/2,D/2,D/2. There are 8 models x 4 laws x 2 policies x 5 moments x 2 methods x 2 fixed looks = 1280 one-sided events. Four independent two-sided parameter intervals each consume one delta event because their formula already divides delta between tails. Set delta=.05/1284 for all. Conditional on historical fitting artifacts, with probability at least .95, all reported known and estimated certificates hold simultaneously:

    known:     G_P >= L_A-a U_W-epsilon U_Ra;
    estimated: G_P >= L_A-aU U_W-epsilon U_RaL.

Conditioning on any parameter sample leaves all 1280 concentration failures bounded by their delta allocation, including random R_aL; integrating this conditional statement and adding the four parameter failures justifies 1284 events. Shared policies, nested looks, and reuse of samples for five moments need no independence among events. The epsilon grid is display only: once the moments hold, the formulas are uniform over epsilon in [0,1]. The old formula, recomputed with the same conservative event allocation, is L_A-epsilon U_B-D(2a/pi), and adds no event because it uses existing moments.

The bound D=min(1.4+C,1.4L) comes from the frozen compact residual model: |h-x0|<=C and Lipschitz constant L imply |h(x)-h(Tx)|/2<=min(1.4+C,1.4L). No observed maximum replaces it. These are mathematically justified real-arithmetic parameter bounds, with ordinary floating-point numerical evaluation; this is not a machine-checked interval proof. Very large image-network interval constants do not supply either r or posterior epsilon on natural images.

## 6. Why estimating one harmonic does not validate a family

Consider density (1+b cos 3phi)/(2 pi), b=.6. It has E cos phi=0, so the first-harmonic parameter estimator concentrates at a=0 even though r=b cos 3phi. Take h(x)=k cos 3phi, k=.2, h(Tx)=-h(x), constant label posterior p=pT=1, and a fixed top-quarter quota selected by input-independent tie keys. Then epsilon=0 and

    true gain = (1/4)[ E_uniform log cosh(k cos 3phi/2)-bk/4 ] < 0,

because log cosh v<=v^2/2 gives E log cosh(k cos 3phi/2)<=k^2/16=.0025, while bk/4=.03. The bound one would incorrectly compute under a=0 is positive, (1/4) E log cosh(k cos 3phi/2). A narrow CI for a therefore cannot establish the exact harmonic family. This is a deterministic counterexample, not a use of the theorem outside its assumptions. Its quadrature is merely a numerical illustration of the analytic sign proof.

More generally, a discriminator supplies a lower-bound witness for TV, not the upper bound needed by the older theorem. Empirical reflected point clouds or a fitted density score are not a valid replacement for r without explicit uniform estimation control. The preceding study's unrestricted TV obstruction remains unchanged. Posterior ambiguity also remains: even known P does not reveal labels from unlabeled data. The extremal posterior world in this study is intentionally model-dependent and must not be described as one common task for every model.

## 7. Prior work and the project specialization

- Chen, Dobriban and Lee, JMLR 2020, [A Group-Theoretic Framework for Data Augmentation](https://jmlr.org/papers/volume21/20-163/20-163.pdf), develops group averaging and approximate-invariance analysis, especially Section 6. That framework precedes this project. Our result concerns fixed-model logit averaging with a coordinate-invariant top-quota mask and an explicit orbit imbalance, rather than a general new augmentation theory.
- Maurer and Pontil, COLT 2009, [Empirical Bernstein Bounds and Sample Variance Penalization](https://arxiv.org/pdf/0907.3740), Theorem 4, supplies the empirical Bernstein constants; Theorem 1 also states the Hoeffding inequality used for the one-parameter CI. We verified those formulas against the paper. Their concentration results are not novel here.
- McDiarmid, 1989, [On the method of bounded differences](https://doi.org/10.1017/CBO9781107359949.008), provides the bounded-differences method. The top-q replacement sensitivity and its use for these particular even moments are established above.
- Ben-David, Lu, Luu and Pal, AISTATS 2010, [Impossibility Theorems for Domain Adaptation](https://proceedings.mlr.press/v9/david10a.html), studies insufficient assumptions and indistinguishable adaptation tasks. This is context for the limitation, not attribution of our harmonic counterexample or a universal impossibility of structured density estimation.

The derivation (1), the sharp constrained posterior coefficient, the same-P top-quota moment specialization, and the estimated-a propagation are the concrete additions in this research round. Numerical simulation checks their consequences; it cannot prove the theorem or establish the assumptions for natural data.
