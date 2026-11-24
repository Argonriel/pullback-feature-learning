# Multiclass gain under input-law and posterior defects

This is a specialized extension of the previous elementary symmetrization argument, not a first general theorem about approximate invariance. All losses below are cross-entropies in nats. Classification error is a separate empirical outcome.

## Assumptions and exact identity

Condition on the already fitted predictor, selector, calibration tables, and clipping threshold. Let T be a measurable involution. Inputs X_i are independent with the same law P; auxiliary tie keys are independent across records and from the inputs. A batch has N=256 records and a fixed quota q=64. The selected mask M_i must be invariant when any one input is replaced by T X_i, with all other records and keys held fixed. Frozen pointwise T-even scores with top-q selection have this property. Whole-batch symmetry or exact quota alone does not establish it. The deterministic transform and model do not depend on verification samples.

For K logits z=z(x), zT=z(Tx), write m=(z+zT)/2, d=(z-zT)/2, and F(z)=log sum_k exp(z_k). Let p(x) be a specified version of the class posterior on support(P) union support(T#P). Define J=(F(z)+F(zT))/2-F(m)>=0. If f_i=M_i[ell(z,p)-ell(m,p)], then

    (f_i + f_i composed with T_i)/2 = M_i[J - (p-pT) dot d / 2].

This is pointwise algebra with all other batch records fixed. In particular, p need not be inferred from model predictions. If P=T#P and p=pT, the expected deployment gain equals E[M J], despite dependence among the mask entries within a batch.

## Shift-invariant range and the two defects

Use posterior TV = ||p-pT||_1/2 <= epsilon and input TV eta = TV(P,T#P) = sup_A |P(A)-T#P(A)|. The posterior vector difference sums to zero, so

    |(p-pT) dot d| <= epsilon span(d),  span(d)=max_k d_k-min_k d_k.

This range is invariant under any common scalar added to all logits at x and another common scalar at Tx. Absolute logit norms are not needed. Since both F(m+d)-F(m) and p dot d lie between min d and max d,

    |ell(z,p)-ell(m,p)| <= span(d) <= D.

D must be a deterministic bound on the entire union of the two supports, not a maximum observed on test samples. Conditional on the other records and all keys, f_i lies in [-D,D]. Its expectation under P and T#P differs by at most 2D eta. The half-symmetrization factor therefore costs D eta. Averaging the coordinates introduces neither a batch-size factor nor a quota discount. With any fixed c>0,

    G_P >= A_P - epsilon B_P/2 - D eta,
    A_P = E_P[M min(J,c)],  B_P = E_P[M span(d)].

All expectations include the IID batch and keys. The A and B moments MUST be measured on the same P as the deployment risk. The displayed bound also holds when epsilon and eta are conservative upper bounds. The final confidence statement is uniform over their admissible values once moment bounds hold; the grid used for display needs no extra multiplicity payment.

For binary logits (0,h), span(d)=|(h-hT)/2|, recovering the previous binary statement exactly. For this compact residual model, |h-x_0|<=C and h has Lipschitz constant L. Hence D=min(1.4+C,1.4L). For a positively rescaled predictor beta*h, the correct bound is beta times this entire expression, not 1.4+beta*C.

A useful independent Jensen bound is J<=log cosh(D/2). Center d into [-D/2,D/2], take s=softmax(m), and write J=.5 log[(E_s exp(d))(E_s exp(-d))]. The product is at most (E_s cosh(d))^2, which is at most cosh(D/2)^2. This bound and the loss-difference bound are tight for suitable two-class examples. They are numerical checks here, not a post-outcome replacement for the frozen calibration-only clipping rule.

If every CNN logit satisfies deterministic bounds z_k in [l_k,u_k] on a transform-closed input domain, let w_k=u_k-l_k. Then d_k is in [-w_k/2,w_k/2], and D equal to half the sum of the two largest distinct-class widths bounds span(d). A positive temperature multiplier scales D. This can be valid yet enormous. It supplies neither a natural-image input-TV bound nor a posterior-TV bound; no natural-data certificate follows from interval bounds alone.

## Finite-sample confidence, without independent-mask fiction

For n independent batches, A_b is in [0,c/4] and B_b in [0,D/4]. The empirical Bernstein one-sided radius for a batch statistic of range R is

    sqrt(2 s^2 log(2/delta)/n) + 7 R log(2/delta)/(3(n-1)).

Use sample variance with denominator n-1. Apply the same bound to the negated variable for an upper bound. Separately, replacing one independent input/key record can change a fixed-score top-q selected sum of weights in [0,c] by at most c: other records retain their ordering, so at most one selected item is exchanged. The McDiarmid radius is c sqrt(log(1/delta)/(2 n N)); replace c by D for B. This argument does not require independent selected inputs and remains valid under asymmetric P.

Take L_A=max(0,L_EB,L_McD) and U_B=min(D/4,U_EB,U_McD). Our 8 fixed models, 4 input laws, 2 policies, 2 moments, 2 concentration methods, and 2 fixed sample-size looks give 512 events, each at delta=.05/512. Conditional on all historical fitting artifacts, with probability at least .95 all reported lines

    G_P >= L_A - epsilon U_B/2 - D eta

hold simultaneously. Verification shared between policies and nested looks does not invalidate the union bound. Each moment's independent unit for the empirical Bernstein calculation is a whole batch. McDiarmid uses the independent underlying input/key records with its proved replacement sensitivity.

## Controlled asymmetric input law and diagnostic correction

Keep r1,r2 independent uniform on [.4,1.4], and x=(r1 cos(phi),r1 sin(phi),r2 cos(2phi),r2 sin(2phi)). The density of phi is (1+a cos(phi))/(2pi), and T adds pi. Thus eta=(1/2) E_uniform|2a cos(phi)|=2|a|/pi. Rejection sampling uses this exact density. No estimated discriminator substitutes for this known defect.

For q_i=a cos(phi_i), define H_i=(f_i+f_iT)/2 and C_i=H_i+q_i(f_i-f_iT)/2. Conditional on the other records and keys, C_i is even. Hence its integral under P_a equals its integral under the uniform orbit law, which equals the integral of f_i under P_a. Other records retain law P_a throughout this argument. This establishes E C_i=G_P without a product of joint-batch importance weights. The empirical C estimator is an explicitly model/posterior-dependent known-density diagnostic, NOT a replacement label-free certificate. Direct IID gains are also retained.

Because the odd density tilt integrates to zero against any even function, the joint distribution of all even scores and weights is unchanged across a. Therefore the population A and B moments in this particular family remain the same. We nevertheless independently sample each actual P_a as required, and do not reuse uniform-law numerical moments. Actual risk can change because its unpaired component is odd. This family tests sensitivity to a controlled defect, not every kind of distribution shift.

## Why the natural-data assumptions cannot be supplied for free

Posterior ambiguity is exact: the same input law supports different label posteriors, including an invariant posterior and a noninvariant one with different augmentation risk. Unlabeled observations cannot identify that missing label relation. For unrestricted natural continuous or arbitrarily complex input laws, there is also no useful distribution-free finite-sample upper bound on TV(P,T#P) without structural/regularity assumptions or a specified law family. A discriminating set or classifier provides a lower-bound witness for TV, not the upper bound needed by this theorem. Original and reflected empirical point clouds can be disjoint and have empirical TV=1 even when the population law is invariant. This is not an impossibility claim for finite known alphabets, known parametric families, or explicitly bounded-complexity density models.

The guarantee is about fixed-model averaging, not online adaptation with feedback. It does not assert that a semantically label-preserving transform also preserves the input law. Deliberately adding transformed copies changes the evaluation distribution.

## Primary literature and scope

- Chen, Dobriban and Lee, JMLR 2020, [A Group-Theoretic Framework for Data Augmentation](https://jmlr.csail.mit.edu/papers/volume21/20-163/20-163.pdf), especially Section 6, develops approximate-invariance bias analysis. This is prior art, not a claim that input symmetry is automatic.
- Ben-David et al., AISTATS 2010, [Impossibility Theorems for Domain Adaptation](https://proceedings.mlr.press/v9/david10a.html), proves indistinguishable adaptation tasks with differing outcomes under insufficient assumptions. It supports the need for assumptions, not a direct theorem about our specific selector.
- Jhawar and Wang, September 2026 preprint, [When Is Test-Time Adaptation Identifiable From Unlabeled Evidence?](https://arxiv.org/html/2609.11235v1), distinguishes unavailable evidence from an imperfect selector. It is a recent preprint, and its adaptive setting differs from this fixed-model experiment.
- Maurer and Pontil 2009, [Empirical Bernstein Bounds and Sample Variance Penalization](https://arxiv.org/pdf/0907.3740), Theorem 4, supplies the bounded independent-variable concentration constants used here.
- McDiarmid 1989, [On the method of bounded differences](https://doi.org/10.1017/CBO9781107359949.008), supplies concentration after the top-quota replacement sensitivity is established.

The TV non-estimability caution above is a qualified structural limitation of unrestricted finite-sample inference; it is not attributed as a literal theorem of the recent TTA preprint.
