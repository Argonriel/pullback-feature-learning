# Quantitative tolerance to posterior symmetry error

Let T be an involution, and assume the input law is exactly T-invariant. The model, positive temperature, calibration tables, and pointwise ranking score are frozen before verification. The batch contains 256 IID inputs and independent per-input tie keys. Each symmetric policy selects exactly 64. Its mask is unchanged if T is applied to any input coordinate with the same keys.

Write a=h(x), b=h(Tx), d=(a-b)/2, m=(a+b)/2, and

    J(x) = [softplus(a)+softplus(b)]/2 - softplus(m) >= 0.

For logistic risk with posterior p(x), pair the input with Tx. The selected mean risk gain is exactly

    G_p = E[ M { J - (p(x)-p(Tx)) d/2 } ].

The expectation includes the full batch and keys; M denotes the selected indicator for a uniformly chosen batch coordinate. The minus sign and factor one half follow from averaging the terms -p(x)d and +p(Tx)d. The identity requires input-law symmetry and mask invariance, but does not require posterior symmetry.

If a known bound |p(x)-p(Tx)|<=epsilon holds almost surely, then

    G_p >= A - epsilon B/2,
    A = E[M min(J,c)],  B = E[M |d|].

The fixed clipping level c may depend on independent calibration information. We use c=min(0.5,max(1e-6,95th percentile of the saved grid Jensen field)). This is a calibration-grid percentile, not a claim about a population quantile. The same c is used for both symmetric selectors within a predictor variant.

For this compact two-amplitude task, the unscaled network has |h(x)-x_0|<=C and Lipschitz constant L. The first amplitude is at most 1.4. For positive inverse temperature beta, a valid difference bound is

    |d| <= D = beta min(1.4+C,1.4L).

Both the residual term and the skip coordinate scale by beta. Substituting beta C into 1.4+C without scaling the coordinate would be incorrect. D is computed from the frozen parameters and is not an observed maximum.

## Simultaneous affine lower bound

For each independent batch record A_j=mean_i M_i min(J_i,c) and B_j=mean_i M_i |d_i|. Their batch ranges are [0,c/4] and [0,D/4]. We obtain an empirical-Bernstein lower bound for E A_j and upper bound for E B_j, using unbiased sample variances and the Maurer--Pontil one-sided penalty

    sqrt(2 sample_variance log(2/delta)/n)
    + 7 range_width log(2/delta)/(3(n-1)).

A second valid method uses the pointwise top-quota replacement lemma. Replacing one input-key pair preserves the relative order of other items, changing the selected weight sum by at most its weight range. Thus the corresponding bounded-difference penalties are

    c sqrt(log(1/delta)/(2 n*256)),
    D sqrt(log(1/delta)/(2 n*256)).

This is concentration of a dependent selection functional of independent underlying records. It does not treat selected contributions as IID. For a directly signed weight min(J,c)-epsilon |d|/2, the replacement sensitivity would be its range width c+epsilon D/2, not its largest absolute endpoint. The primary implementation instead bounds the two moments separately, enabling one line valid for every epsilon.

Take L_A=max(0,EB lower,McDiarmid lower), and U_B=min(D/4,EB upper,McDiarmid upper). There are 32 predictor variants times two policies times two moments times two methods times two sample-count looks =512 elementary events, each at delta=.05/512. On their common event, probability at least .95,

    G_p >= L_A - epsilon U_B/2

simultaneously for every epsilon in [0,1] and every posterior obeying its stated defect bound. The plotted epsilon grid incurs no further multiplicity. The positive-gain limit is min(1,2L_A/U_B) when U_B>0; the bound is strictly positive below the uncapped crossing, and equals zero at an exact crossing. A zero crossing is retained as failure to certify positive gain. Fixed-sample looks are 512 and 2048 batches, with 2048 primary; arbitrary additional stopping is not covered.

## Stress worlds and scope

We evaluate p_epsilon(x)=p0(x)+min(epsilon/2,p0(x),1-p0(x))*q(x). Its partner posterior changes the sign of q, and therefore its defect is at most epsilon while probabilities stay in [0,1]. The fixed world uses q=cos(phi). The adversarial world uses q=sign(d), making its correction nonpositive. This world is different for different fitted predictors; it is a model-dependent stress diagnostic, not one common test population across models. It is shared across all selectors for the same predictor.

The input distribution stays exactly symmetric throughout. No guarantee for an approximately invariant input law is asserted. An upper bound on posterior defect needs label or domain evidence: identical unlabeled input laws can support all the different posterior worlds above. The experiment knows epsilon by synthetic construction. It does not estimate or certify epsilon from unlabeled images, confidence, or augmentation agreement. Confidence ranking is an empirical comparator only; its mask is generally not invariant, so the affine guarantee does not cover it.

## Primary literature and relation to this result

[Chen, Dobriban and Lee, JMLR 2020](https://jmlr.csail.mit.edu/papers/volume21/20-163/20-163.pdf), Section 6, Lemma 15 and Theorem 17, explicitly study approximate invariance and additional bias terms controlled by distributional discrepancy. Our narrower calculation holds the input law invariant and isolates posterior asymmetry for a frozen logistic predictor and exact batch quota. It is not a general replacement for their approximate-distribution theory.

[Maurer and Pontil, COLT 2009](https://arxiv.org/pdf/0907.3740), Theorem 4, supplies the empirical-Bernstein inequality. [McDiarmid, 1989](https://doi.org/10.1017/CBO9781107359949.008) supplies independent bounded differences. The pointwise ranking replacement lemma is what makes the latter applicable despite the quota dependence.

[Schirmer et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/746960ad49ddb47248970a0e1404230c-Paper-Conference.pdf) obtain label-free test-time adaptation monitoring under explicit loss-proxy assumptions; these assumptions are not automatically provided by augmentation consistency. [Han et al., October 2026 preprint](https://arxiv.org/abs/2610.07038) show how monitoring assumptions can fail under feedback and dependence. Here the model is frozen and verification streams are independent; neither online adaptation nor arbitrary real-data safety follows.

[Shanmugam et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shanmugam_Test-time_Augmentation_Improves_Efficiency_in_Conformal_Prediction_CVPR_2025_paper.html) separate labeled data for learning augmentation and conformal calibration. Their guarantee concerns prediction-set coverage. Our guarantee concerns expected logistic gain under a supplied posterior-defect assumption, so these guarantees should not be conflated.
