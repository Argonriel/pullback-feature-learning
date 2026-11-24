# Independent-batch certificates for symmetric allocation

Literature checked on 21 November 2026. The certificate uses frozen predictors and frozen input-only allocation rules. It does not update network parameters during verification.

## Primary literature and what is used

1. [Maurer and Pontil, *Empirical Bernstein Bounds and Sample Variance Penalization* (COLT 2009)](https://arxiv.org/pdf/0907.3740), Theorem 4 and its proof through Theorem 11. For bounded independent observations, the confidence width combines unbiased sample variance with a deterministic range penalty. We use its one-sided form after affine rescaling and reflection. The independent observations below are complete batches, not their individual selected points. The paper also explains finite-class union bounds.
2. [Howard, Ramdas, McAuliffe and Sekhon, *Time-uniform, nonparametric, nonasymptotic confidence sequences* (Annals of Statistics 2021)](https://arxiv.org/abs/1810.08240). Their confidence sequences permit inspection at arbitrary times; ordinary fixed-sample intervals do not. Our implementation instead fixes three batch counts and explicitly allocates error probability across them. It makes no claim of validity at unlisted stopping times.
3. [Waudby-Smith and Ramdas, *Estimating means of bounded random variables by betting* (JRSS B, 2024)](https://doi.org/10.1093/jrsssb/qkad009), [author preprint](https://arxiv.org/abs/2010.09686). Betting methods adapt to variance and can improve bounded-mean intervals and confidence sequences. They are a credible future numerical improvement; they do not remove the need for a correct independent unit, bounded support, or a valid symmetry-to-risk identity. We do not implement an unreviewed betting substitute in this confirmation.
4. [Angelopoulos, Bates, Candes, Jordan and Lei, *Learn then Test* (Annals of Applied Statistics 2025)](https://arxiv.org/abs/2110.01052). Risk control can be formulated as multiple hypothesis testing. This motivates separating development from verification and counting every tested model, contrast, and sample-size look. Its general framework does not itself provide our label-free risk identity.
5. [Chen, Dobriban and Lee, *A Group-Theoretic Framework for Data Augmentation* (JMLR 2020)](https://jmlr.csail.mit.edu/papers/volume21/20-163/20-163.pdf), and [Lyle et al., *On the Benefits of Invariance in Neural Networks* (2020)](https://arxiv.org/abs/2005.00178). Group averaging and convex losses provide the relevant foundation. The batch quota requires an additional mask-invariance argument: a generic confidence gate is not covered just because its partner transformation preserves labels.
6. [Schirmer et al., *Monitoring Risks in Test-Time Adaptation* (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/746960ad49ddb47248970a0e1404230c-Paper-Conference.pdf). Its label-free monitoring relies on an explicit relation between loss-proxy false positives and false negatives (Assumption 1); it is not an unconditional guarantee for every confidence proxy. The paper notes that checking the assumption still needs labeled test data. Our use of an exact symmetry identity is a different assumption and mechanism.
7. [Han et al., *The Premise Is the Problem: Exchangeability Failure in Self-Monitored Test-Time Adaptation* (October 2026 preprint)](https://arxiv.org/abs/2610.07038). The paper studies monitoring failures when feedback and overlapping time-series targets invalidate assumptions. It reinforces the need to state the actual independence and feedback conditions. We use independent batches and a policy frozen before the verification stream; we do not extrapolate the present theorem to online adaptation or dependent time series.

## Symmetry identity with a batch quota

Let T be an involution preserving the joint law of (X,Y). Equivalently, the input law is T-invariant and the conditional label probability p satisfies p(Tx)=p(x). Full rotational invariance is sufficient but unnecessary. Let h be a fixed logit and let a batch contain B=256 independent inputs. All calibration and training are conditioned upon and independent of the verification stream.

The allocation M_i is a measurable function of the entire input batch and independent auxiliary tie/action keys. It selects exactly q=64 inputs. Assume the mask is unchanged under applying T to any input coordinate, with the same keys. An input-only score s(Tx)=s(x) followed by deterministic lexicographic top-q selection satisfies this condition, including ties.

Write a=h(x), b=h(Tx), m=(a+b)/2, and A(t)=log(1+exp(t)). The logistic loss is A(t)-yt. The deployed rule replaces a by m only at selected points. Pairing a batch with its T-transformed batch keeps the mask fixed, and the signed label term cancels. Therefore the expected risk reduction is exactly

    G(M) = E[ B^(-1) sum_i M_i J(X_i) ],
    J(x) = (A(h(x))+A(h(Tx)))/2 - A((h(x)+h(Tx))/2) >= 0.

Thus G(M)>=0 is structural, without a statistical certificate. Verification estimates how much gain can be guaranteed. For two such masks M and F,

    G(M)-G(F) = E[ B^(-1) sum_i (M_i-F_i) J(X_i) ].

The latter can have either sign. Structural safety does not establish superiority over another safe selector. These identities hold for every T-invariant posterior, including angular posteriors invariant only under the half-turn. They are not claims about classification error.

## Valid bounded statistics

For a clipping level c fixed before fresh verification, define

    U_c = B^(-1) sum_i M_i min(J_i,c).

Then 0<=U_c<=qc/B=c/4 and E U_c<=G(M). This gives an unconditional quantitative lower bound even for the unbounded Gaussian-noisy ring input law. Clipping sacrifices estimand size; it does not prove that all of the true gain has been retained.

Clipping both sides of a signed comparison has no useful order relation. We instead use V_R=B^(-1)sum_i(M_i-F_i)J_i 1{r_i<=R}, computing both masks on the full original batch. Truncation never rejects inputs, reranks a smaller batch, or changes the deployed policy. If J<=Jmax inside the region, V_R lies in [-Jmax/4,Jmax/4], with range width Jmax/2. The expectation outside the region is bounded below by minus E[J 1{r>R}].

For d=(a-b)/2 and any midpoint m, direct hyperbolic-function algebra gives

    J = 0.5 log(1+sinh(d/2)^2 / cosh(m/2)^2)
      <= log cosh(|d|/2) <= |d|/2.

The residual network satisfies |h(x)-x_0|<=C and is L-Lipschitz. Therefore |d|<=min(r+C,rL) on rings, <=min(1+C,L) on the one-amplitude harmonic task, and <=min(1.4+C,1.4L) on the two-amplitude task. These are parameter-derived deterministic bounds, never observed sample maxima.

For rings x=r0 u+epsilon, r0 in {.5,1.2}, epsilon~N(0,sigma^2 I), sigma=.15, let S=||epsilon|| and t=R-r0>0. Then r<=r0+S and r>R implies S>t. Since P(S>s)=exp(-s^2/(2sigma^2)), integration of the Rayleigh tail and the bound integral_t^infinity exp(-s^2/(2sigma^2)) ds <= sigma^2/t exp(-t^2/(2sigma^2)) give

    E[J 1{r>R} | r0]
    <= 0.5 [R+C+sigma^2/(R-r0)] exp(-(R-r0)^2/(2sigma^2)).

The mixture average is the deterministic tail allowance. We fix R=3. The masks remain those of the original unbounded law. The compact harmonic tasks need no tail allowance.

## Concentration, multiplicity, and actual computation

Let Z_1,...,Z_n be IID complete-batch statistics of deterministic range width W, with unbiased sample variance s_n^2. The one-sided lower bound is

    mean(Z) - sqrt(2 s_n^2 log(2/delta)/n)
            - 7 W log(2/delta)/(3(n-1)).

For absolute clipped gain, W=c/4. For signed truncated gain, W=Jmax/2, and the deterministic tail allowance is subtracted afterward. The study allocates total error .05 across all fresh models, two absolute gains, one signed comparison, and the three fixed looks n in {128,512,2048}. Shared batches and nested looks do not need to be independent for this union bound; the batches within any given bound do.

Because both masks are input-only, verification computes them before querying the network. It evaluates both h(x) and h(Tx) only on the union of their selected sets. One selector alone requires exactly128 forward examples per batch; two selectors together require at most256. The runner records the actual union size and every network invocation. Coordinate draws, calibration forwards, verification forwards, and deployment forwards remain separate costs.

No rank CDF, interpolation-error certificate, or interval-derivative construction is needed for these structurally even rules. This is a change in the admissible policy and certificate pipeline, not a numerical acceleration of the previous asymmetric-rank certificate. The relative certificate may remain nonpositive because its deterministic range is much larger than its mean advantage. That outcome is retained.
