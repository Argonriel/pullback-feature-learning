# Independent finite-sample, all-posterior certificate

The original continuous field and its one-of-four selector have marginal q(x)=1/4+u(x)/8, with q in [1/8,3/8]. Let a=h(x), a'=h(-x), m=(a+a')/2, d=(a-a')/2, and A=softplus(a)-softplus(m). Let b(r)=E[q(X)d(X)|R=r], and let the independently derived local deterministic bound be |b(r)|<=B(r). The field is zero outside radius 3, so b=0 there.

Center the unknown radial posterior at 1/2. For every measurable p(r) in [0,1], the gain is

    Delta(p) = E[q(A-d/2)] - E[(p(R)-1/2)b(R)].

Consequently Delta(p)>=E[q(A-d/2)]-E[B(R)]/2. This uses no prevalence assumption. It improves the conservative C-E|b| bound and does not estimate the hidden labels.

For each independent input x form the antithetic observation

    W(x) = {q(x)[A(x)-d(x)/2] + q(-x)[A(-x)+d(x)/2]}/2,
    Z(x) = 1_{||x||<=3} [W(x)-B(||x||)/2].

Rotational invariance makes E[W]=E[q(A-d/2)]. Outside radius 3, q=1/4 and W is one quarter of a nonnegative Jensen gap. Therefore Delta(p)>=E[Z] simultaneously for every hidden posterior. The two views of each x are one observation; the sample size counts independent x values, never both views separately. The certificate does not require a known radial density.

Let D=3+C_h, where C_h is the existing sum of residual output weight bounds. Then |d|<=D on the retained disk. The function softplus(t)-t/2 is convex and 1/2-Lipschitz. Its two differences from its value at m each lie in [-D/2,D/2], and their sum is nonnegative. Since both q values lie in [1/8,3/8], W lies in [-D/16,3D/16]. Thus Z lies in [-D/16-B_max/2,3D/16], an interval of length R=D/4+B_max/2. Adding the omitted zero observation does not enlarge this interval.

Use the one-sided empirical Bernstein inequality (Maurer and Pontil, 2009, Theorem 4), applied after shifting and scaling this bounded variable:

    E[Z] >= mean(Z) - sqrt(2*s2*log(2/delta)/n)
                         - 7*R*log(2/delta)/(3*(n-1)).

Here s2 is the unbiased sample variance. Each frozen model gets n=262144 fresh IID unlabeled inputs, seed 941000+model_index. Set delta=0.05/16, so a union bound provides family confidence at least 95% for all 16 model certificates, conditional on the frozen models and local bounds. Model independence is not needed for this union bound. No certificate sample fits a field, local bound configuration or threshold.

Source: https://www.cs.mcgill.ca/~colt2009/papers/012.pdf (Theorem 4). The confidence claim is a real-arithmetic mathematical statement; the ordinary floating-point implementation is separately audited and does not constitute formal directed-rounding numerical certification.

Report the finite-sample lower bound, empirical gain at p=1/2, empirical local penalty, statistical deduction, deterministic range, sample seed and actual verification forwards separately. A positive certificate concerns expected risk under the symmetry assumptions, not every realized prediction or arbitrary distribution shift.
