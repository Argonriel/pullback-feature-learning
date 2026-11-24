# Explanatory extension: an input-law asymmetry penalty

Status: post-outcome mathematical analysis, added after the real-image
negative-transfer result. This was not a prespecified empirical claim, was
not used to change any frozen policy, and is not a real-data certificate.
The primary robustness experiment retains its exactly invariant input law.
Root and the robustness reviewer independently checked the argument below.

Let T be an involution and P the input law. Use the convention
eta=TV(P,T#P)=sup_A abs(P(A)-T#P(A)), half the L1 density distance when
densities exist. Suppose a version of the binary posterior p is specified
on support(P) union support(T#P), with abs(p(x)-p(Tx))<=epsilon there.
Also require abs((h(x)-h(Tx))/2)<=D on that union, for a frozen predictor.

Each batch input and its auxiliary tie key is an independent record.
Require the allocation mask to be invariant under transforming any single
input coordinate, with the other records fixed. Whole-batch invariance alone
would be insufficient. The fixed pointwise even scores in this study satisfy
the stronger condition.

Write d=(h-hT)/2, m=(h+hT)/2 and J for the logistic Jensen gap. For coordinate i,
let f_i=M_i[ell(h(x_i),p(x_i))-ell(m(x_i),p(x_i))]. The derivative of expected
logistic loss with respect to its logit lies in[-1,1]. Therefore
abs(f_i)<=abs(d_i)<=D. The algebraic pairing identity is

    (f_i + f_i composed with T_i)/2
        = M_i[J_i - (p(x_i)-p(Tx_i))*d_i/2].

Conditional on all other records, the difference between expectations of f_i
under P and T#P is bounded by its range2D times eta. The half-pairing factor
then contributes D*eta. Averaging coordinates introduces no additional batch
size or quota factor. Consequently, with A=E_P[M min(J,c)] and B=E_P[M abs(d)],

    gain on P >= A - epsilon*B/2 - D*eta.

The independent-record concentration arguments for A and B do not themselves
require an invariant input law. If their bounds L_A,U_B are computed from
verification inputs drawn from this same P, the conditional certificate is

    gain on P >= L_A - epsilon*U_B/2 - D*eta.

This gives a useful explanation of the real-image result, not permission to
reuse synthetic numbers. Existing synthetic moment estimates cannot be
transferred to another input law. The CNN is multiclass, while this displayed
formula is for the binary logistic setting. No CNN global bound D, natural-
image eta, or posterior epsilon is established by this study. A multiclass
analogue would need its own norm/range statement and evidence for those
assumptions. Semantic label preservation alone does not make the input law
invariant; deliberately adding mirrored copies changes the evaluation law.

Approximate-invariance bias penalties are established prior art, including
Chen, Dobriban and Lee, JMLR2020, Section6. This elementary specialized
derivation is not claimed as a first general robustness theorem.
