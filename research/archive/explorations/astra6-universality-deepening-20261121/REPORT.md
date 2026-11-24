# Spectral neutrality beyond analyticity: exact rigidity, its sharp exception, and response conditioning

Date: November 21, 2026 (America/Los_Angeles).

**Completion status: complete. Final disposition: retain, with a narrower innovation claim.**

This independent deepening produces a stronger theorem than the analytic finite-jet argument, a sharp counterexample to an overbroad extension, and an exact response-identification result with an explicit conditioning limitation. The result remains one specialized research candidate. It is not a new classification of Jordan algebras, a new quantum metric family, or a new quadrature method, and publication priority is not certified.

The main new question resolved is substantive: does exact optimizer neutrality depend on real analyticity, or can a perturbation invisible to every Taylor coefficient still change the endpoint? Analyticity is unnecessary. What matters is whether the kernel actually agrees with BKM on a neighborhood of equal eigenvalues. This distinction is proved without Taylor expansion.

## 1. Precise main theorem

Let `S` be any real linear subspace of `Sym(n)`, with Frobenius projection `P`, normal projection `Q=I-P`, and `J=Jordan(S,I)`. Fix `s>0`. Write a symmetric, normalized, degree-minus-one homogeneous spectral kernel as

    m(x,y)=a^(-1) h(v),
    a=(x+y)/2,  v=(x-y)/(x+y),
    h(0)=1,  h(-v)=h(v).

Let `M_X` be its spectral multiplier and `R_X=Dlog_X-M_X`. Define

    L(v)=atanh(v)/v,  L(0)=1,
    g(v)=L(v)-h(v).

Assume `h` is continuous and `g` is not identically zero on any neighborhood of zero. Then the following statements are equivalent:

1. There is `r>0` such that `Q R_(sI+Z) U=0` for every `Z,U in S` with `||Z||_F<r`.
2. For every `Z,U in S`, both `[Z,[Z,U]]` and `{Z,[Z,[Z,U]]}` belong to `S`.
3. `S` is an associator ideal of the generated unital Jordan algebra `J`; equivalently `J intersect S-perp` is central.
4. `Q R_X U=0` for every `X in (sI+S) intersect SPD` and every `U in S`.

If additionally `h>0` and `h` is `C^1`, these are equivalent to local and global closedness of the normal one-form `Q M_X dX`, and to the following precisely local static-weight condition. For the flow

    Xdot=-(I+lambda M_X)^(-1) B(P(X-sI)-Delta),  X(0)=sI,

let `C_B(Delta)` be the first damping derivative of its interpolating endpoint at `lambda=0`. There exist `r,eta>0` such that, for every target `Delta in S` with `||Delta||_F<r` and every self-adjoint positive `B` with `||B-I||<eta`,

    C_B(Delta)=C_I(Delta).

It is already sufficient that all first weight variations at the identity vanish on one common target neighborhood. When the algebraic conditions hold, neutrality extends to any fixed SPD weight, and uniformly to compact families of such weights with a sufficiently small target neighborhood depending on the family. No uniform neighborhood over arbitrarily ill-conditioned weights is asserted.

At the purely continuous level, the theorem concerns normal cancellation and the exact primitive identity; it does not assign an undefined classical exterior derivative to a merely continuous form. `C^1` functional calculus and a compact-SPD flow argument separately supply the classical and dynamical statements.

The complete proof is [REGULARITY_THEOREM.md](independent-regularity/REGULARITY_THEOREM.md). The complete inherited generator-to-ideal proof is included read only in [ALGEBRA_REDUCTION.md](reference_inputs/endpoint-classification-audit-20261121/ALGEBRA_REDUCTION.md). Thus the review package contains the algebraic bridge rather than assuming it from the name of a classical structure.

## 2. What replaces the analytic proof

For a fixed matrix `Z`, each nonzero spectral-gap component of `R_(sI+tZ)` has the scalar multiplier

    (s+bt)^(-1) g(at/(s+bt)),  a>0,

where `a` is half the eigenvalue gap and `b` is half the eigenvalue sum. A continuous functional-equation argument shows that two distinct `(a,b)` pairs cannot have the same germ unless `g` vanishes on a neighborhood of zero. The proof uses contracting or parabolic iterations of a fractional linear map, not a first nonzero derivative.

There are only finitely many spectral-pair classes. Finitely many evaluations of this commuting self-adjoint family separate them, so its generated operator algebra contains their projectors. The two old operators `ad_Z^2` and `(L_Z+R_Z)ad_Z^2` are recovered from those projectors. This proves necessity of the same cubic and quartic conditions even for flat or oscillatory discrepancies. Repeated eigenvalues and the zero-gap block are included.

Sufficiency uses Peirce invariance of a matrix Jordan algebra and centrality of its normal part. Neither step needs analyticity. This also verifies the finite-damping conserved coordinates

    P_(J intersect S-perp)[X+lambda log X].

Whenever an actual finite-damping trajectory reaches an SPD interpolating limit, these coordinates identify the unique Bregman projection for `||X||_F^2/2+lambda tr(X log X-X)`. General global convergence for arbitrary kernels is not claimed here.

## 3. Exact fixed-weight identification, and its limits

For nonzero `E in S` and arbitrary `V in S`, a self-adjoint `D` can always be chosen with `DE=V`. Every sufficiently small fixed `B_tau=I+tau D` is SPD. Differentiating the exact zero-damping path gives

    partial_tau C_(B_tau)(epsilon E)|_0
      =epsilon integral_0^1 K(u) Q R_(sI+epsilon u E)V du,
    K(u)=1+log(1-u)/u.

The integral transform is locally injective on bounded continuous functions. Two independent proofs reduce it to a sum of positive resolvent energy forms. A fixed-sign assumption is unnecessary. This closes the gap between arbitrary path integrability and what an actual fixed positive quadratic loss can detect.

A single fixed two-dimensional measurement space gives a concrete identification formula. Set

    E=diag(1,-1),  H=[[0,1],[1,0]],
    V=I+H,  N=H-I,  S=span{E,V}.

In the orthonormal basis `E/sqrt(2),V/2`, take `D` with zero diagonal and off-diagonal entries `sqrt(2)`. Then `B_tau` is SPD for `|tau|<1/sqrt(2)`. For `z=epsilon/s`,

    F(z)=(1/2)<N,partial_tau C_(B_tau)(s z E)|_0>
        =z integral_0^1 K(u) g(zu) du.

The complete profile on any small amplitude interval identifies the kernel discrepancy there. For analytic `g(v)=sum b_j v^(2j)`,

    F(z)=-sum a_j b_j z^(2j+1),
    a_j=(H_(2j+1)-1)/(2j)>0.

This is complete-profile identification, not a finite measurement protocol. Moreover, the inverse is **not locally Lipschitz at BKM in uniform norms even within normalized symmetric quantum monotone metrics**. Positive Gaussian approximants have nonnegative moment errors beginning at `j=2m`, so for each fixed `0<r<1`,

    ||F_m||_[0,r] <= r a_(2m) ||g_m||_[0,r],
    a_(2m) -> 0,  ||g_m||_[0,r] -> 0.

Thus no constant `C` near BKM can uniformly bound `||g||` by `C||F||` in these norms. This does not assert discontinuity, rule out a weaker modulus of continuity, or imply that every finite diagnostic is impossible. Full derivation and independent checks: [TRANSFORM_AUDIT.md](independent-transform/TRANSFORM_AUDIT.md).

## 4. Two sharp boundary examples

**Flat but nonzero.** The positive smooth kernel defined by

    h(v)=L(v)+exp(-1/v^2),  with the added term zero at v=0,

has exactly the same Taylor coefficients as BKM at every order, yet its exact neutral spaces are still precisely the associator ideals. Its first-damping weight contrast is flat in target amplitude but is not identically zero on a nonneutral space. This is outside the quantum monotone class, whose kernels are analytic. Within the analytic monotone class, the correct statement remains arbitrary finite-order delay, not infinite-order flatness.

**A BKM plateau.** Set `delta=1/2` and add to `L(v)` the smooth nonnegative function that is zero for `v^2<=delta^2` and equals `exp(-1/(v^2-delta^2))` otherwise. This globally non-BKM kernel makes every sufficiently small affine observation slice locally neutral. More strongly, the following six-dimensional space fails Lie-triple closure while its normal one-form is globally exact on its entire affine SPD slice:

    S={diag(a E+b(I_2+H),4a,-4a,4b,-4b): a,b real},  X0=I6.

The scalar blocks force `|a|,|b|<1/4`; hence the active two-by-two block has spectral ratio less than `sqrt(2)/3<1/2` throughout the slice. The kernel therefore agrees with BKM on every active tangent component. But the double commutator of the two generators is `diag(4H,0,0,0,0)`, outside `S`. This is an exact global counterexample to replacing the nonzero-germ condition by mere global inequality from BKM.

Both examples and their proofs are in the main theorem note. The six-dimensional identity has an independent exact-arithmetic check; it does not rely on sampling the SPD domain.

## 5. Analytic order hierarchy and bounded complexity

The earlier all-`k` static-weight result survives the [independent all-order audit](independent-regularity/OLD_ALL_K_AUDIT.md). If the first kernel discrepancy is `b_k v^(2k)`, then a fixed SPD weight contrast first appears at target order `2k+1` when Lie-triple closure fails, or `2k+2` when Lie-triple closure holds but the quartic condition fails. The weight is fixed throughout each trajectory. The directions and weights may depend on the space and kernel. Positivity of `ad_E^2`, its finite-dimensional spectral calculus, and a pseudoinverse on its range justify the reduction for every `k`; it is not extrapolated from the fifth- and sixth-order examples.

If the kernel is a positive resolvent mixture with at most `m` atoms, classical moment theory gives `k<=2m`. The sharp detection ceilings are therefore `4m+1` and `4m+2`. Gaussian rules uniquely achieve maximal matching and have first unmatched moment

    h_m=2^(4m)/[(4m+1) binom(4m,2m)^2].

This is a useful complexity qualification: a known atom bound restores a finite Taylor-order ceiling. It also explains why the unrestricted finite-jet limitation must not be advertised as absence of a finite algebraic test. The support theorem, its exact constant, and a nonsharp quantitative moment bound are proved in [MOMENT_CEILING.md](independent-literature/MOMENT_CEILING.md). These are supporting applications of classical quadrature, not separate innovation claims.

## 6. Primary literature and priority judgment

Nine original sources were inspected at theorem level and preserved with byte hashes. The full mapping is [PRIORITY_AUDIT.md](independent-literature/PRIORITY_AUDIT.md); executed queries and limitations are in [SEARCH_LEDGER.md](independent-literature/SEARCH_LEDGER.md).

- The spectral/resolvent family and positivity are established by [Lesniewski–Ruskai, Theorems 2.9–2.13](https://arxiv.org/pdf/math-ph/9808016).
- Full-manifold mixture/exponential duality singles out BKM in [Grasselli–Streater, Theorem 18](https://math.mcmaster.ca/~grasselli/GrasselliStreater01.pdf). This is not normal closedness on an arbitrary slice: the latter is vacuous for full `S`.
- The algebraic object is explicitly classical in [Brešar et al., Section 3](https://www.impan.pl/shop/en/publication/transaction/download/product/90340).
- The closest newly checked subspace classification is [Ohara–Ishi–Tsuchiya, Theorems 1–3](https://link.springer.com/article/10.1007/s41884-023-00116-x). At an isotropic point it requires Jordan-subalgebra closure. Traceless `Sym(2)` separates that condition from this candidate's associator-ideal condition.
- General smooth spectral-kernel metrics are already treated by [Thanwerdas–Pennec, Definitions 2.3–2.4 and Theorems 4.1–4.2](https://arxiv.org/pdf/2111.02990). The metric family itself is not new.
- Gaussian moment/Taylor matching is explicit in [Fawzi–Saunderson–Parrilo, Appendix B.1, Proposition 5](https://arxiv.org/pdf/1705.00812).

The inspected originals do not supply the stated equivalence between continuous-germ spectral separation, exact normal rigidity, and the restricted fixed-weight endpoint response. That bounded negative finding is not an originality proof. The appropriate retained claim is this precise equivalence and its sharp local exception; the conditioning consequence strengthens its interpretation. General conservation-law analysis, positive-energy uniqueness, functional calculus, and quadrature remain known tools.

## 7. Minimal validation and independent review

No training, dataset, or reserved sample was used. Seven deterministic cases compare the scalar transform with an independently assembled two-by-two matrix path and central differences of actual fixed SPD weights. BKM and a plateau below its threshold give exactly zero. A flat discrepancy and the same plateau above threshold give nonzero responses. Two quadrature settings and two weight steps are saved. These numerical examples multiply the added smooth bump by `1/4` and use plateau threshold `1/4`; the exact six-dimensional counterexample uses bump coefficient `1` and threshold `1/2` as stated above. The distinction is explicit in the source and records.

The finest weight-difference relative error is at most `5.6462e-6`; halving the step decreases the discrepancy by about four. Maximum scalar quadrature-refinement discrepancy is `2.17e-19` in absolute value. The flat example at amplitude `.2` has response `4.777431854862219e-14`, computed from the discrepancy directly to avoid subtracting almost equal kernels. State positivity is also bounded analytically for the declared targets. The scalar tail has an explicit conservative bound; quadrature refinement is an empirical error check, not a certified quadrature bound.

The saved aggregate pass flag covers its declared gates. Additional properties, including the exact first missed moment, error-ratio behavior, and analytic tail interpretation, were checked separately in [VALIDATION_REVIEW.md](independent-transform/VALIDATION_REVIEW.md). Exact rational calculations verify the Gaussian constants for one through four atoms and the guarded plateau commutator. None of these finite checks substitutes for the universal proof.

The response transform and spectral separation were derived or checked independently by two reviewers. A separate primary-literature reviewer supplied the closest theorem comparisons. All agents inherited the requested model settings; no model override or accelerated mode was requested.

## 8. Reproducibility, preservation, and final decision

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md), [FAILED_ATTEMPTS.md](FAILED_ATTEMPTS.md), `provenance.json`, and `manifest.json`. The validation code and usage instructions are copied into `source_snapshot/`. Earlier theorem and dynamics inputs are copied into `reference_inputs/` without altering their contents; they are explicitly prior work. Source PDFs and publisher/author HTML are preserved in the independent literature folder with retrieval hashes. The two source archives are checked against their existing manifests. No public dependency, other direction's code, frozen configuration, or historical result was modified.

**Retain this direction, with the nonanalytic rigidity theorem and its sharp exception as the main candidate.** Keep the analytic order hierarchy, support ceiling, and conditioning result as connected consequences. Reject the broader claims that globally non-BKM alone suffices in the smooth class, that all finite diagnostics fail, that quadrature or associator ideals are newly discovered, or that any neural-training benefit has been established. The package is independently reviewable; remaining uncertainty is external priority and publication significance, not an unfinished numerical search.
