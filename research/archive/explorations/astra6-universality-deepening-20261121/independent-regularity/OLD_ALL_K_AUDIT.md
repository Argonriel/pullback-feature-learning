# Independent audit of the inherited all-k statement

Date: November 21, 2026 (America/Los_Angeles).

Verdict: no defect found in the inherited analytic-kernel hierarchy after preserving its local and existential quantifiers. The new continuous-germ theorem strengthens the exact classification without changing these finite-jet constants.

## Kernel expansion

If `g(v)=b_k v^(2k)+O(v^(2k+2))`, the `(i,j)` multiplier at `X=sI+Z` is

    b_k (z_i-z_j)^(2k) /
      [2^(2k) (s+(z_i+z_j)/2)^(2k+1)]
      +O(||Z||^(2k+2)).

Thus with `r_k=b_k/[2^(2k)s^(2k+1)]`,

    R_(sI+Z)=r_k A_Z^k
      -(2k+1)r_k C_Z A_Z^k/(2s)
      +O(||Z||^(2k+2)).

The next even coefficient of `g` cannot contribute to either displayed term. Symmetry and homogeneity, not an unexplained extrapolation from k=1, determine the second term.

## Reducing high powers

On the real symmetric matrix space, `A_Z=ad_Z^2` is positive self-adjoint: `<U,A_ZU>=||[Z,U]||_F^2`. Invariance under `A_Z^k` implies invariance under `A_Z`, since the positive root is a polynomial in `A_Z^k` on its finite spectrum. The zero eigenspace is included.

If `A_Z` preserves `S`, then its pseudoinverse and every integer power of that pseudoinverse preserve `S`. For any `U in S`, set `W=(A_Z^dagger)^(k-1)U`. Then `A_Z^k W=A_ZU`. Therefore invariance under `C_Z A_Z^k` forces invariance under `C_Z A_Z`. This is valid even with repeated eigenvalues and nontrivial kernels. No inversion on the zero eigenspace is performed.

## Static weights and exact coefficients

The exact response formula established in `REGULARITY_THEOREM.md` is

    partial_tau C_(I+tau D)(epsilon E)|_0
      =epsilon integral_0^1 K(u) Q R_(sI+epsilon uE)V du,
    V=DE,  K(u)=1+log(1-u)/u.

For integer `p>=1`, direct integration gives

    integral_0^1 K(u)u^p du=-(H_(p+1)-1)/p.

Substituting the kernel expansion yields precisely

    -r_k (H_(2k+1)-1)/(2k) epsilon^(2k+1) Q A_E^k V
    +r_k (H_(2k+2)-1)/(2s) epsilon^(2k+2) Q C_E A_E^k V
    +O(epsilon^(2k+3)).

Every `V` is achieved by a self-adjoint `D` with `DE=V`, and `I+tau D` stays SPD for a sufficiently small fixed nonzero `tau`. A nonzero derivative of the leading coefficient at `tau=0` ensures there is such a fixed `tau` with nonzero coefficient. This supplies actual fixed quadratic weights; it is not a deduction from arbitrary control paths.

The exact detection-order statement is existential over target directions and weight pairs. It does not say every target or every pair of distinct weights realizes the first possible order. For a space failing Lie-triple closure, some pair first differs at order `2k+1`. If Lie-triple closure holds but the quartic condition fails, some pair first differs at order `2k+2`. All lower orders vanish for every fixed pair in the respective case.

For the positive mixture with `k=2`, `r_2=-1/(120s^5)`, the constants reduce to `77/(28800s^5)` and `-29/(4800s^6)`, in agreement with the old report. This arithmetic is an audit, not a new numerical experiment or a second innovation.

## Scope and exclusions

- The Peirce-invariance argument is valid in a concrete matrix Jordan algebra and is reproduced explicitly in the new theorem note.
- The BKM exception gives `R=0` identically, hence all spaces pass the normal condition. Full metric duality is a different question.
- Analyticity guarantees a finite first nonzero jet for a globally non-BKM homogeneous kernel. The new result replaces it with a nonzero discrepancy germ for exact classification, but not for a finite-order hierarchy.
- Positive Gaussian quadrature supplies unbounded delay orders. The inherited construction explicitly gives `k=2m`; the phrase 'arbitrarily large k' does not assert that this particular construction realizes every integer k.
- Conditional SPD endpoint selection at finite damping is valid; global convergence for every kernel has not been proved here.
- For the analytic class, the diagnostic limitation remains restricted to target-amplitude Taylor coefficients of the first-damping endpoint-weight response. The full algebraic membership test is finite and exact, so no assertion about all possible finite diagnostics is justified.
