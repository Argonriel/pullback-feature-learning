# Independent referee review of the global-convergence proof

Date: 2026-11-21. Reviewed `THEOREM.md` and the companion primary-source novelty audit. This review is mathematical; it does not rerun the numerical validation or independently certify novelty. The previously proved classification `N_J=J intersect S-perp subset center(J)` is taken as a stated premise.

## Verdict

The proof is sound under its stated hypotheses. I found no circularity in the chain from boundedness to analytic convergence, convex optimality, and strict positivity. The two quantitative factor estimates and the eventual loss-rate constant are correct. The global statement genuinely removes the earlier small-target/assumed-convergence restriction within the specified class. Its proof mechanisms remain classical, so this conclusion does not strengthen the separate novelty assessment.

## Dependency audit

1. **Strict feasibility in the generated algebra.** The proof that the orthogonal projection onto a unital matrix Jordan algebra is positive is valid. The negative spectral projector of `P_J A` belongs to `J` by polynomial functional calculus. Pairing it with a PSD matrix is nonnegative. Applying positivity to `A-alpha I` proves preservation of a strict positive lower bound. Thus a feasible ambient SPD target yields an SPD feasible `Y in J`, without assuming a minimizer of the Bregman problem.

2. **Product boundedness before global existence.** On the maximal finite-time existence interval the factor is invertible, and the central logarithmic invariants are valid. If the product were unbounded there, a normalized subsequence has a central PSD normal limit `V`. On every support block of `V`, every eigenvalue of `X_k` grows proportionally to its norm. Consequently the logarithmic term in `F_V` cannot cancel its diverging positive linear term. Blocks outside that support have exactly zero coefficient in `F_V`. This rules out escape without presupposing convergence, strict positivity of the limiting matrix, or coercivity of the measurement loss. Representation multiplicities do not affect the argument: a central element is scalar on each represented simple summand.

3. **Global factor existence and analytic convergence.** For arbitrary real `F`, `K_(FF^T)+lambda I` is positive definite as an operator, with lower bound `lambda`. The factor vector field is therefore analytic on the entire factor space when `lambda>0`. Bounded products imply bounded factors via `||F||_F^2=tr X`; bounded factors prevent finite-time blowup. The linear equation for `F` preserves invertibility at each finite time. The analytic finite-length argument is invoked only after boundedness and uniform angle/descent estimates have been established. It gives a limiting factor and hence a limiting product and generator. The relation `H_infinity F_infinity=0` then follows from continuity of the autonomous vector field: a convergent solution cannot have a nonzero limiting velocity.

4. **Excluding a negative limiting generator.** The cone argument is valid even if the complementary spectrum contains zero. Choose the positive spectral subspace of `-H_infinity` and a perturbation norm below one fourth of its smallest positive eigenvalue. On the cone boundary the negative contribution from the expanding subspace dominates the perturbation; inside the cone its component grows exponentially. Invertibility of `F(T)` represents that expanding solution as a fixed linear combination of columns of the bounded fundamental matrix. Thus every negative eigenvalue of `H_infinity` is impossible. This is deterministic and does not require a generic-start or almost-everywhere argument.

5. **Interpolation precedes boundary exclusion.** With `H_infinity` PSD and `H_infinity F_infinity=0`, one obtains `G_infinity=lambda H_infinity` PSD and `<G_infinity,X_infinity>=0`. Convexity and a feasible PSD matrix then force zero loss. Only after this step is `N=Y-X_infinity` known to be normal and central. If the limit were singular, strict positivity of `Y` makes the scalar coefficient of `N` positive on every singular block. Negative coefficients can occur only on blocks with positive-definite limits. Therefore `<N,log X(t)>` diverges negatively and cannot be canceled in the conserved quantity. This proves an SPD limit without assuming boundary divergence of entropy itself.

6. **Ambient projection.** The difference `grad Phi_lambda(X_infinity)-grad Phi_lambda(X0)` belongs to `J` by functional calculus and is orthogonal to `N_J` by conservation, hence belongs to `S`. This is the full ambient affine-constraint KKT condition, not merely a constrained condition within `J`. Strict convexity proves the stated unique ambient Bregman projection. There is no need to assume its existence in advance.

## Constant check

Diagonalize `X` with eigenvalues `x_i>=0`, and write the symmetric gradient entries as `g_ij`. For `X=FF^T`,

\[
\|GF\|_F^2=\frac12\sum_{i,j}(x_i+x_j)g_{ij}^2,
\qquad \operatorname{grad}_F f(FF^T)=2GF.
\]

Writing `H=(K_X+lambda I)^-1G` gives

\[
\|HF\|_F^2=\frac12\sum_{i,j}
 \frac{(x_i+x_j)g_{ij}^2}{(x_i+x_j+\lambda)^2}
 \le \lambda^{-2}\|GF\|_F^2.
\]

If `||X||op<=R`,

\[
-\dot f=\sum_{i,j}\frac{x_i+x_j}{x_i+x_j+\lambda}g_{ij}^2
\ge \frac{\|\operatorname{grad}_F f\|_F^2}{2(2R+\lambda)},
\qquad
\|\dot F\|_F\le\frac{\|\operatorname{grad}_F f\|_F}{2\lambda}.
\]

These are precisely the stated constants. If eventually `X>=aI`, the product mobility is bounded below by `2a/(2a+lambda)`. In orthonormal measurement coordinates with quadratic weight `B>=beta I`, `||G||_F^2>=2 beta f`, giving the reported exponent `2 beta * 2a/(2a+lambda)`.

## Scope and minor exposition points

No mathematical correction is needed. For a formal manuscript, identify the analytic finite-length theorem used, or include its short standard argument from the displayed angle/descent bounds and a Lojasiewicz gradient inequality on the compact limiting set. Also state explicitly that the factor is square and initially invertible, as required by the fundamental-matrix argument; this is already implicit in `X0 SPD` and the displayed factorization. A positive-definite loss weight on redundant raw measurements should be interpreted as the induced positive-definite operator on `S`; the reported rate uses its eigenvalue in orthonormal coordinates.

The review does not extend the theorem to rank-deficient initial factors, zero damping, arbitrary measurement spaces, incompatible initializations outside `J`, inconsistent targets, or discrete-time training. It supplies no uniform target-independent lower eigenvalue bound or complexity estimate.
