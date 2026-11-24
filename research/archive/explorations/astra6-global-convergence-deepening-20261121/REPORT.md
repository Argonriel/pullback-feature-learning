# Classified damped flow: convergence on the PSD cone and exact endpoints

Date: November 21, 2026 (America/Los_Angeles).

**Completion status: complete. Final disposition: retain with narrowed novelty claims.** The original interior theorem survives review. The main result is now a self-contained, boundary-complete theorem for arbitrary data. A computable spectral/rate supplement is proved for geometry-adapted weights. The tempting arbitrary-weight reverse-Bregman Lyapunov claim is explicitly refuted. No claim of certified originality or a new general optimization principle is made.

## 1. Main theorem: the strict-feasibility restriction was stronger than necessary

Let `A:Sym(n)->R^m`, `S=range(A*)`, and `J=Jordan(S,I)`. Assume `N_J=J intersect S-perp` is central in `J`. For every fixed `lambda>0`, every fixed SPD observation weight `W`, every data vector `y`, and every compatible SPD initial matrix `X0 in J`, consider

```
f(X) = (A X-y)^T W (A X-y)/2,
K_X(U) = XU+UX,
Xdot = -K_X (K_X+lambda Id)^(-1) grad f(X).
```

The solution exists globally, is positive definite at each finite time, and converges to `X*>=0`. Define the ambient cone-optimal set

```
C_W = argmin_{Z>=0} f(Z),
Phi_lambda(Z) = ||Z||_F^2/2 + (lambda/2) tr(Z log Z-Z),
```

where `0 log 0=0`. Then `C_W` is nonempty and

```
X* = argmin_{Z in C_W} D_Phi_lambda(Z,X0).
```

This minimizer is unique. Every `Z in C_W` has `ker(X*) subset ker(Z)`: the selected limit has the maximal support of the optimal set. Other optimizers can have smaller supports. The endpoint is SPD **if and only if** the optimal set contains an SPD matrix.

For consistent PSD-feasible targets, `C_W={Z>=0:A Z=y}` is independent of the weight, so exact weight-independent endpoint selection extends to singular feasible sets. For inconsistent targets, the optimal set and endpoint can depend on `W`. Invertible compatible initialization and positive damping remain essential hypotheses.

Full proof and all quantifiers: [THEOREM.md](THEOREM.md). Independent reviews: [root proof audit](ROOT_REVIEW.md) and [second boundary review](independent_spectral/BOUNDARY_THEOREM_SECOND_REVIEW.md).

### What changed in the proof

The original chain through bounded factors, analytic finite length, and the expanding-cone argument already proves global PSD optimality without a feasible target. Its escape argument also needs no feasibility: a normalized escaping direction is a fixed central PSD normal matrix, so the conserved linear/log pairing diverges on its support.

The new boundary step compares `X*` with the Jordan projection of any other optimum. Their difference is central. Negative coefficients cannot occur on a singular limiting subspace; positive coefficients there would make the conserved logarithmic pairing diverge. This forces every optimal matrix into the face supported by `X*`. The Bregman objective is differentiable relative to that face, and the remaining conserved coordinates give its ambient first-order optimality condition. This avoids applying an ambient logarithm to a singular endpoint.

The contribution is this complete result for the specified non-Hessian matrix flow, not separate claims of new KL theory, saddle exclusion, PSD KKT, Jordan structure, maximal-rank selection, or Bregman geometry.

## 2. Computable spectral protection, with the weight condition stated explicitly

Write the induced measurement Hessian as `B:S->S`, with least eigenvalue `beta>0`. Let `Y in J` be a supplied SPD feasible point, `eta=lambda_min(Y)`, and `D0=D_Phi(Y,X0)`. If `B` is scalar on each complete traceless simple Jordan component, arbitrary SPD on the measured center, and has no coupling between those summands, the reverse divergence `D_Phi(Y,X(t))` decreases.

Let `a=lambda/2` and define `m in (0,eta]` by the one-dimensional monotone equation

```
(eta-m)^2/2 + a [eta log(eta/m)-eta+m] = D0.
```

If `D0=0`, take `m=eta`. For every `t>=0`,

```
X(t) >= m I,
f(X(t)) <= f(X0) exp[-2 beta (2m/(2m+lambda)) t].
```

A weaker closed-form bound is `m_exp=eta exp[-1-2D0/(lambda eta)]`. The proof also bounds the product above and gives an explicit endpoint-distance tail. For an inconsistent target admitting an SPD optimizer, the same statement applies to excess loss when that optimizer is supplied. The sufficient weight condition is not asserted necessary.

This is a concrete, non-Gronwall spectral certificate derived from an SPD-witness Bregman sublevel. The subsequent rate is an ordinary PL estimate. It is retained as a quantitative supplement, not an independent innovation. The commuting case is standard Hessian-flow theory.

Full derivation and limitations: [quantitative review](independent_spectral/QUANTITATIVE_REVIEW.md).

## 3. Exact obstructions and necessary boundaries

- **Arbitrary SPD weights do not preserve this Lyapunov function.** A classified `2x2` traceless observation problem has `d D_Phi(Y,X(t))/dt > 11/15` at an admissible SPD start, while the ordinary loss decreases. The independently evaluated derivative is `+0.733543189715`. Thus positivity of the metric-comparison operator and of `B` cannot justify positivity of their product. This refutes the shortcut, not global convergence or every possible alternative spectral bound.
- **Boundary endpoints can converge only polynomially.** For the exact scalar target zero, `xdot=-2x^2/(2x+lambda)`, one has `t x(t)->lambda/2` and `t^2 f(x(t))->lambda^2/8`. No positive global exponential exponent is available in general at the boundary.
- **No target-independent interior margin exists.** With scalar target `epsilon>0`, the endpoint is `epsilon`, and its local state-decay exponent is `2epsilon/(2epsilon+lambda)`, tending to zero with the feasibility margin.
- **Rank-deficient starts can stick.** The scalar factor initialized at zero stays there for target one, although its loss is not optimal.
- **Zero damping can hit the boundary in finite time.** For traceless observations, `X0=I` and traceless target `diag(a,-a)`, `a>1`, the Euclidean solution loses positivity at `-log(1-1/a)`, even though SPD interpolants exist.
- **Inconsistent-data weight independence is false.** The diagonal target `(-1,1)` has endpoints `(0,1)` and `(0,1/2)` under two SPD weights; both are certified by exact PSD KKT conditions.

Proofs: [exact controls](independent_boundary/EXACT_CONTROLS.md), [quantitative counterexample](independent_spectral/QUANTITATIVE_REVIEW.md), and [independent literature-side obstructions](independent_literature/EXACT_OBSTRUCTIONS.md). Failed proof routes and environment failures were preserved in [FAILED_ATTEMPTS.md](FAILED_ATTEMPTS.md).

## 4. Closest primary literature and the residual claim

| Primary result inspected | Why it does not directly supply this theorem |
|---|---|
| Bah et al., Theorems 5, 39, 40 | Full-row-rank regression covariance, a different product metric, and almost-everywhere global statements do not yield this central-kernel, every-compatible-SPD-start claim. |
| Zhang–Peyre, Theorem 3.1 | Quadratic observation nullspaces are allowed, so underdetermination is not a distinction. The tied PSD covariance has exactly zero signed support gap; the theorem requires a positive gap. Its damping extension is momentum. |
| Zhang–Fattahi–Zhang, PrecGD | Different factor preconditioner; local restricted convexity and a perturbed global phase do not give the stated deterministic all-start flow theorem. |
| Alvarez–Bolte–Brahic | Convex Hessian flows already have boundary convergence and Bregman machinery. The present noncommuting metric is not a Hessian metric in these affine coordinates; its usual reverse-Bregman divergence can increase. |
| Wu–Rebeschini | Commuting measurements already have PSD-feasible convergence and entropy endpoint results. Allowing a singular feasible target alone is therefore not new. |

The comparison is checked against the original [Bah paper](https://www.instmath.rwth-aachen.de/~mwest/files/linear.pdf), [Zhang–Peyre paper](https://arxiv.org/html/2610.09356), [PrecGD paper](https://jmlr.org/papers/volume24/22-0882/22-0882.pdf), [Hessian-flow paper](https://isr.umd.edu/Labs/ISL/HELMKE/AlBra_SICON04.pdf), and [Wu–Rebeschini paper](https://proceedings.neurips.cc/paper/2021/file/abf0931987f2f8eb7a8d26f2c21fe172-Paper.pdf). Detailed assumptions, theorem numbering by source version, the exact tied-gap calculation, and retrieval limits are in [PRIMARY_AUDIT.md](independent_literature/PRIMARY_AUDIT.md) and the [optimal-face addendum](independent_literature/OPTIMAL_FACE_NOVELTY_ADDENDUM.md).

The defensible residual candidate is **global optimal-face convergence and exact endpoint selection for this classified noncommuting resolvent flow, for every compatible SPD start and arbitrary data**. The restricted-weight rate and negative Lyapunov example clarify its scope. Failure to locate a theorem-level match is not proof of originality. No practical neural-training advantage has been established.

## 5. Minimal validation and error accounting

| Check | Purpose | Result |
|---|---|---|
| One coupled-center `5x5` PSD-only target; two weights, two RK4 steps | Tests a genuinely singular feasible set with an unresolved regular center | Endpoint prediction has rank 3. Step refinement at most `1.2114e-10`; maximum invariant drift `4.9723e-10`. |
| Same boundary check at times 100 through 1600 | Separates slow transient from numerical error | Final distance to the proved endpoint is about `0.0010074`; finite-time inter-weight gap is `0.00034746`. These are **not** different limiting endpoints. |
| One noncommuting `2x2` geometry-adapted trajectory; two tolerances | Checks explicit spectral/rate certificate | Certified lower bound `0.02848255`, sampled minimum `0.59807976`, loss exponent `0.06716766`; refinement difference `3.62e-10`. The bound is conservative. |
| Exact arbitrary-weight Lyapunov witness and independent alternative | Rejects the full-Hessian shortcut | Positive derivative formulas confirmed by finite differences, with final errors about `2.70e-9` and `3.51e-9`; isotropic, commuting-residual and zero-damping controls retained. |
| Rational scalar/KKT controls | Checks assumption boundaries without integration | Exact checks pass; no truncation or sampling error. |

These are deterministic algebra/ODE consistency checks. Continuous-time conclusions come from the proofs; sampled extrema are not formal trajectory certificates. No old large-target run was repeated, no training sweep was enlarged, and no reserved validation sample was used.

## 6. Reproducibility and final judgment

All research notes, source snapshots, raw results, refinement errors, failed attempts, exact controls, and source/record hashes are retained. [DATA_DICTIONARY.md](DATA_DICTIONARY.md) maps files and reproduction steps. `manifest.json` hashes package content; `ARCHIVE_CHECK.json` records integrity and source-result checks. Saved web-retrieval hashes identify extracted records, not complete publisher PDF bytes. Historical input files were read only and their hashes are recorded separately.

**Retain, with narrowed claims:** the boundary-complete all-data theorem is the main reviewable result. The geometry-adapted explicit rate is a useful corollary. **Reject** the arbitrary-weight reverse-Bregman monotonicity claim and any target-independent spectral/rate claim without a quantitative interior condition. General-weight explicit uniform constants, priority certification, and practical significance remain unresolved; they are not prerequisites falsely marked as completed here.

The requested deepening is complete: there is a self-contained theorem, explicit assumptions and quantifiers, theorem-level primary comparisons, two independent proof reviews, minimal discriminating checks, reproducible records, and a final innovation judgment.
