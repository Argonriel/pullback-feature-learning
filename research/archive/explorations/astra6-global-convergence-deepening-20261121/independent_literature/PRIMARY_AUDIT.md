# Primary-source audit of the classified damped flow

Date: 2026-11-21 (America/Los_Angeles). Independent literature review.
Scope: original global theorem and attempted direct reductions. This is not an exhaustive priority certificate.

## Disposition

**Retain only as a narrowly scoped convergence/selection result.** The inspected primary theorems do not directly imply the all-compatible-SPD-start theorem for the metric
`M_X = I + lambda K_X^{-1}`, `K_X U = XU+UX`, on an associator-ideal measurement space. Most proof ingredients and the route from spectral protection to a PL rate are established. New quantitative claims need their own problem-specific spectral protection, not a repackaged Gronwall step.

An additional exact negative result strengthens the reduction audit: the usual reverse Bregman distance for the claimed endpoint potential can **increase**, even for this classified family and a strictly feasible target. This rules out a tempting direct import of the standard Hessian Lyapunov proof. It does not refute endpoint selection or global convergence.

## Candidate quantified

For every finite dimension, every unital real matrix Jordan algebra J, every measurement space S contained in J with N=J intersect S-perp contained in the center of J, every fixed lambda>0, every X0 in J strictly positive definite, and every fixed SPD quadratic measurement weight, consider
`Xdot=-K_X(K_X+lambda I)^-1 G(X)`.
The strict version assumes that the affine zero set has a strictly positive definite point. Its endpoint is asserted to minimize the Bregman divergence for
`Phi(X)=||X||_F^2/2+(lambda/2)tr(X log X-X)`
over the ambient affine SPD set.
The claim concerns this product metric, not a generic factor preconditioner or momentum damping.

## Theorem-level primary comparisons

### Bah–Rauhut–Terstiege–Westdickenberg

Primary source: [Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers](https://www.instmath.rwth-aachen.de/~mwest/files/linear.pdf), arXiv v5, 15 October 2020. Inspected Theorems 5, 39 and 40 (PDF pages 5 and 21).

Theorem 5 assumes full rank of the input covariance (XX^T); it concludes globally bounded factor flows and convergence to a critical point. Theorem 39 retains that assumption and gives almost-everywhere global conclusions: for general depth, a global minimum on a possibly lower-rank stratum; for depth two, a global minimum of the factor objective. The authors explicitly explain that balanced initializations form an ambient measure-zero set, so Theorem 39 cannot simply be restricted to them. Theorem 40 separately treats almost every initial product on a fixed-rank manifold, allowing convergence to a critical point of a lower-rank stratum.

**Reduction obstruction (our comparison):** the regression operator (W\mapsto WX) is injective under that covariance assumption, whereas the classified observation operator can have nonzero central kernel. Their balanced product metric also lacks the present resolvent damping. Therefore neither observation assumptions nor the all-SPD-start quantifier transfer. Their analytic bounded-flow machinery remains prior art.

### Zhang–Peyre

Primary source: [Global Exponential Convergence of Two-Layer Linear Network Training](https://arxiv.org/html/2610.09356), v1, 7 October 2026. Inspected Assumption 1.1, Definition 2.2, Theorem 3.1 and Appendix C.3.

Assumption 1.1 is smoothness plus a global PL inequality; it explicitly permits a positive-semidefinite quadratic Hessian with a nontrivial kernel. Definition 2.2 sets (a_\pm(\Sigma)=\max\{a\ge0:\Sigma\mp aJ\succeq0\}), (g=a_++a_-). Theorem 3.1 requires (g(\Sigma_0)>0) and gives global exponential loss convergence with exponent (2\kappa g). Their flow is the split Bures covariance flow. The damped extension in Section 5 is heavy-ball momentum, not the present product-metric blend.

**Reduction obstruction (proved below):** a tied PSD embedding has (g=0), even with a strictly positive definite product. Underdetermination alone is therefore not a novelty distinction. Conserved quantities yielding spectral protection and then PL rates are already central to this paper.

### Zhang–Fattahi–Zhang / PrecGD

Primary source: [Preconditioned Gradient Descent for Overparameterized Nonconvex Burer–Monteiro Factorization with Global Optimality Certification](https://jmlr.org/papers/volume24/22-0882/22-0882.pdf), JMLR 24 (2023). Inspected Theorem 4, Theorem 8 and Corollary 9 (PDF pages 9, 11–12).

PrecGD uses the factor update (F^+=F-\alpha\nabla f(F)(F^TF+\eta I)^{-1}). Theorem 4 assumes gradient Lipschitzness, restricted strong convexity of the product objective, a sufficiently small objective gap, and damping comparable above and below to product error. Theorem 8 is a randomized perturbed-iteration guarantee for approximate second-order stationarity, using a finite radius of a containing sublevel set. Corollary 9 adds restricted strong convexity, a global strict-saddle condition and overparameterization, then combines perturbed global and unperturbed local phases.

**Reduction obstruction (our comparison):** at full representation rank the restricted-convexity requirement excludes a nonzero observation kernel; its factor preconditioner is not the Lyapunov resolvent. Fixed positive lambda and every compatible full-rank start are not supplied by these results. Convex PSD optimality certification and preconditioning-based rates must nevertheless be credited as established tools.

### Alvarez–Bolte–Brahic / Hessian Riemannian flows

Primary source: [Hessian Riemannian gradient flows in convex programming](https://www.math.tecnico.ulisboa.pt/~brahic/Legendre10.pdf), 2004. Inspected Theorem 3.1, Theorem 4.1, Proposition 4.1, Theorem 4.2 and Corollary 4.1.

Their metric is a Legendre Hessian. Theorem 4.1 proves well-posedness under its regularity assumptions and one of bounded objective sublevels, suitable Bregman sublevels plus quasi-convexity, or linear growth of inverse metric. Proposition 4.1 uses the exact reverse-Bregman derivative identity to obtain an objective (O(1/t)) comparison. Theorem 4.2 adds Bregman-function properties and gives trajectory convergence to a solution for convex objectives. Corollary 4.1 identifies a Bregman projection for a linear objective.

**Reduction obstruction (proved below):** the present metric is not the Phi Hessian on a noncommuting block, and the central equality does not restore their full Lyapunov identity. The commuting subclass is ordinary Hessian flow. Bregman geometry and entropy barriers are not new contributions.

### Wu–Rebeschini

Primary source: [Implicit Regularization in Matrix Sensing via Mirror Descent](https://proceedings.neurips.cc/paper/2021/file/abf0931987f2f8eb7a8d26f2c21fe172-Paper.pdf), NeurIPS 2021, Proposition 5 and Corollary 6, PDF page 8.

Proposition 5 assumes pairwise commuting symmetric sensing matrices and, for spectral entropy, a positive definite initialization commuting with those matrices. It identifies the mirror and exponentiated-gradient dynamics. Corollary 6 gives factor-flow convergence and an entropy-type endpoint characterization for isotropic positive product initialization, assuming a PSD feasible matrix exists.

**Comparison:** this already occupies commuting factor-flow convergence and entropy selection, including a PSD boundary regime. It has a different metric from fixed finite damping, and does not supply the noncommuting associator-ideal theorem. A PSD-only extension whose mechanism reduces entirely to commuting coordinates has correspondingly limited innovation value.

## Additional primary screens

- [Fan et al., ICML 2024](https://proceedings.mlr.press/v235/fan24b.html), *On the Convergence of Projected Bures-Wasserstein Gradient Descent under Euclidean Strong Convexity*: the official abstract and accessible indexed primary excerpts state Euclidean strong convexity and concern projected BW descent. These exclude an ambient central loss kernel and do not match the resolvent metric. The linked raw PDF could not be retrieved by the web reader; the OpenReview fallback hit verification. This is a **screen**, not a completed theorem audit.
- [Altschuler–Chewi–Gerber–Stromme](https://arxiv.org/abs/2106.08502), v3 (2023), and [Chewi–Maunu–Rigollet–Stromme](https://proceedings.mlr.press/v125/chewi20a/chewi20a.pdf), COLT 2020: the original abstracts/theorem excerpt concern BW barycenters, spectral conditioning and PL arguments. They are important precedents for obtaining spectral control before a rate, not a direct result for arbitrary weighted measurement quadratics.
- [Nylöf et al.](https://arxiv.org/html/2611.08468v1), 5 November 2026: Assumption 1 and Theorem 1 require a convex smooth affine lift of the smooth objective, with a separately convex proximable part. The deterministic algorithm is BW-FISTA. This is a different algorithm and its convex-lift condition does not hold for general sensing squares. An elementary obstruction is the scalar lift ((u^2-y)^2/2), whose second derivative at (u=0) is (-2y<0) for (y>0); it is also not globally gradient-Lipschitz.
- Broad searches for full-rank convex factor flows primarily recovered the commuting implicit-bias results and works assuming convergence or special observation/initialization structure. Search nonappearance is not evidence of priority. Generic saddle exclusion, analytic convergence and convex KKT are not claimed as new.

## Exact tied-covariance zero-gap calculation

This independent calculation uses the primary definition above. Write
\[
\Sigma=\begin{bmatrix}X&X\\X&X\end{bmatrix},\qquad
J=\operatorname{diag}(I,-I),\qquad X\succ0.
\]
For any nonzero \(v\), \(z=(v,-v)\) is in the kernel of \(\Sigma\) and \(z^TJz=0\).
If \(\Sigma-aJ\succeq0\), then \(z^T(\Sigma-aJ)z=0\), so positivity implies
\((\Sigma-aJ)z=0\). But this vector equals \(-a(v,v)\), forcing \(a=0\).
The same argument applies to \(\Sigma+aJ\). Thus \(a_+=a_-=g=0\).
Equivalently \(J\Sigma\) has square zero. No numerical rank threshold is involved.

## Exact failure of the ordinary Bregman Lyapunov route

Let \(J=\operatorname{Sym}(2)\), \(S=\{Z:\operatorname{tr}Z=0\}\), and \(N=\mathbb RI\).
Use Frobenius-orthonormal measurement matrices
\[
D=\operatorname{diag}(1,-1)/\sqrt2,\qquad
O=\begin{bmatrix}0&1\\1&0\end{bmatrix}/\sqrt2.
\]
For \(X=\operatorname{diag}(a,b)\), with \(a,b>0\) and \(a\ne b\), define
\[
B_X=\nabla^2\Phi(X)=I+(\lambda/2)D\log_X,\quad
M_X=I+\lambda K_X^{-1},\quad Q_X=B_XM_X^{-1}.
\]
On the \(D,O\) coordinates, \(Q_X=\operatorname{diag}(1,q)\), where
\[
q=\frac{1+(\lambda/2)\log(a/b)/(a-b)}
        {1+\lambda/(a+b)}>1.
\]
The strict inequality follows from the strict logarithmic-mean/arithmetic-mean inequality.
Choose any \(c\in(2\sqrt q/(1+q),1)\), so
\[
W=\begin{bmatrix}1&c\\c&1\end{bmatrix}\succ0.
\]
Set \(d=\epsilon(-c(1+q)/2,1)\) and \(Y=X-d_1D-d_2O\).
For every sufficiently small \(\epsilon>0\), \(Y\succ0\).
Take the measurement map \(\mathcal AZ=(\langle D,Z\rangle,\langle O,Z\rangle)\) and
\[
f(Z)=\tfrac12(\mathcal AZ-\mathcal AY)^TW(\mathcal AZ-\mathcal AY).
\]
At initialization \(X\), its gradient has coordinates \(Wd\). Along the actual damped flow,
\[
\frac{d}{dt}D_\Phi(Y,X(t))\Big|_{t=0}
=-d^TQ_XWd
=\epsilon^2\left(\frac{c^2(1+q)^2}{4}-q\right)>0.
\]
This derivative is unchanged by replacing \(Y\) with another feasible comparator in \(J\):
their difference \(N\) is central, \(B_XN=M_XN\), and the derivative change is
\(-\langle N,G\rangle=0\).
It therefore also applies to the selected Bregman endpoint once its existence is independently established.

The metric is not a Hessian in these affine matrix coordinates. For \(E=E_{11}\),
the necessary symmetry of third derivatives would require
\[
\partial_E\langle O,M_XO\rangle
=-\frac{\lambda}{(a+b)^2}
=\partial_O\langle E,M_XO\rangle
=-\frac{\lambda}{2a(a+b)},
\]
which fails if \(a\ne b\).
This rules out an affine-coordinate Hessian potential for \(M\), not arbitrary transformations,
auxiliary Lyapunov functions, or loss-dependent constructions.


## Minimal check and failures preserved

The exact construction was evaluated once with (a=256,b=1,\lambda=257,c=.99,\epsilon=.1).
It yields (q=1.8971672580698504), target minimum eigenvalue (0.8985743803274744), and derivative (+0.0015946922108501412).
Central finite differences of the independently evaluated matrix Bregman expression at four step sizes agree within (3.51\times10^{-9}).
The three discriminating controls have negative derivatives: isotropic weight, commuting residual at the tested state, and lambda zero.
The finite-difference errors are roundoff limited and are retained individually; no asymptotic numerical order is claimed.
No ODE integration or training was performed.

Files:
- `lyapunov_counterexample.cjs`: dependency-free Node check.
- `lyapunov_counterexample_results.json`: all values, controls, errors recoverable from four raw differences, Node version, source SHA-256.
- `sources/*_web_retrieval.json`: extracted tool-return snapshots for the four main papers.
- `SOURCE_MANIFEST.json`: SHA-256 of the snapshots and local evidence files.
- `SEARCH_LEDGER.md`: queries, retrieval limitations, and failed routes.

**Provenance limit:** direct original-PDF/HTML downloads failed DNS in this execution environment. Saved hashes are hashes of the retained web-tool extraction records, not hashes of publisher PDF bytes. They must not be labeled full-source content hashes.

## Final novelty judgment

The main theorem remains a plausible, narrow synthesis conditional on the root proof audit.
The original global argument's classification-based escape control and strict-feasibility boundary exclusion are the relevant additions; generic PL/KL, PSD optimality, Jordan structure and Bregman projection cannot be separately counted.
The Bregman counterexample is a useful proof obstruction and scope clarification, not by itself a new optimization method or a strong standalone innovation.
The proper disposition is **retain, with narrowed novelty language**, pending an independently checked quantitative extension and broader priority review.

