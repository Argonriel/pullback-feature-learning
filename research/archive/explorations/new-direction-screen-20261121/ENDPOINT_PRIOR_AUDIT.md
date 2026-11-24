# Independent prior-art audit: damping and loss-weight-dependent endpoints

Date: November 21, 2026 (America/Los_Angeles).

**Verdict: the specific endpoint observation survives the direct theorem-overlap check, but research novelty remains provisional.** The inspected primary papers do not state the proposed two-fixed-loss endpoint separation or a small-damping selection coefficient for this exact metric. They do cover its broad ingredients: matrix-factorization geometry, noncommuting obstructions to a universal mirror flow, non-Hessian Riemannian optimization, damping-dependent spectral dynamics, and implicit bias. A numerical endpoint difference alone is a narrow example inside those frameworks. The meaningful residual contribution would be a proved, nonzero, quantitatively controlled endpoint-selection law under the specified full-representation metric.

This is a read-only literature and logical audit. No training or numerical experiment was run by this reviewer. Numerical values below were supplied by the depth investigation and should be assessed against its own verification records.

## Exact candidate being compared

The model is a balanced two-factor matrix parametrization with SPD product X, initialized at X_0=0.1 I. The pullback metric is formed from the Jacobian of **all entries of X**, with Euclidean parameter damping lambda I. It is not the Gauss–Newton metric of just the two observed measurements. In the balanced SPD reduction,

\[
 S_X(A)=XA+AX,\qquad H_X=I+\lambda S_X^{-1},\qquad
 \dot X=-H_X^{-1}\nabla L_w(X).
\]

The fixed loss is

\[
 L_w(X)=\tfrac12\big[(\langle A_1,X\rangle-1)^2+
 w(\langle A_2,X\rangle-1)^2\big],
 \quad A_1=E_{11},\quad A_2=\tfrac12\begin{pmatrix}1&1\\1&1\end{pmatrix}.
\]

The comparison changes w from 1 to 9. Both losses have exactly the same affine interpolation set. The depth investigator reports refined differences in the final off-diagonal entry X_12 of approximately 2.01294e-5, 2.00625e-4, and 1.64528e-3 at lambda=0.003, 0.03, and 0.3 respectively; the undamped difference is approximately 7e-12. This is endpoint dependence after interpolation, not merely a difference at a shared number of updates.

## What the primary papers actually establish

**Li, Wang, Lee and Arora, NeurIPS 2022.** Theorem 4.8 gives a loss-independent mirror representation for commuting regular parametrizations. Theorem 4.9 provides a necessary Lie-algebra condition when one mirror map must work for all time-dependent losses; its condition is weaker than literal commutativity. Corollary 4.11 supplies a matrix-factorization obstruction using a time-dependent loss. Theorems 3.8 and 4.16 give an interpolation regularizer under the positive mirror/commuting assumptions. None of these statements is a converse saying that failure of commutativity forces two particular *fixed weighted quadratics* to choose distinct endpoints. Nor do they evaluate the metric I+lambda S_X^{-1}. Thus it would be inaccurate either to claim their theorem already proves this endpoint example or to present generic noncommutativity as new. Inspected: original full conference version, Sections 3–4 and Appendix D proof of Corollary 4.11. [Primary full paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Supplemental-Conference.pdf).

**Azulay et al., ICML 2021.** Section 5 shows that multiplying a non-Hessian metric by a suitable scalar integrating factor can produce a Hessian metric without changing paths or endpoints; Section 6 applies this to fully connected linear-network cases. Consequently, failure of the raw Hessian condition alone does not imply loss-weight-dependent endpoint selection. This is a substantive alternative that a proposed impossibility argument must address. The paper does not supply the present full-matrix damped metric's selection law. [Primary paper](https://proceedings.mlr.press/v139/azulay21a/azulay21a.pdf).

**Gunasekar, Woodworth and Srebro, AISTATS 2021.** Section 2.4 gives the standard integrability test for a metric to be a Hessian in a fixed affine chart. Section 3 develops a potential-free flow view for general metrics. This directly occupies a broad claim that a positive metric need not admit an ordinary mirror potential. The paper's framework does not, by itself, calculate the present interpolant or its dependence on row weights. [Primary paper](https://proceedings.mlr.press/v130/gunasekar21a/gunasekar21a.pdf).

**Gunasekar, Lee, Soudry and Srebro, ICML 2018.** Section 2 studies interpolation bias for linear problems and distinguishes mirror descent, natural gradient, and other geometries. The Hessian-metric continuous-flow case gives the familiar Bregman-projection description. This establishes that loss-independent interpolation selection is an existing benchmark, rather than a new hypothesis introduced here. It does not turn an arbitrary non-Hessian flow into an equivalent mirror flow. [Primary paper](https://proceedings.mlr.press/v80/gunasekar18a/gunasekar18a.pdf).

**Han, Mishra, Jawanpuria and Gao, NeurIPS 2021.** Table 1 and Equation (2) identify the Bures–Wasserstein metric on SPD matrices as g_BW,X(U,V)=0.5 tr[S_X^{-1}(U)V], using the inverse Lyapunov operator. Hence, in these conventions, the candidate metric is exactly the sum of the Frobenius metric and 2 lambda times the BW metric. Their work analyzes optimization geometry and conditioning; it does not state the proposed fixed-loss endpoint separation. The BW component and its factorization interpretation are prior art. [Primary paper](https://papers.neurips.cc/paper_files/paper/2021/file/4b04b0dcd2ade339a3d7ce13252a29d4-Paper.pdf).

**Sabanayagam and Lucey, October 2026 preprint.** The inspected original abstract explicitly assumes balance and alignment and studies singular-mode learning laws; it states that damping restores slow phases below a spectral threshold. The depth investigator separately reports that Appendix E.9 already covers the relevant scalar joint Gauss–Newton reduction. Therefore scalar damping/depth dynamics should remain excluded. This review did not independently reread that appendix; the abstract alone does not establish coverage of the present noncommuting, underdetermined, fixed-loss endpoint comparison. [Primary record](https://arxiv.org/abs/2610.04249).

A bounded additional search covered combinations of Bures–Wasserstein, implicit bias, non-Hessian metrics, damping, matrix factorization, loss weighting, and metric perturbations. It found no directly matching small-lambda endpoint-selection law. This is not an exhaustive novelty certification. Search results about BW losses, JKO discretization bias, stochastic natural-gradient priors, or damping an inverse hypergradient concern different mathematical objects; they should not be used as if they proved this particular statement.

## Logical boundaries for the claim

1. A universal mirror-flow representation is a statement about an entire family of vector fields. An endpoint comparison is a statement about two particular autonomous flows. Failure of the first property does not logically imply failure of endpoint agreement.
2. Conversely, two verified distinct interpolants from identical initialization and identical constraints rule out any common **unique-minimizer** variational characterization for that constraint set, including a strictly convex common regularizer. They do not rule out every imaginable regularizer: a non-strict regularizer can tie both points, and a constant regularizer is vacuous.
3. Changing the measurement weights in an observed-output Gauss–Newton metric changes both the objective and the metric. That is a different experiment. The full-representation metric must remain explicit throughout any theorem or report.
4. On the commuting diagonal subspace the Lyapunov inverse is entrywise 1/(2 x_i). Then H has a separable Hessian potential, up to affine terms,

   \[
   \Psi(x)=\tfrac12\sum_i x_i^2+
   \tfrac\lambda2\sum_i(x_i\log x_i-x_i).
   \]

   This elementary positive control is not novel. It also explains why noncommuting matrix behavior must be separated from already occupied scalar-depth claims.
5. Nonzero Riemannian curvature, non-Hessian integrability, and noncommuting measurement matrices are different properties. None should be used interchangeably. Calling an ordinary endpoint response “Bures holonomy” would require specifying an actual connection and parallel-transport construction, which has not been done here.

## What a small-lambda result would add—and what is routine

Let A map a symmetric matrix to its observed measurements, and let P_N be the Frobenius orthogonal projection onto ker A. Because P_N grad L_w=0, the reduced equation immediately gives the exact identity

\[
 P_N(X_{\lambda,w}(T)-X_0)=
 -\lambda\int_0^T P_N S_{X_{\lambda,w}(t)}^{-1}
             [\dot X_{\lambda,w}(t)]\,dt.
\]

This is direct projection and integration of the ODE. If convergence, an SPD lower bound, and uniform tail control justify the limit and first-order expansion, it implies

\[
 P_N(X_{\lambda,w}(\infty)-X_{0,w}(\infty))
 =-\lambda\int_{\gamma_{0,w}}P_N S_X^{-1}[dX]+O(\lambda^2).
\]

The undamped endpoint is the common Euclidean projection, so the leading difference between the two weights is the difference of the two undamped path integrals. When appropriate, it can be expressed as the integral of a one-form around the loop formed by the two paths, and then by its exterior derivative over a surface. These steps are ordinary perturbation and differential-form calculus. The fact that such a formula can be written is not enough to claim novelty.

The work specific to this candidate is to establish that the coefficient is nonzero for the declared *fixed* losses, determine its size/sign or a useful class where it is nonzero, and justify the infinite-time remainder. Smooth dependence on lambda for every fixed finite time alone does not justify exchanging lambda->0 with T->infinity. A coefficient inferred only from three endpoint numbers also does not prove the expansion.

The numerical gaps supplied by the depth investigator are consistent with a nonzero first-order coefficient at small damping, but this literature review independently verifies neither their integration error nor the coefficient. The parallel mathematical review is the appropriate place for that verification.

## Retention recommendation

Retain the narrow statement as a **candidate mathematical example with quantitative evidence**, subject to the independent response calculation. Do not announce a new optimizer, a new metric, generic failure of mirror descent, generic loss-weight dependence, or an established publishable innovation. A theorem with explicit nonzero selection coefficient and controlled assumptions would be a concrete increment beyond the cited framework statements; its broader novelty and significance would still need assessment. If the final result is only the displayed routine projection identity plus one small numerical gap, stop at a documented observation rather than promoting it as a major new direction.
