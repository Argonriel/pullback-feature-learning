# Audit addendum: the all-data optimal-face extension

Date: 2026-11-21. This addendum evaluates the root's strengthened candidate. It is not a substitute for the separate boundary proof review.

The root proposes removing strict feasibility from the convergence part: for every data vector, every positive damping and every compatible SPD start, the bounded factor flow reaches a PSD minimizer of the fixed quadratic loss. Its limit is the unique minimizer of the closed-PSD extension of the anchored Phi Bregman objective over the entire PSD optimal set. The limit has maximal support among that set. In the consistent PSD case the endpoint is weight-independent; in the inconsistent case the optimal measurement vector may depend on the quadratic weight.

This is more useful than merely restating the former interior theorem, but the novelty must still be scoped.

- The noncommuting resolvent metric and deterministic every-compatible-SPD-start quantifier remain outside the inspected Bah, Zhang–Peyré and PrecGD results for the same exact reasons documented in PRIMARY_AUDIT.md.
- PSD-only feasible boundary convergence is already present in Wu–Rebeschini Corollary 6 for commuting measurements. Thus “allowing a singular target” is not by itself an innovation.
- Convex boundary convergence in full Hessian metrics is covered by Alvarez–Bolte–Brahic Theorem 4.2. The ordinary conserved-normal-coordinate/KKT route to Bregman selection is established mirror geometry. The all-data optimal-face selection conclusion for this specific non-Hessian metric is the remaining distinction.
- Maximum support of an entropy minimizer is a standard consequence of the derivative of x log x at zero, not a newly discovered rank-selection principle. If a PSD optimum Z adds mass to a kernel direction of another optimum X, moving from X toward Z lowers an entropy-regularized objective at first order with derivative minus infinity.
- “All minimizers share the support face” must mean their ranges are contained in the range of the selected limit. It must not assert that all minimizers have equal support or equal rank.

One useful projection detail for the root proof is exact and dimension-free. If P is the kernel projector of the limit, then P belongs to J. If the Jordan projection of an ambient PSD optimizer Z has zero P trace, then
tr(PZ)=tr(P P_J Z)=0.
Since Z is PSD, this implies ZP=0. Thus the support inclusion passes from Jordan-projected optimizers to ambient optimizers without requiring individual rank-one projectors to belong to J.

The strengthened claim should be described as **a retained but narrow optimal-face convergence and endpoint-selection theorem**. It should not be decomposed into separate “new” global convergence, maximum-rank selection, Bregman projection and entropy-barrier mechanisms. No theorem-level direct match was located in the inspected sources; this is not a priority proof.

