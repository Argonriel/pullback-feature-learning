# Geometry-only poisoning: original-literature and reduction audit

Date: November 21, 2026 (America/Los_Angeles).

**Decision: reject the generic direction as a new research contribution.** A separate unlabeled geometry sampler is a meaningful software trust boundary, and the bounded search did not locate a paper with precisely that interface. Nevertheless, the proposed mathematical effect—changing an inverse metric while holding the supervised gradient fixed—is completely explained by elementary matrix perturbation. Neither a new name nor a visible slowdown/rotation would overcome this reduction. No neural network was trained.

## Proposed threat model and the distinction that was checked

The supervised data and labels remain fixed. Only the inputs used to form the representation pullback metric are replaced or appended. At one parameter value, write

\[
 q=\nabla L(\theta),\qquad
 M_G=\lambda I+\mathbb E_{x\sim G}[J_\theta(x)^TJ_\theta(x)],\qquad
 \dot\theta=-M_G^{-1}q,\quad\lambda>0.
\]

The attacker can change the geometry distribution G but cannot change L or q directly. This is different from ordinary feature/label poisoning, because poisoned inputs need never enter the supervised objective. It is also different from changing the loss Hessian or from inserting a backdoor into supervised examples. It is not different algebraically from contaminating a second-moment estimate that is inverted by a downstream algorithm.

For a linear representation h_theta(x)=theta^T x, J_theta(x)=x^T. Therefore arbitrary geometry feature vectors in the two-dimensional audit below are actual Jacobian rows; they are not fictitious matrices that a model could never realize. This simple realization is enough to reject a broad existence claim.

## Original literature read before the minimal check

1. **Rubinstein et al., ANTIDOTE: Understanding and Defending against Poisoning of Anomaly Detectors, IMC 2009.** The original paper studies variance injection into the unlabeled traffic data used by a PCA detector, distorting its learned subspace, and defenses based on robust statistics. Sections 3–4 were checked, including the covariance explanation and constrained chaff optimization. It does not have a separate supervised-gradient stream, but already establishes the covariance/subspace contamination mechanism and robust-estimation response. [Author-hosted original PDF](https://people.eecs.berkeley.edu/~tygar/papers/SML/IMC.2009.pdf).

2. **Liu, Carr and Sun, Empirical Perturbation Analysis of Linear System Solvers from a Data Poisoning Perspective, 2024.** Sections 4–5 define feature perturbations in a least-squares data matrix, including a label-free condition-number objective. The perturbed features affect the fitting problem, so this is not the same separate-sampler threat model. It nevertheless directly occupies the broad framing of adversarial matrix perturbations causing solver sensitivity or slower optimization. This audit does not rely on its quantitative convergence theorems. [Original arXiv full text](https://arxiv.org/html/2410.00878v1).

3. **He et al., Deferred Poisoning: Making the Model More Vulnerable via Hessian Singularization, arXiv 2024 / AAAI 2026.** The original arXiv text, Section III and Algorithm 1, was read. Its perturbed examples enter the supervised cross-entropy update; an input-Hessian penalty shapes later vulnerability while attempting to preserve clean validation behavior. The distinction from a geometry-only sampler is real. Consequently it must not be cited as an identical method, but a generic claim that poisoning can manipulate curvature or remain invisible on clean validation is already occupied. [Original arXiv full text](https://arxiv.org/html/2411.03752v1), [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/download/39318/43279). The arXiv and final publication author lists differ; the argument here refers to the explicitly inspected arXiv version.

4. **Diakonikolas et al., Robust Estimators in High Dimensions without the Computational Intractability, FOCS 2016 / SIAM J. Comput. 2019.** The original abstract/problem statement allows an adversary to corrupt an epsilon fraction of samples and develops efficient high-dimensional robust distribution estimation. It is foundational evidence that generic contamination resistance is already a substantial field. It is not a plug-in theorem for arbitrary, parameter-dependent neural Jacobian distributions: its distributional assumptions cannot be silently dropped. [Original arXiv record](https://arxiv.org/abs/1604.06443), [author/institution manuscript](https://www.cse.wustl.edu/~bjuba/cse519t/f19/papers/Diakonikolas16-conf.pdf).

The rank-one inverse identity has a historical original source in Sherman and Morrison, *Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix*, Annals of Mathematical Statistics 21(1), 124–127 (1950), [DOI](https://doi.org/10.1214/aoms/1177729893). The publisher viewer did not expose its full text to the web reader. The identity and all claims used below are independently derived and numerically checked, rather than inferred from an inaccessible article.

Searches also included the exact phrases `poisoning preconditioner`, `poisoning natural gradient`, `poisoning Fisher matrix`, `poisoning covariance whitening`, and adversarial curvature/preconditioning combinations. Search absence is not proof of literature absence. No secondary summary was used to establish a technical novelty claim.

## Reduction 1: rank-one suppression and rotation are exact old algebra

Let M be positive definite and add one geometry contribution alpha aa^T, alpha >= 0. Define p=M^{-1}q, so the optimization update is -p. Direct multiplication gives

\[
 p_\alpha=p-
 \frac{\alpha M^{-1}a\,a^Tp}{1+\alpha a^TM^{-1}a}.
\]

The change in instantaneous supervised progress is exactly

\[
 q^Tp-q^Tp_\alpha=
 \frac{\alpha(a^Tp)^2}{1+\alpha a^TM^{-1}a}\geq0.
\]

The direction can rotate because one component is suppressed more strongly than another. In the M-whitened coordinates this is simply a rank-one contraction. An arbitrarily large-norm geometry row can dominate an empirical second moment; a dramatic example with no norm or leverage restriction therefore tests an already understood lack of robustness. It does not establish a new learned-geometry mechanism.

At every nonstationary point q^T p_alpha > 0. A positive-definite metric alone cannot turn the infinitesimal update into ascent on the same supervised objective. This does not protect finite steps from overshoot, nor protect population risk, nor establish a common path or limit in nonconvex or underdetermined problems.

## Reduction 2: even unrestricted target-direction control is elementary

Suppose the proposed objective is to produce a target descent vector p by adding an arbitrary PSD geometry matrix B. Then Bp=q-Mp=:r. If r is nonzero, existence requires p^Tr>0; PSD Cauchy–Schwarz rules out nonzero r when p^Tr=0. Conversely, when p^Tr>0,

\[
 B=\frac{rr^T}{p^Tr}\succeq0
\]

satisfies (M+B)p=q exactly. Thus, aside from the unchanged solution, unrestricted achievable vectors satisfy p^TMp<p^Tq and require only one rank-one addition. This is a direct PSD secant construction, not a new control theorem. A finite-dimensional example of targeted rotation under arbitrary Jacobian rows would fail the novelty gate as well.

## Reduction 3: constant-size contamination and damping

Appending geometry rows while retaining a sum is different from replacing a fixed fraction of an averaged geometry batch. For a normalized replacement model,

\[
 M_\varepsilon=\lambda I+(1-\varepsilon)C_0+\varepsilon C_A,
 \quad \Delta=M_\varepsilon-M_0=\varepsilon(C_A-C_0).
\]

The difference includes a removal term, so monotone suppression relative to M_0 is not guaranteed. The correct identity is

\[
 p_\varepsilon-p_0=-M_\varepsilon^{-1}\Delta M_0^{-1}q.
\]

If each allowed Jacobian row has norm at most R in this scalar-representation example, both C_0 and C_A lie between 0 and R^2 I. Hence ||Delta||_op <= epsilon R^2 and

\[
 \|p_\varepsilon-p_0\|\leq
 \varepsilon R^2\|q\|/\lambda^2.
\]

This is a deliberately crude resolvent bound, not a sharp or novel robustness guarantee. A practical high-leverage cap or robust second-moment estimator would need evaluation, but those generic remedies cannot be presented as a new principle.

Finally, because M_G is invertible, the stationary points are exactly the stationary points of the unchanged L. For a strongly convex objective with a uniformly bounded metric and suitable continuous-time regularity, convergence cannot produce a different unique minimizer merely by changing G. A finite-training-budget slowdown is a different claim from a persistent backdoor. Nonconvex basin selection, underdetermined implicit bias, or loss of uniform bounds require separate analysis; none is shown here.

## Minimal verification and its limits

Source: `experiments/new_direction_screen_20261121/poisoning.py` in the repository. Output: `poisoning_checks.json` alongside this note. The script uses NumPy in float64, contains no training loop, and refuses to overwrite its output file.

The check was run only after the original-paper comparison above. It verifies the algebra rather than collecting a favorable model endpoint:

| Check | Result |
|---|---:|
| Rank-one inverse prediction versus direct solve, maximum absolute error | 1.11e-16 |
| Exact supervised-progress loss | 0.08130579098 |
| Progress identity error | 1.94e-16 |
| Rank-one example direction change | 11.7886 degrees |
| Fixed-size mixture: poison fraction | 0.02 |
| Fixed-size mixture: poison-row norm | 30 |
| Mixture direction change | 41.7603 degrees |
| Mixture supervised progress, clean / poisoned | 1.818181818 / 0.978336827 |
| Mixture resolvent identity error | 1.11e-16 |
| Rank-one target construction error | 2.78e-17 |
| Unchanged quadratic minimizer, both update norms | below 1e-16 |

The 41.76-degree rotation is intentionally a large-leverage example, not evidence of a subtle, realistic, or novel attack. Every measured effect is explained to floating-point precision by the elementary predictions. This is an exclusion result, not a general robustness result for neural PB.

## Relation to previous exclusions and stopping rule

The local archive already retired unused-coordinate penalties and readout decoupling as a novelty claim after fixed-preconditioner/clock and ridge-head controls. Merely calling such a metric contribution an attacker would recreate the same evidence with new terminology. This audit therefore does not launch another PB-versus-GD training comparison.

The only possible residual question is a genuinely restrictive one: characterize which *bounded, semantically valid input perturbations* can realize a targeted parameter-dependent Jacobian second moment, and prove an effect not explained by leverage, damping, a fixed metric, or ordinary implicit bias. Neither a literature separation nor a nonroutine theorem for that restricted problem has been established here. It is not retained as a promising result. Reopening would require those ingredients before spending training compute.
