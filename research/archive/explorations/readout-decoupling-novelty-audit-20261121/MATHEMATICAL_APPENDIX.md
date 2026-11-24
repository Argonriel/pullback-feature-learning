# Theory audit: persistence of readout and unused-feature decoupling

Date: 21 November 2026. This is an analytical audit, not a training experiment. It reads the existing pivot memo and `tmp/rings-mechanism-20261120/stage2_local.md`; historical files are unchanged.

## Decision

**Retire the current candidate as a main novelty claim.** Uniformly small coupling plus stability yields finite-trajectory control by the resolvent identity and ordinary continuous-dependence estimates. Exact persistence on the proposed specially initialized architecture is block decoupling plus symmetry preservation. Most decisively, the headline combination of approximately unchanged predictive path length and a 49-fold reduction in unused-feature motion has an exact two-parameter linear analogue in which prediction changes only through the training clock.

An unexplained observation may still justify investigation, but neither a general persistence theorem with assumed uniform small coupling nor the displayed exact witness is a substantial new mechanism. A nontrivial architecture-specific claim would have to prove how low coupling is generated or maintained under relevant initialization/data conditions and establish a predictive consequence beyond time reparameterization. No such result is established here.

## 1. Fixed-state identities and the hidden-state distinction

Let f(theta) be the vector of predictive training outputs, h(theta) the unused outputs, A=Df, B=Dh, and g=grad_f L. Put

\[
 M_\gamma=A^TA+\gamma B^TB+\lambda I,\quad
 F_\gamma=-M_\gamma^{-1}A^Tg,\quad
 K_\gamma=AM_\gamma^{-1}A^T,\quad
 C_\gamma=AM_\gamma^{-1}B^T,
\]

with gamma>=0 and lambda>0. F_gamma is the parameter vector field, while V_gamma=A F_gamma=-K_gamma g is the predictive velocity. All derivatives in this paragraph hold theta, A, B, g and lambda fixed. Direct differentiation gives

\[
 \partial_\gamma K_\gamma=-C_\gamma C_\gamma^T,\qquad
 \partial_\gamma F_\gamma=M_\gamma^{-1}B^TC_\gamma^Tg,\qquad
 \partial_\gamma V_\gamma=C_\gamma C_\gamma^Tg. \tag{1}
\]

Thus small C makes the same-state predictive sensitivity quadratic in C, whereas the full parameter sensitivity can be only linear in C. If M is poorly conditioned, even that comparison is unfavorable. Hidden parameter drift can subsequently change A, B and g.

There is an important exact-case qualification. If C_gamma^T g=0, the parameter sensitivity also vanishes. Indeed, if s_gamma=M_gamma^{-1}A^Tg and B s_gamma=0, then M_beta s_gamma=A^Tg for every beta>=0, so the entire fixed-state update is identical for all weights. Conversely, C_gamma C_gamma^T g=0 implies C_gamma^T g=0 by taking its inner product with g. Therefore one must not describe an exact same-state predictive-null intervention as changing hidden parameters within this specific SPD family. The issue is approximate insensitivity or later departure from the initial zero-coupling set. The old stage-two proof already contains the identical-update observation.

For two weights alpha,beta, the resolvent identity yields

\[
 F_\alpha-F_\beta=(\alpha-\beta)M_\alpha^{-1}B^TC_\beta^Tg,
 \qquad
 K_\alpha-K_\beta=-(\alpha-\beta)C_\alpha C_\beta^T. \tag{2}
\]

These are exact at one common state. They do not compare operators evaluated at different learned parameter states.

## 2. What finite-trajectory control follows routinely

Assume a common region D contains both trajectories through time T, lambda>=lambda0>0, and on D the relevant weights satisfy

\[
 \|B\|\le B_*,\quad \|g\|\le G_*,\quad
 \|C_\alpha\|,\|C_\beta\|\le c.
\]

Suppose F_alpha is L-Lipschitz on the comparison region. Equation (2) gives a same-state perturbation bound

\[
 \|F_\alpha-F_\beta\|\le |\alpha-\beta|b,
 \qquad b=B_*G_*c/\lambda_0.
\]

For the same initial state, the integral equations and Gronwall's inequality imply

\[
 \|\theta_\alpha(t)-\theta_\beta(t)\|
 \le |\alpha-\beta|b\frac{e^{Lt}-1}{L}. \tag{3}
\]

The L=0 expression is the continuous limit |alpha-beta|bt. If f is A_*-Lipschitz on the same region, multiply the right side by A_* to control prediction differences. The region/stability assumptions must be established; otherwise this is only a bound up to the first exit time. Sampling a few common states does not prove them along either full trajectory.

For Euler updates with common nonnegative step sizes eta_k, the same argument gives

\[
 e_{k+1}\le(1+\eta_kL)e_k+\eta_k|\alpha-\beta|b,
\]

\[
 e_n\le |\alpha-\beta|b
 \frac{\prod_{k<n}(1+\eta_k L)-1}{L}
 \le |\alpha-\beta|b\frac{\exp(L\sum_{k<n}\eta_k)-1}{L}. \tag{4}
\]

These formulas are standard perturbation bounds, not new persistence theory. [Teschl, *Ordinary Differential Equations and Dynamical Systems*](https://www.mat.univie.ac.at/~gerald/ftp/book-ode/ode.pdf), Lemma 2.7 and Theorem 2.8, provides Gronwall and continuous dependence on the vector field; Theorem 2.11 treats parameter dependence.

The sensitivity equation makes the feedback explicit. For differentiable dependence on gamma,

\[
 \dot Z=D_\theta F_\gamma(\theta_\gamma)Z+
          \partial_\gamma F_\gamma(\theta_\gamma),\qquad Z(0)=0,
 \quad \partial_\gamma f(\theta_\gamma)=A(\theta_\gamma)Z. \tag{5}
\]

Bounding only the second equation in (1) after multiplying by A misses the transport term D_theta F_gamma Z. For example, if V_alpha is L_V-Lipschitz in theta, integration of (2) and (3) gives, for L>0,

\[
 \|f(\theta_\alpha(t))-f(\theta_\beta(t))\|
 \le |\alpha-\beta|\left[
 c^2G_*t+L_Vb\left\{\frac{e^{Lt}-1}{L^2}-\frac tL\right\}\right]. \tag{6}
\]

The second term is generally only linear in c. A purely quadratic output bound needs extra structure, for example a closed predictive dynamics V_gamma(theta)=v_gamma(f(theta)) with a uniform Lipschitz bound in f. Then an output-space Gronwall argument is available. Such closure says that states with the same predictive outputs have the same relevant predictive velocity; it is not automatic for nonlinear models.

Likewise, matching the predicted output-step norm changes the vector field by a gamma- and state-dependent scalar. Its regularity requires a denominator bounded away from zero and cannot be silently inherited from (3). Closeness on the training outputs alone also does not establish population-risk closeness. One needs an evaluation-map bound, uniform input control, or an independent statistical argument.

## 3. Zero initial coupling does not persist automatically

Consider the smooth two-parameter model

\[
 f(u,v)=u(1+v),\qquad h(u,v)=v,\qquad
 L=\tfrac12(f-1)^2,\qquad\lambda=1,
\]

initialized at (0,0). Here A=(1+v,u), B=(0,1), and C_gamma=0 initially for every gamma. All weights have the same first update and the same initial predictive velocity.

Nevertheless, with Euler step size one, the first iterate is (1/2,0), where coupling is nonzero. A second update gives exactly

\[
 \theta^{(2)}_0=(13/18,1/9),\quad f(\theta^{(2)}_0)=65/81;
\]

\[
 \theta^{(2)}_1=(25/34,1/17),\quad f(\theta^{(2)}_1)=225/289.
\]

The prediction difference is 560/23409>0. These are rational substitutions into the stated optimizer, not results from a training run.

The continuous flow shows the same effect without finite-step error. At zero time,

\[
 f'=1/2,\qquad f''=-1/4,\qquad
 f'''=1/8+\frac{1}{4(1+\gamma)}.
\]

Hence f_0(t)-f_1(t)=t^3/48+O(t^4). The hidden coordinate first separates at order t^2. The zero-coupling set at initialization is not an invariant set. This example only disproves automatic persistence; scalar monotone prediction may still be reconciled by a time change, so it is not evidence of a useful beyond-clock effect.

## 4. Equal path lengths do not identify predictive paths

A linear example already suffices. Let f(theta)=theta in R^2, h(theta)=theta_1, lambda=1, and at a common state let g=(1,1). Then

\[
 -F_0=(1/2,1/2),\qquad -F_2=(1/4,1/2).
\]

Choose each scalar learning rate so that its predictive displacement has norm one. The actual displacements are

\[
 (-1/\sqrt2,-1/\sqrt2),\qquad(-1/\sqrt5,-2/\sqrt5).
\]

Their path lengths are exactly equal, while their endpoints and directions differ. The distance between them is sqrt(2-6/sqrt(10))>0. No nonlinear approximation is involved, and these directions cannot be made equal by a scalar clock change. Thus similar cumulative norms in the archive cannot establish similar prediction trajectories, even before nonlinear complications are considered.

Conversely, a large change in unused motion need not indicate a change of predictive path at all, as the next example shows.

## 5. Exact linear reproduction of the 49-fold headline effect

The root review proposed the following counterexample; I independently checked its matrix inversion and conclusions. Let theta=(u,v), f=u, h=a u+b v, L=(u-1)^2/2, and start at zero. Then

\[
 M_\gamma=\begin{pmatrix}
 1+\lambda+\gamma a^2&\gamma ab\\
 \gamma ab&\lambda+\gamma b^2
 \end{pmatrix},
\]

\[
 D_\gamma=(1+\lambda)\lambda+
 \gamma[(1+\lambda)b^2+\lambda a^2],\qquad
 k_\gamma=\frac{\lambda+\gamma b^2}{D_\gamma}.
\]

The flow solves

\[
 u'=k_\gamma(1-u),\qquad
 v'=-\frac{\gamma ab}{D_\gamma}(1-u),
\]

so

\[
 u=1-e^{-k_\gamma t},\qquad
 v=-\frac{\gamma ab}{\lambda+\gamma b^2}u,\qquad
 h=\frac{a\lambda}{\lambda+\gamma b^2}u. \tag{7}
\]

For a nonzero a, at convergence the predictive path length is one for every gamma, while the ratio of unused path lengths at weight zero versus weight gamma is

\[
 1+\gamma b^2/\lambda.
\]

The same ratio holds at any matched predictive progress u. Taking lambda=0.1, gamma=9, b^2=8/15 gives exactly 49. At a common finite time the ratio also includes u_0(t)/u_gamma(t); it approaches 49 as both converge. All predictive curves in (7) are the same scalar path with different clocks. This does not reproduce every archived statistic, but it reproduces the purported distinctive mechanism without nonlinear representation learning. An exact finite-step comparison matched in predictive progress has the same cancellation because f and h are linear.

Therefore the archived 49-fold versus 0.04-percent path-length contrast is not, by itself, evidence of an unexplained architecture-dependent decoupling phenomenon. A linear preconditioned model is already a sufficient alternative explanation for that type of contrast.

## 6. The exact architecture witness is standard block and symmetry preservation

Under the pivot memo's restrictions, the two unused output rows are zero and the second residual block ignores its second input. Consequently B has support only on the unused output-row parameters, while A vanishes on those parameters. The damped metric separates these coordinates from the others, and their update is zero. On reflection-paired data, the objective and metric respect the specified parameter permutation/sign symmetry. A symmetric state has a symmetric update, so the paired parameters and forbidden cross-column remain constrained under common finite steps. This validates the sufficient witness in exact arithmetic, subject to its full paired-data and optimizer assumptions.

The reasoning is ordinary invariant-subspace preservation: if an update map U commutes with an orthogonal symmetry R, then R theta=theta implies R U(theta)=U(theta). A block with zero task gradient and no metric cross-block has zero update. Positive isotropic damping preserves these properties. The witness does not establish that generic random starts or unpaired minibatches approach this set.

The general equivariance-to-flow-invariance argument is reviewed in Section II.B of [Russo and Slotine, *Symmetries, Stability, and Control in Nonlinear Systems and Networks*](https://web.mit.edu/nsl/www/preprints/symmetry_pre2011.pdf). There is also close neural-network theory: [Nordenfors, Ohlsson and Flinth, *Optimization Dynamics of Equivariant and Augmented Neural Networks*](https://arxiv.org/html/2303.13458v5), Theorem 3.7 and Remark 3.8, treat exact preservation under their compatibility, embedding and augmentation assumptions; Proposition 3.13 derives tangential/transverse dynamics near the equivariant subspace. Their optimizer is not the present state-dependent pullback rule. Nevertheless, exact preservation and a nearby-subspace expansion are already established forms of argument.

An approximate version of the witness is also routine once its stability is assumed. Compare the perturbed dynamics to an invariant reference trajectory: an initial displacement delta and a uniform symmetry defect rho produce the usual delta exp(Lt)+rho(exp(Lt)-1)/L bound. A contracting transverse dynamics improves this to a bounded neighborhood. Proving contraction or persistence for the actual model and data could require serious work; simply assuming it and applying the bound is not that work.

## 7. What remains open, and the stop rule

The only nontrivial remaining hypothesis is that the actual nonlinear architecture and task produce a robust constraint on coupling evolution that is not explained by a linear preconditioner, matched training progress, or proximity to a deliberately invariant initialization. A useful result would need to derive that constraint from independently checkable architecture/data conditions and predict a consequence beyond the scalar clock. For example, a change transverse to the current predictive velocity is a relevant discriminator; its mere definition or one-state calculation is elementary.

The present evidence does not establish such a constraint. Large unused motion, small local C, similar accumulated readout norms, or an exact special-case witness are individually insufficient. Uniformly small C and stable dynamics yield only the standard result in Section 2. Accordingly, this audit recommends **not launching another broad experiment round under the current decoupling novelty claim**. Preserve the old findings as conditional empirical observations and the identities as technical facts; reopen the candidate only after identifying a specific claim that survives the linear mimic and standard stability reductions.

No new executable verifier was needed: the counterexamples and bounds are explicit algebra. No training, dataset evaluation, or modification of historical sources was performed.
