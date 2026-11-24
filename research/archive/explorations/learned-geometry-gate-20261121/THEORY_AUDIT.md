# Theoretical elimination audit for learned representation geometry

Prepared November 21, 2026. This audit supports the separately frozen minimal discrimination protocol. It does not revise that protocol, resurrect retired candidates, or claim an innovation. The author inspected repository AGENTS.md, core.py, the formal/representation/probe findings, and both previous novelty-rejection reports. Historical results and junctions were preserved.

## Decision and interpretation

The broad question of when learned geometry helps remains worth screening. It is not a contribution merely because nonlinear and tangent models differ. The four-arm gate can establish a residual beyond initial tangent preconditioning, but cannot attribute it to metric evolution. The nonlinear-predictor/fixed-initial-metric intervention is necessary for that attribution. Generic spectral whitening, nonlinear feature learning, and the proposed Taylor formulas below are not new results.

## Exact full tangent reduction

For every input x, including evaluation inputs, define the complete two-coordinate affine representation

    h_lin(x, delta) = h(theta0, x) + J0(x) delta.

The predictor is its first coordinate. Both coordinates remain linearized even though only the first appears in the logistic objective. With the mean representation gradient g, tangent GD uses -J0^T g and tangent PB uses -(J0^T J0 + lambda I)^(-1) J0^T g. The Jacobian and metric stay fixed for the entire trajectory. Evaluating the nonlinear network at theta0+delta would be the wrong comparator.

If A0 selects the predictive rows of J0 and M0=J0^T J0+lambda I, tangent PB is logistic optimization with fixed kernel K0=A0 M0^(-1) A0^T and the corresponding fixed cross-kernel at evaluation inputs. This is ordinary initial-feature preconditioning. Comparing against a probe on the initial two-dimensional representation does not perform this control.

For matched progress, the protocol's B=R_tangentPB-R_nonlinearPB measures nonlinear residual benefit. Its D interaction compares how much nonlinearity changes PB's advantage over GD. B alone can reflect generic model capacity. A positive D is stronger but still does not isolate metric adaptation. Nonlinear prediction with fixed M0 preserves current nonlinear gradients while suppressing preconditioner evolution; it is GD under a fixed reparameterization, not a new method.

Matching loss removes a scalar progress discrepancy, not every possible predictor difference. Step refinement is needed because discrete finite updates need not follow the same orbit as their flow. Missing target crossings must remain censored outcomes. If the geometry sample is replicated k times while the objective remains a mean, both sum-metric damping and PB learning rate must be multiplied by k to retain exactly the same update. Independent geometry inputs do not remove label dependence from J(theta) once theta has been fitted using labels.

## Primary-source boundaries

Amari et al., *When Does Preconditioning Help or Hurt Generalization?*, Sections 2-3, explicitly distinguish population Fisher from sample Fisher. Their principal risk calculations use time-invariant preconditioners in overparameterized least-squares regression. Damped sample-Fisher updates reach GD's minimum-Euclidean-norm interpolant there; population Fisher may reach another interpolant and has a distinct variance conclusion. Appendix A.3 considers estimating the population covariance with additional unlabeled inputs. Our complete representation metric is neither the logistic Fisher nor necessarily the predictive sample Fisher. Their conclusions defeat a broad preconditioning/bias-variance novelty claim but cannot be imported as a universal noise ranking for this model. [Primary text, version 4](https://arxiv.org/html/2006.10732v4).

Wadia et al., *Whitening and Second Order Optimization Both Make Information in the Dataset Unusable During Training, and Can Reduce or Prevent Generalization*, Theorem 2.1.1, assumes a dense isotropically initialized first layer. Section 3.3 connects exact Newton optimization to whitening for linear least squares and the wide linearized regime. Their experiments include regularized optimization. The paper already occupies the generic explanation that removing a spectral learning hierarchy can damage generalization. It does not give a universal ranking for finite damped nonlinear representation-pullback training. [Primary paper, ICML 2021](https://proceedings.mlr.press/v139/wadia21a/wadia21a.pdf).

These are scope comparisons, not claims that either paper proves an identical statement about the present residual architecture. Chizat, Arbel, and Yoshida-Nitanda are covered by the separate literature audit.

## Exact curvature counterexample

Use parameters (u,v), training representation

    h_train(u,v) = (u, a*u + b*v + c*u^2/2),

and read out u. Choose the prediction at an unseen input to be v. This deliberately arbitrary train/test map is an alternative baseline and a counterexample to universal claims. It is not an architecture theorem, an estimate of the residual network, or evidence about rings.

For any differentiable scalar training loss L(u), damped PB flow has

    M(u) = [[1+(a+c*u)^2+lambda, b*(a+c*u)],
            [b*(a+c*u),          b^2+lambda]],
    (du/dt,dv/dt) = -M(u)^(-1) (L'(u),0).

Where du/dt is nonzero, division cancels the loss derivative:

    dv/du = -b*(a+c*u)/(b^2+lambda).

Starting from zero, integration gives

    v_adaptive(u) = -b*(a*u+c*u^2/2)/(b^2+lambda),
    v_frozen(u)   = -a*b*u/(b^2+lambda),
    delta_v(u)   = -b*c*u^2/[2*(b^2+lambda)].

Thus the result holds for logistic loss at any attained finite u, without comparing unequal training progress. The numerical check uses the positive training label and the same positive unseen label. Changing c from -1 to +1 reverses the unseen logistic-risk effect. Both curvatures have exactly the same h0, J0=[[1,0],[a,b]], and initial metric spectrum. The benefit/harm difference cannot be predicted universally by those initial spectral quantities alone. The unseen target or readout could also reverse the interpretation, underscoring that geometry alone does not determine generalization.

The verifier independently integrates the actual logistic gradient-flow vector field with RK4 at step sizes 0.005 and 0.0025. It shortens the final actual RK4 step to attain u=1; it does not interpolate predictions. Its numerical checks compare the resulting v to the analytical relation, verify the loss target, and verify opposite signs of the unseen risk contrast. The full small JSON records include initial spectra, all endpoints, step counts, and source provenance.

### Separate exact noise expectation

For a transparent label-noise calculation only, replace logistic loss by squared loss to y=s+epsilon and run to its endpoint u=y. If E epsilon=0, then

    E delta_v = -b*c*[s^2+Var(epsilon)]/[2*(b^2+lambda)].

This is a different, explicitly labeled squared-loss endpoint calculation. It is not a claim about the limiting solution of noisy logistic classification. A two-point symmetric noise law verifies it by exact enumeration. It shows why zero conditional mean cannot eliminate noise after the label has changed the metric and trajectory. The identity is elementary; it is not offered as a novel mechanism theorem.

## Routine curvature baseline and a provisional hypothesis

Let M(theta)=J(theta)^T J(theta)+lambda I, P=M(theta0)^(-1), q=grad L(theta0), and v=-Pq. Adaptive-metric and fixed-M0 nonlinear flows initially have the same velocity. Differentiating their vector fields yields the acceleration difference

    a_adaptive - a_frozen = -P DM(theta0)[v] v.

The fixed metric and adaptive metric share the derivative of the nonlinear objective; it cancels in this difference. For q^T v != 0, removing the infinitesimal time correction needed to match training loss gives

    a_transverse = (I - v*q^T/(q^T v)) (a_adaptive-a_frozen).

These are ordinary Taylor and inverse-matrix differentiation identities. Before progress matching, write q=mu+xi with E[xi|X]=0 and conditional covariance C. Since DM is linear in its direction argument,

    E[DM[Pq]Pq | X]
      = DM[Pmu]Pmu + sum_ab C_ab DM[P e_a]P e_b.

The quadratic covariance contribution can remain nonzero. After progress matching, the projection itself depends on labels; the preceding simple covariance decomposition cannot simply be reused for that projected quantity. This is another reason not to carry an initialization independence argument into a fitted state.

The protocol's conditional C1 is an appropriately narrower hypothesis: early conditional-label-replica variation of learned geometry might predict the later sign of adaptive-versus-fixed-metric population-risk benefit beyond initial geometry and training progress. A successful version must freeze early available information, the decision rule, primary evaluation criterion, and independent validation seeds/noise levels. It must predict more than an initial-spectrum baseline or a direct population-gradient/update inner product. A general stability statement or this Taylor calculation alone would fail the novelty screen. No useful finite-progress condition or prospective predictor has been established here.

## Signed spectral bookkeeping and its limitations

For a fitted full representation Jacobian J=U Sigma V^T and representation-gradient g, the PB contribution of singular mode j to the population-risk derivative is

    -[sigma_j/(sigma_j^2+lambda)] (u_j^T g) (v_j^T grad R).

The corresponding instantaneous training-risk contribution is

    -[sigma_j^2/(sigma_j^2+lambda)] (u_j^T g)^2.

The training-only square has lost the population sign. A large training contribution need not be useful signal. Eigenvector sign changes cancel in each product, but rotations inside a repeated-eigenvalue subspace change individual-mode allocations. Exact multiplicities require subspace/projector sums; near multiplicities warrant stable clusters and sensitivity checks. Modes should not be tracked through eigenvalue crossings solely by sorted index. The total is the already known whole-update derivative identity, not a new forecast.

At a fitted state, J and its eigenspaces depend on the labels. Separating signal and noise requires an explicit conditional-label-replica construction or the known synthetic law with that dependence accounted for. A zero-mean noise claim conditioned on the original inputs alone is insufficient. Extra unlabeled geometry samples do not by themselves cure this dependence.

## Reproduction and retained status

Reusable script: `experiments/learned_geometry_theory_20261121/verify_curvature_example.py` in the repository. It needs only Python's standard library and refuses to overwrite an existing JSON. Run it with `--output <fresh-output-json>`. This audit's output is `curvature_example.json` in the same staging directory; the parent study will archive both files.

- Full-tangent preconditioning reduction: standard, used as an elimination baseline.
- Same initial spectra with either learned-metric effect: verified elementary counterexample.
- Zero-mean noise surviving learned geometry: verified exact toy expectation.
- A new finite-progress predictive mechanism for the actual architecture: not established.
- Whether the neural experiment justifies further investigation: decided by the separately frozen empirical gate, not by this toy.
