# Curvature-sign screening in an exact residual-network slice

Decision: reject activation-concavity, fixed mixed-partial signs, or an initially favorable non-clock coefficient as sufficient stand-alone conditions for a persistent population benefit. A concrete constrained slice of the actual architecture supplies a sign reversal on one fixed population. More decisively, this slice's ostensibly adaptive metric reduces exactly to a fixed preconditioner plus a scalar clock. It therefore does not provide an irreducible learned-geometry mechanism or a new contribution.

This is a deterministic mathematical counterexample construction, not a new training sweep on the failed moons/rings gate. All 78 unlisted coordinates are frozen; no statement about the original all-80-parameter flow follows automatically.

## Exact embedding and metric

Let c=0.5*tanh(1). Keep one active neuron per block. The first block has A1=0, b1[0]=1, B1[1,0]=u. The second block has A2[0,1]=1, b2[0]=b, B2[0,0]=v. All other coefficients vanish. The only trainable original-coordinate indices are 24 and 56. For input (0,x), the original two-block residual map is exactly

    z(u,v;x) = (0.5*v*tanh(x+c*u+b), x+c*u).

The first coordinate is read out. On the single training input x=0 with label +1, write g=grad_(u,v) f(0). Its full two-coordinate Jacobian and summed damped metric are

    J = [[0.5*v*c*sech(s)^2, 0.5*tanh(s)], [c,0]],
    s = c*u+b,
    M = g*g^T + D,    D=diag(c^2+lambda,lambda).

The code verifies these predictions and both Jacobian rows against the original 80-parameter implementation, restricted to the two active columns.

Sherman-Morrison gives the exact identity

    M^(-1) g = D^(-1)g / (1+g^T D^(-1)g).

Consequently adaptive PB on this training sample has exactly the same continuous parameter orbit as nonlinear GD preconditioned by the constant matrix D. The remaining factor is a positive scalar clock. Comparing adaptive PB to frozen M0 can still yield different predictions, because M0 and D are different fixed preconditioners. Such a difference cannot, by itself, establish irreducible geometry feedback. The reduction is elementary, not innovative.

For the adaptive path, cancellation of the scalar loss derivative also yields

    v^2-v0^2 = [sinh(s)^2-sinh(s0)^2]/kappa,
    kappa = lambda*c^2/(c^2+lambda).

This first integral independently checks the numerical orbit.

## A fixed-population sign reversal despite fixed activation curvature

Use lambda=0.1, b=0.25 and (u0,v0)=(0,0.5). The fixed population has inputs x=0,-0.1,1 with respective probabilities 0.1,0.675,0.225; every label is +1. The training observation x=0 has positive probability under this same law. Thus no deliberately different train/test distribution is required. The example is an intentionally simple slice witness, not a realistic classification benchmark or an estimate of rings behavior.

Both flows are integrated using the common training logit r=f(0) as clock:

    dtheta/dr = P(theta)g/(g^T P(theta)g),

where P=M(theta)^(-1) for adaptive PB and P=M0^(-1) for its frozen-metric counterpart. Matching r exactly matches the scalar logistic training loss. All active tanh arguments on every population atom stay at least 0.15. Therefore tanh is concave throughout these trajectories, and the mixed derivative f_uv=0.5*c*sech(s+x)^2 remains positive. This is activation concavity and a mixed-partial sign condition; it is not joint concavity of f(u,v), whose Hessian need not be semidefinite.

| Matched training logit | Adaptive population risk | Frozen-M0 population risk | Adaptive minus frozen |
|---|---:|---:|---:|
| 0.1731828212 | 0.5979876522 | 0.5980994833 | -0.0001118311 |
| 3.0 | 0.04823031980 | 0.04822388829 | +0.00000643151 |

Thus a fixed activation-curvature sign and fixed positive mixed derivative do not preserve the benefit sign. The population and labels are unchanged between the two progress levels. This reverses an actual risk contrast, not merely a coordinate or parameter norm.

At initialization, the ordinary matched-logit Taylor calculation gives

    R_adaptive(r0+dr)-R_frozen(r0+dr)
       = -0.02378566969 * dr^2 + O(dr^3).

Independent small-progress ratios approach that coefficient: -0.0236320, -0.0237093 and -0.0237476 for dr=0.001,0.0005,0.00025. Hence an initially favorable transverse coefficient also does not ensure persistent benefit to r=3. Local continuity still guarantees the usual sufficiently small neighborhood when the leading coefficient is nonzero. That generic Taylor consequence is not the sought useful finite-progress theorem.

RK4 resolutions 0.002,0.001,0.0005 preserve both endpoint signs. The maximum train-logit matching error is 1.17e-11; the finest-grid error is below 4.5e-14. The exact adaptive path invariant agrees within 1.84e-10 at the coarsest grid and 8.6e-13 at the finest. The metric-reduction error is 2.22e-16. These are high-accuracy numerical counterexample checks, not interval-certified mathematics or a rigorous uniform discretization bound.

The first candidate was selected by an explicitly recorded bounded construction grid of 120 combinations. Every attempted combination, including failures and sampled curves, is retained in `curvature_search_full.json`. The selected case was then checked at three smaller resolutions. Selection is appropriate for finding a universal-claim counterexample; it supplies no statistical generalization or prevalence estimate. Historical `curvature_results.json` and its source snapshot are preserved; `curvature_results_v2.json` adds the local-coefficient verification.

## Closest geometric prior work

[Transtrum, Machta and Sethna (2011), Sections II, IV, VI-VIII](https://sethna.lassp.cornell.edu/pubPDF/SloppyGeometry.pdf) already interprets J^T J as the model-manifold metric, damping through a model graph, and optimization difficulties through parameter-effect and extrinsic curvature. It derives geodesic acceleration corrections for least-squares optimization. This is direct prior art against a generic claim that state-dependent pullback curvature explains deviations from initial linearization. It does not prove the present logistic held-out risk comparison or a sign criterion for label-dependent full-parameter learning.

[Transtrum and Sethna (2012), Section II, equation (11)](https://arxiv.org/pdf/1207.4999) derives the geodesic-acceleration correction -0.5*(J^T J+lambda D^T D)^(-1)J^T r'' by a small-curvature expansion. The generic adaptive-minus-frozen field derivative is not literally the same expression: DM[v]v contains both DJ[v]^T Jv and J^T DJ[v]v. Distinguishing those terms is ordinary differentiation, not a novelty result. Their optimization/convergence claims concern least squares and do not license a population-risk sign claim for this logistic setting.

## Independent review of the commuting-metric witness

The parent study's `commuting.py` uses original parameter indices 0 and 3 in A1, fixed B1[0,0:2]=(2,2), and an identity second block. On e1,e2, its logits are exactly (1+tanh u,tanh v). The restricted full metric is diagonal with entries sech(u)^4+lambda and sech(v)^4+lambda, so every metric commutes with M0. The mean-logistic gradient's factor 1/2 and the loss-drop clock -Pq/(q^T Pq) are correct. Nonproportional diagonal scaling can change direction even with an unchanged eigenbasis, so commutation does not imply a scalar clock.

The two population laws with masses (0.8,0.2) and (0.2,0.8) both assign probability 0.32 to an i.i.d. two-sample event with one observation of each atom. Thus the identical observed sample need not come from an artificial train/test shift. The resulting opposite population contrasts exclude a population-prior-free sign claim, while the same-data commuting metric comparison excludes eigenvector rotation as a necessary cause of non-clock differences. Both are elementary elimination examples. As with the present curvature slice, 78 frozen coordinates mean neither is a theorem about the full 80-coordinate training flow.

## Retained conclusion

No positive mechanism survives this lane. A future condition would have to control prediction transfer under the task law in addition to curvature, and must not silently assume the desired future population alignment. It must also rule out exact fixed-preconditioner/clock reductions such as the one above. Merely strengthening the local Taylor coefficient with a generic Lipschitz remainder, or selecting a helpful test-atom mixture after the fact, would fail the innovation standard.
