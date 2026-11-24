# Joint asymmetry novelty test and change of research direction

Date: 21 November 2026.

## Decision

**Retire simultaneous input and posterior asymmetry as the main innovation claim.** The existing formula is mathematically useful, but the claimed scope is already inside approximate invariance of a joint supervised distribution, and the interaction penalty follows from a standard total-variation support calculation. Independent numerical checks reproduce this reduction. No new large experiment is justified by this claim.

This decision does not assert that an earlier paper prints our identical formula or proves our exact batch-quota certificate. It distinguishes a correct specialization from a substantial new scientific contribution. Historical theory and experimental archives remain valid within their stated conditions and are preserved.

The research focus changes to a provisional mechanism question in the original representation-optimization project: **when can changing the penalty on features unused by the readout substantially alter representation motion while barely changing predictive learning, and when does nonlinear training break that decoupling?** This is a new priority for investigation, not a newly established innovation.

## The decisive prior-art finding

Chen, Dobriban and Lee explicitly define a supervised observation to include both features and outcome on printed page 7. Approximate invariance in Section 6 therefore concerns their joint law. Lemma 15 decomposes averaging effects into a convexity benefit and a distribution-discrepancy term; Theorem 17 gives an approximate-invariance excess-risk result under bounded-loss and Lipschitz assumptions. It does not require the feature marginal or conditional label law to be separately exactly invariant. Its displayed theorem is not our particular log-loss/quota certificate, but it defeats the broad claim that allowing both defects is a new scope. [Primary paper](https://jmlr.org/papers/volume21/20-163/20-163.pdf).

Worst-case expectations under total-variation constraints and their signed-mass/oscillation formulations also have established literature. The 2014 paper by Charalambous et al. supplies this general mathematical precedent, with a different TV normalization and ambiguity-set formulation. The exact free-pair specialization below is derived here rather than attributed to an identical printed equation. [Primary paper](https://www.site.uottawa.ca/~sloyka/papers/2014/Charalambous.Tzortzis.Loyka.14.pdf).

## Reduction 1 Both defects are a joint distribution defect

Let the full law be Q of (X,Y), and let T act on X while leaving the label coordinate unchanged. On a two-point orbit, write its orientation probabilities as

    a=(1+r)/2, b=(1-r)/2, -1<=r<=1,

and its two label posteriors as p and q. The orbit law is (a p,b q); its transformed law is (b q,a p). With TV(p,q)=||p-q||_1/2,

    TV((a p,b q),(b q,a p)) = ||a p-b q||_1
      <= |r| + (1-|r|) TV(p,q).

For r>=0, use a p-b q=r p+b(p-q); the negative case follows by interchanging orientations. Thus the two imperfections are not outside a joint-distribution formulation. Local bounds on them can carry stronger information than a single integrated discrepancy, but that is additional structure, not a previously excluded kind of noninvariance.

For a fixed logit predictor z, let m=(z+zT)/2, and let Qbar=(Q+T#Q)/2. The original-versus-averaged log-loss gain is the sum of its Jensen gain under Qbar and a signed expectation under Q-Qbar. This is an exact change-of-measure identity; it does not rely on either symmetry being exact. A label-free, orbit-even selection mask can be included. For IID records, condition on all other records and independent tie keys, and require that each record's selection mask be unchanged when that record alone is transformed. Fixed even-score top-quota selection satisfies this condition; a quota alone or whole-batch invariance does not.

## Reduction 2 The exact interaction is a small linear optimization problem

Set d=(z-zT)/2, S=max(d)-min(d), b0=(max(d)+min(d))/2, K=(F(z)-F(zT))/2, t=K-b0, and F=log-sum-exp. The label-dependent support problem is

    sup_{p,q in simplex, TV(p,q)<=epsilon}
        [(1+r)p-(1-r)q] dot d / 2,      0<=epsilon<=1.

Any feasible pair has a common subdistribution of mass 1-epsilon. Hence it can be written

    p=(1-epsilon)c+epsilon u,
    q=(1-epsilon)c+epsilon v,

with c,u,v probability vectors. The objective becomes

    r(1-epsilon)c dot d + epsilon[a u dot d-b v dot d].

Maximize u at a largest coordinate, v at a smallest coordinate, and c at the largest or smallest coordinate according to the sign of r. This both bounds and attains the maximum:

    r b0 + [|r|+(1-|r|)epsilon] S/2.

Subtracting rK yields exactly the old penalty

    h_epsilon(r)=[|r|+(1-|r|)epsilon] S/2-r t.

The joint coefficient |r|+epsilon-|r|epsilon improves on separately adding |r| and epsilon, but the cross-term follows from the same common-mass decomposition. It is not a new optimization principle. The function is convex in r, so the interval worst case is at an endpoint. Independent estimation, union bounds, and standard concentration complete the existing certificate without changing this novelty judgment.

## Independent numerical verification

The verifier constructs the feasible polytope from probability nonnegativity, simplex equalities, and all subset inequalities for total variation. It enumerates intersections of active constraints and retains feasible vertices. It does not optimize by using the displayed closed form. The final simplex coordinate of each posterior is eliminated algebraically. The computation is floating point, with explicit feasibility tolerances; the algebraic derivation supplies the proof.

| Check | Scope | Result |
|---|---|---|
| Exact support and gain extrema | 300 two- and three-class cases; both signs and endpoints of r; epsilon from zero to one | Maximum formula-versus-vertex error 1.78e-15 |
| Joint-law identity and posterior decomposition | 250 random finite joint laws, four orbits each, arbitrary unequal posteriors and orbit masses | Maximum identity error 4.44e-16 |
| Shared-posterior constraint example | Two equal-mass orbits, opposite antisymmetric logits, common posterior pair | Pointwise lower bound -0.07988549; actual feasible gain +0.12011451 |

The final example demonstrates conservatism, not an invalid certificate. With r=0 and epsilon=0.4, each orbit has Jensen gain log(cosh(0.5)). The posterior terms cancel when the same p and q apply to both orbits. Independently optimizing them orbit by orbit loses that cancellation and subtracts 0.2. Therefore local sharpness cannot be promoted to global sharpness with shared constraints. Solving a joint constrained optimization instead is not, by itself, a new statistical method either.

The random checks are reproducible numerical verification, not new evidence that an averaging algorithm improves a real prediction task. They do not prove a historical priority claim.

## Why adding a validation layer does not rescue the innovation

For a frozen candidate, directly evaluate the paired loss difference against the original predictor on independent labeled deployment data. This requires no label-preserving interpretation of T: both predictions are evaluated against the original observed label. With valid range or tail assumptions and appropriate batch sampling, the difference can enter existing risk tests. Learn then Test already controls selection from such tests. Semi-supervised risk control can combine a surrogate mean with labeled corrections, with efficiency depending on surrogate quality. [Learn then Test](https://arxiv.org/html/2110.01052); [Semi-Supervised Risk Control](https://arxiv.org/html/2412.11174v2).

These methods do not automatically establish our exact local bound. They show why combining a standard local bound with held-out acceptance, or adding unlabeled data, does not independently create a new framework. More restrictive structural knowledge must be charged or supplied equally to a fair comparator.

## The replacement question

The most specific unexplained observation in the original archive concerns the unused representation coordinate. Changing its metric weight from 0 to 9 reduced its mean path length about 49-fold while the mean path length of the readout coordinate changed by about 0.04 percent. Population-risk differences at the primary endpoint were inconclusive. Similar path lengths do not mean identical paths, and inconclusive risk contrasts do not establish equivalence. [Existing mechanism study](C:/Users/liwei/Documents/ChatGPT/277/tmp/rings-mechanism-20261120/README.md).

The new question is whether there is a persistent, architecture-dependent decoupling mechanism that explains this effect and predicts its breakdown. The immediate matrix identities are already understood: at a fixed parameter state, hold readout Jacobian A, unused-feature Jacobian B, and damping lambda>0 fixed while varying gamma>=0,

    M_gamma=A^T A+gamma B^T B+lambda I,
    K_gamma=A M_gamma^{-1} A^T,
    dK_gamma/dgamma=-C_gamma C_gamma^T,
    C_gamma=A M_gamma^{-1} B^T.

This is a partial derivative at a fixed state; differentiating across nonlinear training trajectories also introduces changing-Jacobian terms. The previous local study already derived the corresponding training-improvement derivative. Re-deriving it will not count as an innovation. The unresolved part concerns persistence along finite nonlinear trajectories, a prediction about when coupling develops, and separation from a change in learning rate or training clock.

This direction must also face existing natural-gradient and Gauss-Newton work. Arbel et al. already study feature-learning regimes and hidden useful features; Buffelli et al. show that tangent-kernel changes need not correspond to useful representation changes in their published study. Thus neither a changing representation nor a better probe is enough. [Arbel et al., NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/6a14c7f9fb3f42645cfa6bd5aa446819-Paper-Conference.pdf); [Buffelli et al., NeurIPS 2024 final proceedings](https://proceedings.nips.cc/paper_files/paper/2024/hash/f144ab9985c739a5091ec188a2688644-Abstract-Conference.html).

Two other alternatives were screened and not promoted: generic cheap-score routing is close to existing conditional-decision/deferral theory, and generic adaptive curvature refresh is already covered by lazy second-order methods and recent adaptive damping/refresh work. [Learning to Defer](https://proceedings.mlr.press/v119/mozannar20b.html); [Lazy Hessians](https://proceedings.mlr.press/v202/doikov23a.html); [FOAM](https://arxiv.org/abs/2606.02365).

## Research status

- Joint-asymmetry main-innovation candidate: rejected as insufficiently distinct.
- Existing conditional bound and historical experiments: retained with their original scope.
- Replacement priority: finite-trajectory coupling between unused-feature geometry and predictive learning.
- Replacement novelty: unresolved; no new method or theorem is claimed.
- New training or dataset evaluation in this review: none.
- New work completed: source inspection, explicit reductions, independent finite-dimensional checks, a constraint counterexample, and a screened change of research priority.

The next research gate is to establish a finite-trajectory persistence or breakdown mechanism, compare it against prior work, and derive a prediction beyond changing the step length or training clock. If the proposed mechanism reduces to the already known one-state derivative or a standard invariant-subspace example, this replacement must also be retired before a large experiment.

Supporting files contain consolidated independent-review findings, numerical records, and the executable verifier. See README.md for reproduction.
