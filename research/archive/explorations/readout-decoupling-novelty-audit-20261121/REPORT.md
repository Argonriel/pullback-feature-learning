# Readout and unused-feature decoupling: novelty audit

Date: 21 November 2026.

## Decision

**Retire the current decoupling candidate as the main innovation.** It has not passed the same elimination standard applied to joint asymmetry. Broad representation/function dissociation and symmetry-preserved dynamics have close prior art. The proposed local identities, exact persistence construction, and generic trajectory perturbation argument reduce to established mathematics. Most decisively, a two-parameter linear model reproduces the motivating 49-fold path reduction without a new nonlinear mechanism.

This is a research-priority decision about the contribution currently available. It is not a theorem that every architecture-specific trajectory result is impossible or already published. Nor does the linear example prove that all differences in the original rings models are a training-clock effect. The retrospective audit finds nonzero predictor differences and ordinary local compensation; it does not identify a complete causal explanation of those trajectories.

No additional neural-network training, validation evaluation, held-out evaluation, or population-risk integration was performed. New computations comprise a small analytical-model simulation and diagnostics on stored training inputs and checkpoints. Historical files were checked and preserved.

## The same elimination standard

The candidate was split into four possible contributions rather than treated as an open-ended topic:

| Proposed contribution | Test | Judgment |
|---|---|---|
| Representations can change substantially while predictions remain similar | Check existing representation/function dissociation work | Already established broadly |
| Unused-feature weighting has an exact local coupling formula | Reduce the formula to matrix calculus and prior project work | Routine; already partly proved in this project |
| Exact or approximate decoupling persists over training | Check symmetry invariance, perturbation theory, and the proposed assumptions | Exact witness is elementary; generic uniform-control theorem is routine; a distinctive applicable theorem is absent |
| The 49-fold observation reveals a new nonlinear mechanism | Construct a simple alternative and inspect actual predictions | Simple alternative reproduces headline; archived predictions are not identical |

No large experiment is warranted by any of these formulations. An unexplained aspect of an experiment is not, by itself, a demonstrated innovation.

## Closest primary sources and their actual limits

- **Braun, Grant and Saxe, ICML 2025:** analytically characterize functional/representational dissociation in two-layer linear networks, with nonlinear illustrations. Theorem 3.3 and Corollary 3.9 in the later author manuscript describe solution freedom and representational geometry. This defeats the broad dissociation claim, not an exact theorem about our weighted optimizer. [Published record](https://proceedings.mlr.press/v267/braun25a.html), [author manuscript inspected](https://arxiv.org/html/2609.38998v1).
- **Nordenfors, Ohlsson and Flinth, manuscript v5, 2024:** Theorem 3.7 and Remark 3.8 preserve the equivariant subspace under gradient flow and deterministic finite-step GD under compatibility and parameterization assumptions. Proposition 3.13 gives tangential/transverse dynamics with quadratic remainder terms near that subspace. The assumptions involve augmented objectives, commuting projections, and partially unitary embeddings. It is not literally a theorem for our state-dependent metric, but the generic exact-symmetry/nearby-decoupling narrative is already occupied. [Primary text](https://arxiv.org/html/2303.13458v5).
- **van Oostrum, Mueller and Ay:** distinguish invariance of the pushed-forward natural-gradient direction from parameter-space invariance and finite-step trajectory invariance. Their result concerns reparameterization; positive damping and our change of representation metric are not covered automatically. It prevents treating the distinction between local and finite invariance as a new discovery. [Primary text](https://tore.tuhh.de/dspace-cris-server/api/core/bitstreams/30f20c71-2c2f-44bd-b5b7-67172f7118fd/content).
- **Huang et al., September 2026 preprint:** Hessian Null Space Continuation changes representations through function-preserving exploration and explicitly addresses higher-order drift with restoration and relinearization. It works around trained anchors and finite probe sets, not our ordinary supervised metric-intervention trajectory. [Primary text, Sections 2.1-2.2 and 7](https://arxiv.org/html/2609.38081v1).
- **Mueller and Montufar, online 2023 / journal issue 2024:** Definition 4 and Theorem 5 formulate natural gradients for a chosen model-space factorization and arbitrary Riemannian geometry. The metric choice need not be determined uniquely by the objective. This supplies the general framework, not a claim of invariance to changing that metric. [Primary text](https://link.springer.com/article/10.1007/s41884-023-00106-z).

The audit also compared the previously identified natural-gradient and Gauss-Newton literature. No identical published theorem for the complete original residual-network intervention is claimed. The rejection rests on specific overlap plus explicit reductions and insufficient distinctive evidence, not on an asserted exhaustive literature search.

## A linear model reproduces the motivating phenomenon

Let parameters be (u,v), the predictive output be f=u, the unused feature be h=a*u+b*v, and the loss be (u-1)^2/2. Use exactly the same form of damped weighted pullback metric:

    A=[1,0], B=[a,b], M_gamma=A^T A+gamma B^T B+lambda I.

For gradient flow from (0,0), direct inversion of the 2-by-2 system gives

    D_gamma=(1+lambda)*lambda
            +gamma*((1+lambda)*b^2+lambda*a^2),
    k_gamma=(lambda+gamma*b^2)/D_gamma,
    u(t)=1-exp(-k_gamma*t),
    v(t)=-gamma*a*b/(lambda+gamma*b^2)*u(t),
    h(t)=a*lambda/(lambda+gamma*b^2)*u(t).

All predictive trajectories become identical after the time change tau=k_gamma*t. At convergence their predictive path length is one; the unused path is a*lambda/(lambda+gamma*b^2) for a>0. The reduction relative to gamma=0 is exactly

    1+gamma*b^2/lambda.

Choose a=1, lambda=0.1, b^2=8/15, and gamma=9: the ratio is 49. These are deliberately selected illustrative coefficients, not estimates fitted to the rings data.

Direct numerical updates with step size 0.05 for 500 steps reproduced this behavior:

| Quantity | Result |
|---|---:|
| Unused-path reduction, gamma 0 versus 9 | 49.00000011-fold |
| Relative predictive-path difference | 2.28e-9 |
| Maximum difference from exact discrete solution | 3.55e-15 |
| Readout difference at matched continuous clock | 0 within floating-point evaluation |

The exact discrete solution is u_n=1-(1-eta*k_gamma)^n. The other coordinates retain their displayed linear relationships. This is weighted least squares and a scalar clock change; it requires no nonlinear feature-learning mechanism. Therefore the path-length headline cannot distinguish a new nonlinear explanation from a basic alternative.

## The standard mathematics already available

At a fixed parameter state, hold readout Jacobian A, unused-feature Jacobian B, task gradient g, and lambda>0 fixed. Set

    M_gamma=A^T A+gamma B^T B+lambda I,
    K_gamma=A M_gamma^-1 A^T,
    C_gamma=A M_gamma^-1 B^T.

Then partial_gamma K_gamma=-C_gamma C_gamma^T. The previous project study already derived the corresponding predicted-improvement derivative q'(gamma)=-||B M_gamma^-1 A^T g||^2. This is neither new mathematics nor a new project result.

There is also a crucial limitation of the earlier exact persistence witness. If B M_0^-1 A^T g=0 at a state, then M_gamma^-1 A^T g=M_0^-1 A^T g for every gamma. If this holds throughout a shared trajectory, the entire parameter update sequence is identical, including unused-feature motion. The previously constructed symmetric witness has zero unused motion. It cannot explain substantially different nonzero paths.

For near-decoupling, the parameter vector field F_gamma=-M_gamma^-1 A^T g satisfies

    partial_gamma F_gamma=M_gamma^-1 B^T C_gamma^T g.

Uniform bounds on this derivative and on the state Lipschitz constant give an ordinary Gronwall estimate. For example, if ||F_alpha-F_beta||<=delta and F_alpha is L-Lipschitz on a common trajectory domain, equal initial states yield

    ||theta_alpha(t)-theta_beta(t)|| <= delta*(exp(L*t)-1)/L,

with delta*t when L=0. Discrete updates admit the corresponding product bound. These assumptions must hold along the relevant domain; small coupling at initialization does not supply them. A useful new result would need to derive persistent control from architecture, data, and initialization, not assume the desired stability throughout training.

## What the stored models actually show

The audit reconstructs 16 original training sets, verifies 288 stored parameter-state hashes and 384 logical checkpoint records, and reproduces their training losses exactly. It computes 480 within-seed endpoint contrasts and 576 local diagnostic records at 144 stored/initial states. These are descriptive, retrospective diagnostics; no new confidence interval or population claim is made.

The primary descriptive comparison is native-rate, full-pullback, step 500, gamma=0 versus gamma=9:

| Measurement | Result |
|---|---:|
| Ratio of mean unused-coordinate path lengths | 49.1046 |
| Relative difference of mean predictive path lengths | +0.0379% |
| Mean training-logit RMS difference | 0.1294 |
| Mean relative training-logit RMS difference | 4.97% |
| Mean absolute class-probability difference | 0.00642 |
| Largest individual probability difference | 0.40538 |
| Classification disagreements on paired training records | 9 / 4,096 |

The average probability change is small, but not uniformly small across records. Neither these numbers nor the old inconclusive risk intervals establish prediction equivalence over an entire trajectory. The approximately equal mean path lengths are also aggregate quantities; per-seed relative path differences range from -1.19% to +0.56%. Stored endpoints cannot reconstruct the full intervening predictive paths. The archived local step-length-matched control is not an alignment of whole trajectories at equal progress.

The original architecture has an exact, ordinary compensation mechanism: the eight parameters in the final block's second output row affect the unused feature and never the readout. Partition the Jacobians as A=(A_c,0), B=(B_c,H), and the update as (w_c,w_h). Eliminating the final head by its normal equation gives

    w_h=-gamma*(gamma H^T H+lambda I)^-1 H^T B_c w_c,
    B_c w_c+H w_h=lambda*(gamma H H^T+lambda I)^-1 B_c w_c.

This is standard ridge cancellation. At the 16 baseline step-500 states, for the gamma=9 direction, a mean 98.32% of the body's unused-velocity squared norm lies in the head's span. The mean body/head velocity cosine is -0.9914, and the remaining unused velocity has 9.77% of the body-only norm. These local measurements agree with the algebraic compensation explanation.

However, the final unused head accounts for only 3.35% of the endpoint parameter-difference squared norm on average. The entire trajectory difference is not confined to a trivial readout-null parameter block. Thus the audit supports a local standard explanation and rejects the claim of proven trajectory equivalence; it does not overstate a complete explanation of the nonlinear experiment.

## Research consequence

The joint-asymmetry claim remains retired. The replacement decoupling claim is now also retired in its current form. Existing experiments remain useful as exploratory evidence and methodological checks, not as proof of a new principle.

The next stage is problem selection, before additional training. A candidate must name a concrete prediction or algorithmic advantage absent from the closest prior work; survive linear, nullspace-compensation, and time-rescaling alternatives; and state how evidence could refute it. A specific trajectory result could be reconsidered only if it establishes a nonroutine, quantitatively useful condition and predicts an effect beyond these alternatives. No third candidate is promoted merely to maintain momentum.

The archive includes source code, numerical records, original-input snapshots needed for reproduction, independent mathematical and empirical review details, and a hash manifest. Detailed literature-search working notes remain in the workspace; this is the curated decision record.
