# Input-map topology: a different setting, but no retained innovation

## Candidate and original-source check

The new candidate asks whether the loss of invertibility of the learned input-to-representation map creates an optimizer-specific phase transition that explains when representation pullback gains predictive power. This concerns the input Jacobian D_x R, not the parameter Jacobian D_theta R studied in the earlier spectral screens.

[Dupont, Doucet and Teh, Augmented Neural ODEs, NeurIPS 2019, Propositions 2-3 and the following discrete-points discussion](https://proceedings.neurips.cc/paper/2019/file/21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf) already establishes the obstruction for nested continuous regions under a same-dimensional homeomorphic flow followed by a linear readout. It also explicitly discusses fitting finite point samples through gaps and the resulting complexity/generalization issue. Its discrete-ResNet discussion explains why finite residual steps can cross trajectories. Therefore 'rings require topology change' or 'finite-sample fit hides a narrow gap' is not a new explanation.

[Behrmann et al., Invertible Residual Networks, ICML 2019](https://proceedings.mlr.press/v97/behrmann19a.html) gives the standard invertible-residual construction via Lipschitz-controlled residual blocks. Our two-block unconstrained tanh model does not satisfy its invertibility restriction automatically. Imposing a Lipschitz bound and observing the known nested-class limitation would not establish a novel PB result.

[Naitzat, Zhitnikov and Lim, Topology of Deep Neural Networks, JMLR 2020](https://www.jmlr.org/papers/volume21/20-345/20-345.pdf) directly studies changes in data topology through trained neural representations. Its discussion of coordinatewise tanh homeomorphisms must not be extended to arbitrary sums, dimension-changing linear maps, or the unconstrained residual map used here. This work already occupies generic topology-change-as-feature-learning narratives; it does not establish the present optimizer comparison.

These original sources were inspected before the static witness calculation. No topology experiment on the old benchmark trajectories was launched because the broad proposed explanation is already occupied.

## Minimal falsification of a parameter-metric warning criterion

A narrower possibility is that a representation pullback spectrum detects an approaching input-map fold and thereby supplies a useful early criterion. The following exact restriction shows that no universal implication follows from parameter conditioning alone.

In the original architecture, set A1[0,0]=1 and B1[0,0]=2a, with all other parameters zero. Only a is allowed to vary; the remaining original coordinates are frozen. Then

    R_a(x)=(x1+a*tanh(x1),x2),
    partial_a R_a(x)=(tanh(x1),0),
    det D_x R_a(x)=1+a*sech(x1)^2.

The summed one-parameter pullback metric is sum_i tanh(x_i1)^2+lambda, independent of a. It remains strictly positive with the same spectrum as the input map passes from a diffeomorphism (a=-0.5), to a zero input derivative at the origin (a=-1), to a non-injective folded map (a=-2). The latter has critical input coordinates +/-acosh(sqrt(2)).

The implementation checks both the exact core forward map and parameter Jacobian, and independently checks the input derivative by finite differences. There is no training, optimizer sweep, population comparison, or claim about the unrestricted 80-parameter metric. This witness distinguishes two different Jacobians; it is elementary calculus, not an innovation.

## Quantitative variants and decision

A proposed refinement could combine a topological escaping-path argument with a logit Lipschitz bound to lower-bound the width of an error corridor on an outer ring. Such a statement would require explicit margin and derivative assumptions and would be a direct topology-plus-Lipschitz calculation. It would constrain all qualifying classifiers, not establish a PB-specific learned-geometry mechanism. It is not promoted merely because an identical numerical constant was not found in the bounded literature check.

Reject the broad topology-transition candidate as already covered. Reject a universal parameter-pullback-spectrum fold-warning rule by the static witness. The distinct known difficulty of nested classes remains useful context for architecture choice; it does not justify claiming a new optimizer mechanism or reopening the failed deterministic gate.

Artifacts: `topology_results.json` and repository `experiments/new_direction_screen_20261121/topology.py`.
