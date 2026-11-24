# Conditional-label geometry feedback: noise-condition screen

Date: November 21, 2026, America/Los_Angeles. Decision: **reject the universal training-only replica-variability sign rule; retain the covariance formula as routine bookkeeping, not an innovation.** No known-law rings predictor or general architecture/data condition is disproved by the restricted impossibility argument below.

This screen follows the stopped empirical gate. It does not reopen the moons/rings trajectories, run the reserved confirmation cohort, or search for a new favorable endpoint. It uses one initial state of the actual 80-parameter residual model, four fixed inputs, exact enumeration of sixteen conditional Bernoulli label configurations, and short local numerical probes. Repository AGENTS.md and the previous literature/decision audits were read. Existing files were preserved.

## Information and candidate

The sharp candidate examined here is:

> Initial geometry, the conditional training-label law at the observed inputs, and arbitrary training-only label-replica variability statistics determine the sign of the later adaptive-versus-frozen metric population-risk difference without additional population-task information.

This is stronger than the allowed project question and is false. It would include rules based only on a norm, trace, covariance, or instability of replica metrics. The counterexample conditions on the observed training inputs. The project explicitly allows a known synthetic population law; such additional information can distinguish the counterexample tasks, so the result does **not** eliminate a predictor that uses that law. It also does not imply a statistical lower bound for typical iid samples: the common observed-input event in the witness has positive but small probability.

## Exact raw acceleration identity

At a deterministic initialization independent of labels, write A for the first-coordinate Jacobian on n fixed training inputs, J for the complete two-coordinate representation Jacobian, M=J^T J+lambda I, and P=M^(-1). Let T_i be independent Bernoulli labels in {0,1} with conditional means p_i, and let s_i=sigmoid(f_initial(x_i)). The mean-logistic gradient and PB velocity are

    q(T) = A^T(s-T)/n,
    v(T) = -P q(T) = v_bar + sum_i w_i (T_i-p_i),
    v_bar = -P A^T(s-p)/n,  w_i = P A_i^T/n.

Adaptive PB and nonlinear prediction with frozen P start with the same velocity. Their initial parameter-acceleration difference is

    b(T) = -P DM[v(T)] v(T) = Q(v(T),v(T)),
    Q(u,w) = -P DM[u] w.

DM is linear in its direction, so Q is bilinear, generally not symmetric. Expanding and using conditional independence gives the exact identity

    E[b(T) | X,theta0] = Q(v_bar,v_bar)
                       + sum_i p_i(1-p_i) Q(w_i,w_i).

For correlated conditional labels, replace the final sum by sum_ij Cov(T_i,T_j) Q(w_i,w_j). The zero-mean cross terms vanish; the quadratic covariance contribution remains. Its vector or a generic scalar contraction has no universal positive-semidefinite sign. This is a standard second-moment expansion, not a new theoretical contribution. It holds at the specified label-independent initialization; it must not be copied to a fitted state while pretending the fitted geometry is independent of the same labels.

## Removing the clock is not a linear expectation operation

For a label realization with q nonzero, let d=q^T Pq=v^T Mv>0. Matching attained training loss removes the acceleration component parallel to the initial velocity:

    b_perp(T) = [I - v(T) v(T)^T M / (v(T)^T M v(T))] b(T).

The bracket is a label-dependent rational map. In general, E[b_perp] differs from projecting E[b] with the mean velocity. Moreover, a common small **training-loss decrease** rho across replicas has leading parameter contrast

    theta_adaptive(rho)-theta_frozen(rho)
       = (rho^2/2) b_perp(T)/d(T)^2 + O(rho^3).

Thus the quantity relevant to an across-replica, common-progress risk comparison is E[b_perp/d^2], not simply E[b], and not the covariance contraction projected once after averaging. These are ordinary inverse-matrix differentiation, clock matching, and Taylor expansion. They identify a possible confound in proposed noise diagnostics but do not provide a new finite-horizon prediction theorem. The expansion is local, with nonzero descent for every realization considered; it does not claim uniform control near stationary points or across arbitrary label distributions.

## Actual-network numerical verification

The executable `noise.py` uses the existing full 80-parameter, two-block tanh residual network at initialization seed 404027, summed full representation metric and lambda=0.1. Inputs are (-0.7,0.2), (0.4,-0.6), (0.8,0.5), (-0.2,-0.8), with conditional positive-label probabilities (0.2,0.7,0.4,0.8). These are a finite local audit fixture, not the previously screened moons/rings task law.

DM is evaluated by a complex-step derivative of the analytic full Jacobian. An independent central-difference derivative of J^T J at three step sizes checks it. The exact weighted sum over all sixteen label configurations gives:

| Check | Value |
|---|---:|
| Maximum error in raw covariance identity | 7.63e-17 |
| Norm of nonzero noise-covariance contribution | 0.064328 |
| Norm of mean raw acceleration difference | 0.101503 |
| Norm of soft-target, mean-label raw difference | 0.057836 |
| Norm of E[b_perp] minus projection of E[b] at mean velocity | 0.005924 |

These checks establish that neither mean-zero cancellation nor interchange of projection and averaging is valid in this fixture. They do not establish predictive harm from the nonzero term.

## A limited non-identifiability witness

Keep the same initialization, observed X, training conditional probabilities and all training/replica procedures. Add an unseen atom x_star=(0.1,0.9). Give the four common atoms total input mass 0.01 and x_star mass 0.99 in both population laws. The initial probability at x_star is 0.5279166661. The two population laws differ only in its conditional positive-label probability:

| Law | Probability at x_star | Expected leading adaptive-minus-frozen risk coefficient of rho^2 |
|---|---:|---:|
| Lower posterior | 0.2918749991 | -0.7035980070 |
| Upper posterior | 0.7639583330 | +0.6932285047 |

Every training-only statistic, including statistics from arbitrarily many conditional-label replicas at observed X, has the same conditional distribution in these two laws. The full network and metric are also identical. Yet the expected local equal-loss population-risk signs are opposite. A three-input finite witness search selected the atom with largest nonzero predictive projection; this is disclosed counterexample construction, not held-out predictor selection.

An independent RK4 integration of the full network in training-loss-decrease coordinates probes both flows. All sixteen label configurations are integrated for sixteen RK4 steps at each of two fixed tiny progress values:

| Loss decrease rho | Expected risk contrast, lower law | Expected risk contrast, upper law | Maximum loss-target error |
|---|---:|---:|---:|
| 1e-4 | -7.3490e-9 | +7.2418e-9 | 1.23e-16 |
| 5e-5 | -1.7983e-9 | +1.7719e-9 | 2.28e-16 |

Dividing by rho^2 approaches the displayed local coefficients as rho decreases. The finite contrasts confirm the sign in these small probes; their tiny absolute size is not a practical performance claim. This is a standard conditional-information limitation with an actual-network witness. It is not a new no-free-lunch theorem and does not explain rings or refute a predictor that knows the full population law.

## Closest primary-source threats and boundaries

[Zhang et al., ICML 2025, Section 3 equations (4)-(5), Section 3.1](https://arxiv.org/html/2502.01763v1) already studies learned layerwise preconditioning with independent update batches and fresh-batch analysis. Therefore learned geometry plus replica/sample-splitting noise control is not enough for novelty. Its linear representation and one-step nonlinear results do not automatically prove the present all-parameter, repeated-label metric-feedback formula or a finite-progress sign condition. The formula here is nevertheless elementary, so that scope difference does not create a contribution.

[Damian, Ma and Lee, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/e6af401c28c1790eaef7d55c92ab6ab6-Paper.pdf) analyzes label-noise SGD through an explicit implicit-regularization effect, with extensions to classification/general noise covariance. It already demonstrates that zero-mean stochastic label perturbations can systematically affect parameter evolution. Its injected iteration-wise stochastic noise differs from our fixed sampled labels and deterministic full-batch metric adaptation. One must not import its regularization sign into this setting; neither should surviving quadratic noise drift be advertised as a new principle.

[Huang et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/ffab50f3cad7cb5733ca324e5be20976-Abstract-Conference.html) proves that introduced label noise can suppress noise memorization and improve signal learning in a specified low-SNR two-layer network model. This further defeats generic claims that greater label noise must worsen learning. It does not analyze the present representation-pullback rule.

[Amari et al., ICLR 2021, Sections 2-3](https://arxiv.org/html/2006.10732v4) distinguishes population from sample Fisher and shows that preconditioning/noise rankings depend on signal and model assumptions in time-invariant linear regression. Its conclusions do not establish a universal ordering for label-dependent full-parameter geometry.

## Decision and admissible remainder

Reject the training-only universal sign candidate. Reject proposed novelty based only on the raw covariance contraction, mean-zero-noise survival, independent replicas, or the matched-loss projection identity. The exact bookkeeping remains useful to prevent an invalid argument in any later analysis.

What remains open is a **known-law**, architecture-specific structural condition that makes the population-facing contraction of the progress-normalized feedback sign-controlled and keeps it so beyond a purely infinitesimal interval, without assuming the desired future population derivative sign. The present audit neither derives nor validates such a condition. Its existence is not inferred from the fact that the universal shortcut fails.

## Artifacts and checks

- `noise.py`: reusable full-network local computation and short-flow checks in the repository.
- `noise_results_verified.json`: final raw numerical data and source/core hashes.
- `noise_results.json`: preserved initial algebra-only output.
- `noise_initial_snapshot.py`: exact original source version matching that initial output hash.
- All sixteen label weights sum to one; all initial descent rates are positive.
- Raw identity, independent metric derivative, nonzero projection gap, opposite-sign witnesses, and short-flow target checks pass.
- No large training sweep, benchmark extension, prospective-predictor validation, or new contribution claim was made.
