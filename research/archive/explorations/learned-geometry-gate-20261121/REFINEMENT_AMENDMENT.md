# Controller-refinement amendment

Recorded before any new held-out or population scores were generated or inspected. The independent training audit identified a limitation of the original refinement: halving the target logit RMS from 0.01 to 0.005 while keeping the parameter cap at 0.1 can leave the actual steps unchanged. Rings tangent-PB trajectories demonstrate this exactly. Agreement under that check is not evidence of Euler-flow convergence.

Preserve the original run and the training-only coverage supplement. Add one explicit numerical-refinement run for all four arms, both tasks and all four discovery pairs, with target logit RMS 0.005 AND parameter cap 0.05. All other settings, loss endpoints, seeds, damping and acceptance tolerances stay unchanged. Its fixed update ceiling is 120000, twice the coverage supplement's 60000 to preserve the maximum cumulative parameter-displacement allowance. There will be no further budget extension.

At the same parameter state, both positive scalar upper bounds are now halved. Armijo/Taylor backtracking can still be active, so two resolutions provide a numerical sensitivity check rather than a convergence theorem. Report any censored trajectory and the actual controller diagnostics.

The final supplemental gate compares original 0.01/cap-0.1 trajectories (using extended rings tangents) to these 0.005/cap-0.05 trajectories. All original primary endpoint, effect-size, paired-sign, and refinement criteria remain unchanged. Decisions from the original frozen protocol and the two transparent supplements must be separately labeled. This is a pre-evaluation design repair, not a favorable-endpoint search.

For any conditional independent confirmation, declare the same coarse and refined controllers before training that cohort. The independent confirmation is required before treating a surviving exploratory difference as replicated. No new prospective predictor or novelty claim is licensed by this repair.
