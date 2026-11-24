# Why unrestricted finite-sample residual certification fails

This supplemental analytic argument does not change the frozen experiment or its assumptions. It sharpens the distinction between estimating two coefficients and establishing a support-wide residual envelope.

Let mu be the invariant reference law and let m be any finite number of independent unlabeled observations. Consider all measurable odd imbalances r with |r|<=1 and with zero projection on the two retained harmonics. Any reported upper bound Rhat for the essential sup norm ||r||_infinity that has coverage at least 1-alpha uniformly over this unrestricted class must satisfy

    Pr_mu(Rhat<1) <= alpha.

Thus at the perfectly symmetric reference law, no such uniformly honest procedure can usually certify a nontrivial residual envelope. This statement concerns unrestricted measurable residuals. A justified smoothness, basis-complexity, or minimum-width restriction changes the statistical problem.

To prove the claim, choose three disjoint positive-measure small angular sets in a fundamental half-circle, with their three transform partners. Make their combined mu mass at most delta. On the three original sets assign constant coefficients c1,c2,c3, and on their partners the negatives. Orthogonality to cos(phi) and cos(3phi) imposes two homogeneous linear equations on three coefficients, so a nonzero solution exists. Normalize its largest absolute coefficient to one. This gives an odd r_delta with essential sup norm one, both retained moments zero, support mass at most delta, and a valid density 1+r_delta. Discontinuities are allowed in the unrestricted class.

Its single-observation total variation distance from mu is E_mu|r_delta|/2 <= delta/2. The m-fold product total variation is at most m delta/2, by coupling each coordinate (or telescoping product measures). Consequently, if beta=Pr_mu(Rhat<1), then

    Pr_(1+r_delta)mu(Rhat<1) >= beta - m delta/2.

The event on the left is a coverage failure because the true essential sup norm is one. Uniform coverage therefore implies beta <= alpha+m delta/2 for every delta>0. Let delta decrease to zero to obtain the claim. Randomized procedures satisfy the same argument after adjoining independent randomness. The independent radial coordinates do not alter it.

This is a direct indistinguishability argument, not a claim of a new general impossibility theorem. It also explains why merely adding more data or more finite moment diagnostics cannot certify the supplied envelope without further restrictions. The experiment's covered residual is valid because its amplitude is part of the specified generator family; that fact is not inferred from its two coefficient estimates.

The analogous label limitation is different and already appears in the frozen THEORY.md: equal averages in eight cells do not establish a pointwise posterior condition. A sine posterior can have invariant cell averages and pointwise defect one. The label-supported certificates are therefore conditional on the explicitly assumed within-cell posterior constancy.
