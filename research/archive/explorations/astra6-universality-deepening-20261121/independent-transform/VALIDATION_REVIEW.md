# Independent minimal-validation review

Date: November 21, 2026 (America/Los_Angeles).

Reviewed source: `experiments/astra6_universality_deepening_20261121/validate_nonanalytic.py`.

Reviewed result: `validation-v1/results.json` in the new research staging package.

The current source SHA-256 is `7ffb28148569c65142e4d96fe3ccb12db9764444ed12e5cddc6e7e41f3d85814`, exactly matching the hash saved in the result. No source, result, or prior archive was modified in this review. No repeated run was necessary.

## Findings

The scalar-transform implementation, the independent full matrix zero-damping path calculation, and the central finite difference in a fixed SPD weight are consistent with the exact formulas in `TRANSFORM_AUDIT.md`. The canonical matrix normalizations are correct. The matrix calculation uses the direct discrepancy to avoid subtracting two nearly equal spectral kernels; this is particularly appropriate for the flat discrepancy. It does not simulate positive-damping trajectories and does not validate a finite-damping claim.

The seven cases include exact BKM zero, an analytic SLD control, a nonzero C-infinity flat defect at three amplitudes, and a plateau defect below and above its activation threshold. The flat and plateau metrics are positive because their h functions add a nonnegative bump to the positive BKM kernel. They are not claimed to be monotone quantum metrics.

All five nonzero central finite-difference errors decrease by a factor between 3.999948 and 4.000004 when tau halves from .002 to .001. This is the expected second-order central-difference truncation. The largest fine relative discrepancy is 5.646109610061633e-6, in the smallest flat response, whose magnitude is 4.777431854862219e-14. Both exact-zero controls are zero in both routes.

The reported quadrature-refinement differences compare two orders and two horizons. They are useful numerical evidence, not rigorous quadrature-error bounds. A reported difference of zero means equality at the returned floating precision, not an exact quadrature identity.

## Tail and domain controls

For t>=1,

    |h'-h f'/f|
      = exp(-t)[t-1+t/(exp(t)-1)]
      <= t exp(-t).

All selected |g| are increasing in |v|. Thus the scalar omitted tail after T>=1 is bounded by

    epsilon |g(epsilon)| (T+1) exp(-T).

The implemented `(T+3)exp(-T)` bound at T=64 is therefore conservative. It bounds this scalar integration tail only. It is not a bound on quadrature discretization, fixed-weight finite-difference truncation, or the independent matrix-path tail. Those distinctions should be preserved in the report.

The SLD discrepancy is evaluated by its positive series through index 159. At the scalar test amplitude .2, its omitted kernel tail is at most `.2^320/[321(1-.2^2)]`, far below the reported errors. For all declared matrix paths, a crude continuous bound gives |v|<=sqrt(2)epsilon; even the .2 SLD matrix test therefore has an immaterial series tail. Flat exponential underflow at extremely small |v| suppresses quantities below floating representability and cannot explain a response of order 1e-14.

The saved state eigenvalues are sampled. A separate exact bound certifies the entire tested zero-damping paths: for SPD B, the operator I-exp(-Bt) has norm at most one on S, so

    ||X(t)-I||_op <= ||X(t)-I||_F <= ||Delta||_F = sqrt(2)epsilon.

The largest epsilon is .6, giving a positive continuous lower bound `1-.6sqrt(2)>0.1514`. The weights have eigenvalues 1 plus/minus sqrt(2)tau, so both declared tau values are rigorously SPD. No intermediate loss of positivity is hidden by discrete state sampling.

## Classical atom checks and aggregate gate

The exact first missed BKM moment for m Gaussian atoms,

    h_m = 2^(4m)/[(4m+1) binom(4m,2m)^2],

is correct. It follows from the squared norm of the monic degree-2m Legendre polynomial under the normalized uniform measure on [-1,1]. The first missed moments recorded for m=1,2,3,4 agree with their exact values to floating precision. The canonical response coefficient uses the correct harmonic multiplier, including its sign.

The lower bound `h_m/4^m` for the maximum among the first 2m moment errors of any measure supported on at most m nodes in [0,1] is a conservative classical consequence: square the monic node polynomial p, use the minimum monic polynomial norm h_m, and bound the sum of absolute coefficients of p^2 by 4^m. It is not a new moment-theory result and need not be sharp as a simultaneous-error constant.

The saved `passed` Boolean checks the canonical normal coefficient, fine fixed-weight errors, quadrature refinement, matched moment errors, exact zero controls, and positive flat/plateau response signs. It does **not** gate the numerical first-missed-moment agreement with the exact formula, the error-ratio convergence, or every tail inequality. Those additional statements were independently inspected here; the aggregate Boolean should not be described as covering them automatically.

## Verdict

No implementation defect affecting the stated narrow validation was found. These are bounded consistency checks of the exact first damping response and the classical atom formulas. The all-subspace theorem, continuous injectivity, and novelty judgment remain mathematical and literature questions, not outputs of these seven calculations.
