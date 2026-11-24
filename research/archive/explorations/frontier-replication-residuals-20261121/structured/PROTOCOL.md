# Prospective residual-envelope and label-evidence study

Freeze this protocol, theorem, all runtime sources and copied inputs before every fresh 243-family sample. Use exactly the prior eight historical 144-parameter models and their unchanged interpolation and folded policies. No model/selector/temperature/clip fitting or search occurs. All old files are immutable. Deployment uses N=256 inputs, exactly64 selected, and 320 model example evaluations. Fixed pointwise even scores make every single-coordinate replacement invariant.

## Prespecified questions and endpoints

Primary: at 2048 batches on covered_third (r=.10cos3phi), report every one of 16 model-policy estimated-coefficient lower bounds at assumed epsilon=0, and at the independently labeled epsilon_U from the invariant finite-cell posterior audit. The corresponding known-coefficient and global-TV bounds are fixed comparators. A useful primary result requires at least one strictly positive estimated-coefficient certificate; report the two primary positivity counts separately, with the label-audit result conditional on posterior constancy within cells. No post-outcome epsilon, sample size, harmonic, rho or policy changes. Also report all other laws, all epsilon values, both looks and every negative result. A violated-envelope row is never counted as theorem-supported success.

## Fixed laws and budgets

Radii are independently uniform on [.4,1.4]. Angular density is (1+theta1 cos phi+theta3 cos3phi+b5 cos5phi)/(2pi). Rejection sampling uses the analytic sum-of-absolute-amplitudes envelope.

|Index|Name|theta1|theta3|b5|Supplied rho|Covered?|
|---|---|---:|---:|---:|---:|---|
|0|uniform|0|0|0|0|yes|
|1|small_first|.01|0|0|0|yes|
|2|covered_third|0|.10|0|0|yes|
|3|covered_mixed|.04|.04|0|0|yes|
|4|covered_residual|.01|.04|.025|.025|yes|
|5|strong_third|0|.60|0|0|yes|
|6|violated_residual|0|0|.60|0|NO, deliberately invalid sensitivity|

Each law has an independent 262144-input coordinate-only coefficient sample, seed2431000+j. Estimate theta1 and theta3 using twice the sample cosine mean and separate two-sided Hoeffding CIs. No network calls or labels are used. The coefficients define an odd interval with a deterministic supplied residual bound; no residual magnitude is estimated from a discriminator or empirical maximum.

Independent label evidence is collected only for laws2 and4, for both true finite-cell defects0 and.1. Each of these four audits uses131072 fresh IID labeled examples. Input seed2436000+100*j+k and Bernoulli-label seed2437000+100*j+k, k=0,1 for defects0,.1. The eight cells are2x2 radial bins cut at.9, crossed with a canonical half-circle and its T partner. Base probabilities are.8 in radial cells00/11 and.2 in01/10; add or subtract defect/2 by orientation. Actual sampled labels, counts, successes, intervals and raw inputs' hashes are saved. Cells are frozen; no data-dependent merging. Empty cells receive[0,1]. Cell probability lower bounds are documented but never substituted for observed counts. Exactly32 two-sided cell intervals produce four epsilon_U values. Label samples are independent of parameter, verification and truth streams, and shared across the already fixed models.

Verification uses2048 batches per model/law and retains the fixed nested512 look. Inputs use2432000+100*j+i, ties2433000+100*j+i, model i=1..8. Every moment comes from that actual P_j. Pair evaluations only cover the union of both masks, charged as two forwards per union input. Store all three harmonic features on that union, full cell IDs, packed masks, union IDs, selected logits, raw per-batch A/B and every H moment, all call ledgers and hashes. Whole-batch and four single-coordinate invariance checks precede each record's calculation. c is unchanged historical q95 clipped into[1e-6,.5]; D=min(1.4+C,1.4L) from frozen weights. Commit both before fresh verification.

The fixed epsilon grid is0,.025,.05,.1,.2,.4,.8,1. For known and estimated coefficient modes, H uses the exact joint endpoint maximum from THEORY.md and conditional range D*[epsilon/2+(1-epsilon/2)rbar]. The label-supported moments use their independently estimated epsilon_U, not the true generator defect. For grid rows the true generator defect is only a later evaluation setting. The total family allocation is8622 events at delta=.05/8622:8576 moment/method events,14 coefficient CIs and32 cell CIs. The bound is L_A-U_H. Global-TV comparison is L_A-epsilon U_B-D*min(1,2(|theta1|+|theta3|)/pi+rho). All results on violated_residual are explicitly invalid sensitivity values, excluded from coverage claims.

## Independent truth stress, negative controls and numerical checks

Truth inputs use2434000+100*j+i, ties2435000+100*j+i, 65536 inputs per model/law. Both views are evaluated on every input, shared among raw, full, interpolation and folded. One deployment batch is replayed for each limited policy per model/law and charged640 extra forwards. Save logits, p0, cos features, true imbalance, cell IDs and masks. Posterior worlds are cosine perturbation of p0, model-adversarial sign(d) perturbation and orbit_worst, at the full epsilon grid. Orbit_worst uses pbar=.5+.5(1-epsilon)sign(r_true*d) and p-pT=epsilon sign(d), with cos phi as orientation when r_true=0. It is model-dependent and attains the sharp un-clipped bound. Also evaluate the two exact finite-cell posterior laws on the label-audited input laws. Record expected NLL/error, ordinary IID gains, same-law density-corrected gains, descriptive batch standard errors, sharp expression and all sign failures. True posterior probabilities are evaluation information, never observed-label replacements.

Random algebra seed2438000 checks K2/3/10, scales.1/1/5/20,4096 records each: orbit identity, endpoint versus dense interval extremum, posterior simplex optimum, even interval/penalty, conditional global range and shared offsets. Fixed angular quadrature1048576 points records two analytic counterexamples: the prior cos3 example is covered by the expanded basis; cos5 is covered by rho=.6 but not by rho=0. The code also records the finite-cell averaged-posterior obstruction from THEORY.md. These quadratures illustrate independently derived sign/identifiability arguments and do not set a certificate parameter.

## Release, integrity and accounting

Run assets.py once, run.py freeze, and request internal root scientific release binding FREEZE.json. Then prepare, checks, evidence, certificate, stress, analyze and report. This is an internal methodological checkpoint, not a new user-permission requirement. Freeze precedes all new randomness; copied historical assets are hash-verified. Runtime writes use exclusive creation. Preserve failures and do not overwrite completed outputs. Raw sample, label, coordinate, model-forward, scalar-statistics and historical fitting costs are distinct. No timing comparisons are inferred from concurrent wall time. Pause promptly for another branch's exclusive timing window. Final standalone replay copies frozen sources, inputs and approval into a fresh directory. The final report distinguishes proved mathematical implications, ordinary floating-point checks, conditional assumptions and observed Monte Carlo estimates.
