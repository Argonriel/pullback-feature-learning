# Independent adapter and interpolation audit

Date: 2026-11-21T10:53:01.037976+00:00. Status: PASS for the specified development checks.

The executable `task_geometry.py` reads adapters and frozen model/task/selector source but changes none. It uses only seeds 1201001 through 1201010, all three tasks, no training and no fresh state. Thirty task/seed cases check 7,680 training labels (exactly 128 per class), 15,360 population inputs, model dimensions (80 for rings; 144 for each harmonic task), two-valued label encoding, geometry reconstruction, quotient and posterior invariance, exact partner involution, and pointwise/masked expected-loss Jensen identities. The rings training generator takes one seed argument and returns exactly `(x,y)`; it has no `n` parameter.

Canonical folded angles are bitwise equal under the partner, including signed-zero axis cases. All 60 quota batches select exactly 64 of 256, and arbitrary individual input flips preserve the whole mask with common tie keys. Independent scalar interpolation checks 3,870 points using explicit corner sums. This audits the same multilinear formula used by the two-invariant trilinear selector, rather than comparing it only with itself.

Maximum errors: `{"group_quotient_invariance": 4.440892098500626e-16, "helper_nll": 4.440892098500626e-16, "interpolation": 2.581268532253489e-15, "jensen_identity": 2.220446049250313e-16, "masked_jensen_identity": 2.220446049250313e-16, "partner_involution": 0.0, "partner_vs_pi": 3.469446951953614e-16, "posterior_group": 3.497202527569243e-15, "posterior_partner": 0.0, "quotient_invariance": 0.0, "reconstruction": 7.771561172376096e-16}`.

The evaluator's angular world `p=.5+.4*cos(2 phi)` is half-turn invariant but not full-circle invariant. The selective Jensen guarantee needs the former; this world must not be described as a full-orbit-invariant label task. The two-amplitude quotient is two-dimensional only on its specified phase-coupled support. All source hashes, per-case results, and checks are in `task_geometry.json`.

Separate stress note: Current ring_posterior uses unscaled i0 and is nonfinite at extreme tail radii 15 or 40; ordinary sampled inputs and calibration [0,3] pass. This is a numerical API robustness limitation, not observed experiment failure. If the posterior implementation is repaired, that change must be documented and hashes refreshed before fresh scoring; this audit does not modify it.
