# Frozen local-rank certificate protocol

This verification leaves the ranked selector, its fitted tables, action keys,
scores, exact batch quota and deployed predictions unchanged. The target is the
actual global top-64 inclusion probability, not the preceding group-of-four
formula. Only the previous 921-series models and the separately declared
waveform development models are used to choose numerical settings.

## Primary known-law configuration

Use all 16 new PB models in each task: rings and the two-harmonic four-dimensional
waveform. Verify the frozen `rank_balanced` policy. Split every original radial
interval into eight bands. Use at least 256 angular cells per action for rings
and 512 for waveform, adding splits at every exact sawtooth jump. Each angular
cell uses 64 monotone probability subintervals. The deterministic radial-law
partition has a target of 131,072 cells allocated according to variation of the
fixed score-location function. Round-up may add fewer than 128 cells. Beta-CDF
interpolation uses 262,145 nodes with its analytic error bound.

The signed moment and derivative enclosures are retained for every radial band.
The real-arithmetic pointwise envelope covers every radius, not only fitting
nodes. Even-frequency action contributions are zero analytically. Source
floating-point padding is numerically checked, not formally directed-rounded.

Known radial input laws are verification assumptions. The policy is still the
one fitted using empirical calibration; true score CDF values are never used
to refit or deploy it. Unknown posterior labels do not enter verification.

The utility target is a positive independent lower confidence bound on expected
gain for all 32 fresh models. The root's independent sample contains 262,144
inputs/model, with a jointly allocated 95% family confidence level. Every model
and every failure is reported; a failed target cannot change the numerical
configuration, selector or fresh-model inclusion.

## Optional empirical-law configuration

Use exactly 1,048,576 new verification-only IID radii for each fixed model.
Rings seed is 1013000+model index; waveform seed is 1053000+model index. These
streams are separate from training, selector calibration, performance testing
and the root's finite-sample confidence observations. The uniform DKW event
uses alpha=.05/64 per model, accounting for 32 CDF events and 32 later gain
confidence events in a separate 95% family statement. The remaining numerical
settings equal the primary configuration.

This empirical pointwise envelope depends only on the independent radius
sample and the fixed network/policy. Its `integrated_bound`, `mass` and related
summary fields use the known synthetic law for descriptive comparison only.
They are not inputs to the empirical-law guarantee. The root instead evaluates
the pointwise envelope on an additional independent sample, so its empirical
guarantee does not require the analytic radial distribution. Both uncertainty
events and their costs must be reported.

The empirical branch is a prespecified additional result, not a replacement for
a failed known-law primary result. No larger verification sample is selected
after fresh outcomes.

## Development record and costs

The first coarse rings-model-1 attempt (two radial subdivisions, 64 angles and
32 inner probability intervals) gave integrated bound 0.171998, too broad for
the intended gain certificate. It is preserved with source `pilot/local_rank_v1.py`.
The improved algorithm intersects direct action-integral ranges with ranges
implied by both endpoint enclosures and the local derivative interval.

At eight subdivisions and 256 angles, rings development models 1/8/16 give
integrated bounds 0.00312286/0.000688518/0.00282477. The waveform model-1 attempt
at 256 angles gives 0.0121692; the fixed 512-angle refinement gives 0.00345244.
The empirical 1M-radius rings-model-1 attempt gives 0.00338413. These attempts,
including weaker settings, remain in `pilot/`. All numerical settings were
chosen before access to new confirmation models.

Measured single-thread setup in these pilots is approximately 6–38 seconds per
model and branch. Cost depends on the amount of odd-action support. The model-1
rings final setting executes 4,714,076 five-component interval-jet propagations
and 54,964,483 coordinate-only CDF query inputs. The waveform final setting
executes 7,739,188 interval-jet propagations and 89,848,841 CDF query inputs.
These are offline verification costs, distinct from ordinary network forwards,
selector fitting, deployment and the root's gain-confidence observations.

Independent numerical development checks test four saved configurations, 96
interior moment points, 96 moment derivatives and 384 local network-jet boxes.
The checks use 4,722,432 ordinary forward examples and 768 interval propagations.
All checks pass; they supplement rather than replace the mathematical argument.

## Files and immutable staging

`experiment.py freeze` records source hashes and confirms that fresh validation
states have not been accessed. New validation training requires the root's
subsequent release. `experiment.py run --task rings --kind known` and its
waveform/empirical counterparts verify those hashes and the validation policy
commit before reading a fresh policy.

Known-law outputs are `fresh/{rings,waveform}/{seed}_local.npz` and
`{seed}_distribution.npz`. Empirical outputs use `fresh_empirical/` with the same
structure. Each task/branch has a start record, complete record, per-model
provenance and CSV summary. The runner never overwrites an existing output.
All exception attempts are preserved. Raw bounds and distribution tables allow
the root to recompute the pointwise certificate independently.
