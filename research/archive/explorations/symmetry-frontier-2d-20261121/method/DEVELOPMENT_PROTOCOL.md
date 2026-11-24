# Fixed development comparison

This protocol precedes the new baseline development evaluation. It uses the
previous 16 ring models and 16 one-invariant harmonic models solely as
development data, together with the four already trained two-invariant pilot
models selected by the task's independent training protocol. It does not use
any new confirmation model.

Each policy has a calibration budget of 65,536 model-forward examples.
The Jensen grid uses 128 radial anchors by 512 full-orbit midpoint angles for
one invariant, and 16 by 16 quotient anchors by 256 angles for two invariants.
Its interpolation is linear in each quotient coordinate and periodic-linear
in the canonical folded angle. Quotient coordinates outside the grid are
clamped. A nearest-grid control reuses exactly the same table.

Histogram policies share a single pool of 32,768 IID inputs and their partners.
The fixed candidate resolutions are (16,8), (32,16), (64,32) for one invariant
and (4,4,8), (8,8,16), (16,16,16) for two invariants. Entries specify quotient
quantile bins followed by folded-angle bins. Empty cells use the pooled mean.
Select exactly one resolution per cohort by the greatest mean selected true
Jensen gain on the development batches, averaging models equally. Resolve
exact ties in favor of the earlier, smaller resolution. Retain all results.
There is no search over grid resolution, interpolation, or model training.

Each model receives 32,768 independent evaluation inputs in batches of 256.
All budgeted methods select exactly 64, breaking score ties with the same
independent random keys. The exact-Jensen oracle queries both views on every
input and is only an evaluation upper envelope for invariant selectors.
Confidence and historical rank policies are secondary contextual controls;
their posterior-dependent realized gains do not equal selected Jensen gains
unless their individual-coordinate invariance holds.

The development streams are 1191000+offset+pair (evaluation inputs),
1192000+offset+pair (tie keys), 1193000+offset+pair (histogram calibration),
1194000+offset+pair (rank CDF coordinates), and 1195000+offset+pair (rank
action keys); offsets are 0 for rings, 100 for one-invariant transfer, and 200
for the two-invariant task. Model training seeds are inherited and recorded.
Known posteriors are evaluator-only; the histogram choice uses selected J,
never posterior risk. Rank policies use their frozen tau=1 family with an
in-memory 128-anchor configuration, leaving all previous files untouched.

For each model, write a hash commitment for calibration tables before reading
evaluation inputs. Check exact quotas, individual-input flip invariance with
unchanged tie keys, and an executed 256+64 deployment batch. Preserve source,
states, outputs, failed attempts, timing, and call counts. Final histogram
choices and source hashes are frozen before any fresh confirmation access.
