# Scope and interpretation notes

These notes clarify the frozen study without changing sources, configurations, models or outcomes.

- The posterior guarantee ranges over all measurable functions p(r) of the fullSO(2) orbit invariant. It does not range over arbitrary functions invariant only under the half-turn, such as p(r,phi modpi).
- The empirical-law method does not need the analytic radial density for its pointwise bound or root finite-sample certificate. It does assume uniform recoverable orientation conditional on the invariant. For rings this is a regular-orbit assumption: P(R=0)=0. The actual continuous rings distribution satisfies it. The waveform task has a unit first harmonic and no corresponding angular singularity. No claim is made for arbitrary nonsymmetric inputs.
- The waveform actionR_theta direct-sumR_2theta is a linear representation on4D observations. Its orbit parameterization is nonlinear. The group remainsSO(2), and its quotient on the task support remains scalar and monotone with total input norm. This is geometry/noise/architecture transfer, not arbitrary-group or multidimensional-quotient validation.
- Exact-even selection already has a population Jensen nonharm guarantee without CDF verification. A positive learned-selector certificate relative to the raw predictor does not prove superiority over the even selector, the folded symmetric gate or confidence allocation.
- Known-law and empirical-law certificates are separate95% families. They are not one combined95% family. Empirical bootstrap intervals describe model-replication uncertainty; they are distinct from the all-posterior finite-sample certificates.
- Continuous-radius interval reasoning is mathematically conservative in real arithmetic. The implementation is paddedfloat64 with independent checks, not a directed-rounding or machine-checked proof.
- The controlling fresh control list is the nine policies in the validation freeze, as clarified in the root release before fresh training. Broader generic control examples in the transfer protocol were not additional confirmation policies.
- Descriptive integrated bounds in the empirical branch use the known synthetic radial law only to summarize size and decomposition. The actual empirical-law guarantee uses the full pointwise bound and independent coordinate observations; it does not use those descriptive integrals.
