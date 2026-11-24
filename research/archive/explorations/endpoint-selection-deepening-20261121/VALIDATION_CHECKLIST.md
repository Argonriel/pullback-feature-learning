# Validation checklist

## Mathematical scope

- Repository instructions and the previous retained/rejected decisions were read before continuing.
- The theorem states isotropic initialization, a real symmetric SPD product, full-representation pullback, fixed positive quadratic weights, and sufficiently small targets.
- Existence and interpolation of the local flow and the right derivative in damping are justified by a uniform interior bound and exponential decay.
- The forced radial primitive and finite two-condition invariant-subspace proof have been independently reviewed.
- The static-weight mixed coefficients 5/288 and 13/576 were independently derived by the root and both mathematics agents.
- Static positive weights are explicitly realized; arbitrary time-dependent control is not substituted for the claimed training problem.
- General SPD weights, diagonal weights in an orthonormal basis, arbitrary nonorthogonal measurements, and fixed versus existential targets are distinguished.
- Normal projected-form exactness is distinguished from Hessian integrability of the induced tangent metric.
- A Lie-triple-only criterion is rejected by an exact three-by-three example with a nonzero quartic response.
- First-order damping neutrality is not promoted to universal finite-damping endpoint independence.

## Numerical checks

- Exact-path response quadrature uses fixed amplitude and weight-perturbation grids with refinement.
- The quartic witness's coefficients and normalization were checked independently.
- Geometry checks cover both passing spaces and counterexamples, using finite polynomial tests rather than a model sweep.
- The finite-damping control meets residual <1e-13, SPD, and step-refinement checks.
- A separate original 18-parameter implementation agrees with the product implementation within 1.25e-11.
- The established entropy-Hessian model supplies a weight-neutral control; its matrix-log derivative is independently checked by finite differences.
- All configured cases and both quadrature/step resolutions are retained. No cases were silently selected away.

## Literature and preservation

- Original primary sources were inspected before new minimal validation.
- Closest quantum-geometry results are compared at the level of projection, domain, distribution, and quantifiers.
- Known metric, commutator, Jordan, and mirror ingredients are not claimed as new.
- Source, imported-source, and referenced-data hashes are verified by the archive packager.
- The three immediately preceding archives are checked against their original manifests and their known manifest hashes.
- A fresh archive manifest is generated and reread; final machine-readable status is in `ARCHIVE_CHECK.json`.

No broad neural experiment, previous failed benchmark reanalysis, reserved confirmation cohort, external message, or population-benefit claim was included.
