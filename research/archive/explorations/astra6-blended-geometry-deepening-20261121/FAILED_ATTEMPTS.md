# Failed approaches, negative controls, and refinements

Date: November 21, 2026 (America/Los_Angeles). These records are retained rather than replaced by successful checks.

## Mathematical overclaims rejected

1. **Bracket generation automatically implies fixed-loss endpoint controllability.** Rejected as an argument. The permitted time-dependent controls are a larger class. The new result instead proves a C1 endpoint expansion for constant SPD quadratic weights and a full-rank area map.
2. **Every coherent initial state has invertible first curvature.** False. Coherent equal-diagonal 2x2 states have zero first curvature; a fully coherent equicorrelated 3x3 example has first-bracket rank five rather than six. Higher brackets repair these examples, but this does not remove genericity in arbitrary dimension.
3. **Each initial zero off-diagonal entry is preserved.** False. An exact sparse connected 3x3 example creates its missing edge. The component partition, rather than individual edges, is the invariant object.
4. **An isotropic transformed weight supplies all leading endpoint directions.** False for n>=3. Its area differential has rank n-1. The complete rank formula requires Krylov dimension at least n-1 for full leading rank. No conclusion about higher-order exact rank is inferred from this control.
5. **The exact area spectral criterion is independent new matrix theory.** Rejected after finding the direct Ballantine factorization reduction. The criterion remains useful and is explicitly credited.
6. **A bounded leading coefficient range determines the complete finite-target endpoint range.** Unsupported. The expansion is local in a compact weight family and not uniform as weights become singular or unbounded.
7. **Connected support gives full Lie rank at every state in all dimensions.** Unresolved, not retained as a theorem. The two-step version is false; block invariance alone does not exclude other exceptional orbits.

## Preserved numerical check failure

`validation_v1/results.json` has `sensitivity_fd_converges=false`. The centered endpoint differences at weight increments 0.002 and 0.001 disagree with the integrated sensitivity by 3.14696e-13 and 2.48955e-13. They are small in absolute terms but fail the chosen factor-four-decay criterion. Both the original source and this first result remain unchanged.

The focused follow-up `endpoint_difference_refinement/results.json` uses increments 0.04,0.02,0.01 to resolve the truncation regime. Errors are 4.21756e-11,1.04377e-11,2.65805e-12. All perturbed weights have positive minimum eigenvalue (at least 1.94). This supports cancellation/roundoff as the explanation of the first ratio failure, without claiming a rigorous floating-point error bound. No additional training or broad parameter sweep was performed.

## Environment and retrieval failures

The Windows `python` alias was inaccessible; the existing bundled Python 3.12.14 / NumPy 2.3.5 runtime was used. A SymPy probe in the exact-rank subtask found it unavailable, so standard-library Fraction arithmetic was used. No package was installed. See `exceptional/FAILED_ATTEMPTS.md` and `static_loss/FAILED_ATTEMPTS.md`.

Initial restricted-network source retrieval failed, direct MDPI source snapshots were denied, and one Taussky DOI lookup failed. Successful primary-source replacements, version-number corrections, and the distinction between web-inspected text and byte-hashed snapshots are recorded in `literature/SEARCH_LEDGER.md`. No hash is invented for an unavailable snapshot.

## Packaging-parser false positive

The first packaging pass correctly checked old archives and source hashes, then stopped because its simple Markdown-link expression interpreted the mathematical code expression `[H_i,H_j](X)` as a file link to `X`. The failed parser source stays unchanged as `source_snapshot/package_results.py`; `packaging_attempt_v1/` preserves the exception record and first provenance file. The follow-up `finish_package.py` ignores code blocks and inline code when checking prose links. No mathematical or numerical output was changed for this correction.
