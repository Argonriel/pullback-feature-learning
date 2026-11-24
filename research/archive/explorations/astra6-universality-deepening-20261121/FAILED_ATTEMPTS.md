# Failed attempts and excluded inferences

Date: November 21, 2026 (America/Los_Angeles).

## Environment probes

1. The default `python` command resolved to the WindowsApps execution alias and
   failed to launch. No result was produced. The bundled runtime was located
   through the desktop dependency tool and used explicitly thereafter.
2. A read-only import probe found NumPy 2.3.5 but no SciPy. A separate probe found
   no mpmath. No installation was attempted. The final bounded validation needs
   only NumPy and the Python standard library.
3. Original-source download failures and their later recovery are preserved
   separately in `independent-literature/FAILED_ATTEMPTS.md` and
   `independent-literature/FAILED_SOURCE_RETRIEVAL.json`.

## Mathematical avenues not promoted

- Finite nonzero Taylor jets yield a routine weakening of analyticity but do not
  handle flat defects. That preliminary route was superseded by the exact
  spectral-germ separation and injective response transform.
- The sign of the transform kernel alone is insufficient to prove injectivity
  for sign-changing functions. Two independent derivations supply a positive
  energy identity instead.
- A globally non-BKM smooth kernel need not have a nonzero germ at equal
  eigenvalues. Local BKM plateaus disprove the naive extension; a guarded
  six-dimensional example also disproves a global closedness version.
- Positive quadrature, maximal polynomial exactness, and finite-support moment
  lower bounds are classical tools. Their optimizer-response translations are
  useful calibration results, not separate mathematical discoveries.
- Exact response-profile identification is not a finite-sample or uniformly
  stable recovery result. The complete target-amplitude profile is infinite
  data. No assertion that all finite diagnostics are impossible is made.

## Numerical disposition

The first actual validation run is retained as `validation-v1/results.json`.
All seven declared checks passed. There was no numerical failure, hidden
parameter search, or selective replacement of failed outputs. Any subsequent
source correction must preserve this run and its original source hash.
