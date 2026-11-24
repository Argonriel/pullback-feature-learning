# Evidence and reproducibility map

Date: 2026-11-21 (America/Los_Angeles).

## Reading order

1. `REPORT.md`: completion status, final narrowed judgment, results and limits.
2. `THEOREM.md`: authoritative copy of the independently drafted all-data closed-PSD theorem.
3. `ROOT_REVIEW.md` and `independent_spectral/BOUNDARY_THEOREM_SECOND_REVIEW.md`: two mathematical checks of the new theorem.
4. `independent_spectral/QUANTITATIVE_REVIEW.md`: explicit lower eigenvalue/rate theorem, exact additional weight condition, counterexample to the unrestricted shortcut.
5. `independent_literature/PRIMARY_AUDIT.md`, `OPTIMAL_FACE_NOVELTY_ADDENDUM.md`, and `SEARCH_LEDGER.md`: primary comparisons and access limitations.

## Numerical and exact evidence

| Location | Content | Interpretation |
|---|---|---|
| `validation_boundary_v1/results.json` | Four runs of one new PSD-only coupled-center problem | Two weights times two steps; all errors and sampled invariants retained |
| `validation_boundary_v1/matrices.npz` | Measurement basis, initial point, target, exact-predicted endpoint, weights, normal matrices, final numerical matrices | No external data or reserved samples |
| `independent_spectral/results.json` | One scalar-weight noncommuting example at two tolerances; one exact Lyapunov failure | Root equation, rate constant, matrices, finite differences and refinement |
| `independent_boundary/exact_checks.json` | Rational scalar identities and inconsistent-data KKT | Exact arithmetic controls; not a proof certificate |
| `independent_literature/lyapunov_counterexample_results.json` | Independently derived alternative witness, four finite differences and three controls | Confirms sign and rejects accidental numerical explanation |

The 5-by-5 boundary case has `J=Sym(2) direct-sum R direct-sum Sym(2)`. Measurement order is first-block traceless diagonal, first-block off-diagonal, last-block traceless diagonal, last-block off-diagonal, and normalized center `(I2/2,1,0)/sqrt(1.5)`. The first target block is `[[.8,.4],[.4,.2]]`; the scalar target is zero. The observed center is `c+z=.5`, which together with the first traceless spectral radius `.5` forces both boundary conditions. The last block's center is independently solved from its conserved trace/log determinant. Two zero endpoint eigenvalues are therefore an exact constraint consequence, not a numerical rank-threshold choice.

The finite-time boundary errors and inter-weight difference must not be interpreted as endpoint-selection violations. The archive reports them explicitly because polynomial transients dominate refinement errors at the chosen finite horizon. No experiment was extended simply to make that distinction visually disappear.

## Reproduction

The reusable sources are in the dedicated repository directory
`C:/Users/liwei/Documents/ChatGPT/277/pullback-feature-learning/experiments/astra6_global_convergence_deepening_20261121`.
Exact preserved copies are in `source_snapshot/`.

The boundary script accepts a fresh output directory:

```
python verify_boundary.py --output <new-directory>
```

It requires Python 3 and NumPy only. Defaults: damping 2, horizon 1600, steps .05 and .025. No SciPy or package installation is required.

The three independent checks write their result JSON beside their script. **Copy the desired source into a new empty directory before rerunning** to preserve the archived outputs:

```
python check_quantitative.py
python exact_controls_check.py
node lyapunov_counterexample.cjs
```

The first requires NumPy; the rational check uses only the Python standard library; the Node check uses only built-in modules. Runtime versions and source hashes are embedded in their result records. The bundled Python used by the root is recorded in `provenance.json`; the boundary agent's exact-rational check used an independently available Python runtime. This does not change exact rational results.

`package_audit.py` computes package/source checks without rerunning the ODEs. It is bookkeeping, not formal verification. `manifest.json` excludes itself and `ARCHIVE_CHECK.json` to avoid self-hash recursion.

## Provenance limits

`provenance.json` records hashes of all required prior artifacts and the repository instructions, plus new source locations. `historical_input_hashes.json` is a baseline taken before final package preparation. The closing audit checks those same files again. No historical file was modified by this work.

The literature agent could inspect original-source content through the web tool but direct file downloads encountered DNS/access limits. Its four saved source snapshots are tool-return extraction records; `SOURCE_MANIFEST.json` hashes those records, not publisher PDF bytes. Other inspected primary sources have URLs and theorem locations but no claimed complete-byte capture. Supplemental sources that were not fully accessible are explicitly labeled screens.

`FAILED_ATTEMPTS.md` and the independent failure notes preserve environment failures and rejected mathematical approaches. Duplicate independent derivations were retained rather than silently selecting only the more favorable numerical witness.

The code and research package are isolated to this assigned direction. Other directions, frozen configurations, published history, and old research results were not changed.
