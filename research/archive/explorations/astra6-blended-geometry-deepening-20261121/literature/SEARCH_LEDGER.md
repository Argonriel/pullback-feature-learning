# Search and failure ledger

Date: November 21, 2026 (America/Los_Angeles). Scope: finite-blend diagonal-observation conservation and fixed-SPD-loss endpoint realizability. Researcher: independent primary-source review subtask. No authors were contacted and no external messages were sent.

## Discovery queries

The following literal queries were run with the web research tool, mostly in batches of three or four. Results from secondary summaries were used only to locate original papers, not as authority for theorem scope.

1. `"Bures" "Euclidean" "integrability"`
2. `"Bures" "bracket generating"`
3. `"covariance" "diagonal" "Lie algebra" control`
4. `"gradient flow" "holonomy" "implicit"`
5. `"implicit bias" "loss weights" "metric"`
6. `"covariance steering" "diagonal" "controllability"`
7. `"Bures" "regularized" "Euclidean metric"`
8. `"damped natural gradient" "conservation"`
9. `"implicit regularization" "commuting parametrization"`
10. `"gradient flow" "loss function" "noncommuting" mirror descent`
11. `"Bures" "mixture" "metric" gradient`
12. `"gradient flow" "endpoint" "curvature" "quadratic" implicit`
13. `"covariance" "controllability" "diagonal controls"`
14. `"Bures" "damping" "conservation"`
15. `"implicit bias" "reweighting" "quadratic"`
16. `"implicit regularization" "path-dependent" gradient flow`
17. `"gradient flow" "weighted loss" "endpoint"`
18. `"Bures" "Frobenius" "sum" metric`
19. `loss reweighting implicit bias matrix factorization gradient flow`
20. `Riemannian gradient flow path dependence conservation loss weighting`
21. `sum Riemannian metrics integrable horizontal distribution nonintegrable`
22. `"Lyapunov" "Gramian" "area" "skew"`
23. `"Lyapunov" "commutator" "rank one"`
24. `"linear system" "signed area" controllability`
25. `"gradient flow" "endpoint map" submersion`
26. `"Ornstein Uhlenbeck" "stochastic area" Lyapunov`
27. `"Lyapunov equation" "antisymmetric" "inverse" covariance`
28. `"Gramian" "simple eigenvalues" "single input"`
29. `"signed area" "linear" "Gramian"`
30. `"Lyapunov" "rank-one" "simple spectrum"`
31. `"symmetric" "drift" "area production"`
32. `"matrix" "rank-one symmetric part" eigenvalues`
33. `"dissipative" "rank one" "real spectrum"`
34. `"Lyapunov" "positive real eigenvalues" "symmetric"`
35. `"skew symmetric" "rank one perturbation" eigenvalues inverse`
36. `Taussky matrix product two positive definite matrices positive eigenvalues theorem`
37. `"matrix" "product of two positive definite" "diagonalizable"`
38. `"symmetrizable" "real diagonalizable" positive definite Taussky`
39. `"Taussky" "positive definite" "1968" products`
40. `"Ballantine" "Products of positive definite matrices" pdf`

## Reduction attempts and decisions

- Natural-gradient preservation theorem: failed to reduce the finite blend to the same-observation gradient span. Keep only the general known criterion.
- SLD e-parallel theorem: commuting scores settle the pure limit; the finite resolvent is not the same distribution. No general mixed-metric extension was found.
- Monotone quantum group actions: the finite metric lacks the homogeneous kernel required for the direct specialization. No claim about all possible group actions follows.
- Covariance steering with symmetric gains: the admissible control class differs from a fixed quadratic loss, and endpoint reachability cannot be transferred without a new proof.
- Stochastic area / Lyapunov commutators: the formula is established machinery. Read Friz--Gassiat--Lyons as primary authority rather than relying on the review article discovered in search.
- Classical SPD factorization: successful direct reduction. The exact cyclic area image follows from Ballantine Corollary 1.1' plus the rank-one Lyapunov relation. This was sent to the main derivation task as an attribution boundary and alternative proof.
- Recent searches returned many irrelevant or automatically summarized results. Their presence/absence is not used as priority evidence.

## Inspected additional leads, not treated as direct theorem coverage

- `https://arxiv.org/html/2601.06597v3`, *Inverse design of implicit bias by symmetry, stochasticity and holomorphic parametrizations*: inspected for damping overlap; no matching damping statement was present in the retrieved text. Not used to support a retained theorem.
- `https://www.mdpi.com/2075-1680/13/12/820`, *A Theoretical Review of Area Production Rates...*: located the known Lyapunov-commutator area formula and useful references. This is a review; the final mathematical comparison uses the original Friz--Gassiat--Lyons source.
- `https://doi.org/10.1103/PhysRevResearch.6.033103`: inspected SGD covariance/current formulation. It supplies related Lyapunov/circulation context, not a constant-loss endpoint-submersion result.
- `https://arxiv.org/pdf/1506.08962`, *Products of positive semi-definite matrices*: inspected for factorization antecedents and followed references backward to Ballantine's original.

## Retrieval and environment failures preserved

1. Attempted `python -c` to inspect available runtime. The `python.exe` WindowsApps alias could not execute. No analysis or data were lost; the audit used the available shell and web tool.
2. The first shell download of arXiv 2508.07370v1 failed with DNS unavailable under restricted network access. The normal approved network workflow succeeded; no source bytes were replaced.
3. Direct shell requests for both MDPI pages returned server access-denied responses. The browser research tool successfully displayed their contents. No full-byte hash is asserted for either page.
4. Opening the DOI `https://doi.org/10.1016/0001-8708(68)90020-0` for Taussky's 1968 article failed in the web tool. Instead, the directly inspected original Ballantine 1968 paper gives the exact real SPD-product statement as Corollary 1.1'. Taussky is not cited as if its original text had been examined.
5. The first inspected `https://arxiv.org/html/2508.07370` page was a later renumbered version. The saved source is explicit v1; the audit corrects its theorem numbering accordingly.
