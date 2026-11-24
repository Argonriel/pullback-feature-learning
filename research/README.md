# Research record and shared files

Snapshot: November 24, 2026. This index covers the original proposal study and all 28 subsequent exploration packages. Reports include positive findings, failed replications, rejected hypotheses, and scoped theoretical candidates.

## Start here

- [Original proposal](archive/docs/proposal.tex)
- [Proposal completion and experiment plan](archive/docs/project_plan.md)
- [Core mathematical derivation](archive/docs/mathematics.md)
- [Formal GD versus pullback findings](archive/docs/formal_findings.md)
- [Frozen-feature probe findings](archive/docs/probe_findings.md)
- [Feature motion and spectral analysis](archive/docs/representation_findings.md)

The core study found early feature-quality and matched-motion advantages for pullback, but no overall advantage in final accuracy or elapsed time. The extensions below do not reverse that conclusion into a general optimizer superiority claim.

## Current theoretical candidates

| Study | Reviewable result | Entry point |
|---|---|---|
| Spectral neutrality beyond analyticity | Exact rigidity under a nonzero local discrepancy condition, a sharp exception, and response-conditioning limits | [Report](archive/explorations/astra6-universality-deepening-20261121/REPORT.md) |
| Finite Euclidean--Bures blend | Generic local endpoint freedom under a single fixed positive quadratic loss weight | [Report](archive/explorations/astra6-blended-geometry-deepening-20261121/REPORT.md) |
| Classified flow on the PSD cone | Arbitrary-data convergence and exact Bregman endpoint selection under the stated algebra and initialization assumptions | [Report](archive/explorations/astra6-global-convergence-deepening-20261121/REPORT.md) |

Each package contains assumptions, proof reviews, counterexamples, primary-literature comparisons, and minimal checks. These are retained research candidates, not certified priority claims, formal machine proofs, or established neural-network performance benefits.

## Complete exploration index

| Package | Scope and disposition |
|---|---|
| [early-pullback](archive/explorations/early-pullback/README.md) | Early directions, warm starts, and metric reuse; gains depend on the comparison budget |
| [rings-crossed-controls](archive/explorations/rings-crossed-controls/README.md) | Learning-rate and wall-clock controls remove much of the initial headline advantage |
| [rings-mechanism](archive/explorations/rings-mechanism/README.md) | Radial/angular population-risk decomposition and unused-feature interventions |
| [budgeted-rotation](archive/explorations/budgeted-rotation/README.md) | Selective second views; fewer calls do not automatically mean lower latency |
| [switch-time](archive/explorations/switch-time/README.md) | Routing preferences across training schedules, including unsuccessful confirmation |
| [rate-clock](archive/explorations/rate-clock/README.md) | Matched training time and matched loss substantially reduce apparent rate effects |
| [predictive-symmetry-20261120](archive/explorations/predictive-symmetry-20261120/README.md) | Local update forecasts and unlabeled angle selection, with long-horizon limitations |
| [calibration-quota-20261121](archive/explorations/calibration-quota-20261121/REPORT.md) | Calibration-call budgets and fixed inference quotas |
| [continuous-confirmation-20261121](archive/explorations/continuous-confirmation-20261121/REPORT.md) | Independent rings/ellipse confirmation and continuous balancing |
| [local-balance-20261121](archive/explorations/local-balance-20261121/REPORT.md) | Conditional risk certificates, stronger routing baselines, and hardware costs |
| [rank-certificate-transfer-20261121](archive/explorations/rank-certificate-transfer-20261121/REPORT.md) | Exact-quota ranking certificates and transfer to a harmonic task |
| [symmetry-frontier-2d-20261121](archive/explorations/symmetry-frontier-2d-20261121/REPORT.md) | Strong-baseline frontier and a task needing two invariants |
| [calibration-symmetry-realdata-20261121](archive/explorations/calibration-symmetry-realdata-20261121/REPORT.md) | Calibration and approximate symmetry; untrained image reflections can be harmful |
| [pipeline-calibration-augmentation-shift-20261121](archive/explorations/pipeline-calibration-augmentation-shift-20261121/REPORT.md) | Final-output calibration, training augmentation, and distribution-shift limits |
| [budget-distillation-structured-bounds-20261121](archive/explorations/budget-distillation-structured-bounds-20261121/REPORT.md) | Label/compute allocation, one-pass distillation, and conditional structured bounds |
| [frontier-replication-residuals-20261121](archive/explorations/frontier-replication-residuals-20261121/REPORT.md) | Independent allocation confirmation, failed external distillation replication, and residual-aware bounds |
| [literature-novelty-audit-20261121](archive/explorations/literature-novelty-audit-20261121/literature_review.md) | Primary-source audit narrowing the broad selective-symmetry innovation claim |
| [joint-asymmetry-novelty-test-20261121](archive/explorations/joint-asymmetry-novelty-test-20261121/REPORT.md) | Rejected as a core innovation: reduction to established joint-distribution methods |
| [readout-decoupling-novelty-audit-20261121](archive/explorations/readout-decoupling-novelty-audit-20261121/REPORT.md) | Rejected as a core innovation: a small linear model reproduces the headline |
| [learned-geometry-gate-20261121](archive/explorations/learned-geometry-gate-20261121/REPORT.md) | Complete tangent-model controls did not pass the predeclared effect gate |
| [geometry-condition-screen-20261121](archive/explorations/geometry-condition-screen-20261121/REPORT.md) | Several proposed geometric explanations rejected by reductions or counterexamples |
| [new-direction-screen-20261121](archive/explorations/new-direction-screen-20261121/REPORT.md) | Broad alternatives screened; underdetermined matrix endpoint selection retained |
| [endpoint-selection-deepening-20261121](archive/explorations/endpoint-selection-deepening-20261121/REPORT.md) | From a small witness to explicit static-weight neutrality conditions |
| [endpoint-classification-audit-20261121](archive/explorations/endpoint-classification-audit-20261121/REPORT.md) | Algebraic classification and theorem-level novelty review |
| [three-candidate-exploration-20261121](archive/explorations/three-candidate-exploration-20261121/REPORT.md) | Initial versions of the three retained theoretical candidates |
| [astra6-universality-deepening-20261121](archive/explorations/astra6-universality-deepening-20261121/REPORT.md) | Strengthened nonanalytic spectral-neutrality result |
| [astra6-blended-geometry-deepening-20261121](archive/explorations/astra6-blended-geometry-deepening-20261121/REPORT.md) | All-dimensional autonomous-loss endpoint theorem and exceptional initializations |
| [astra6-global-convergence-deepening-20261121](archive/explorations/astra6-global-convergence-deepening-20261121/REPORT.md) | Boundary-complete convergence and endpoint selection |

## What is in Git and what is in the release

The readable record is under `archive/`: reports, PDFs, Word/LaTeX drafts, figures, CSV/JSON results, protocols, and source snapshots. Reusable new code is also under the top-level `experiments/` directory.

Numerical arrays, checkpoints, and compressed run tables are in independently extractable ZIP attachments on the [research snapshot release](https://github.com/Argonriel/pullback-feature-learning/releases/tag/research-snapshot-2026-11-24). Extracting all selected attachments into `research/archive/` restores their original relative paths. Their total size is approximately 10.6 GB; individual study downloads are supported.

From the repository root:

```sh
python scripts/download_research_data.py --list
python scripts/download_research_data.py --study learned-geometry-gate-20261121
python scripts/download_research_data.py --study learned-geometry-gate-20261121 --verify-only
```

Omit `--study` to download all data. The helper verifies archive and member hashes and refuses to overwrite a different existing file. It does not train models or evaluate held-out data.

## Integrity and historical context

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) records every source-archive file, its original SHA-256, and whether it is in Git, in a release attachment, or intentionally local-only. [DATA_ASSETS.json](DATA_ASSETS.json) records attachment sizes, download links, and hashes.

Local instruction files, interpreter caches, and a historical Git backup are not published. Third-party paper copies and original external product photographs are not redistributed; citation URLs, acquisition manifests, licenses, selections, and evaluation outputs are retained. The publication manifest makes each omission explicit rather than presenting the shared copy as the complete original byte archive.

Original study files are preserved without rewriting historical claims or frozen hashes. Accordingly, some older reports describe the then-current local-only storage arrangement, and some run metadata records original workstation paths. Those statements are provenance, not instructions about the current repository. Original per-study manifest checks may report the deliberately omitted source copies; the publication manifest is the authoritative inventory for this shared snapshot.

Some later studies reuse earlier checkpoints or data. Package counts and result rows must not be added together as counts of independent experiments or seeds.
