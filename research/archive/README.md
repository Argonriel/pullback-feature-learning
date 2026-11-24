# Pullback Feature Learning: local research archive

This folder is the primary local home for project analysis, research writing, experiment data, plots, and reproducibility records. Repository code, configurations, and tests remain in the separate source checkout. All project materials are in English.

## Start here

- [Predicting optimizer updates and choosing the second view](explorations/predictive-symmetry-20261120/README.md)
- [Does a smaller GD learning rate just need more steps? Matched-time and matched-loss controls](explorations/rate-clock/README.md)
- [When the preferred second-view rule changes during training](explorations/switch-time/README.md)
- [Which rings inputs deserve a second view? Final insert, proof, and experiments](explorations/budgeted-rotation/README.md)
- [How networks learn radius and angle, and the effect of the unused feature](explorations/rings-mechanism/README.md)
- [Crossed learning-rate and budget controls on concentric rings](explorations/rings-crossed-controls/README.md)
- [New experiments on early pullback updates](explorations/early-pullback/README.md)

- [Project plan and proposal alignment](docs/project_plan.md)
- [Original proposal](docs/proposal.tex)
- [Project overview before the repository cleanup](PROJECT_OVERVIEW.md)
- [Mathematical derivation](docs/mathematics.md)
- [Literature review](docs/literature.md) and [references](docs/references.bib)
- [Formal experiment interpretation](docs/formal_findings.md)
- [Long-training interpretation](docs/long_training_findings.md)
- [Frozen-feature probe findings](docs/probe_findings.md)
- [Representation and displacement findings](docs/representation_findings.md)

## Data and figures

`reports/` preserves the verified report archives, raw data, metrics, checkpoint weights, audits, SVG figures, and result summaries. `runs/` preserves the original local execution outputs, including PNG previews. `configs/` is the configuration snapshot at the time of this move. The layout preserves relative links between research documents and report artifacts.

- [Formal study](reports/formal-v1/README.md)
- [Long-training diagnostic](reports/long-training-v1/README.md)
- [Frozen-feature probes](reports/probes-v1/README.md)
- [Feature clouds and spectra](reports/representation-analysis-v1/README.md)
- [Refined displacement comparison](reports/displacement-refinement-v1/README.md)

## Preservation and future work

The archive was copied and checked file by file using SHA-256 before any repository files were removed. [The manifest](provenance/archive_manifest.json) records original locations, archived paths, sizes, and checksums. `provenance/workspace_tmp/` includes the independent audit scripts and a verified Git bundle containing the pre-cleanup repository history.

Future analysis notes, report drafts, and generated outputs belong here. The source checkout may contain ignored directory links named `docs`, `reports`, and `runs` pointing into this folder so existing experiment commands keep working. Those links and their contents are not included in new Git commits.

This cleanup removes research materials from the current GitHub file tree. It does not erase historical Git commits. The local Git bundle preserves the original research provenance independently of the remote repository.
