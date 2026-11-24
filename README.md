# Pullback Feature Learning

NumPy code for comparing parameter gradient descent with damped representation pullback updates in a small residual network on synthetic two-moons data.

This repository contains the core implementation, experiment configurations, tests, and subsequent research code. The November 24, 2026 snapshot also publishes research reports, theorem drafts, figures, result tables, and reproduction records.

Start with the [research index](research/README.md). Large numerical arrays and model checkpoints are provided as [release attachments](https://github.com/Argonriel/pullback-feature-learning/releases/tag/research-snapshot-2026-11-24), with file hashes and a download helper. The original local research archive remains unchanged.

The original controlled study found early feature-quality advantages for pullback, but not an overall advantage in final accuracy or elapsed time. Later empirical extensions include negative results. The latest three matrix-flow results are scoped theoretical candidates, not certified originality claims or demonstrated neural-network performance improvements.

## Installation

Use Python 3.10 or newer in a virtual environment:

```sh
python -m pip install -e .
```

For the recorded dependency versions, install `requirements-lock.txt` first, then install this package with `python -m pip install --no-deps -e .`. No GPU or external dataset download is needed.

## Verify and run

Run these commands from the repository root:

```sh
python -m unittest discover -s tests -v
python -m pullback_feature_learning.verify --output runs/verification.json
python -m pullback_feature_learning.pilot --config configs/pilot.json --output runs/pilot
```

Continue to the pilot only after tests pass and verification reports `"passed": true`. Output paths must be fresh; the commands refuse to overwrite earlier results.

## Layout and research data

- `src/pullback_feature_learning/`: model, optimizers, verification, and experiment runners.
- `configs/`: fixed experiment settings.
- `tests/`: numerical correctness, protocol, and execution tests.
- `experiments/`: subsequent empirical studies, independent checks, and matrix-flow investigations.
- `research/archive/`: published reports, proposal, figures, tables, source snapshots, and historical run metadata.
- `research/PUBLICATION_MANIFEST.json`: per-file storage locations and SHA-256 hashes, including explicit local-only exclusions.
- `research/DATA_ASSETS.json`: downloadable data archives and their SHA-256 hashes.
- `scripts/download_research_data.py`: verify or restore the numerical files under `research/archive/`.
- `docs/`, `reports/`, and `runs/`: ignored legacy working paths. Existing workspaces may link these to the original local archive; they are not included as links in this repository.

Advanced experiment runners expose their options through `python -m pullback_feature_learning.<module> --help`. Protocol-locked analyses require the corresponding archived inputs and referenced Git commits. A source clone includes the readable research record; download the relevant release data before replaying a study. Historical commands and recorded paths describe the original workstation layout and may need to be redirected to `research/archive/` on another machine.

All shared project documentation is in English. Historical integrity manifests describe the complete local study packages; the public snapshot manifest separately identifies local-only instruction files, caches, backups, and third-party source copies that are not redistributed. Original papers and external product photographs remain available through the cited sources and acquisition records.
