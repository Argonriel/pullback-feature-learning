# Pullback Feature Learning

NumPy code for comparing parameter gradient descent with damped representation pullback updates in a small residual network on synthetic two-moons data.

This repository contains implementation, experiment configurations, and tests. Research notes, proposals, results, figures, and raw run records are maintained in a separate local research archive.

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

## Layout and local storage

- `src/pullback_feature_learning/`: model, optimizers, verification, and experiment runners.
- `configs/`: fixed experiment settings.
- `tests/`: numerical correctness, protocol, and execution tests.
- `docs/`, `reports/`, and `runs/`: ignored local paths for research materials and generated outputs. They may be directory links to an external research archive.

Advanced experiment runners expose their options through `python -m pullback_feature_learning.<module> --help`. Protocol-locked analyses require the original local protocol files, archived inputs, and referenced Git commits; a fresh clone alone does not include those research materials. The local archive preserves the complete project overview and reproduction commands.

Keep generated data, figures, research analysis, and report drafts out of Git. All repository content remains in English.
