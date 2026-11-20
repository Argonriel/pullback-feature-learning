# Pullback Feature Learning

A reproducible, small-scale study of how optimization geometry changes learned representations. We compare ordinary parameter gradient descent with a **damped pullback of the Euclidean representation metric**, using an 80-parameter residual network and noisy two-moons data.

The research question is whether the pullback update produces more useful feature motion, and whether any benefit remains after accounting for computational cost. This is a Math 277 course project by Jacob Cai and Jiayi Liwei.

## Current status

Stages 1-3 are implemented and verified:

- A fixed experimental specification, explicit NumPy implementation, and reproducibility records.
- Mathematical derivations and 10 passing independent numerical tests.
- A 200-step paired CPU pilot, with learning curves, feature clouds, checkpoint spectra, timings, and raw records.
- A review of six primary research papers, with precise distinctions from Fisher and Wasserstein natural gradients.

The pilot uses one paired seed and illustrative hyperparameters. **It is not a tuned comparison or a claim of optimizer superiority.** Test data are specified but neither generated nor evaluated. Equal-budget tuning, at least three formal paired seeds, and held-out linear probes remain subsequent stages.

## Read the project

| Resource | Contents |
| --- | --- |
| [Experimental protocol](docs/protocol.md) | Data, initialization, timing boundaries, pilot exit criteria, and subsequent stages |
| [Mathematics](docs/mathematics.md) | Update derivation, SVD factors, local descent, nullspaces, and damping |
| [Literature review](docs/literature.md) | Six primary papers and their specific consequences for this experiment |
| [References](docs/references.bib) | Verified BibTeX metadata |
| [Original proposal](docs/proposal.tex) | The source project scope |
| [Numerical verification](reports/verification.json) | Measured errors and acceptance tolerances |
| [Pilot results](reports/pilot/README.md) | Recorded results, curves, and raw artifacts |
| [Pilot interpretation](docs/pilot_findings.md) | What the measurements establish and the next experimental budget |

## Installation

Use Python 3.10 or newer. The recorded run used Python 3.12 on Windows, float64, CPU, and one BLAS thread. No GPU or deep-learning framework is required.

Create and activate a virtual environment. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

On macOS/Linux, replace the activation command with `source .venv/bin/activate`. The lock file records the measured environment. For a compatible dependency installation on other Python/platform combinations, use `python -m pip install -e .`; record the resulting versions when comparing timings.

## Verify, then run

Run the numerical tests and verification first. Continue only when both commands succeed and verification reports `"passed": true`.

```sh
python -m unittest discover -s tests -v
python -m pullback_feature_learning.verify --output runs/verification.json
```

Then execute the short pilot:

```sh
python -m pullback_feature_learning.pilot --config configs/pilot.json --output runs/pilot
```

Each command refuses to overwrite its existing output. Choose a fresh path for subsequent runs. The pilot is not a hyperparameter sweep and does not run the later formal experiments.

Generated outputs include the resolved configuration, train/validation arrays, software and source fingerprints, per-step CSV records, initial/middle/final checkpoints, singular values and gradient alignment, separate training/diagnostic timings, and SVG/PNG figures. The checked-in pilot omits redundant PNG copies; fresh runs produce both formats.

## Method at a glance

For representations `Z = R_X(theta)`, full Jacobian `J`, and representation loss gradient `g`, solve

```text
GD:        u = -J.T @ g
Pullback:  (J.T @ J + lambda * I) @ u = -J.T @ g
Update:    theta_next = theta + learning_rate * u
```

The code uses independent reverse-mode differentiation for GD, so GD is not charged for an unnecessary full Jacobian. Pullback training includes the required Jacobian, Gram matrix, and linear solve. It uses a solve rather than an explicit matrix inverse. Both representation coordinates contribute to the metric, even though the fixed readout uses only the first coordinate.

The representation metric sums over samples; the logistic objective averages over samples. Damping regularizes the update direction, not the model weights. The two methods have separate learning rates. Detailed qualifications and connections to prior work are in the mathematics and literature notes.

## Repository layout

```text
configs/                         Saved experiment settings
docs/                            Protocol, derivations, literature, interpretation
src/pullback_feature_learning/   Model, data, verification, pilot runner
tests/                           Independent numerical correctness checks
reports/                         Checked-in verification and pilot evidence
runs/                            New local runs (ignored by Git)
```

All source, documentation, plot labels, and experiment records are in English. No external dataset download is needed. New experiments should preserve prior outputs and clearly distinguish pilot evidence from final evaluation.
