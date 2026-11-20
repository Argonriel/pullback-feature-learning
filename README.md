# Pullback Feature Learning

A reproducible, small-scale study of how optimization geometry changes learned representations. We compare ordinary parameter gradient descent with a **damped pullback of the Euclidean representation metric**, using an 80-parameter residual network and noisy two-moons data.

The research question is whether the pullback update produces more useful feature motion, and whether any benefit remains after accounting for computational cost. This is a Math 277 course project by Jacob Cai and Jiayi Liwei.

## Current status

Stages 1-4 are implemented and verified:

- A fixed experimental specification, explicit NumPy implementation, and reproducibility records.
- Mathematical derivations and 26 passing numerical, selection-policy, and execution tests.
- A 200-step paired CPU pilot, with learning curves, feature clouds, checkpoint spectra, timings, and raw records.
- A review of seven primary research papers, with precise distinctions from Fisher and Wasserstein natural gradients and an explicit treatment of selection bias.
- A predeclared validation-only tuning batch: 12 configurations per method, three paired development seeds, and 2,000 updates per run (72 runs, all completed).

The pilot uses one paired seed and illustrative hyperparameters. **Neither the pilot nor the selected development scores establish optimizer superiority.** Stage 4 selected GD learning rate **0.125** and pullback learning rate **8** with damping **0.1**, using mean terminal validation loss. Test and formal data remain ungenerated and unevaluated. At least three fresh formal paired seeds and held-out linear probes remain subsequent stages. The selected damping is at the upper boundary of the declared search grid; no post-hoc expansion was performed.

## Read the project

| Resource | Contents |
| --- | --- |
| [Experimental protocol](docs/protocol.md) | Data, initialization, timing boundaries, pilot exit criteria, and subsequent stages |
| [Mathematics](docs/mathematics.md) | Update derivation, SVD factors, local descent, nullspaces, and damping |
| [Literature review](docs/literature.md) | Seven primary papers and their specific consequences for this experiment |
| [References](docs/references.bib) | Verified BibTeX metadata |
| [Original proposal](docs/proposal.tex) | The source project scope |
| [Numerical verification](reports/verification.json) | Measured errors and acceptance tolerances |
| [Pilot results](reports/pilot/README.md) | Recorded results, curves, and raw artifacts |
| [Pilot interpretation](docs/pilot_findings.md) | What the measurements establish and the next experimental budget |
| [Predeclared tuning protocol](docs/tuning_protocol.md) | Candidate grids, equal trial budgets, selection rules, and formal endpoints |
| [Tuning results](reports/tuning-v1/README.md) | All 72 runs, candidate rankings, raw records, and measured costs |
| [Tuning interpretation](docs/tuning_findings.md) | Selected settings, limitations, and what is now frozen |
| [Frozen formal configuration](configs/formal.json) | Fresh seeds and separate iteration/time endpoints; not yet executed |

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

To reproduce the fixed stage-4 tuning batch, use its preregistration commit:

```sh
python -m pullback_feature_learning.tuning --config configs/tuning.json --output runs/tuning --preregistration-commit 26e278dbd708b7eab6b687f1e398e0741c225164
```

The command verifies that the JSON settings match that Git commit before running. It records a fixed shuffled schedule and refuses to overwrite earlier output. It generates only the development train/validation splits. `configs/formal.json` is a frozen specification for the next stage; the formal experiment runner is not implemented or invoked by the tuning command.

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
