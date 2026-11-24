# Reproducing the local rank verification

Use the main NumPy environment recorded in the study root. No GPU, external
service or labels are needed by this component. The archived run is immutable;
recreate experiments in a new study directory with the same relative source
layout and use the root validation instructions for new training and policies.

Before any new validation states are generated, run:

```text
python certificate/experiment.py freeze
```

The root then verifies all method freezes and releases training. After
`validation/policy_commit.json` exists, run the four fixed verification stages:

```text
python certificate/experiment.py run --task rings --kind known
python certificate/experiment.py run --task waveform --kind known
python certificate/experiment.py run --task rings --kind empirical
python certificate/experiment.py run --task waveform --kind empirical
```

They can run in parallel. Each process limits BLAS to one thread. Every stage
checks the frozen source hashes and the input-policy manifest, refuses to
overwrite an output directory, retains failures, and writes a completion record.
The optional empirical stages retain their independent verification radii and
their DKW confidence allocation. They do not use labels or the analytic radial
law in the pointwise enclosure. Their known-law integrated summary fields are
descriptive only.

The output contracts are:

```text
certificate/fresh/{rings,waveform}/{seed}_local.npz
certificate/fresh/{rings,waveform}/{seed}_distribution.npz
certificate/fresh_empirical/{rings,waveform}/{seed}_local.npz
certificate/fresh_empirical/{rings,waveform}/{seed}_distribution.npz
```

`local_rank.pointwise_bound` evaluates the signed moment enclosure at any
radius. `known_cdf.q_bracket` evaluates the inclusion-probability enclosure at
any score. `root_audit/certificate.py` performs the separately frozen expected
gain confidence calculation; its input seeds and confidence events are distinct
from policy fitting and empirical-CDF verification. Consult its protocol before
running that stage.

To aggregate the saved results and regenerate the figure:

```text
python certificate/analyze.py
```

This updates descriptive summaries and figures only. After both distribution
branches exist, `python certificate/empirical_overlap_check.py` performs a
zero-forward finite-grid overlap check against the known synthetic law. That
diagnostic is descriptive, not a replacement for the uniform DKW theorem.

Portable copies of the three old rings pilot tables are in `development/` and
their hashes are listed in `pilot_input_manifest.json`. The waveform pilot is
in `transfer/pilot_policy/`. To replay the frozen development checks without
requiring an adjacent old study directory, use a new output filename:

```text
python certificate/replay_development_checks.py --output development_checks_replayed.json
```

This wrapper calls the unchanged frozen check implementation. It does not
refit models or change any fresh output. The original coarse and refined pilot
settings remain in `pilot/`, including the first source version. The theoretical
assumptions and the real-arithmetic versus float64 distinction are stated in
`THEORY.md` and `PROTOCOL.md`.
