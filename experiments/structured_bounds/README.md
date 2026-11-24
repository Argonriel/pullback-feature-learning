# Structured transformation-bound experiment sources

These Python files are byte-identical copies of the completed structured-bound study's runtime and reporting sources. The `reference` directory contains the frozen residual-model and selector implementation reused by the experiment. `run.py` defines the prospective configuration and verifies a source/input manifest before execution.

Research protocols, derivations, fitted arrays, raw verification streams, reports and figures belong in the separate research archive, not in Git. Execute from a prepared research directory containing the matching `FREEZE.json`, `ROOT_APPROVAL.json`, historical arrays, `PROTOCOL.md` and `THEORY.md`.

From the archived standalone branch, replay into a fresh directory with:

```text
python reproduce.py --output ../structured_replay
```

The replay copies committed inputs and sources, verifies hashes, then runs deterministic constants, algebra checks, independent parameter intervals, same-law certificate streams, independent posterior stress and analysis. It never overwrites an existing result directory. Python dependencies are NumPy, Matplotlib and threadpoolctl. The original workspace environment was `.venv/Scripts/python.exe`; the archived ENVIRONMENT.json records exact versions.

`assets.py` documents the one-time historical input copy. It is not needed to replay the standalone archived branch. Model training and selector fitting are unchanged historical work; this runner performs no new fitting.
