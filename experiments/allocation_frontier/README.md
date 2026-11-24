# Two-stage PB allocation frontier

This reusable study compares four prespecified training/calibration allocations under 4,352 accepted labels per fitted procedure and the unchanged nominal PB arithmetic reference. Eight development pairs select exactly one recipe using sampled-label calibrated raw NLL. Sixteen new confirmation pairs compare the committed recipe with the 2,304-training baseline. If the baseline wins development, confirmation reports its absolute performance without promoting another candidate or making a redundant improvement test.

Read the archived `PROTOCOL.md` before executing. Primary files are created exclusively and existing studies must never be overwritten. The portable interface accepts an explicit output directory and works from either repository or archived source:

```text
PYTHON reproduce.py preflight --out NEW_RESULTS
PYTHON reproduce.py freeze --out NEW_RESULTS
```

After internal scientific review, write the `ROOT_APPROVAL.json` matching that new source freeze. Then run:

```text
PYTHON reproduce.py train --phase development --out NEW_RESULTS
PYTHON reproduce.py calibrate --phase development --out NEW_RESULTS
PYTHON reproduce.py score --phase development --out NEW_RESULTS
PYTHON reproduce.py select --out NEW_RESULTS
PYTHON reproduce.py train --phase confirmation --out NEW_RESULTS
PYTHON reproduce.py calibrate --phase confirmation --out NEW_RESULTS
PYTHON reproduce.py score --phase confirmation --out NEW_RESULTS
PYTHON reproduce.py analyze --out NEW_RESULTS
```

After all outcomes are committed, the separately frozen complete-procedure benchmark may run only in a coordinator-granted exclusive CPU window:

```text
PYTHON reproduce.py timing --out NEW_RESULTS --exclusive-confirmed
PYTHON reproduce.py account --out NEW_RESULTS
PYTHON reproduce.py report --out NEW_RESULTS
```

NumPy, threadpoolctl and Matplotlib are required. Numerical stages run with one BLAS thread. Ordinary fit elapsed times can include concurrent work and are descriptive only; the timing stage measures complete training plus raw/full calibration on one fixed development pair.

The primary freeze covers `runner.py`, `model.py`, `preflight.py`, `timing.py` and `reference/`. The portable wrapper and postprocessor do not modify fitted states or primary outputs. All fitting/calibration is label-only; posterior expected metrics are restricted to final confirmation evaluation. Full development selection cost, including 524,288 labeled evaluation items, is reported separately from the per-procedure 4,352-label budget.

The inherited model, data law and development-state reference are preserved byte-for-byte from the prior allocation implementation. No earlier study files are changed.
