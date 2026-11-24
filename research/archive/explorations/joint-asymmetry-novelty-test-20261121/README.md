# Joint asymmetry novelty audit

This archive records the decision to retire joint input and posterior asymmetry as the main innovation claim and to investigate a different mechanism in representation optimization. The replacement is a research hypothesis, not an established innovation.

- `REPORT.md`: English decision report, reductions, numerical evidence, and replacement question.
- `REVIEW_NOTES.md`: consolidated independent-review findings and limitations.
- `source/verify_reduction.py`: independent polytope and joint-law verifier.
- `numerical_audit/audit_data.json`: all 300 support cases and 250 joint-identity records, plus the shared-constraint example.
- `numerical_audit/audit_summary.json`: numerical results, environment, and verifier digest.
- `DECISION.json`: structured research decision and scope.
- `manifest.json`: file hashes and sizes for all content files.

Reproduce with Python 3.12 and NumPy, using a new output directory:

```text
python source/verify_reduction.py --output path/to/new-output-directory
```

The script refuses to overwrite an existing output directory. The recorded runtime is Python 3.12.14 and NumPy 2.5.5. Numerical checks verify the algebra and conservatism example; they are not a formal interval-arithmetic proof or an experiment demonstrating prediction improvement.

No historical files were overwritten, no models were trained, and no prior test dataset was rescored. Detailed working review memos remain in the workspace; this archive contains the curated decision package.
