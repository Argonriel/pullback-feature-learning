# Independent validation of the frontier/replication/residual study

The portable code snapshot is archived with the full study under
`frontier-replication-residuals-20261121/root_audit/`. Execute those archived
copies so their relative paths resolve the committed branch data. The scripts
here preserve reusable implementation; large research data do not belong in Git.

Audits default to read-only. `--record` deliberately writes a new audit JSON and
should be used only in a fresh working copy. CPU audits require NumPy and
threadpoolctl; the CNN audit also requires PyTorch/CUDA and Pillow. The source
selection audit accepts `--zip PATH` for the original official UT Zappos ZIP.

The scripts independently reconstruct selection, forward outputs, temperatures,
uncertainty, evidence, moment bounds, stress metrics and timing arithmetic.
They do not independently repeat every full training trajectory. The theory
checks are ordinary floating-point verification, not formal directed-rounding
proof. Refer to the archived English report for statistical/structural limits.
