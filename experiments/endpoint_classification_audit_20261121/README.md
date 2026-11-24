# Finite-damping classification check

`verify_finite_damping.py` checks one new 6x6 direct sum with coupled central
measurements. Two fixed SPD quadratic loss weights are compared at two RK4 step
sizes, and an independent solve of the central invariant equations predicts the
endpoint. This is numerical consistency evidence for a proved structural result.

Requires Python and NumPy. Pass a fresh output directory:

```powershell
python verify_finite_damping.py --output C:\path\to\fresh-output
```

The script preserves all four endpoint matrices, measurement and invariant
residuals, the bases, target, and weights. It refuses to reuse an existing output
directory. Research proofs and the novelty review are maintained in the separate
research archive under `explorations/endpoint-classification-audit-20261121`.
