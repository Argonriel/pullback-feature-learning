# Finite blended geometry deepening

Standalone reproducible checks for the generic fixed-quadratic-loss endpoint
submersion theorem and its initialization qualifications. Research prose and
results live in the separate research package; this directory contains only new
reusable source and this usage note. No old result, source, or shared dependency
is modified or imported.

Requirements: Python 3.12+, NumPy (validated with Python 3.12.14 / NumPy 2.3.5).
The exact exceptional-state checks use standard-library Fraction arithmetic.
No packages are installed by these scripts.

Run with an available Python executable; the Windows application alias may be
inaccessible. All output paths below must be new directories.

```text
python validate_static_range.py --output NEW_ROOT_CHECK
python refine_endpoint_difference.py --input NEW_ROOT_CHECK/results.json --output NEW_FD_REFINEMENT
python static_loss/check_area_rank.py --output NEW_AREA_CHECK
python exceptional/exact_rank_checks.py --output NEW_EXACT_CHECK
python exceptional/finite_difference_check.py --output NEW_BRACKET_FD_CHECK
```

The first root check intentionally preserves the original tiny-increment endpoint
finite-difference diagnostic, which was roundoff limited and had a false decay
flag. The separate follow-up recovers the expected truncation regime at larger
increments; it does not replace or overwrite the first output. Source/dependency
hashes and full inputs are embedded in each result.

No training, random search, data split, or held-out cohort is involved. The 3x3
endpoint check, 4x4 Krylov-rank check, and exact exceptional-state certificates
have distinct discriminating purposes; they are not performance experiments.
