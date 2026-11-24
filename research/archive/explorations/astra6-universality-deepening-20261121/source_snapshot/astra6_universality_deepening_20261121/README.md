# Nonanalytic spectral universality: bounded validation

This directory contains the new reusable validation source for the independent
November 21, 2026 universality deepening. All prose, results, literature records,
and provenance belong to the separate research package.

`validate_nonanalytic.py` requires Python 3.12 and NumPy. It compares the exact
scalar response transform with symmetric differences in a fixed positive loss
weight along the independently computed matrix path at zero damping. Its seven
cases cover BKM, SLD, a smooth flat defect, and a smooth BKM plateau, with two
quadrature settings and two weight-difference steps. It also checks exact
classical Gaussian first-missed moments for one through four atoms.

Run from any directory, supplying a fresh research output directory:

```text
python validate_nonanalytic.py --output NEW_RESEARCH_DIRECTORY
```

The research package records the actual bundled Python executable used. No
training, reserved validation samples, finite-damping experiment, package
installation, or shared dependency modification is needed. The script does not
prove the continuum theorem or certify its originality.
