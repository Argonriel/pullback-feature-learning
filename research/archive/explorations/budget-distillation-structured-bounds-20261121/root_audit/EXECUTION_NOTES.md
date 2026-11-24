# Audit execution notes

The independent image audit first stopped before any calculation because the GPU environment lacked Pillow. A package download failed because the execution environment could not resolve its package-index host. The already installed Pillow 13.0.0 package was copied from the workspace's NumPy environment into the GPU environment; both use Python 3.12.14. The next run passed all complete image, target, checkpoint, fit, risk and bootstrap replays. No experiment source, sample, fit or evaluation choice changed.

Cross-review by the structured-branch researcher identified three coverage additions in the independent image audit: bind temperatures to their model commit, check calibration IDs/labels against the frozen Fashion-MNIST split, and independently reconstruct preservation success. The first two checks are included in the reusable audit source. `distillation_criteria.py` separately executes all three additions without repeating the already passed 618,000 image forwards. Its result is retained alongside the complete replay result.

All mathematical checks use ordinary floating point. Their purpose is to detect implementation and bookkeeping errors; they are not a formal proof of machine rounding or confidence coverage.
