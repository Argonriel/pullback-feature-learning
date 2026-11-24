# External mathematical and runtime references

- Andreas Maurer and Massimiliano Pontil (2009), [Empirical Bernstein Bounds and Sample Variance Penalization](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf), Theorem 4. The root certificate uses its one-sided bound with the unbiased sample variance, after scaling a bounded antithetic observation. Model-family confidence uses an explicit union bound.
- Pascal Massart (1990), [The Tight Constant in the Dvoretzky-Kiefer-Wolfowitz Inequality](https://doi.org/10.1214/aop/1176990746), Annals of Probability 18(3), 1269-1283. The rank-method calibration analysis uses the uniform empirical-CDF inequality with constant 2.
- PyTorch, [previous-version installation instructions](https://pytorch.org/get-started/previous-versions/). The isolated benchmark environment installs torch 2.7.1 with the official CUDA 12.8 wheel index. Actual device execution and package versions are recorded locally in gpu/setup_record.json and the training commit; compatibility is not inferred solely from documentation.

All performance numbers, policy comparisons, local derivative enclosures, quotas, replay checks and timing results are produced by this study, not taken from these sources.
