# Search and abandoned-route record

Date: November 21, 2026 (America/Los_Angeles).

This sub-audit primarily supplies an independent proof. The parent task performs the broader quantum-information/Jordan priority audit.

## Primary source inspection

- Query: `Daleckii Krein integration differentiation functions Hermitian operators C1 Frechet derivative original 1965`.
- Inspected [Vanni Noferini, 2016 author preprint, A Daleckii-Krein formula for the Frechet derivative of a generalized matrix function](https://eprints.maths.manchester.ac.uk/2462/1/gendk.pdf), section 2.4 and Theorem 2.10. This primary research source restates the classical finite-dimensional divided-difference derivative under a continuously differentiable scalar function. It is used only to identify the standard functional-calculus tool behind `h in C1 -> X maps to M_X in C1`; no new regularity result for matrix functions is claimed.
- Attempted original Daletskii-Krein translation DOI [10.1090/TRANS2/047/01](https://doi.org/10.1090/TRANS2/047/01). The web tool could not retrieve it. The original article was therefore not inspected in this sub-audit, and the inspected preprint is explicitly a later restatement.
- Search-engine secondary bibliographic pages were used only to locate those primary links. Their summaries were not used as independent mathematical support.

## Routes considered and not used

1. **Finite nonzero jet only.** Replacing analyticity by existence of a finite nonzero Taylor coefficient is valid but does not address flat kernels. It was superseded by the continuous spectral-pair separation proof.
2. **Unqualified global non-BKM condition.** This was initially tempting. The guarded six-by-six plateau construction rigorously refutes it, even for global closedness of the entire affine SPD slice.
3. **One sample separates all spectral classes.** Distinct continuous germs need not be simultaneously distinct at one parameter. The proof instead selects a finite set of sample parameters, then a generic linear combination of their commuting multipliers.
4. **Mellin transform or Titchmarsh citation alone.** The exact response transform has a nonvanishing symbol on the relevant half-plane, but treating that as injectivity of a half-line future convolution would need an additional theorem. This shortcut was abandoned. A direct weighted L2 energy proof using positive resolvents proves injectivity for all continuous germs.
5. **An unguarded two-by-two plateau example.** It shows local equivalence can fail while global closedness fails too. It does not refute global universality. Four scalar guard coordinates were added to bound the active spectral ratio throughout the whole affine SPD slice; the resulting n=6 example gives the required global counterexample.
6. **Flat monotone metric claim.** Explicitly rejected. The smooth flat examples lie outside the analytic Petz monotone class, so they do not invalidate the older analytic-class formulation or justify a broader claim about all finite diagnostics.

## Execution failure and recovery

The first exact-arithmetic validation invocation used the shell's `python` alias. It resolved to `C:\Users\liwei\AppData\Local\Microsoft\WindowsApps\python.exe` and failed before script execution with `ResourceUnavailable: the file cannot be accessed by the system`. No result file was produced by that failed invocation. The parent supplied the already-installed bundled runtime at `C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1118-904-0_26.1118.904.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe`; the same unchanged check script was then run with that executable. No dependency installation was attempted.

No numerical failure was hidden. No training, held-out validation cohort, new dependency installation, external message, or old-archive mutation occurred in this sub-audit.
