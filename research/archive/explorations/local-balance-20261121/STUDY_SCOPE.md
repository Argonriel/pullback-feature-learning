# Local balance, stronger allocation, and independent confirmation

This new study follows the completed continuous-confirmation-20261121 archive. Historical code, data, failures and reports remain unchanged.

The primary question is original Idea 4: can conditional-balance analysis for an unknown radial posterior become useful for a continuous neural predictor? The work has three required parts: local deterministic bounds; richer exact-64-of-256 allocation with its true selection bias included; and independently seeded validation after a method freeze. A GPU efficiency extension is conditional on substantive progress in the first parts. It concerns actual inference costs, not scaling the current dense pullback training matrix.

Development may use all previously recorded models. New PB models use training seeds 921001 through 921016 and initialization seeds 922001 through 922016. No new model is generated or read before the new methods and evaluation protocol are fixed. The architecture, 500 pullback updates, learning rate 8 and damping 0.1 remain unchanged. Test labels are forbidden during method fitting. Failed methods, negative contrasts and costs are retained.

The three work areas are bounds/, selection/, and validation/. GPU readiness and any conditional benchmark belong in gpu/. The root independently reviews the mathematics, evaluates a finite-sample certificate, checks quota/call accounting and freezes, and assembles the final report. Frozen dependencies are copied into source/, legacy/, and reference/; they are not silently changed.

The local bound theorem is in real arithmetic. Ordinary floating-point interval-style calculations are not described as formal directed-rounding certification. Numerical estimates of conditional imbalance are not promoted to exact identification. A known symmetry, known input law, and an arbitrary hidden posterior are distinct assumptions and are reported separately.

Primary reporting units are independent trained models. Inputs, angular nodes and timing repeats are not counted as independent model replications. Deployment costs include routing and actual second predictions; offline calibration, verification and diagnostic costs are itemized separately.

All research materials are in English. The final user-facing explanation is plain Chinese.
