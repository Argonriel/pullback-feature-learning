# Training-only coverage amendment

Recorded after discovery training ended (2026-11-22 01:51:05 UTC) and before any new held-out or population scores were generated or inspected. The original protocol and all 6000-step outputs remain frozen and authoritative for the original gate.

The original 64 trajectories contain 12 censored rings tangent runs: all eight tangent-GD runs and four tangent-PB runs. All nonlinear runs and all moons runs attained the primary loss. Censoring is not a rejection of the nonlinear hypothesis, proof of a capacity barrier, or evidence of predictive benefit. The common 0.1 parameter-step cap can particularly slow a tangent model whose useful initial directions are small.

This supplement resolves training-progress coverage only. Rerun all rings tangent arms for all four discovery pairs from the same initialization, keeping the same two step sizes, damping, controller, targets, architecture, data and objective, with a fixed limit of 60000 updates. There is one extension, with no further increase if it remains censored. Verify original shared target states agree exactly. Nonlinear runs are reused unchanged. Evaluation still occurs only after training finishes.

The original C0 decision continues to report missing primary endpoints. A separate explicitly supplemental C0 decision may use the extended tangent states and the original nonlinear states. It uses the original effect floor, sign agreement, refinement tolerance and primary endpoint. No new loss endpoint or score is selected. This amendment is an exploratory coverage repair, not a retroactive claim that the original frozen gate passed.

If the supplemental gate survives, the previously reserved independent confirmation cohort tests the same completed comparison with this 60000-update cap declared in advance; conditional fixed-metric and damping controls also retain their original definitions. If it fails, no expansion is authorized by an attractive secondary endpoint. Neither iteration counts nor elapsed times from the unequal coverage caps support speed/cost claims.

This action is motivated solely by training censoring, not predictive results. The extension's source/config hashes, start time, and exact overlap replay must be archived.
