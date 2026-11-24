# Literature-informed symmetric allocation frontier

This new cycle was authorized by the user on2026-11-21. Prior completed studies and their archives remain unchanged. The previous32 primary models are now development data, not fresh replications.

## Three objectives

1. Compare a stronger input-only symmetric Jensen-gain scorer with a genuinely competitive folded symmetric baseline, and quantify remaining oracle headroom.
2. Test a genuinely two-dimensional invariant on a new harmonic-amplitude task whose posterior cannot be reduced to total input norm.
3. Measure the cost and tightness of independent, quantitative gain certificates. Exploit exact selector symmetry when valid; do not describe this as an acceleration of the same asymmetric policy's certificate.

Primary literature is searched before primary methods are frozen. Group averaging, learned test-time augmentation, and unlabeled-identifiability obstructions are existing ideas. The report must state the narrower implementation/verification contribution and distinguish inference-time augmentation from online parameter adaptation.

## Common candidate and strong controls

The candidate ranks a multilinearly interpolated, exactly half-turn-even Jensen-gain table and selects exactly64/256 inputs for one additional model view. It uses65,536 calibration forwards:128quotient anchors×512full-orbit angles in1D, or16×16invariant anchors×256angles in2D. The folded baseline uses32,768IID calibration inputs plus their partners, also65,536forwards. These are matched forward budgets, not identical independent-sample counts or CPU times.

A declared histogram development grid prevents a deliberately weak comparator. In1D, quotient×folded-angle bins are(16,8),(32,16),(64,32). In2D they are(4,4,8),(8,8,16),(16,16,16). Choose one configuration per task from independent development selected-Jensen gain, without labels. The legacy128-cell histogram and nearest-grid interpolation are retained as controls. An exact-Jensen oracle may evaluate all partners, but is charged two model calls/input and is never presented as a deployable1.25-call policy.

## New two-invariant task

R1,R2 are independent Uniform[0.4,1.4]; phase is independent uniform. X=(R1cosphi,R1sinphi,R2cos2phi,R2sin2phi). The SO(2)action rotates the first pair bytheta and the second pair by2theta. Its half-turn flips only the first pair. The two amplitudes separate orbits on this support.

With Zj=2(Rj−0.9), the posterior is0.5+0.4tanh(4Z1)tanh(4Z2). It depends on both amplitudes, with prevalence0.5. The task law and bounded four-seed pilot recipes were frozen before pilot execution. The pilot compared PB8/500,PB4/500 andPB8/1000 with damping0.1; all outcomes are retained. A supervised development-risk rule selects training only. This must not be described as strong raw-model calibration if its fresh outcomes show overconfidence or underfitting.

## Independent streams and replications

Three fresh primary cohorts each contain16trained models. Rings keep PB8/500 and80parameters. The scalar harmonic task keeps PB8/500 and144parameters. The two-invariant task uses its separately frozen pilot-selected144parameter recipe. Each model receives65,536fresh IID performance inputs. No GD context cohort is added this cycle.

| Cohort | Train/init bases | Folded calibration | Performance/ties/labels bases | Certificate inputs/ties |
|---|---|---|---|---|
| rings |1161000/1162000|1163000|1164000/1165000/1166000|1121000/1122000|
| waveform |1111000/1112000|1113000|1114000/1115000/1116000|1123000/1124000|
| two_invariants |1141000/1142000|1143000|1144000/1145000/1146000|1151000/1152000|

Add model index1..16 to each base. Two-invariant pilot bases1131000/1132000/1133000 are disjoint. Method-development streams1191000/1192000 plus cohort offsets are also disjoint. Root approval after all primary freezes is required before fresh-state generation; this is an internal scientific release, not a new request for user permission.

Pre-freeze correction: the initially proposed rings1101/1102 bases overlapped an already completed old-model certificate pilot. This was caught before any fresh state existed. All six rings streams were moved to1161..1166; the original pilot is preserved. Adapter checks use1201 streams. The eight fixed policies are raw, full half-turn, confidence, legacy histogram, development-selected histogram, interpolated grid, nearest grid, and evaluation-only Jensen oracle. Historical asymmetric policies and alternative-angle actions are outside this cycle's fixed-half-turn comparison.

## Certificate and interpretation boundaries

For structurally even input-only selectors, expected logistic-loss improvement equals expected batch-mean selected Jensen gap under an invariant input law and a half-turn-invariant posterior. This works for any invariant dimension. Verification can evaluate both views only on the selected-set union; it need not predict unselected inputs. Actual counts must be recorded. It is an offline verification saving, not a measured serving speedup.

Independent batches, rather than individual mask-correlated inputs, are the empirical-Bernstein statistical units. Fixed clipped nonnegative Jensen observations give lower bounds on absolute gain; signed relative gain needs its own valid range/tail treatment. The declared batch counts are128,512,2048, with primary512. Clipping constants are0.25forrings and0.5forbothharmonictasks, chosen or inherited from development. A pre-fresh addendum also exploits top-quota bounded differences, with a separate proof accounting for within-batch selection dependence. All48models×threeclaims(twoabsolute andonepairedrelative)×threebatchlooks×twoboundmethods share a0.05family budget (864events). The original432-event source is preserved as historical; final_compare.py recalculates both methods at0.05/864 and alone supplies final family claims. No uncorrected stopping or fresh retuning is allowed.

Certificates relative to the strong folded baseline may fail to be positive even if empirical comparisons favor the candidate. Preserve and explain that distinction. A positive loss guarantee is not a classification-error guarantee. Synthetic transfer does not establish real-data utility, formal floating-point certification, or novelty over the complete literature.
