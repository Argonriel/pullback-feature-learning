# Independent PB confirmation protocol

This study confirms frozen allocation methods on 16 newly trained original-ring models. Development uses previously archived models only. No model using training seeds 921001--921016 or initialization seeds 922001--922016 may be generated or opened before the root records approval after all method freezes.

## Models and observations

The unchanged 80-parameter, two-block residual tanh model is initialized with seed 922000 + pair. The unchanged balanced 256-example training generator uses seed 921000 + pair. Pullback training takes 500 updates at rate 8 with damping 0.1. Initial, step-100, and step-500 parameters and all training data are retained. Nonfinite states or failed policies are recorded and are never silently excluded.

Each model is evaluated on 65,536 independently sampled ring-mixture observations using seed 926000 + pair, arranged in 256 batches of 256. First predictions are available to all methods. Limited policies must select exactly 64 observations in every batch and make one extra prediction at the selected transformed inputs. The original observation law and its half-turn symmetry are known. The radial class posterior is used only after the policy commit, for expected logistic risk and separately generated label checks. It is never a calibration input.

## Calibration and controls

The new top-64 selectors and all tuning choices are fixed by the selection workstream using old models. Their API and frozen sources enter the validation freeze. Calibration uses seed 923000 + pair. The three frozen candidates are `rank_balanced`, `rank_even`, and `rank_unbalanced`; they share 66,048 network evaluations and an additional 131,072 coordinate-only draws for the score distribution. Coordinate draws do not count as network forwards. The unchanged previous continuous policy is independently fit on its 129 radial anchors and 512 angular nodes, with all resulting tables retained. Its 66,048 network evaluations are charged as offline calibration. Grouped balanced, unbalanced, even, uniform and confidence controls use the unchanged one-of-four construction.

The legacy radius and folded half-turn gates use 33,024 IID calibration observations (seed 925000 + pair) plus their antipodes, also costing 66,048 forwards. These are strong invariant gate controls. A secondary confidence plus radius-specific angle policy uses 64 IID radii (seed 923100 + pair), 128 angular nodes and independently sampled phases (seed 924000 + pair), costing 8,192 forwards. All full and raw references are retained. These budgets differ and will be reported explicitly; the comparison does not imply equal calibration cost.

Ties use seed 927000 + pair; grouped categorical choices use 928000 + pair; new action mixtures use 928100 + pair; sampled labels use 928200 + pair; timing order uses 928300 + pair. These are separate from test coordinates and the root certificate stream 941000 + pair. Policy tables, model hashes and fitting records are committed before any fresh test risk is computed.

## Outcomes and audits

Expected logistic risk is primary. Independently sampled-label logistic risk and classification error are secondary checks. The independent unit is the trained model, not angular nodes, test observations or timing repeats. Paired contrasts use the same test observations, first predictions, ties and label draws within a model. Report all 16 model outcomes, means, wins and percentile bootstrap intervals over 16 paired model clusters using 20,000 draws (seed 928999). The primary family comprises the new balanced selector against global confidence and against the new symmetric selector; use 97.5% intervals for those two contrasts, with ordinary 95% intervals also shown. Other comparisons are descriptive.

Three secondary posterior-world diagnostics use the same fixed IID test inputs: p(r)=1/2, p(r)=sigmoid((r-0.85)/0.15), and its complement. They illustrate posterior sensitivity and are not empirical identification of an unknown posterior, sharp extrema, or substitutes for the separate deterministic bounds.

Every scored deployment batch records actual prediction-call sizes and quota. The full data path is executed, including routing and selected second predictions. The standard limited policy requires [256, 64] forward examples; raw uses [256]; full uses [256, 256]. Hashes of policies, generated observations, masks and final predictions support replay. Calibration, mathematical verification, certificate samples and test scoring are separately charged and never presented as deployment work.

Resident CPU latency uses actual batch deployments, randomized policy order and warmups. It includes routing and second calls. Full half-turn and global confidence implementations use direct negation and avoid unnecessary radius/angle work. A separate equivalent full-half-turn timing control concatenates x and -x into one 512-example forward invocation, with output equivalence verified. Latency break-even uses the faster median of the ordinary two-call and fused full implementations for each model. No reduction in forward counts is called a speedup unless measured latency improves. Calibration amortization is reported only against a faster per-batch saving; otherwise the latency break-even is absent.

The GPU extension is conditional. Its readiness assessment does not install a runtime or claim a benchmark. Any later real-network GPU experiment requires its own frozen protocol and an explicit root go-ahead after progress in the main study.
