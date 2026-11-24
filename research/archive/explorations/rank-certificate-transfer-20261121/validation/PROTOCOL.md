# Independent rings and harmonic-task confirmation

No fresh state may be generated or opened until root approval links all completed method, task, certificate and validation freezes. Prior models and the four-seed task pilot are development data. Neither validation nor certificate outcomes may change a scientific setting.

## Cohorts and fixed training

The primary confirmation has two independently seeded cohorts of 16 trained models. Original rings retain the 80-parameter two-block residual network, 256 exactly class-balanced training examples, and 500 pullback updates at rate 8 and damping 0.1. Seeds are training 1001001--1001016 and initialization 1002001--1002016.

The transfer task is a four-dimensional two-harmonic representation: x=(cos(phi),sin(phi),r cos(2phi),r sin(2phi)). The action rotates the first pair by alpha and the second pair by 2alpha. Its exact half-turn partner flips only the first pair. The scalar quotient r=.4+u follows a balanced Beta(2,5)/Beta(5,2) mixture, with posterior u^3/[u^3+(1-u)^3]. The task changes the group representation, input law, label noise and architecture; it is not an equal-difficulty or equal-architecture comparison with rings. The quotient is still scalar and monotone with total norm on this manifold; this does not establish transfer to arbitrary multidimensional quotients.

The task/model workstream freezes a 144-parameter, two-block width-8 residual network after four development seeds and all predeclared pilot recipes. The primary transfer model uses 500 PB updates at rate 8 and damping .1. A secondary GD context uses rate .3 and 500 steps from the same initial state/data, chosen on the four pilot models. GD contributes raw and full-half-turn context only, not an additional selector-confirmation cohort. Equal update counts are not matched computation, and no optimizer-superiority claim is intended. Fresh transfer training seeds are 1041001--1041016; initialization seeds are 1042001--1042016. All initial, step-100 and step-500 states, training inputs and labels are retained, including any failure.

## Policies and label-free calibration

Each primary PB model evaluates raw, full half-turn, global confidence, folded symmetric gate, rank-balanced, rank-even, rank-unbalanced, confidence plus quotient-specific angle, and full quotient-specific angle. Every limited policy selects exactly 64 of 256 inputs. All second views use the cohort's actual group action; four-dimensional full negation is never used as its half-turn.

The frozen rank family uses tau=1, 129 quotient anchors and 512 angular nodes (66,048 model forwards), plus 131,072 coordinate-only quotient draws. The three rank policies share this fit. The folded gate uses 33,024 IID coordinates plus their partners (66,048 forwards), 16 quotient bins and 8 half-angle bins. It uses the first-pair canonical half-plane angle, not a four-dimensional polar angle. Confidence needs no calibration.

The angle control is defensible because quotient-dependent rotations preserve the observation law and every invariant posterior. It uses 64 independent quotient draws, 128 angular nodes and 64 nonzero candidate rotations between 2pi/128 and pi, totaling 8,192 forwards. Independent phases are used for each orbit. Sixteen quotient bins require at least four independent quotient draws; otherwise the bin uses the global calibration curve. Exact unlabeled log-cosh/softplus Jensen gain selects angles. Confidence-dependent masking remains an empirical policy, without a uniform unknown-posterior guarantee. The full-angle reference is not a universal risk ceiling.

## Streams and scoring

Rings streams use bases 1003000 rank calibration, 1004000 folded calibration, 1005000 test coordinates, 1006000 ties, 1007000 action-mixture keys and 1008000 sampled labels, each plus pair index. Transfer uses corresponding bases 1043000 through 1048000. Angle quotient draws use rank-calibration base+100+pair; phases use rank-calibration base+200+pair. The two optimizers in the transfer cohort share test inputs and random keys. Root certificate probes use disjoint 1011000/1051000 bases, with 1012000/1052000 crosschecks, and are not part of this evaluation sample.

Each model is scored on 65,536 fresh IID inputs (256 batches). The analytic posterior enters evaluation only. Independently seeded Bernoulli labels provide sampled-label loss and classification checks. All calibration tables and model hashes are committed before any fresh test risk. Actual call sizes and exact selected counts are recorded for every scored batch. Raw uses [256], limited policies [256,64], full references [256,256]. Input-only fusion is a possible equivalent implementation, but this study does not claim wall-time gains from these call counts and performs no new GPU experiment.

Expected logistic loss is primary. Report expected and sampled classification error separately. The independent unit is the trained-model seed. Primary empirical contrasts are rank-balanced minus confidence and rank-balanced minus rank-even in each of the two PB cohorts: four contrasts total. Paired model-cluster bootstrap uses 20,000 draws with seed 1009000. Report ordinary 95% intervals and 98.75% intervals for this four-comparison family. Transfer GD and all other contrasts are descriptive. Test observations and angular nodes do not multiply the number of model replications.

Secondary invariant-posterior worlds are constant one-half, increasing normalized quotient, decreasing normalized quotient, and a fixed midpoint threshold. For rings the normalized quotient is clip((r-.25)/1.25,0,1); for transfer it is r-.4. These are prespecified sensitivity diagnostics, not sharp extrema or identification of an arbitrary hidden posterior. Known-law and posterior-free certificates are owned and evaluated separately by the root.

## Artifact contract

States are `validation/states/{rings|transfer}_{pair}.npz`, with x,y,initial,pb100,pb; transfer additionally includes gd100,gd. Primary tables are `validation/policies/{cohort}_{pair}_pb_{rank|folded|angles}.npz`. The policy commit precedes scoring. Raw predictions, masks, sample hashes, per-batch call records, per-model/world risks, source/config hashes and all failed attempts are retained. Scientific scripts refuse to overwrite completed outputs. Shared dependencies are relative to the study root.
