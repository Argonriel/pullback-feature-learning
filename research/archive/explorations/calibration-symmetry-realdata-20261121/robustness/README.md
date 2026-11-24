# Imperfect posterior symmetry: results

The input law remains exactly half-turn invariant. The guarantee below assumes a supplied uniform bound on the posterior difference, |p(x)-p(Tx)| <= epsilon. Synthetic construction supplies that bound; unlabeled input data do not establish it for real applications.

The exact gain identity is E[M(J-(p-pT)d/2)], with d=(h-hT)/2. The certificate is the affine line L_A-epsilon U_B/2. All 64 policy/predictor lines at both 512 and 2048 batches are simultaneously covered by a 0.05 family budget over 512 elementary bounds. Coverage holds for every epsilon in [0,1], not only the plotted values. Primary results use 2048 independent batches.

| Predictor | Selector | Positive at epsilon zero | Certified limit, mean | Minimum | Maximum |
|---|---|---:|---:|---:|---:|
| legacy_raw | symmetric_jensen | 16/16 | 0.210883 | 0.177009 | 0.247467 |
| legacy_raw | folded | 16/16 | 0.201520 | 0.170163 | 0.238171 |
| pb_temp | symmetric_jensen | 16/16 | 0.120789 | 0.106555 | 0.158064 |
| pb_temp | folded | 16/16 | 0.117812 | 0.102635 | 0.154865 |

A positive guarantee holds strictly below an uncapped crossing; the line equals zero at the crossing. The limit is a sufficient tolerance bound, not an estimate of the actual point where augmentation becomes harmful. Neither the confidence selector nor approximately invariant input distributions are covered.

## Calibration-only clip and scaling

| Predictor | Mean clip | Clip range | Mean inverse temperature |
|---|---:|---:|---:|
| legacy_raw | 0.297041 | [0.223545, 0.391642] | 1 |
| pb_temp | 0.0863643 | [0.0573238, 0.134319] | 0.805783 |

The clipping rule was fixed before fresh access and uses the 95th percentile of the independent calibration grid. All clip choices were committed before certificate sampling. The network bound scales both the skip coordinate and residual contribution by inverse temperature. Training and temperature fitting use labels; this study does not describe the complete pipeline as label-free. The fresh groups differ in fitted parameters as well as temperature, so their contrast does not isolate a causal temperature effect. The old fixed-beta diagnostics are the controlled scaling check.

## Independent stress tests

The fixed cosine world is a common population across models. The adversarial sign(d) world depends on each frozen predictor and is shared across its selectors; its cross-model averages describe sensitivity rather than a common population. Posterior probabilities remain valid and their uniform defect is bounded by epsilon at every tested setting. Both direct IID and antithetic expected losses and classification errors are preserved. The antithetic gain identity is checked numerically for both symmetric policies.

| Predictor | World | Selector | Gain at epsilon 0 | Gain at epsilon 1 | Negative models at epsilon 1 |
|---|---|---|---:|---:|---:|
| legacy_raw | cosine | symmetric_jensen | 0.0531028 | 0.0380572 | 0/16 |
| legacy_raw | cosine | folded | 0.0481324 | 0.0336249 | 0/16 |
| legacy_raw | cosine | confidence | 0.0067301 | 0.0029732 | 6/16 |
| legacy_raw | cosine | full | 0.0721222 | 0.0506858 | 0/16 |
| legacy_raw | adversarial | symmetric_jensen | 0.0531028 | -0.0243797 | 16/16 |
| legacy_raw | adversarial | folded | 0.0481324 | -0.0226100 | 16/16 |
| legacy_raw | adversarial | confidence | 0.0067301 | -0.0259077 | 16/16 |
| legacy_raw | adversarial | full | 0.0721222 | -0.0797537 | 16/16 |
| pb_temp | cosine | symmetric_jensen | 0.0153661 | -0.0054583 | 15/16 |
| pb_temp | cosine | folded | 0.0146017 | -0.0055899 | 15/16 |
| pb_temp | cosine | confidence | 0.0073228 | 0.0020164 | 5/16 |
| pb_temp | cosine | full | 0.0211724 | -0.0115522 | 15/16 |
| pb_temp | adversarial | symmetric_jensen | 0.0153661 | -0.0210365 | 16/16 |
| pb_temp | adversarial | folded | 0.0146017 | -0.0200823 | 16/16 |
| pb_temp | adversarial | confidence | 0.0073228 | -0.0071992 | 16/16 |
| pb_temp | adversarial | full | 0.0211724 | -0.0521763 | 16/16 |

## Cost, audit and reproduction

Certificate computation used 10,036,010 actual forwards on selected unions. Independent stress evaluation used 4,194,304 forwards, plus 30,720 separately counted deployment-audit forwards. Full pair evaluation is charged as research cost; each limited deployment audit uses exactly 256 first and 64 second views. The original training and calibration costs are reported by the training branch.

Raw batch moment observations, masks/input hashes, clipping commit, exact calls, per-model limits, stress outputs, and first negative tested epsilon or unreached are preserved. Bootstrap intervals in stress_summary.csv are pointwise 95% model-seed descriptions, separate from simultaneous certificate coverage. Floating-point identity checks do not amount to interval-arithmetic verification of every operation.

After the training branch writes the committed model/policy manifest and root releases this branch, run these stages in a fresh output tree:

```text
python robustness/run.py prepare
python robustness/run.py certificate
python robustness/run.py stress
python robustness/analyze.py
```

THEORY.md supplies the derivation and primary literature. PROTOCOL.md and FREEZE.json preserve the prospective design. The old-only development uses a separately supplied previous-study directory; fresh confirmation depends only on current study sources and committed training artifacts.
