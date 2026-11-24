# Can a pre-update derivative choose a useful post-update view?

The derivative improved forecasts of how valuable a second view would be. It usually did not change which global angle should be used. The main PB100 experiment was an exact policy null: every stale, forecasted and freshly recalibrated global selector chose the half-turn. A separately frozen PB500 follow-up produced a few beneficial angle changes, but their mean risk benefit was about 0.00001 nats and its seed-bootstrap interval included zero. These are separate conclusions about forecast accuracy and decision usefulness.

## Prediction and controls

At the pre-update state θ, let d be the GD or damped pullback direction, and let `a = −∇L_train(θ)·d > 0`. For requested training-loss decrease δ, the fully pre-update forecast uses

`v(x) = (δ/a) J_h(θ,x) d`, and `h_predicted(x) = h_θ(x) + v(x)`.

Independent unlabeled calibration orbits supply 512 radii × 128 angles. We choose the global angle maximizing the exact Jensen score of `h_predicted`, over the same 64 candidates as the main rotation study. The chosen angle is saved before evaluating the updated model's population risk. The actual update is chosen by a training-only root search along d, matching the requested loss decrease to numerical precision, with a displacement-norm cap of 4. These endpoints were fixed independently of the forecast's population outcome.

Four comparators are retained: the angle chosen from pre-update logits (stale); the angle chosen after observing updated logits (post-update recalibration); the fixed half-turn; and a first-order expansion of the Jensen gain itself. A secondary tangent forecast uses the actual train-only matched step α instead of δ/a. This is outcome-blind with respect to population risk but is not a fully pre-update forecast, because α is found by testing the training objective at candidate updated states.

For the first-order-gain variant, if `m=(h+h_rot)/2` and `v_m=(v+v_rot)/2`, the curve is

`G_pre(α) + E[φ′(h)v − φ′(m)v_m]`.

No post-update labels or population risk enter any of these calibrations. Every deployed policy has the same two-network-call cost. The analytic forward tangent implementation is checked against the repository's independent full Jacobian on every branch.

## Main PB100 bridge: improved forecasts, no angle change

This bridge reuses direction1's original-rings cohort trained with seeds 611001–611016 and initialized with 612001–612016. It is not an independent replication of those training runs. The pre-state is PB100; the target drop is 0.02 nats. Calibration seeds 615001–615016 are independent of main angle-selection calibration and test draws. All 16 pullback endpoints were reachable; 12 GD endpoints were reachable. Four GD root failures remain in `bridge_new/unreachable.csv` and direction1's records.

| Branch | Reachable | Stale gain MAE | Pre-update forecast MAE | Post-update recalibration MAE | Forecast angle changes |
|---|---:|---:|---:|---:|---:|
| GD | 12/16 | 0.006149 | 0.000771 | 0.000121 | 0/12 |
| Pullback | 16/16 | 0.001501 | 0.000241 | 0.000192 | 0/16 |

All five learned selector variants, including post-update recalibration, chose π on all 28 reachable branches. Therefore all had exactly the same post-update risk as the fixed half-turn. This is not evidence of a failed derivative: the gain forecasts improved substantially while the optimal decision remained in the same discrete candidate. It is also not evidence that prediction improved deployment, because it did not change the decision.

## Bounded PB500 follow-up and separate confirmation

The all-π PB100 pilot motivated one follow-up at PB500, where the main angle study had shown meaningful competitors to π. The full development sequence is preserved.

1. On the six old direction1 seeds, the original 0.02 target reached zero GD and five pullback endpoints under the same norm cap. Files under `late_bridge_source/old` and `late_bridge_results/bridge_old`, and the original `late_bridge_pilot02.py`, preserve this unsuccessful attempt.
2. Before generating the new follow-up cohort, a training-only feasibility rule tried exactly 0.005 and 0.001: select the largest target feasible for both methods on at least four of six old pairs; if neither qualified, use 0.001. Targets 0.005 and 0.001 were jointly feasible on five and six old pairs, respectively. The rule therefore selected **0.005**. All 24 root trials and the choice are in `late_feasibility_all_roots.csv` and `late_feasibility_choice.json`. No population risk or selector improvement was used in this choice.
3. After old-cohort calibration/evaluation, the code and settings were frozen in `late_bridge_freeze.json`. The confirmation cohort then used new training seeds 621001–621016, initialization seeds 622001–622016 and calibration seeds 625001–625016. These models are independent of the main 611/612 cohort. No settings changed after seeing this third cohort.

At the selected 0.005 target, all 16 pullback and eight GD endpoints were reachable in the confirmation cohort. The other eight GD root failures are retained. Different method-specific success subsets must not be interpreted as an unbiased comparison of GD and pullback mean risk; the policy comparisons below are paired within each reachable branch.

| Branch | Reachable | Forecast angle changes | Wins / losses / ties versus stale | Mean risk improvement | 95% seed-bootstrap interval |
|---|---:|---:|---:|---:|---:|
| GD | 8/16 | 2/8 | 2 / 0 / 6 | 0.00001356 | [0, 0.00004013] |
| Pullback | 16/16 | 1/16 | 1 / 0 / 15 | 0.00001003 | [0, 0.00003008] |

Post-update recalibration produced mean improvements over stale angles of 0.00002197 nats for the eight GD successes and 0.00001003 for pullback. Thus little decision value was available at this small matched loss drop, even to a selector that could observe the updated model. The pre-update forecast captured part of that small GD value and all of the measured pullback value, but three changed decisions are too few for a broad performance claim. Both intervals include zero because most seed-level improvements are exactly zero.

Gain MAE decreased from 0.001914 to 0.001369 for GD and from 0.001805 to 0.001450 for pullback. These are more modest than PB100 improvements. Finite calibration noise contributes to the remaining error: even post-update recalibration has gain MAE 0.001258 and 0.001435, respectively. All primary, matched-step, first-order-gain and post-calibration results remain in the tables, including weaker outcomes.

## Cost and interpretation

Per branch, stale/forecast construction evaluates 65,536 pre-update logits and 65,536 pre-update Jacobian-vector products. The forward tangent explicitly propagates both the forward state and its directional derivative; it is additional work and must not be counted as a free ordinary forward evaluation. All forecast variants reuse these values. Post-update recalibration requires another 65,536 ordinary forward evaluations after the update. At deployment, every policy evaluates the model exactly twice per input. Timing archives distinguish the joint forecasting-and-serialization time from post-update calibration time; they do not pretend that a JVP equals one forward call.

The supported bridge is therefore limited but concrete: the loss-clock derivative can improve advance estimates of augmentation value, and it can occasionally anticipate a useful second-view angle change. The main bridge produced no policy benefit, and the separate later-checkpoint confirmation produced only sparse, small benefits. The much stronger result remains the label-free static choice of an angle once a model is available.

## Reproduction and retained artifacts

Main bridge: `bridge.py fit --cohort new`, followed by `bridge.py evaluate --cohort new`, using direction1's already committed new states/predictions and the supplied `bridge_freeze.json`. Outputs go to `bridge_new`. The original `bridge.py` is frozen and was not changed to implement the smaller follow-up drop.

Follow-up confirmation: retain `late_bridge_freeze.json`, `late_feasibility_choice.json`, `late_bridge.py`, `late_forecast.py`, `rotation_selectors.py` and `experiment.py`, then run:

```text
python late_bridge.py generate --cohort new
python late_bridge.py fit --cohort new
python late_bridge.py evaluate --cohort new
python bridge_analysis.py
```

Use a fresh copied study directory: stages refuse to overwrite their generated directories. The confirmation training/root records go to `late_bridge_source_selected/new`; fitted selectors and risks go to `late_bridge_results_selected/bridge_new`. The `selector_commit.json` files hash all forecast and endpoint arrays before scoring. `forecast_quality.csv` and its per-seed companion separate risk improvement, gain prediction error, changed decisions and diagnostic population regret.
