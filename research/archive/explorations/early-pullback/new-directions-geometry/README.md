# Does early progress come from direction or step length?

The early classification advantage survived a change of step-length rule. At 100 updates the pullback direction outperformed the GD direction on all eight development pairs under both rules. Shrinking or enlarging a GD step to the local pullback length did not reproduce pullback's early performance in this setting.

The directions are compared first using their native step lengths and then with their size rules exchanged. A size rule is evaluated on each arm's own training state at every update. It matches the RMS length predicted by the Jacobian, which can differ from the actual finite movement. The two sides of a cross-trajectory comparison can consequently have different step lengths even if both use the same rule.

| Direction | Local size rule | Validation accuracy at 100 | Validation loss at 100 | Validation accuracy at 500 | Validation loss at 500 |
| --- | --- | ---: | ---: | ---: | ---: |
| GD | GD | 84.38% | 0.3501 | 88.77% | 0.2656 |
| GD | Pullback | 82.42% | 0.3909 | 92.38% | 0.1958 |
| Pullback | GD | 98.10% | 0.1009 | 97.75% | 0.1417 |
| Pullback | Pullback | 96.48% | 0.2808 | 98.10% | 0.1002 |

The table gives mean fixed-readout validation scores. Under the pullback size rule at step 100, changing from GD direction to pullback direction increased accuracy by 14.06 percentage points. Under the GD size rule, the increase was 13.72 points. At step 500 the corresponding gaps were 5.71 and 8.98 points. The pullback-direction arm had higher accuracy in every one of the eight pairs at each of these comparisons. The reciprocal control, pullback direction with the GD size rule, rose rapidly in accuracy, then acquired greater loss and slightly lower mean accuracy between 100 and 500 updates. Its mean cumulative actual training feature motion at 500 was 10.02, compared with a first-order prediction of 9.15. Monitoring actual movement matters for this aggressive rule.

## Shared-state one-step comparisons

From the same parameter state, try each direction at exactly the same predicted RMS length. The two branches are evaluated after one update, without either branch changing the source trajectory. Shared initialization and sampled states immediately before updates 100 and 500 of each native method give five state types per seed. Two length settings at each state give 80 paired contrasts. These repeated observations involve eight seeds.

Changing the direction to pullback yielded a lower post-step training loss in all 80 contrasts. At the shared initial states and both pre-100 state types it yielded a lower post-step validation loss in all 48 contrasts. On late GD states the same held in all 16. On late pullback states it yielded a lower post-step validation loss in only 5 of 16. Thus the connection between the local direction and training improvement persisted when the validation comparison had begun to favor the other direction. This describes the sampled states and the fixed readout on one synthetic task.

## Reproduction and checks

Install the project package in the recorded Python environment. Run `python run_geometry.py --output my_fresh_results` from this folder. The output path must be new. The [fixed design](protocol.md) defines seeds, split sizes, direction formulas, endpoints, and diagnostics. The 32 trajectories use 500 updates each. The maximum absolute difference between prescribed and measured first-order RMS length was 2.22e-15 across 16,000 training updates. At initialization the script checks both candidate increments against the original optimizer code on each seed.

The [seed-level checkpoints](results/checkpoints.csv) contain measured and predicted cumulative movements and training/validation metrics. [Within-seed differences](results/paired_differences.csv) cover the declared endpoints. [Branch measurements](results/branches.csv) contain matched-length one-step outcomes and directional cosines. The [summary](results/summary.json) gives endpoint means, sample standard deviations, software versions, and source fingerprints. Diagnostics use only newly generated training and validation data. Timing and final held-out performance require separate comparisons.
