# Idea 2B: fixed quota and calibrated second-view angles

The primary joint radius policy did not beat the strong first-view confidence baseline. On 16 fresh pullback models, at the 8192-forward calibration cap, joint risk exceeded confidence-plus-half-turn by 0.00165668 nats (95% paired seed-bootstrap interval [0.00035641,0.00291192]); joint was worse on 14/16 seeds. Against the original IID-calibrated folded half-turn gate, joint was better by 0.00066659 nats on average, but the interval [-0.00178090,0.00052721] included zero. Independently sampled labels gave the same conclusions.

There was a useful prespecified factorial result: keeping the confidence mask exactly fixed while replacing the half-turn with a learned radius angle reduced risk by 0.00258288 nats [-0.00364939,-0.00157176], winning 15/16 seeds. Changing the reference radius mask after choosing radius angles added only 0.00001930 nats mean improvement, with an interval crossing zero. The evidence supports angle calibration as an addition to a strong allocator; it does not support replacing confidence with the tested joint radius surrogate. This interpretation does not select or refit a new policy on these results: all arms were frozen beforehand.

## Primary comparison at both frozen calibration caps

|Model|Calibration forwards|Joint minus confidence+pi|95% CI|Joint minus legacy IID folded+pi|95% CI|
|---|---:|---:|---|---:|---|
|gd|8192|+0.0182195|[+0.0117414,+0.0253405]|+0.0111566|[+0.0061973,+0.0167329]|
|gd|65536|+0.0179569|[+0.0115293,+0.0251154]|+0.0110291|[+0.0060997,+0.0167124]|
|pb|8192|+0.0016567|[+0.0003564,+0.0029119]|-0.0006666|[-0.0017809,+0.0005272]|
|pb|65536|+0.0014012|[+0.0001829,+0.0025841]|-0.0009164|[-0.0019327,+0.0001627]|
|hybrid|8192|-0.0028033|[-0.0062864,+0.0002324]|-0.0026383|[-0.0058100,-0.0001370]|
|hybrid|65536|-0.0030400|[-0.0066272,+0.0000338]|-0.0028438|[-0.0060929,-0.0002759]|

The pullback8192 contrasts are primary. Other methods, the 65536 cap, sampled-label risks and all factorial contrasts are reported as prespecified secondary/sensitivity results. Every interval resamples sixteen complete seed records, not batches. Intervals are pointwise; correlated policy/endpoint tests are not independent evidence. Calibration has one independent pool per model/seed here, so this study does not isolate variability over repeated calibration pools; Idea 2A supplies that separate analysis.

## All pullback policies, primary calibration cap

|Policy|Expected risk|Sampled-label risk|New-model calibration forwards|Deployment calls/input|Median ms/batch256|
|---|---:|---:|---:|---:|---:|
|random_pi|0.113751|0.113723|0|1.25|0.1251|
|random_default|0.113925|0.113869|0|1.25|0.1258|
|random_global|0.113390|0.113339|8192|1.25|0.1043|
|random_radial|0.113262|0.113218|8192|1.25|0.1063|
|confidence_pi|0.102932|0.102869|0|1.25|0.1142|
|confidence_default|0.103481|0.103352|0|1.25|0.1130|
|confidence_global|0.101101|0.100999|8192|1.25|0.1231|
|confidence_radial|0.100349|0.100242|8192|1.25|0.1180|
|radius_pi_pi|0.106347|0.106297|8192|1.25|0.1257|
|radius_pi_default|0.106883|0.106777|8192|1.25|0.1198|
|radius_pi_global|0.104931|0.104853|8192|1.25|0.1218|
|radius_pi_radial|0.104608|0.104519|8192|1.25|0.1145|
|folded_pi|0.105490|0.105444|8192|1.25|0.1337|
|radius_angle_default|0.106856|0.106748|8192|1.25|0.1251|
|radius_angle_global|0.104911|0.104834|8192|1.25|0.1143|
|joint|0.104588|0.104498|8192|1.25|0.1276|
|legacy_iid_radius_pi|0.106269|0.106206|8192|1.25|0.1290|
|legacy_iid_folded_pi|0.105255|0.105201|8192|1.25|0.1525|
|raw|0.116684|0.116615|0|1|0.0506|
|full_pi|0.104954|0.104895|0|2|0.1079|
|full_radial|0.102995|0.102886|8192|2|0.1123|

Policy naming: random, confidence and radius_pi prefixes fix the selected mask across pi/default/global/radial angle rules. `default` is frozen from old models: GD180 degrees, pullback143.4375 degrees, hybrid163.125 degrees. `radius_pi` always ranks the estimated half-turn gain. `folded_pi` uses the orbit-derived folded table. The two `legacy_iid_*` rules use the unchanged prior Gate.fit algorithm on IID inputs and antipodes. `radius_angle_*` updates a radius priority for that same angle; `joint` uses the maximizing radius angle and corresponding gain priority.

The full-two-view rows are distinct-budget references, NOT universal risk ceilings. A confidence-dependent mask can skip orientation-specific harmful second views. Indeed pullback confidence+radius-angle risk 0.100349 is below the full-radius reference 0.102995 despite using 1.25 versus 2 calls/input. No universal dominance claim follows from forward counts or Jensen symmetry. The folded half-turn guarantee applies only to that pair and is not extended to arbitrary angles. See THEORY.md.

## Samples, calibration and deployment accounting

The 48 fixed models come from 16 new paired training draws shared with Idea 2A: train721001..721016, init722001..722016; GD500, PB500 and PB100/GD400. This shared model cohort is not an independent replication of Idea 2A. Each seed uses 65536 independent test inputs, arranged into 256 batches 256, shared across all models, budgets and policies. Input seeds 736000+i, independent ties 737000+i and independent Bernoulli-label draws 738000+i are retained by hashes. Conditional expected risk uses the known posterior only in the evaluator.

Primary calibration uses 64 independent radii and 128-point angular orbits, exactly 8192 forwards; secondary uses 512 radii and 65536 forwards, with the primary radii as a prefix. Radius seeds 735000+i and separate phase seeds 745000+i are independent of test data and Idea 2A calibration. All angle/radius/folded orbit tables reuse these forwards. Sixteen radius bins contain at least 4 independent radii; angular samples are correlated quadrature nodes, never counted as independent calibration draws.

The stronger unchanged legacy gate uses 4096 IID inputs and their antipodes at the primary cap, or 32768 at the secondary cap, with seed741000+i and shared prefixes. Its original empty-cell fallback is preserved. Each individual gate respects the same forward cap, but its sample design is different. At8192, IID folding improves over orbit folding by 0.00023461 nats on pullback, with interval [-0.00052762,-0.00003028], showing why retaining this control mattered.

Confidence/random with fixed pi or historical default require zero new-model calibration. Learned policies incur their stated budget. Historical defaults reuse old calibration work and are not free to discover. The two budgets and every losing policy remain reported. Across the final confirmation, actual shared calibration work is 48*(65536+8192+65536)=6684672 model-example forwards: one nested orbit pool and separately executed legacy fits at both caps. This research total is distinct from any one deployed policy's cost. The preserved pre-test phase attempt incurred additional replay work of the same size and is not hidden.

- orbit at 8192 forwards: median 0.0122 seconds per model, 64 independent radii/inputs.
- legacy_iid_antipodes at 8192 forwards: median 0.0017 seconds per model, 4096 independent radii/inputs.
- orbit at 65536 forwards: median 0.0907 seconds per model, 512 independent radii/inputs.
- legacy_iid_antipodes at 65536 forwards: median 0.0183 seconds per model, 32768 independent radii/inputs.

All18 limited policies actually select64 entries per batch after their first logits and coordinates are available. Only those entries receive second-view evaluations; no candidate second views are probed during routing. Every model×budget×policy has an independently counted first-batch deployment audit with calls[256,64], including random allocation. Raw calls[256] and full references[256,256] are separate. Whole-test evaluation vectorizes these identical batch allocations; per-batch risks and actual mask hashes are saved. Timings randomize policy order, include routing/rotation overhead and repeat16 real batches. Local CPU latency does not imply accelerator-service throughput.

## Frozen boundaries and retained attempts

DESIGN.md/design_freeze.json and experiment_v1_frozen.py/freeze.json preserve the initial protocol/code. Before fresh calibration/testing, ADDENDUM.md and freeze_addendum.json added unchanged IID strong baselines for calibration sampling fairness. Before any test data or outcomes, PHASE_ADDENDUM.md and freeze_phase_addendum.json corrected a restarted RNG stream to an independent phase stream. The entire earlier calibration remains in fresh_v2_shared_phase with its own commit and experiment_v2_frozen.py; it was never evaluated for choosing a result. The final fresh/policy_commit.json precedes fresh/score_start.json, and all policy hashes remain immutable through testing. No failed scientific policy was dropped or retuned.

## Reproduction and files

Python dependencies: NumPy and threadpoolctl for experiments; Matplotlib for figures. The numerical model and unchanged gate are imported from the surrounding archived source/ and legacy/ folders. Fresh model states are supplied by ../idea2a/fresh/training/. No network, repository checkout or account access is needed for confirmation. The 48 historical gain curves used for old-only defaults are copied into old_pilot/historical_gain_curves.npz with source hashes and means in historical_defaults.json. Use `python replay_old_pilot.py` to reproduce the original12 old-pilot rows and historical choices entirely from the archived compact curves and old model bank. That portable replay passed. The frozen original oldpilot command remains preserved for provenance and is not required for reproduction.

In a fresh archive-layout copy, preserve the supplied freezes, source and input model states, omit fresh/outputs, then run: `python experiment.py calibrate`, `python experiment.py score`, `python experiment.py summarize`. Existing calibration/results are refused. Reporting is `python report.py`; it reads and verifies the archived compact historical curves and needs no original external exploration path. Existing report tables are verified unchanged before regenerating prose/figures. No scientific fitting depends on report generation.

Primary raw files: fresh/per_seed.csv, fresh/per_batch.csv.gz, fresh/samples.csv, fresh/deployment_audit.csv, fresh/calibration.csv, fresh/latency.csv, frozen policies and full calibration orbit arrays. summary.csv/contrasts.csv contain every prespecified result. supplementary_contrasts.csv completes pairwise fixed-angle mask comparisons and every budget sensitivity; descriptive comparisons against full references are labeled. policy_costs.csv, calibration_cost.csv and latency_summary.csv distinguish calibration, calls and measured time.
