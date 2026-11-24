# Real-network GPU efficiency extension

Subsequent independent CPU verification is now complete for full_fused, folded_fused and confidence_radial on all 16,384 saved test inputs in all nine models. The 27 CPU risk rows have no selected-mask or class-decision changes relative to CUDA; maximum expected-risk difference is 1.76e-9. The original summary tables below remain CUDA-scored, while the direct CPU results and 1,728 audited batch calls are preserved in `../root_audit/cpu_quality_checks.csv` and `cpu_quality_calls.csv`. This additional verification costs 663,552 forward examples and does not alter training or timing.

No limited-view policy was faster than the strongest equivalent full prediction on the GPU in this experiment. This held for all nine models, both resident and transfer conditions, and both implementation variants. The largest speed ratio among all limited GPU paths was 0.303; a ratio above one would indicate a speedup. Therefore no GPU wall-time calibration break-even exists for these policies.

This is a separately frozen experiment on nine ordinarily trained MLPs, not an enlargement of the dense pullback Gram computation. There are three widths (64, 256, 1024), with three seeds per width. Training data and test draws are paired across widths within each seed. Thus each width has three model replications, not nine independent training-data draws. The wider models were fixed in advance and were not selected because of their test accuracy or measured timing.

The GPU extension tests the fixed folded/confidence view policies and their inference implementations. It does not port the new rank-balanced selector or its mathematical certificate to the GPU.

The GPU is an RTX 5070 Ti. The isolated runtime uses PyTorch 2.7.1+cu128, FP32, no autocast, and TF32 disabled. Each model has two tanh hidden layers and trains for 1,000 full-batch Adam updates on 1,024 examples. Parameter counts are 4,417, 66,817 and 1,053,697. Model weights, data and training losses are saved before calibration. All nine models trained and scored successfully.

## Prediction quality

CUDA-scored expected logistic risks on 16,384 independent test inputs per model are:

| Width | Raw | Full half-turn | Confidence + half-turn | Folded gate | Confidence + radius angles |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.033974 | 0.030791 | 0.030942 | 0.030831 | 0.030435 |
| 256 | 0.042714 | 0.036433 | 0.036940 | 0.036459 | 0.036220 |
| 1024 | 0.040882 | 0.034593 | 0.034860 | 0.034633 | 0.034099 |

Each cell is a mean over three trained models. These are logistic losses, not classification error rates. The complete per-model expected and sampled-label losses and errors are retained in `per_model.csv`. The folded gate approaches the full-half-turn loss while selecting only 64 of 256 inputs. The largest architecture does not have the lowest raw test loss. The angle policy and full half-turn implement different predictors; full half-turn is not a universal quality ceiling. The quality columns attached to CPU timing rows in `cost_quality.csv` also refer to this CUDA-scored quality of the same frozen weights; CPU risk was not independently rescored. Original-versus-optimized equivalence was audited separately on each device and does not establish cross-device equality.

## Exact work and implementation controls

| Policy implementation | Actual example counts per call | Total forward examples |
|---|---|---:|
| Raw | [256] | 256 |
| Full, sequential | [256, 256] | 512 |
| Full, fused | [512] | 512 |
| Confidence half-turn | [256, 64] | 320 |
| Folded, sequential | [256, 64] | 320 |
| Folded, fused | [320] | 320 |
| Confidence plus radius angles | [256, 64] | 320 |

Folded routing uses only coordinates and its calibrated table, so it can select partners before any network evaluation and fuse them into one 320-example call. Confidence requires first-view logits and therefore needs sequential prediction. All actual calls and exact quotas pass the recorded deployment audit. The folded and angle policies each use 8,192 offline model forwards. Folded calibration uses 4,096 IID inputs and antipodes; angle calibration uses 64 independent radii and 128 angular views each.

The first frozen implementation used Boolean gathers. The optimized implementation carries fixed-size integer index sets, avoiding dynamic Boolean-index discovery. Its confidence key exactly combines the order of finite absolute FP32 values with independent tie ranks; its folded key combines cell-score ranks with tie ranks. Synthetic checks include duplicates, signed zero, subnormals and extreme finite values. Every selected set and every logit matched the original implementation exactly on all 16,384 test inputs of all nine models, on both CPU and CUDA. Sequential and fused predictions also pass their separate equivalence audit.

The first optimized timing runner would have unnecessarily transferred tie keys for raw/full policies. Root review caught this before timing. The separately frozen v2 runner transfers only needed tensors; raw/full receive no tie tensor. Both preceding sources and freezes remain intact. This correction does not change models, calibration, policies, test inputs or outcomes.

## Timing interpretation

The fixed-index, fused folded implementation has the following mean model speed ratios against the fastest equivalent full reference:

| Width | GPU resident | GPU with transfers | One-thread CPU |
|---:|---:|---:|---:|
| 64 | 0.177 | 0.257 | 0.630 |
| 256 | 0.178 | 0.258 | 1.193 |
| 1024 | 0.151 | 0.221 | 1.505 |

At width 1024, mean resident folded latency is 1.365 ms per batch versus 0.207 ms for the fastest full reference. On one CPU thread, folded latency is 7.024 ms versus 10.569 ms for full. All three models at widths 256 and 1024 show the CPU saving; none at width 64 does. The width-1024 CPU folded speed ratios range from 1.482 to 1.530. These are descriptive three-model results, not a broad hardware theorem. GPU full inference itself can be much faster than CPU full inference; the negative result concerns whether selective extra views beat the strong full-GPU reference.

Final measurements belong to `optimized/latency.csv`, `optimized/cost_quality.csv` and `optimized/summary.csv`, produced by `optimized_benchmark_v2.py`. The original and optimized implementations run in common randomized timing blocks. Each uses 16 real batches, three warmups and seven retained repeats per model and condition. Repeats measure timing variability and are not independent model samples.

The resident condition starts with input tensors, ties, model and calibration tables on the GPU. The transfer condition starts with FP32 CPU input tensors and CPU tie tensors, transfers the inputs and only needed ties, and returns logits to the CPU. It includes synchronization and transfers, but does not include model loading, calibration or conversion from an arbitrary external input format. Tie keys are supplied inputs; random-key generation is outside the measured deployment interval. CPU context uses the identical frozen FP32 model with one CPU thread. CUDA events report the stream interval, including possible gaps between submitted work; they are not asserted to isolate only matrix multiplication.

Cost comparisons use the fastest measured equivalent full implementation across both original and optimized paths in the same condition. A smaller forward-example count is not by itself a speedup. Calibration wall-time amortization is reported only when the candidate saves measured wall time; otherwise the break-even field is blank. The count-only break-even for an 8,192-forward calibration against full two-view prediction is 43 batches, or 11,008 deployed inputs. Against an equally budgeted policy, there is no saving in forward-example count to amortize calibration.

The CPU context uses tables fitted on the GPU. Its wall-time break-even therefore describes that mixed-hardware workflow, not a CPU-only calibration run. For folded-fused, these measured break-evens are 31/12/10 batches at width 256 and 2/2/2 at width 1024. They are **calibration-only amortization under prepared resident tables**, not complete cold-start break-evens: they exclude training, installation, model/table loading, the one-time `prepare_tables` rank construction, random-key generation and all evaluation/audit work. The separate forward-count threshold remains 43 batches. No GPU policy has a wall-time break-even even if its offline calibration were free, because its per-batch deployment is already slower.

All heavy main-study CPU work and the root's isolated CPU benchmark finished before GPU timing began at 09:45:32 UTC. The 2,646 retained timing records completed at 09:47:29 UTC. Root and the selection agent held other heavy work during this interval.

## Reproduction and provenance

`benchmark.py` is the untouched initial frozen program. `optimized_serving.py` and `optimized_benchmark.py` are the first optimized addition. `optimized_benchmark_v2.py` changes only unnecessary tie transfers in the measurement implementation. `freeze.json`, `optimized/freeze.json`, and `optimized/freeze_v2.json` chain the sources, training and policy commits. `PROTOCOL.md`, `OPTIMIZED_SERVING_ADDENDUM.md`, and `TRANSFER_TIMING_AMENDMENT.md` explain the changes. `READINESS.md` preserves the initial absent-runtime and failed metadata-probe observations; the later isolated installation and actual successful GPU execution supersede readiness-only status.

The original 951xxx--958xxx proposal was moved before training to avoid selector-development streams. Actual training/init seeds are 971001--971003 and 972001--972003. Calibration, phase, test and ties use 973000/974000/975000/976000/977000 plus seed. Labels use 978000 plus seed; timing order uses 978100 plus width plus seed. Complete values are frozen in code and protocol.

For a fresh output copy with the same relative study dependencies, stages are:

```text
python gpu/benchmark.py freeze
python gpu/benchmark.py train
python gpu/benchmark.py calibrate
python gpu/optimized_benchmark.py freeze
python gpu/optimized_benchmark.py audit
python gpu/optimized_benchmark_v2.py freeze
python gpu/benchmark.py score
python gpu/optimized_benchmark_v2.py time
python gpu/optimized_benchmark_v2.py analyze
```

The timing stage requires an explicit `TIMING_APPROVAL.json` confirming that the main CPU jobs have finished. Reproduction should retain this isolation step. Output stages refuse to overwrite prior results. This experiment establishes results only for these trained classifiers, this batch size, precision, runtime and machine.
