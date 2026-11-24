# Needed-transfer correction before timing

The first frozen optimized runner and its completed equivalence audit are preserved unchanged. Before any timing was run, root review identified that the transfer condition copied tie keys for raw and full policies even though those policies do not use them. That overhead could unfairly penalize the strongest full reference.

The separately frozen `optimized_benchmark_v2.py` transfers tie keys only for limited policies that need them. Raw and full policies receive no tie tensor. The same unused-tensor omission is applied to CPU and resident GPU calls. Models, logits, selected sets, calibration, precision, data, timing repetitions and the original/optimized scientific policy definitions remain fixed. The existing full-input equivalence audit applies because the omitted argument is unused by raw and full paths. A small no-tie-argument smoke check is recorded before the first timing run.

The v2 timing runner writes the final latency and cost tables in `optimized/`. The original frozen runner was never timed; it is retained as a corrected pre-timing implementation attempt. Its original freeze and all equivalence results remain verifiable.
