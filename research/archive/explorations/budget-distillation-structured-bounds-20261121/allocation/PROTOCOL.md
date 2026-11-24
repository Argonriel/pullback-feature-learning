# Prospective label allocation and PB training work

This branch uses the unchanged two-amplitude data law, 144-parameter two-block residual model, and sampled-label PB mean logistic objective. The previous completed study is immutable and supplies only the model and fixed training/calibration recipe. No fresh outcome, optimizer pilot, or test selection occurs before this protocol, executable sources and development-only preflight are frozen and internally reviewed.

## Procedures and estimands

Sixteen paired replications compare nine procedures, each using exactly 4,352 accepted, unique fitting labels. Four balanced training pools are nested independent class-balanced increments of 256, 512, 512 and 1,024 examples. Each procedure sees only its designated training prefix. Training size 256/768/1,280/2,304 is paired with iid calibration size 4,096/3,584/3,072/2,048. Validation is zero except for the one prespecified validation arm. The strongest previous allocation remains `fixed_2304`: 2,304 training labels, 2,048 calibration labels, 500 PB updates.

| Procedure | Training | Validation | Calibration | Executed updates | Selected update |
|---|---:|---:|---:|---:|---:|
| fixed_256 | 256 | 0 | 4096 | 500 | 500 |
| fixed_768 | 768 | 0 | 3584 | 500 | 500 |
| fixed_1280 | 1280 | 0 | 3072 | 500 | 500 |
| fixed_2304 | 2304 | 0 | 2048 | 500 | 500 |
| flop_256 | 256 | 0 | 4096 | 4332 | 4332 |
| flop_768 | 768 | 0 | 3584 | 1485 | 1485 |
| flop_1280 | 1280 | 0 | 3072 | 896 | 896 |
| examples_256 | 256 | 0 | 4096 | 4500 | 4500 |
| validation_1280 | 1280 | 1024 | 2048 | 896 | minimum validation NLL at 100/250/500/896; ties select earliest |

The fixed-step grid tests label allocation at equal updates, without claiming equal training work. `examples_256` and `fixed_2304` execute exactly 1,152,000 example-updates and 4,608,000 Jacobian rows each, but different solve counts. The three `flop_*` procedures match the baseline's explicitly defined nominal arithmetic training budget to within less than one update. The validation arm shares `flop_1280`'s training schedule and costs four extra validation forward passes (4,096 examples). Its comparison measures exchanging 1,024 calibration labels for independent checkpoint selection. All prospective procedures are reported, including unfavorable outcomes; no data-dependent winning procedure is presented as independently validated.

The experiment reuses common trajectories and saved checkpoints, including between the validation and fixed schedules. This reduces actual experiment cost without crediting individual procedures with free training or extra labels. `actual_training_work.csv` records executed trajectories; `training.csv` records standalone procedure costs. Training-loss monitoring is diagnostic only, with its actual forwards recorded separately. The unique fitting-label union per replication is 7,424 (2,304 training, 1,024 validation, 4,096 calibration). Rejection proposals are synthetic data-generation overhead, retained in sampling metadata; the 4,352-label budget refers to accepted observed examples passed to the fits, consistently with the previous study.

## PB normalization and work accounting

The inherited metric sums the full four-dimensional representation Jacobians while the objective averages loss. Every sample size therefore uses rate `4*n/256` and damping `0.1*n/256`, equal to normalized metric `J.T@J/n + (0.1/256)I` with rate `4/256`. Development-only preflight verifies duplication invariance for k=2,3,9, mean-gradient duplication invariance, analytic gradients against independent finite differences, and normalized-metric equivalence. The base rate/damping are not retuned.

One PB update performs one standalone representation evaluation on n examples, a second forward pass inside the analytic representation Jacobian, 4n Jacobian rows with 144 columns, n scalar loss derivatives, one Jacobian-transpose vector contraction, one Gram matrix, and one 144 by 144 LU linear solve. There is no separate reverse-mode backpropagation call: the loss gradient is formed by the Jacobian contraction. Calling each update a single ordinary backward pass would conceal its work. Ledgers retain forward examples, Jacobian rows and scalar entries, loss-derivative examples, contraction/Gram counts, solve counts and elementary-function counts separately.

Nominal arithmetic counts one addition, subtraction, multiplication or division as one scalar FLOP; a length-k dot product costs 2k-1. Counts follow the executed dense NumPy contractions, including the initially zero Jacobian multiplication. Comparisons, memory traffic, allocations, pivot searches and transcendental internals are excluded. A general LU factorization is modeled by sum over m=1..143 of `(m+2m^2)` = 1,980,264 arithmetic operations; its forward/back substitutions cost `2*144*143+144` = 41,328. These are model counts, not a profiler's hardware instruction counter.

For each residual block (D=4,H=8,P=144), the standalone representation arithmetic is 136n. The analytic Jacobian arithmetic per block is: hidden preactivation 64n; tanh derivative 16n; state derivative contraction/scale/add 400n; propagate old Jacobian 4,032n; A-parameter derivative/add 512n; B-parameter derivative/add 384n; bias derivative/add 96n; residual output update 72n. Total Jacobian arithmetic is 5,576n per block, or 11,152n. The remaining update costs are scalar logistic derivative 5n; Gram `144^2*(8n-1)+144`; negated-Jacobian/RHS contraction `576n+144*(8n-1)`; the LU and substitutions above; and 288 operations for rate scaling and parameter addition. Total `C(n)=179045*n+2001144`.

Each update additionally uses 32n tanh, n exp, and n logaddexp evaluations, recorded but not assigned an arbitrary FLOP conversion. This distinction means nominal arithmetic matching is not an assertion of equal physical computation, special-function cost, elapsed time, or energy. The 500-update baseline nominal budget is 207,260,412,000. For n=256/768/1280 use floor(budget/C(n))=4332/1485/896 updates; each unused budget is less than one update (under 0.112%). Frozen preflight independently reassembles the kernel ledger and checks these floors. An exclusive, isolated timing audit after fitting/scoring commits will measure representative update blocks at a saved development state; no timing result changes training schedules.

## Fitting, deployment and commitments

Each procedure has raw logits h(x) and full mean logits `(h(x)+h(partner(x)))/2`. Fit independent positive inverse temperatures for these actual final outputs on that procedure's iid calibration prefix. Inverse temperature minimizes sampled-label NLL on `[exp(-3),exp(3)]` using frozen derivative bisection. Posterior probabilities are unavailable to training, checkpoint selection and temperature fitting; the generative sampler internally uses the posterior solely to sample labels. There is no selector or interpolation field in this branch.

Five deployment arms per procedure are raw uncalibrated, raw own-temperature, full uncalibrated, full with raw temperature (transfer diagnostic), and full own-temperature. Positive scaling must preserve signs. Uncalibrated diagnostics consume fewer fitting labels and are labeled with their actual costs. Raw deployments pay one forward/example and full deployments pay two, physically replayed in batches of 256. Temperature fitting actually shares paid raw/partner outputs, costing 2n calibration forwards per procedure; standalone raw/full alternatives cost n/2n. Temperature scalar optimization cost and iterations are retained separately. Raw and full test logits, inputs, posterior and sampled labels are saved per item; all scaled outputs reconstruct exactly from committed temperatures.

The source/protocol freeze precedes all reserved streams and requires a matching internal `ROOT_APPROVAL.json`. Training and calibration streams are opened first. `MODEL_COMMIT.json` commits every model/split and training ledger; `FIT_COMMIT.json` commits all temperatures and calibration outputs before test streams are opened. `SCORE_START.json` points to the fit commit. Source and previous-stage manifests are verified before each stage. Numerical failure aborts without seed replacement, with the attempted seed retained by process logs. Existing outputs are never overwritten.

## Random streams, analysis and limits

Pair indices 1..16 add to: training increments 2311000/2311100/2311200/2311300; shuffle 2311800; initialization 2312000; validation 2313000; calibration 2314000; test inputs 2316000; sampled test labels 2317000. Bootstrap seed is 2319000. All are disjoint from prior 215/216/217/218 families. Development preflight uses only the vendored old development state and 2099901/2099902. Each pair has 65,536 common iid test inputs. Expected NLL is primary; expected error, sampled-label NLL/error, Brier and posterior-binned ECE are secondary. The known posterior reduces evaluation noise but does not provide a certificate.

Eight primary expected-NLL contrasts comprise `flop_256`, `flop_768`, `flop_1280` minus `fixed_2304`, and `validation_1280` minus `flop_1280`, separately for raw own-temperature and full own-temperature. Negative favors the first procedure. Use 20,000 paired bootstrap resamples of the 16 model-seed clusters and two-sided alpha 0.05/8 per primary contrast. This is an approximate multiplicity-adjusted bootstrap interval, not a finite-sample theorem. All fixed-step adjacent contrasts, exact-example endpoints, within-n long training, full-temperature transfer, and all error contrasts are secondary with descriptive 95% intervals. No new formal claim follows from an exploratory best grid member. Generalization is limited to this synthetic task, fixed architecture/optimizer, chosen label budget, and modest frozen grid.

## Prior art and reproduction

Temperature scaling is established post-hoc calibration, not a novel method here: Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017. Compute-constrained allocation is established more broadly: Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022. That work studies large-model scaling and does not establish the present small PB task's allocation. This experiment's contribution is a controlled empirical accounting comparison under the preexisting model and invariance structure.

Reusable sources live in `pullback-feature-learning/experiments/budget_allocation/`; results stay in this archive branch. Execute the workspace `.venv/Scripts/python.exe` against `preflight.py`, then `runner.py freeze`; after internal review, run `runner.py train`, `calibrate`, `score`, and `analyze`, in that order. The `--out` option supports a new empty replication directory containing a copied protocol and its own explicit internal approval. Sources and stage manifests are hashed; archived previous results remain untouched.

