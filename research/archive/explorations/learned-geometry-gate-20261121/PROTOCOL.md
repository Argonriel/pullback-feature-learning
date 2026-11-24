# Frozen minimal discrimination protocol

Prepared on November 21, 2026 (America/Los_Angeles), before running or scoring the new study. This is a mechanism-screening protocol, not a claim of innovation and not a replacement for any frozen earlier course-project study.

## Claims and elimination order

The research question is when label-dependent, all-parameter representation-pullback geometry gives predictive benefits beyond fixed initial tangent preconditioning, and when it harms them. No answer is assumed. Broad claims that nonlinearity matters, preconditioning alters learning, or spectra affect generalization are already occupied by prior work. Joint-asymmetry and readout-decoupling candidates remain retired.

Screen C0: at matched attained training loss, nonlinear PB has meaningful predictive behavior not reproduced by full initial tangent PB. The four-arm interaction additionally tests whether PB benefits from nonlinearity more than GD. C0 is a necessary empirical gate, not an innovation claim.

Conditional C1: early label-replica variability of learned geometry might predict the later signed benefit of adaptive over fixed-initial-metric PB beyond initial geometry and training progress. This is a specific candidate, not an established mechanism. It will not be promoted from the four-arm result. A future prospective test would require fixed early information, a trained/frozen decision rule, independent seeds and noise levels, and comparison to initial-geometry and training-progress predictors. The present gate must first justify investigating it.

Elimination order: original-source assumptions; reduction to fixed features/reparameterization/scalar time change; simple counterexamples; four-arm experiment; necessary fixed-metric intervention; only then signed population transfer diagnostics. Routine identities and already completed whole-update population derivative forecasts are not contributions.

## Models and exact normalization

Use the existing 80-parameter, two-block width-8 residual model, fixed first-coordinate readout, mean logistic loss, and summed two-coordinate representation metric. For every training or evaluation input x, the full tangent representation is z0(x)+J0(x)(theta-theta0); both prediction map and representation Jacobian are frozen. Tangent PB uses M0=J0(X)^T J0(X)+lambda I throughout. This is an 80-parameter tangent model, not the old two-dimensional initial-feature probe.

Four arms: nonlinear GD, nonlinear adaptive PB, tangent GD, tangent PB. Same inputs, labels and initialization within every pair. Damping is lambda=0.1 on the summed metric. The training sample count is fixed at 256. No geometry sample-size enlargement is allowed. Repeated-sample normalization is independently checked.

## Tasks, information and cohorts

Moons: existing continuous-angle balanced two-moons generator with Gaussian noise 0.15. Rings: balanced class-conditional circles with radii 0.5 and 1.2 and isotropic Gaussian noise 0.15, using the existing generator law. Model width and task law are not selected from new results.

Discovery pairs d1-d4 use training seeds 221101-221104, initialization seeds 222101-222104, and evaluation seeds 223101-223104; rings adds 10000 to each seed. Conditional independent confirmation uses v1-v4, corresponding seeds 321101-321104 / 322101-322104 / 323101-323104, again adding 10000 for rings. All seeds are predeclared. Discovery scores do not select a task variation, model, damping, endpoint or rate for confirmation.

Training and endpoint selection may use training X,y, initial/current parameters and Jacobians only. Evaluation is a separate pass after states are saved. Moons uses a fresh balanced 20,000-point sample per pair. Rings uses the known population law by polar integration, with the original radial/angular excess decomposition and refinement tolerances 2e-7 absolute plus 2e-6 relative. Evaluation outcomes are unavailable to training and checkpoint selection. A separate Simpson/shifted-angle check audits representative rings endpoints. No label-conditioned independence is asserted for fitted Jacobians.

## Rate, progress, stopping and missing outcomes

All arms retain their prescribed update direction. To remove scalar clock as an explanation, choose the scalar rate to target predicted training-logit RMS step 0.01; repeat with 0.005. Cap parameter displacement at 0.1. Backtrack to satisfy Armijo coefficient 1e-4 and first-coordinate Taylor remainder at most 5% of predicted logit displacement. These controls are explicit experimental scalar clocks, not a claim about the original tuned optimizers' speed.

Record first attained losses at fractions 0.75, 0.5 and 0.25 of that pair's common initial loss. Shorten a crossing update by scalar bisection to target absolute loss error <=1e-10; resume from this actual parameter state. No interpolation of saved parameters, predictions or risk is used. The primary endpoint is 0.25; the other two are secondary diagnostics and cannot rescue the primary gate. Stop after reaching the primary target, 6000 updates, a numerical failure, or inability to accept a descent step. Preserve every failed/censored prefix. Unreachable endpoints are reported, never silently dropped or substituted.

## Estimands and gate

Risk R is logistic loss (population for rings; held-out for moons). Accuracy is descriptive only. At each shared loss target record B=R_tangentPB-R_nonlinearPB (positive means nonlinear PB improves on its own tangent model) and D=(R_nonlinearGD-R_nonlinearPB)-(R_tangentGD-R_tangentPB). Positive D means PB's comparative advantage is amplified by nonlinearity. Neither is a metric-learning causal effect.

C0 survives discovery for a task only if all four pairs attain the primary endpoint, |mean B|>=0.01 nat with at least 3/4 paired signs agreeing with that mean at both step sizes, and the change in mean B between step sizes is <=0.003 nat. Retain positive/negative direction explicitly. Apply the same thresholds to D as a separate, stronger PB-specific finding; D is not required to diagnose a generic nonlinear residual. Display all paired values, means and sample SDs without treating four seeds as population-wide significance.

If C0 fails, stop expansion of this candidate for that task; retain the bounded null/inconclusive result. If C0 passes, run predeclared confirmation pairs for that task at both step sizes, with the same gate. If it does not replicate, report failure and stop. A meaningful negative residual passes the screening gate as harm, not as success for a benefit claim.

## Conditional interventions and second gate

Only for tasks passing discovery C0, run nonlinear prediction with fixed M0 at both step sizes on the same discovery pairs. This preserves nonlinear gradients while removing metric adaptation. Quantify F=R_fixedMetricPB-R_adaptivePB; positive is benefit from adapting the metric. Confirmation uses the same arm if the first gate replicates. Fixed-M0 training is GD under a fixed parameter reparameterization, and is itself not a new method. Compare lambda=0.01 and 1.0 on discovery PB arms at step size 0.005 as damping sensitivity; these are prespecified checks, never post-result replacements for the primary lambda.

Only if the first gate leaves a meaningful replicated difference should signed spectral diagnostics be expanded. For rings the diagnostic contribution per parameter-space eigenvector v_k is (v_k^T grad R_population)(v_k^T grad L_train)/(mu_k+lambda), with all signs retained. The sum is the known whole-update derivative identity. It cannot be called a new prospective predictor. Signal/noise separation must use independent conditional-label replicas or the known law; fitted eigenvectors may depend on the same label noise. A causal or advance predictor claim needs additional independently frozen validation and is not automatically authorized as a conclusion by this protocol.

## Cost and provenance

Record iterations, controller backtracks and elapsed times, but do not rank computational efficiency: the common direction-normalization controller evaluates Jacobians even for GD, while production GD can use reverse mode. A fair end-to-end cost comparison is deferred until a mechanism survives screening.

Code and reusable config live in the repository. Protocol, data, snapshots, evaluation records, literature audit, report and hashes live in a new independent archive directory. Preserve all earlier records and all repository junctions. Save code/config hashes and environment metadata before training. No contact with other people is authorized.

## Required checks

- Full affine tangent representation, both coordinates and out-of-sample inputs.
- Same initialization and first directions for nonlinear and tangent counterparts.
- Multi-step tangent prediction-kernel equivalence.
- Full representation metric is not replaced by readout-only metric.
- Nonlinear fixed metric retains current prediction gradient.
- Summed/mean normalization under duplicated samples.
- Exact loss-target attainment, finite states, censored-run reporting.
- Independent evaluation recomputation and representative numerical replay.
- Population integration refinement and independent quadrature.
- Source/config/data/artifact hashes; junction targets unchanged.
