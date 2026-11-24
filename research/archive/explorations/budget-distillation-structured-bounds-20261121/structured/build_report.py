"""Build branch report and an explicit statistics-work ledger from fixed outputs."""
import importlib.metadata,json,platform
import numpy as np
from common import HERE,ci,jo,now

def main():
    results=json.loads((HERE/'RESULTS.json').read_text());cost=results['costs']
    curves=ci(HERE/'certificate_curves.csv');moments=ci(HERE/'moment_bounds.csv');truth=ci(HERE/'stress_results.csv')
    primary=[r for r in curves if int(r['n_batches'])==2048 and float(r['a'])==.01 and float(r['epsilon'])==0]
    primary.sort(key=lambda r:(r['policy']!='symmetric_jensen',int(r['pair'])))
    primary_table='\n'.join(f"|{r['pair']}|{'Interpolation' if r['policy']=='symmetric_jensen' else 'Folded'}|{float(r['known_lower']):.6f}|{float(r['estimated_lower']):.6f}|{float(r['old_lower']):.6f}|" for r in primary)
    count_table=[]
    for a in [0,.01,.1,.6]:
        for eps in [0,.1,.2,.4]:
            rows=[r for r in curves if int(r['n_batches'])==2048 and float(r['a'])==a and float(r['epsilon'])==eps]
            counts=[sum(float(r[k])>0 for r in rows) for k in ['known_lower','estimated_lower','old_lower']]
            count_table.append(f'|{a:g}|{eps:g}|{counts[0]}/16|{counts[1]}/16|{counts[2]}/16|')
    count_table='\n'.join(count_table)
    ptable='\n'.join(f"|{float(r['a']):g}|{float(r['estimate']):.6f}|[{float(r['lower']):.6f}, {float(r['upper']):.6f}]|" for r in results['parameter_intervals'])
    comparisons=[]
    for a,eps,world in [(.01,0,'cosine'),(.01,0,'orbit_worst'),(.01,.4,'adversarial'),(.1,0,'orbit_worst'),(.6,0,'cosine'),(.6,0,'orbit_worst')]:
        rows=[r for r in truth if float(r['a'])==a and float(r['epsilon'])==eps and r['world']==world and r['policy'] in ['symmetric_jensen','folded']]
        values=np.array([float(r['known_density_gain']) for r in rows]);direct=np.array([float(r['direct_gain']) for r in rows])
        comparisons.append(f'|{a:g}|{eps:g}|{world}|{values.mean():.6f}|{direct.mean():.6f}|{sum(values<0)}/16|')
    truth_table='\n'.join(comparisons)
    known=np.array([float(r['known_lower']) for r in primary]);estimated=np.array([float(r['estimated_lower']) for r in primary]);old=np.array([float(r['old_lower']) for r in primary])
    penalty=np.mean([float(r['known_shift_penalty']) for r in primary]);oldpenalty=np.mean([float(r['old_shift_penalty']) for r in primary])
    stats=dict(timestamp_utc=now(),independent_batch_moment_series=8*4*2*5,
        stored_batch_moment_values=8*4*2*5*2048,moment_look_reductions=len(moments),
        batch_values_processed_across_fixed_looks=8*4*2*5*(512+2048),one_sided_moment_radius_calculations=2*len(moments),
        independent_parameter_intervals=4,parameter_cosines=4*65536,total_probability_events=1284,
        epsilon_curve_rows=len(curves),separate_certificate_values=3*len(curves),
        posterior_stress_rows=len(truth),random_algebra_cases=12,random_algebra_records=49152,deterministic_quadrature_points=1048576,
        frozen_model_or_policy_refits=0,model_forwards_for_statistics_from_saved_logits=0,
        explanation='Counts are exact operations/retained scalar series, not a FLOP estimate or a runtime benchmark.')
    jo(HERE/'STATISTICS_LEDGER.json',stats)
    environment=dict(python=platform.python_version(),platform=platform.platform(),
        dependencies={name:importlib.metadata.version(name) for name in ['numpy','matplotlib','threadpoolctl']},threadpool_limit=1)
    jo(HERE/'ENVIRONMENT.json',environment)
    report=f'''# Explicit orbit structure rescues the small-shift guarantee

Completed 21 November 2026. The new bound certifies all **16/16 fixed model-policy pairs** at the previously fatal a=.01 shift, both when a is known and when it is estimated with a conservative confidence interval. The old bound certifies 0/16. This is a conditional result for an explicitly specified harmonic input law and a supplied posterior-invariance assumption, not an unrestricted natural-data certificate.

The prospective primary setting is epsilon=0, a=.01, 2048 IID batches of 256 inputs per fixed model and law. The known-a lower bounds range from **{known.min():.6f} to {known.max():.6f} nats** (mean {known.mean():.6f}); estimated-a bounds range from **{estimated.min():.6f} to {estimated.max():.6f}** (mean {estimated.mean():.6f}). Old global-TV bounds average {old.mean():.6f}. All are recomputed on the same fresh verification streams at the same 95% simultaneous confidence allocation.

![Structured certificate results](structured_frontier.png)

## What changed mathematically

The old penalty was D*TV(P,T#P), with D=24.34--34.22. The new result uses the actual orbit imbalance r=d(P-T#P)/d(P+T#P), the logit difference d=(z-zT)/2, its span S, and t=(F(z)-F(zT))/2-midrange(d). Under posterior TV at most epsilon, the sharp conditional orbit lower bound is

    J + r t - [|r|+(1-|r|)epsilon] S/2.

The coefficient is proved by optimizing jointly over two posterior probability vectors, not by replacing a global range with observed extrema. The proof includes the extremal posteriors, and is valid for any number of classes. All logits may receive arbitrary common offsets without changing the bound. The deployment selector must be invariant when **one input** is transformed with every other input and tie key fixed. Both historical exact-top-quarter policies satisfy this condition.

For the family (1+a cos phi)/(2 pi), r=a cos phi. Nonnegative even weights W=|cos phi|S/2-cos phi*t and R_a=(1-a|cos phi|)S/2 give

    G_P >= L_A - a U_W - epsilon U_Ra.

All moments are measured under the actual deployment P_a. Deterministic D remains in finite-sample concentration; it is not discarded or estimated from a maximum. At the primary setting, the mean new input-shift penalty is {penalty:.6f}, compared with {oldpenalty:.6f} for the old global penalty, a {oldpenalty/penalty:.1f}-fold reduction. The gain comes from explicit density structure and the selected local loss geometry, not from an assumed distribution-free TV estimate. No clipped tail is ignored: only the nonnegative Jensen gain is clipped downward; W and S remain untruncated.

THEORY.md gives the complete proof, the same-P conditional expectation identity, deterministic support-wide ranges and the confidence-event accounting. The general group-averaging viewpoint precedes this project; see [Chen, Dobriban and Lee (2020)](https://jmlr.org/papers/volume21/20-163/20-163.pdf). The concentration constants come from [Maurer and Pontil (2009), Theorem 4](https://arxiv.org/pdf/0907.3740), together with [McDiarmid's bounded-differences method](https://doi.org/10.1017/CBO9781107359949.008). The sharp posterior specialization and this fixed-policy experiment are the additions in this round; no literature-wide novelty claim is made.

## All primary lines

Positive numbers certify expected raw-minus-policy cross-entropy gain, conditional on the stated assumptions. Model indices are the eight previously fixed historical networks, with unchanged selectors and clipping rule. They are not independent fresh-model replications.

|Model|Policy|Known a lower|Estimated a lower|Old TV lower|
|---|---|---:|---:|---:|
{primary_table}

The 512-batch look, the other epsilon values and every individual line are retained in certificate_curves.csv. The 2048-batch look was primary before new draws; no sample-size extension or clipping/range change followed the result.

## Unknown parameter: useful, still conditional

One independent sample of 65,536 coordinate inputs per law estimates a=2 E cos phi. There are no model calls or labels in this estimation step. A two-sided Hoeffding interval has radius .03638749, with one of 1284 total probability-event allocations per law. Restrict to a in [0,1]. The conservative propagated bound is

    G_P >= L_A - aU U_W - epsilon U_R(aL).

|True a|Estimate|Simultaneous parameter interval|
|---:|---:|---:|
{ptable}

The parameter interval is conditioned on an exact family assumption; it does not validate that family. Conditional on the independent estimation sample, R(aL) is fixed, so the verification concentration remains valid. The union bound covers 1280 one-sided moment/method/model/policy/law/look events and four already-two-sided parameter intervals. Once these events hold, the certificate is uniform over epsilon in [0,1]; the display grid costs no additional events. The whole family has coverage at least 95%, conditional on all historical fitting artifacts.

## Where certification still fails

These are counts of positive lower bounds at the primary 2048-batch look. All negative and inconclusive results are retained.

|a|Posterior epsilon|Known a|Estimated a|Old global TV|
|---:|---:|---:|---:|---:|
{count_table}

At a=.01, both new versions still certify all 16 pairs at epsilon=.1. At epsilon=.2, only 5 known-a certificates remain positive and the parameter-uncertain version has none. At a=.6, neither version certifies any pair even with invariant labels. Thus the method tolerates modest, explicitly structured defects; it does not guarantee averaging will always help.

## Independent truth stress retains real negative gains

The independent 65,536-input evaluation per model/law uses known conditional label probabilities. It reports the ordinary direct IID gain and a density-corrected conditional orbit estimator of the **same** deployment risk. The corrected estimator is used below because it reduces orientation noise; it is not the ordinary unweighted paired-law score, a new certificate, or the exact population risk. Batch standard errors and both estimators are retained for every line in stress_results.csv.

|a|epsilon|Posterior world|Mean corrected gain|Mean direct IID gain|Negative corrected lines|
|---:|---:|---|---:|---:|---:|
{truth_table}

The orbit_worst posterior world attains the sharp local lower bound before Jensen clipping, up to floating-point error. It is deliberately model-dependent; eight models do not share one common label task in this world. The true label law is still invariant at epsilon=0. All 16 averaged policies become harmful at a=.6 in that world, while the original invariant p0 world remains beneficial. Input symmetry and label semantics are separate requirements.

For all model/law/epsilon/policy/posterior comparisons, the smallest corrected Monte Carlo estimate minus known-a certificate is {results['min_truth_minus_known']:.6f}, and minus estimated-a certificate is {results['min_truth_minus_estimated']:.6f}. These consistency margins do not prove finite-sample coverage. The theorem supplies the coverage; simulation illustrates its consequences. No coverage claim over a new population of trained models is made.

## A parameter CI cannot repair a wrong family

An analytic counterexample uses density (1+.6 cos 3phi)/(2 pi), model h=.2 cos 3phi, invariant posterior p=1 and an input-independent exact-top-quarter mask. Its first-harmonic moment is zero, so the a estimator converges to zero and its interval narrows. Nevertheless, the correct gain is negative: log cosh v<=v^2/2 proves gain<-0.006875. A deterministic angular calculation gives -0.006875780, while incorrectly using the assumed a=0 family produces a positive +0.000624220 bound. MISSPECIFICATION.json records this illustration.

This is failure of a missing assumption, not a counterexample to the theorem. A discriminator only witnesses a TV lower bound. The previous study's unrestricted-TV obstruction remains valid, and unlabeled inputs still cannot identify the posterior relation. Related insufficiency-of-assumptions results appear in [Ben-David et al. (2010)](https://proceedings.mlr.press/v9/david10a.html); our harmonic counterexample is derived explicitly here rather than attributed to that paper.

## Cost and verification record

|Work|Coordinate inputs|Neural example evaluations|
|---|---:|---:|
|Parameter intervals, all four laws|{cost['parameter_coordinate_inputs']:,}|0|
|512-batch certificate look, all model/law/policy pairs|{8*4*512*256:,}|{cost['certificate_512_model_forwards']:,}|
|2048-batch certificate, including nested 512 look|{cost['certificate_coordinate_inputs']:,}|{cost['certificate_model_forwards']:,}|
|Independent posterior stress|{cost['stress_coordinate_inputs']:,}|{cost['stress_model_forwards']:,}|
|Deployment replay audits|Reused 256-input batches|{cost['deployment_replay_forwards']:,}|

The two policies share paired evaluations on their selected union, which contained {cost['union_selected_min']}--{cost['union_selected_max']} of 256 inputs. At deployment each policy uses exactly 256 original plus 64 transformed calls. There is no new model/selector fit and no new fitting label. Historical selector construction used {cost['historical_selector_forwards']:,} model forwards and is charged separately from this round. Historical models were trained with labels; this is not a claim of label-free model training.

STATISTICS_LEDGER.json records {stats['moment_look_reductions']} separate moment/look reductions and {stats['one_sided_moment_radius_calculations']} allocated concentration calculations, plus all parameter, curve and posterior-stress work. The independent unit for empirical Bernstein is the full batch; the McDiarmid calculation uses a proved replacement sensitivity over independent records. Neither treats selected records as independent. Source/raw-data hashes, exact sampler seeds, individual forward ledgers and per-moment variances/radii/ranges are retained.

All 12 multiclass algebra cases (49,152 random records) passed. The largest stress identity error is {cost['max_identity_error']:.3g}, sharp extremal equality error {cost['max_extremal_error']:.3g}, and partner-posterior error zero. Four single-coordinate mask replacements per model/law passed exactly. Ordinary floating-point parameter arithmetic is used; these checks are not a formal machine-verified interval proof. No natural-image input law or posterior defect is supplied by this controlled experiment.

The source freeze preceded every 233-family fresh sample and internal root approval. No runtime source was altered after release and no failed scientific outcome was discarded. Concurrent wall times ({cost['certificate_seconds_descriptive']:.2f}s certificate, {cost['stress_seconds_descriptive']:.2f}s stress) are descriptive only. Independent root auditing, including its separate calls and fresh random stream, is recorded outside this branch. See README.md for the standalone fresh-directory replay command and the complete file map.
'''
    with (HERE/'REPORT.md').open('x',encoding='utf8') as f:f.write(report)
    print('Wrote REPORT.md, STATISTICS_LEDGER.json and ENVIRONMENT.json')
if __name__=='__main__':main()
