"""Generate the branch report entirely from committed outputs."""
import importlib.metadata,json,platform
import numpy as np
from common import HERE,ci,jo
from math_utils import LAWS

def main():
    result=json.loads((HERE/'RESULTS.json').read_text());cost=result['costs'];summary=ci(HERE/'certificate_summary.csv')
    curves=ci(HERE/'certificate_curves.csv');truth=ci(HERE/'stress_summary.csv');evidence=json.loads((HERE/'EVIDENCE_COMMIT.json').read_text())
    primary=next(r for r in result['primary'] if r['kind']=='grid');labeled=next(r for r in result['primary'] if r['kind']=='labeled')
    main_rows=[]
    for law in LAWS:
        r=next(r for r in summary if r['law']==law['name'] and int(r['n_batches'])==2048 and r['scenario']=='e0')
        t=next(r for r in truth if r['law']==law['name'] and r['world']=='orbit_worst' and float(r['epsilon'])==0)
        label='valid assumption' if law['covered'] else 'INVALID sensitivity'
        main_rows.append(f"|{law['name']}|{label}|{float(r['known_lower_mean']):+.6f}|{float(r['estimated_lower_mean']):+.6f}|{r['estimated_lower_certified_positive']}/16|{float(t['known_density_gain_mean']):+.6f}|{t['known_density_gain_negative']}/16|")
    law_table='\n'.join(main_rows)
    label_rows=[]
    for audit in evidence['labels']:
        s=next(r for r in summary if r['law']==audit['law'] and int(r['n_batches'])==2048 and r['kind']=='labeled' and float(r['true_defect'])==audit['defect'])
        t=next(r for r in truth if r['law']==audit['law'] and r['world']=='cell' and float(r['epsilon'])==audit['defect'])
        label_rows.append(f"|{audit['law']}|{audit['defect']:.1f}|{audit['epsilon_upper']:.6f}|{audit['minimum_count']:,}|{float(s['estimated_lower_mean']):+.6f}|{s['estimated_lower_certified_positive']}/16|{float(t['known_density_gain_mean']):+.6f}|")
    label_table='\n'.join(label_rows)
    primary_rows=[r for r in curves if r['law']=='covered_third' and int(r['n_batches'])==2048 and r['scenario']=='e0']
    primary_rows.sort(key=lambda r:(r['policy']!='symmetric_jensen',int(r['pair'])))
    label_index={(r['pair'],r['policy']):r for r in curves if r['law']=='covered_third' and int(r['n_batches'])==2048 and r['scenario']=='label0'}
    line_table='\n'.join(f"|{r['pair']}|{'Interpolation' if r['policy']=='symmetric_jensen' else 'Folded'}|{float(r['known_lower']):+.6f}|{float(r['estimated_lower']):+.6f}|{float(label_index[r['pair'],r['policy']]['estimated_lower']):+.6f}|" for r in primary_rows)
    invalid=[r for r in ci(HERE/'certificate_vs_truth.csv') if r['law']=='violated_residual' and r['scenario']=='e0' and r['world']=='orbit_worst']
    false_positives=sum(r['assumptions_covered']=='False' and float(r['estimated_lower'])>0 and float(r['truth_estimate'])<0 for r in invalid)
    p_radius=evidence['parameters'][0]['radius'][0]
    body=f'''# Residual envelopes and independent label evidence

Completed 21 November 2026. The primary covered third-harmonic law gives **{primary['estimated_lower_certified_positive']}/16 positive estimated-coefficient certificates** at assumed posterior defect zero, and **{labeled['estimated_lower_certified_positive']}/16** when an independent finite-cell label audit supplies its defect upper bound. Both positivity counts were fixed primary endpoints. The latter conclusion additionally assumes posterior constancy within the eight specified cells. No model, policy, coefficient basis, residual envelope, clip or sample budget was changed after fresh outcomes.

The advance is a conditional density-envelope result rather than an unrestricted distribution-shift certificate. It includes the previously omitted third harmonic, propagates coefficient uncertainty and a supplied residual bound, and accounts for finite label evidence. Deliberately understating the residual remains unsafe: the outside-envelope sensitivity law produces {false_positives}/16 positive estimated formula values paired with negative independent worst-posterior gain estimates. Those values are explicitly invalid, never counted as certificates.

![Primary fixed-model results](residual_report.png)

## The mathematical extension

Let r=d(P-T#P)/d(P+T#P) be the odd orbit imbalance. For logits z,zT, let d=(z-zT)/2, S=span(d), J their Jensen gap, and t=(F(z)-F(zT))/2-midrange(d). The preceding sharp posterior result yields a per-orbit lower expression J-h_epsilon(r), where

    h_epsilon(r)=[|r|+(1-|r|)epsilon]S/2-r*t.

This penalty is convex in r. If independent evidence and explicit assumptions provide an odd interval I=[l,u] containing r everywhere, the exact worst case is **H_epsilon(I)=max(h_epsilon(l),h_epsilon(u))**. It is even, so it remains compatible with a fixed, single-coordinate-invariant top-quota mask on the same deployment law. No product of batch importance weights is used. The resulting bound is G_P>=L_A-U_H, where A uses the clipped Jensen gain. This is the joint endpoint extremum, not a sum of penalties maximized at incompatible endpoints.

If all interval endpoints have magnitude at most rbar, the deterministic item range is D[epsilon/2+(1-epsilon/2)rbar]. D remains support-wide and parameter-derived, never a test maximum. A small independently bounded rbar therefore improves both the population penalty and its finite-sample radius. Ordinary floating-point evaluation of parameter constants is used; no directed-rounding proof is claimed.

The input model is r=theta1*cos(phi)+theta3*cos(3phi)+e(phi), relative to an explicitly known invariant reference law. The residual is odd, orthogonal to the retained harmonics and bounded pointwise by supplied rho. The orthogonality identifies theta_j=2E_P cos(j phi). Each law receives 262,144 independent unlabeled coordinate inputs, giving two simultaneous Hoeffding intervals of half-width {p_radius:.6f}. The propagated interval includes coefficient error plus rho, clipped symmetrically to [-1,1]. All verification moments are measured on the actual law, never transplanted from a symmetric distribution.

The residual envelope is an assumption or externally justified constraint; the two coefficient estimates do not establish it. The broader reference-law and angular assumptions also remain. A discriminator would provide a TV lower witness, not the needed support-wide upper envelope. Detailed proofs and event accounting are in THEORY.md. LIMITATIONS.md gives a finite-sample impossibility argument beyond the two-moment example: narrow orthogonal odd perturbations can have residual essential supremum one while being arbitrarily close in total variation to the symmetric law. Without regularity restrictions, a uniformly honest 95% procedure can report a residual upper bound below one at that symmetric law with probability at most 5%.

![Covered-law and independent-label overview](residual_overview.png)

The hatched outside-envelope bar counts positive numerical values only; it is not a certificate count. The label panel marks all 16 model-policy lines in each audit, including nonpositive bounds.

## Complete covered and uncovered law comparison

All entries use epsilon=0 and 2048 verification batches. Means and ranges describe eight fixed historical models times two unchanged policies, not a distribution of new trained models. The gain column uses an independent Monte Carlo orbit-corrected estimator of the same deployment risk. The worst posterior is model-dependent.

|Input law|Status|Known-coefficient mean bound|Estimated-coefficient mean bound|Certified positive|Worst-posterior gain estimate|Negative truth lines|
|---|---|---:|---:|---:|---:|---:|
{law_table}

The exact laws and supplied rho values were frozen in PROTOCOL.md: first .01; third .10; mixed first/third .04 each; covered residual first .01, third .04, fifth .025 with rho=.025; strong third .60; and omitted fifth .60 with falsely supplied rho=0. The uniform law is the reference control. Covered does not mean beneficial: an honest negative lower bound on a harmful case is an expected outcome. All eight epsilon values and both fixed looks remain in certificate_curves.csv.

![Covered and invalid sensitivity outcomes](residual_stress.png)

## Independent labels under an explicit eight-cell posterior model

The label audit freezes two radial bins per coordinate (cut at .9), crossed with the two transform-paired angular half-circles. It assumes a constant binary posterior within each of these eight cells. An independent sample of 131,072 actual IID labeled inputs is used for each of two laws and two true cell defects. Conditional on the observed cell counts, each cell receives a two-sided Hoeffding interval. Empty cells receive [0,1]; expected cell counts are never substituted for observed counts. Taking the largest difference allowed by paired intervals supplies epsilon_U. These labels are new observations, not known generator probabilities treated as observed evidence.

|Law|True cell defect|Label-supported epsilon_U|Smallest cell count|Estimated-density bound mean|Certified positive|Independent cell-law gain|
|---|---:|---:|---:|---:|---:|---:|
{label_table}

The bound statistic and its range use epsilon_U, not the true generator defect. This sample is independent of coefficient estimation and verification; the same already frozen model-policy set shares it. Across the four audits this consumes **{cost['actual_new_audit_labels']:,} labeled examples**, separately from fitting labels. There is no new training or selector fitting. All cell counts, successes, raw sampled labels and interval endpoints are retained.

The within-cell assumption matters even with unlimited labels. For p(phi)=(1+sin(3phi))/2, pointwise posterior TV reaches one, but both half-circle means are exactly one-half under every cosine-only input law used here. Thus cell-average evidence could converge to apparent symmetry while the pointwise condition fails. This analytic counterexample is recorded in COUNTEREXAMPLES.json. We have not solved arbitrary posterior identifiability with a coarse label audit.

## All primary model-policy lines

The first two columns assume epsilon=0. The last column instead uses the independent invariant-cell audit's epsilon_U={labeled['epsilon']:.6f}. All three bounds include finite verification uncertainty.

|Model|Policy|Known coefficients|Estimated coefficients|Estimated coefficients + labels|
|---|---|---:|---:|---:|
{line_table}

## Confidence, negative outcomes and interpretation

The 95% simultaneous allocation comprises 8576 moment/method events, 14 coefficient intervals and 32 cell intervals, 8622 total. Empirical Bernstein treats whole batches as independent units. The bounded-differences alternative uses a proved top-quota replacement sensitivity over independent input/key records. Every endpoint penalty and its range are fixed conditional on the independent evidence samples. Shared policies, moments and nested 512/2048 looks are covered by the explicit union bound.

This exact endpoint implementation computes epsilon-dependent statistics, so it pays for all eight reported grid values and four independently estimated label values. It does **not** claim unreported-epsilon uniformity from the finite grid. Only rows satisfying the density and posterior assumptions are risk certificates. Moment concentration itself still applies to the numerical sensitivity formula under a violated envelope, but it cannot repair a false population inequality.

The smallest independent truth estimate minus a valid known-coefficient bound is {result['valid_min_truth_minus_known']:.6f}; for estimated coefficients it is {result['valid_min_truth_minus_estimated']:.6f}. These are descriptive Monte Carlo consistency diagnostics, not proofs of confidence coverage. Truth tables include ordinary IID gains, the exact known-density orbit-corrected estimator, batch standard errors, NLL and classification error. Classification error itself is not certified by this cross-entropy bound. All negative outcomes are retained.

The previous cos3 counterexample is now explicitly covered: density (1+.6cos3phi)/(2pi), model h=.2cos3phi, invariant p=1 and an independent quarter quota have true gain below -.006875. Retaining cos3 recovers its negative un-clipped orbit expression instead of the false positive obtained by pretending r=0. Equality with the true gain refers to this un-clipped expression; arbitrary Jensen clipping only lowers it further. Replacing 3 by 5 demonstrates the residual issue: rho=.6 covers it, while rho=0 does not. Both signs follow analytically from log cosh(v)<=v^2/2; deterministic quadrature only illustrates them.

## Cost, reproducibility and limits

|Work|Inputs|Model example evaluations|
|---|---:|---:|
|Independent coefficient samples|{cost['parameter_inputs']:,}|0|
|Independent label audits|{cost['labeled_audit_inputs']:,}|0|
|Full verification, including nested 512 look|{cost['verification_inputs']:,}|{cost['verification_forwards']:,}|
|Independent truth stress|{cost['stress_inputs']:,}|{cost['stress_forwards']:,}|
|Deployment replay checks|Reused batches|{cost['deployment_replay_forwards']:,}|

The 512-batch prefix costs {cost['verification_512_forwards']:,} forwards. Both policies share evaluations of their selected union, which contains {cost['union_selected_min']}--{cost['union_selected_max']} inputs per batch. Deploying one policy remains 256 original plus 64 transformed evaluations. Historical selector fitting used 1,048,576 evaluations and is distinct from this round. The saved statistics require no additional model evaluations; their scalar-work counts are in STATISTICS_LEDGER.json. Concurrent wall times are descriptive, not equal-hardware comparisons.

The first truth-stage attempt stopped on its first model/law because theoretical probability zero rounded to a tiny negative value. Its source and failure record remain intact. A separately reviewed recovery clips only endpoint excursions below 1e-12 and retains every precommitted seed, sample count and setting. The largest observed excursion was {cost['recovery_endpoint_roundoff']['maximum_absolute_excursion']:.3g}. The failed attempt's {cost['original_failed_attempt_truth_inputs']:,} regenerated truth inputs and {cost['original_failed_attempt_forwards']:,} forward evaluations are implementation overhead, additional to the successful-stage table. They are not independent evidence. See STRESS_RECOVERY.md and its source/release hashes.

Frozen sources and historical assets are checked before execution, and every raw stage has its own hash manifest. All original studies remain untouched. The raw evidence, selected-union logits, features, masks, batch moments, confidence ranges/radii, truth logits, posteriors' definitions and exact call ledgers are retained. The largest truth identity error is {cost['max_identity_error']:.3g}; extremal equality error is {cost['max_extremal_error']:.3g}. Root's independent proof/preflight and outcome reconstruction are separate from this author's diagnostics.

Use `python reproduce.py --output ../fresh_residual_replay` from the archived standalone branch. It refuses to overwrite an existing directory and verifies the frozen source/input hashes. The complete file map and environment are in README.md and ENVIRONMENT.json. No natural-image density envelope, within-cell posterior model or universal unlabeled safety guarantee follows from this controlled study.

## Prior work and the scope of this result

[Chen, Dobriban and Lee (2020)](https://jmlr.org/papers/v21/20-163.html) provides the group-averaging framework. [Maurer and Pontil (2009)](https://arxiv.org/pdf/0907.3740) supplies the bounded-variable concentration constants. [Gine and Nickl (2010)](https://arxiv.org/pdf/1002.4801) and [Kerkyacharian, Nickl and Picard (2012)](https://arxiv.org/abs/1102.2450) develop density confidence bands under explicit regularity assumptions; our simple coefficient intervals with a supplied residual are not replacements for those assumptions or algorithms. The new work is the specialized joint endpoint/range propagation, its independent finite-cell label extension, and this prospective controlled comparison, with no literature-wide priority claim.
'''
    with (HERE/'REPORT.md').open('x',encoding='utf-8') as stream:stream.write(body)
    jo(HERE/'ENVIRONMENT.json',dict(python=platform.python_version(),platform=platform.platform(),
        packages={key:importlib.metadata.version(key) for key in ['numpy','matplotlib','threadpoolctl']},threadpool_limit=1))
    print('Wrote REPORT.md and ENVIRONMENT.json')
if __name__=='__main__':main()
