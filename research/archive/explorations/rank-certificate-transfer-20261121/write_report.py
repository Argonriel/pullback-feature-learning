"""Build the final manuscript from completed, audited result records."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def rows(path):
    with (ROOT/path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def js(path):return json.loads((ROOT/path).read_text(encoding='utf8'))
def mean(data,key):return float(np.mean([float(r[key]) for r in data]))
def num(x):return f'{x:,.0f}'

def run():
    s=rows('validation/summary.csv');contrasts=rows('validation/contrasts.csv')
    lookup={(r['cohort'],r['method'],r['policy'],r['outcome']):float(r['mean']) for r in s}
    known=js('root_audit/certificates/summary.json');emp=js('root_audit/certificates_empirical/summary.json')
    cert_rows={m:rows(f'root_audit/{f}/per_model.csv') for m,f in [('known','certificates'),('empirical','certificates_empirical')]}
    bounds={(m,t):rows(f'certificate/{f}/{t}/summary.csv') for m,f in [('known','fresh'),('empirical','fresh_empirical')] for t in ('rings','waveform')}
    audit=js('root_audit/validation_checks/summary.json');assert audit['all_passed']
    assert js('root_audit/certificate_checks.json')['all_passed']
    names=[('raw','Raw model'),('rank_balanced','Learned balanced rank'),('rank_even','Even rank'),
           ('folded','Folded symmetric gate'),('confidence','Confidence'),('confidence_radial','Confidence + angle'),
           ('rank_unbalanced','Unconstrained rank'),('full','Full half-turn'),('full_radial','Full adapted angle')]
    table='\n'.join(f'| {label} | {1 if name=="raw" else 2 if name in ("full","full_radial") else 1.25:.2f} | {lookup["rings","pb",name,"expected_risk"]:.6f} | {lookup["transfer","pb",name,"expected_risk"]:.6f} |' for name,label in names)
    ct='\n'.join(f'| {"Rings" if r["cohort"]=="rings" else "Harmonic"} | {"Confidence" if r["reference"]=="confidence" else "Even rank"} | {float(r["delta_risk"]):+.6f} | [{float(r["low98_75"]):+.6f}, {float(r["high98_75"]):+.6f}] | {r["wins"]}/16 |' for r in contrasts if r['primary']=='True')
    cert_table=[]
    for mode,label in [('known','Known law'),('empirical','Empirical law')]:
        for cohort,task in [('rings','rings'),('transfer','waveform')]:
            group=[r for r in cert_rows[mode] if r['cohort']==cohort]
            cert_table.append(f'| {label}, {"rings" if cohort=="rings" else "harmonic"} | {sum(r["positive"]=="True" for r in group)}/16 | {mean(group,"gain_lower"):.6f} | {min(float(r["gain_lower"]) for r in group):.6f} | {mean(group,"mean_bound_penalty"):.6f} |')
    cert_table='\n'.join(cert_table)
    setup='\n'.join(f'| {mode}, {task} | {mean(v,"total_seconds"):.1f} | {num(sum(float(r["interval_jet_propagations"]) for r in v))} |' for (mode,task),v in bounds.items())
    gains={c:100*(1-lookup[c,'pb','rank_balanced','expected_risk']/lookup[c,'pb','raw','expected_risk']) for c in ('rings','transfer')}
    alljets=sum(sum(float(r['interval_jet_propagations']) for r in v) for v in bounds.values())
    allqueries=sum(sum(float(r['coordinate_cdf_queries']) for r in v) for v in bounds.values())
    text=f'''# Rank guarantees and harmonic transfer

Completed research cycle, 21 November 2026. The previous local-balance study is preserved unchanged. This report concerns a new frozen study with 32 independent primary models and 2,097,152 independent test inputs across their separate cohorts.

## What this cycle established

The actual global top-64-of-256 selector now has a useful continuous-in-radius analysis and an independent finite-sample verification procedure. The known-law family produced positive lower bounds for **{known['positive_models']}/32** fresh models; the separate empirical-law family produced **{emp['positive_models']}/32**. Each family has a simultaneous 95% confidence allocation under the stated assumptions and valid numerical enclosures. The empirical procedure estimates the score distribution from independent unlabeled inputs; its guarantee does not require the analytic radial density.

Using 1.25 calls/input, learned balance reduced mean expected logistic loss relative to the one-call raw model by **{gains['rings']:.2f}% on rings** and **{gains['transfer']:.2f}% on the harmonic task**. Among policies sharing the 1.25-call budget, the empirical ranking changed across tasks: it lost to confidence selection on rings and beat confidence on the new task. It remained worse than the strong folded symmetric gate on all 16 harmonic models. The main advance is a useful guarantee for an orientation-sensitive global selector, not a new universally best prediction method.

The guarantee concerns average logistic loss for every full-orbit-invariant posterior p(r), relative to the same fixed raw model. It does not concern every realized batch, classification accuracy, arbitrary input shifts or arbitrary non-invariant labels. The computation uses padded float64 intervals with independent checks; it is not a directed-rounding or machine-checked proof.

## Completed work

- Frozen primary task, selector and certificate settings, independent streams and primary comparisons before generating fresh states.
- Evaluated 16 new rings models and 16 new harmonic models, with nine primary policies and exact per-batch cost records. Sixteen underfit GD models provide secondary context only.
- Constructed all 64 fixed local enclosures: two tasks, 16 models, two distribution assumptions. Mean-verification inputs are independent of fitting and CDF estimation; the two certificate families reuse the same input sample within each model.
- Reconstructed all 320 outcome rows and 81,920 deployment ledgers; independently replayed nine training runs and 54 complete deployment paths. Preserved pilots, losing methods, source hashes and raw outputs.

The useful decision now is whether extra learned asymmetry can outperform a strong, already-safe symmetric baseline. More expressive models or more GPU computation would not answer that scientific question by themselves.

<!-- pagebreak -->

## A different task, with a clear limit

The original task observes noisy two-dimensional rings and uses an 80-parameter residual network. The new task observes four harmonic coefficients: X=(cos(phi), sin(phi), r cos(2phi), r sin(2phi)). A phase shift rotates the first pair by theta and the second pair by 2theta. Its half-turn flips only the first pair; full input negation is incorrect.

For balanced classes, r=0.4+u with u drawn from Beta(2,5) or Beta(5,2). Phase phi is uniform independently of r, so the input law is invariant under the action. Its posterior is u^3/[u^3+(1-u)^3], and Bayes classification error is 7/64=10.9375%. The new model has 144 parameters. A supervised four-seed development pilot selected PB learning rate 8; all recipes were retained. Fresh training used 256 class-balanced observations and 500 updates. The selector family and tau=1 transferred unchanged; selector calibration is label-free.

This changes input dimension, orbit geometry, architecture and label noise. The group remains SO(2), and the quotient on this support is still one scalar, monotone with total input norm. It is controlled synthetic transfer, not evidence for arbitrary groups, multidimensional quotients or real-world data.

![Descriptive improvement over each cohort's own raw model. All plotted policies use exactly 64 extra views per 256 inputs. Percentages are ratios of model-mean outcomes, with 16 model replications per cohort.](quality_comparison.png)

Loss reduction and fewer classification errors are different outcomes. On the harmonic task, learned balance had lower loss than confidence selection but higher expected classification error: 14.7665% versus 13.8034%. Its 14.7665% error also exceeded folded selection's 14.2894%. The objective and theorem concern logistic loss.

<!-- pagebreak -->

## Fresh predictive comparisons

All entries below are means across 16 independently trained primary PB models. Smaller loss is better. A call is a model-example evaluation, not wall time. Full two-view references use twice the raw budget and are not same-budget competitors or universal performance ceilings.

| Policy | Calls/input | Rings loss | Harmonic loss |
|---|---:|---:|---:|
{table}

The four prespecified primary contrasts use a paired model-level bootstrap with 20,000 draws. The following 98.75% intervals adjust the four-comparison family. A negative difference favors learned balance. These approximate bootstrap intervals are distinct from the finite-sample certificate on the next page.

| Task | Reference | Balanced minus reference | 98.75% interval | Wins |
|---|---|---:|---|---:|
{ct}

The folded comparison is descriptive but decisive on the harmonic cohort: learned balance lost all 16 pairs, by 0.007275 nats/input on average; its ordinary 95% paired interval was [0.005607,0.009154]. On rings, the balanced-minus-folded interval crossed zero, so no difference was resolved; this does not establish equivalence.

Prespecified posterior-shift diagnostics illustrate why the guarantee can matter. In the constant-one-half world, confidence allocation increased mean loss relative to raw by 0.005245 on rings and 0.014695 on the harmonic task. In the decreasing-posterior world, those increases were 0.013980 and 0.034953. Learned balance and the folded symmetric gate improved every model in all four tested worlds. Those finite scenarios are sensitivity checks, not the uniform-posterior proof.

<!-- pagebreak -->

## A useful guarantee for the actual rank rule

The deployment rule selects the largest 64 scores in every batch of 256. For an input with score s, its true inclusion probability is Q(F(s)), with Q the Beta(192,64) CDF and F the population score CDF. The analysis handles this actual inclusion probability, the randomized action mixture, score wrap jumps and all radii between calibration anchors. It does not substitute a soft gate or assume that fitted anchor balance is exact population balance.

Let b(r)=E[q(X)(h(X)-h(TX))/2 given r]. Local interval derivatives and jump-split integration enclose b at every radius. An antithetic centered-loss observation W gives gain at least E[W]-E[absolute b]/2 for every measurable p(r) in [0,1]. Sign-aware CDF bounds provide a conservative W. Each model then uses 262,144 independent coordinate observations and 524,288 verification model evaluations, with an empirical-Bernstein correction.

| Distribution assumption and task | Positive | Mean gain lower bound | Minimum lower bound | Mean bound penalty |
|---|---:|---:|---:|---:|
{cert_table}

Gain lower bounds and penalties are in nats/input. The last column is the mean B(r)/2 deduction in the independent verification sample. The known-law family allocates failure probability 0.05 across 32 mean bounds. The empirical family separately allocates 0.05 across 32 DKW CDF events and 32 mean bounds, with 1,048,576 additional independent unlabeled radii per model. These are two separate 95% families, not one joint 95% statement across both.

![Positive values certify improvement over the raw model under the stated full-orbit symmetry and numerical-enclosure assumptions. Model seeds are separate replications; orbit nodes are not.](certificate_results.png)

Even rank and folded symmetric selection already have an analytic Jensen nonharm guarantee because their masks are symmetric. The new certificate extends useful guarantees to the learned orientation-sensitive rule. It does not prove that this rule beats those symmetric controls.

<!-- pagebreak -->

## Costs, independent checks and reproducibility

Deployment is unchanged: every limited rule uses exactly [256,64] model-example calls per batch. Rank and folded calibration each use 66,048 model forwards per model: rank uses 129 structured orbits with 512 angles; folded uses 33,024 IID inputs and their partners. This matches forward budget, not independent sample count or CPU cost. Rank additionally uses 131,072 coordinate-only radii; the angle baseline uses 8,192 forwards. Total calibration was 4,489,216 forwards plus 4,194,304 coordinate-only draws. No new GPU speed experiment was performed, and fewer calls are not reported as measured speedup.

| Enclosure branch | Mean construction seconds/model | Total interval-row propagations (16 models) |
|---|---:|---:|
{setup}

These elapsed times were observed under parallel research workers and exclude compressed-output writes. Across all branches, the enclosure workload was {num(alljets)} interval network-row propagations and {num(allqueries)} score-CDF query inputs. The query count excludes auxiliary CDF endpoint lookups, beta-table preparation and distribution construction; it is not a FLOP count. These are offline verification costs, separate from deployment.

The two root confidence passes used 33,554,432 model-example verification calls in total. The empirical CDF stage used 33,554,432 coordinate-only radii. Ordinary performance scoring used 29,360,128 model-example calls across all policies and GD context. Independent checks are recorded separately rather than hidden in the deployment budget.

Root reconstructed all 320 saved risk rows with maximum discrepancy below 9e-16 and checked every one of 81,920 quota/call ledgers. Nine seeded training replays reproduced their saved parameters exactly. Fifty-four full-input deployment replays preserved masks and outputs to floating-point tolerance. An independent investigator reconstructed all bootstrap contrasts by multiplicity-weight resampling, with maximum discrepancy below 4e-16.

Independent certificate checks passed all 222 dense-orbit moment probes, 4,800 point-jet values, 1,542 CDF probes and 1,548 rank-probability/centered-gain input spots on the fixed six-model subset. Their cost was 3,643,896 ordinary model forwards and 1,920 analytic-jet propagations. The maximum angular subsample difference was 0.000192; refinement agreement is a diagnostic, not a substitute for the enclosure proof. All raw checks, frozen source hashes, input hashes, streams and release timestamps are retained.

The original calibration law and verification law are separated. In the empirical branch, analytic radial masses appear only in descriptive bound-size summaries; the actual certificate uses the pointwise envelope and a new independent sample. Rings additionally require regular orbits, with no atom at r=0; both actual study distributions meet the stated symmetry conditions.

<!-- pagebreak -->

## Interpretation and next direction

This cycle strengthens the fourth research item: approximate conditional balance can support a quantitative, useful risk guarantee for an actual exact-budget batch selector, including when its input marginal is estimated independently. The guarantee is no longer merely a very large formal bound. Cross-task replication also shows that confidence allocation is not uniformly the strongest empirical policy.

The strongest limitation is equally clear. The learned rule has not justified replacing the simple folded symmetric gate. On the harmonic task that gate won all 16 loss comparisons, already has structural nonharm, and uses the same deployment and forward-calibration budget. A small advantage over the restricted even-rank family does not resolve this stronger comparison. Better log loss also did not translate into better classification than confidence selection.

The next work should have three concrete objectives:

- **Make the strong symmetric gate the main comparator.** Compare guaranteed gain and ordinary predictive loss against it. Let learned asymmetry earn its extra complexity through a prespecified advantage; retain the simpler gate where it does not.
- **Expand the invariant from one scalar to two.** Use a controlled task whose posterior depends on two independent invariant amplitudes and cannot be reduced to total norm. Freeze the extension and its numerical cost before testing. This addresses a real limit of both completed tasks.
- **Measure the price of a guarantee.** Test a declared sequence of smaller independent verification samples and coarser numerical budgets on development models, then freeze one practical setting for new confirmation. Report the fraction of models certified, the size of the guarantee and all setup costs. Only after this tradeoff is credible should a real-data demonstration be prioritized.

The current evidence supports a methodological result with precise assumptions. It does not yet support a claim of universal allocation superiority, broad real-world robustness, exact point identification, formal floating-point certification or a hardware speedup.

## Sources and archive map

The finite-sample correction uses Maurer and Pontil, [Empirical Bernstein Bounds and Sample Variance Penalization](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf), Theorem 4. The empirical CDF band uses Massart, [The Tight Constant in the Dvoretzky-Kiefer-Wolfowitz Inequality](https://doi.org/10.1214/aop/1176990746). All model outcomes and local numerical results were generated in this study.

`STUDY_SCOPE.md` and the four freeze records define the study. `validation/` retains states, policies, predictions, comparisons and call ledgers. `certificate/` retains all local enclosures and their distribution brackets. `root_audit/` retains confidence observations, independent reconstructions and release records. `transfer/` contains the new task, all pilot recipes and the independent numerical audit. `REPRODUCE.md` gives the execution order. The SHA-256 manifest verifies the completed archive without overwriting previous studies.
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf8')
    print('Wrote final manuscript from completed result records.')

if __name__=='__main__':run()
