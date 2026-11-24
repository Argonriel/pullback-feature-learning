"""Assemble the literature-informed report from completed, audited outcomes."""
from pathlib import Path
import csv,json,re
ROOT=Path(__file__).resolve().parent
def read(p):return list(csv.DictReader((ROOT/p).open()))
summary=read('validation/summary.csv');contrasts=read('validation/contrasts.csv');cert=read('certification/final_summary.csv')
cohorts=['rings','waveform','two_invariants'];labels=['Rings','One-invariant harmonic','Two-invariant harmonic']
def s(c,n,k='expected_loss'):return float(next(r for r in summary if r['cohort']==c and r['policy']==n)[k])
def cr(c,n=512):return next(r for r in cert if r['cohort']==c and r['claim']=='symmetric_jensen' and int(r['n_batches'])==n)
table='\n'.join(f'|{label}|{s(c,"raw"):.6f}|{s(c,"folded"):.6f}|{s(c,"symmetric_jensen"):.6f}|{100*(1-s(c,"symmetric_jensen")/s(c,"raw")):.2f}%|' for c,label in zip(cohorts,labels))
ctable='\n'.join(f'|{label}|{float(r["mean"]):.8f}|[{float(r["lower"]):.8f}, {float(r["upper"]):.8f}]|{r["wins"]}/16|' for c,label in zip(cohorts,labels) for r in contrasts if r['cohort']==c and r['primary']=='True')
costtable='\n'.join(f'|{label}|{float(cr(c,128)["mean_lower"]):.6f}|{float(cr(c)["mean_lower"]):.6f}|{float(cr(c,2048)["mean_lower"]):.6f}|{100*float(cr(c)["mean_2048_lower_bound_retention"]):.1f}%|' for c,label in zip(cohorts,labels))
text=f'''# Symmetric allocation reaches its useful frontier

Literature review, strong controls, genuine two-invariant transfer, and cheaper verification. Completed 21 November 2026. All three authorized objectives are complete; previous studies remain unchanged.

## What the new evidence supports

Across48 newly trained models, an input-only interpolated Jensen-gain table improves expected logistic loss over a development-selected strong symmetric histogram. The rings effect is tiny. Both harmonic tasks show consistent16/16 modelwise wins. This is a restricted fixed-half-turn comparison, not universal dominance over augmentation methods.

|Task (16 new models each)|Raw loss|Strong histogram|Interpolated grid|Reduction versus raw|
|---|---|---|---|---|
{table}

![Mean known-posterior loss on65,536 fresh inputs per model. Limited policies use1.25 forward examples/input; full half-turn averaging uses2.00.](validation/fresh_loss.png)

The new rule captures99.995%,99.991%, and99.953% of the modelwise exact-quota Jensen oracle gain on the three tasks. This oracle evaluates both views everywhere and costs2.00 forwards/input. It is a headroom diagnostic, not a free deployable competitor.

Quantitative absolute gain bounds are positive for all48 models at only128 independent batches. A standalone policy needs16,384 network forward examples at this look. The prespecified512-batch primary analysis costs65,536 and retains86.0%,96.2%, and96.8% of the corresponding2048-batch lower bounds, averaging the per-model ratios. No baseline-relative certificate is positive, even at2048 batches.

<!-- pagebreak -->

## 1. Challenge a genuinely strong simple baseline

The previous32 models and four new-task pilot models are development data only. Each fresh grid uses65,536 calibration forwards; each histogram uses32,768 IID input pairs, also65,536 forwards. Development chooses64x32 bins for each one-invariant task and16x16x16 bins for the two-invariant task. The old128-cell histogram is retained. Nearest-grid and interpolation share exactly the same stored grid.

The primary comparison is model-paired expected loss, interpolation minus strong histogram. Negative favors interpolation. Twenty thousand paired bootstrap draws give98.333% marginal intervals, with Bonferroni nominal95% coverage across the three primary comparisons. These finite-replication bootstrap intervals are approximate; they are distinct from the rigorous gain certificates.

|Task|Mean loss difference|Primary interval|Wins|
|---|---|---|---|
{ctable}

![Missed exact-quota oracle gain, averaged over the16 models in each task. The logarithmic axis exposes how little headroom remains after grid interpolation.](validation/oracle_headroom.png)

The relative loss reductions against the strong histogram are only0.0175%,0.529%, and0.824%. The nearest-grid control already captures99.974%,99.891%, and99.153% of oracle gain. Much of the improvement over a coarse histogram therefore comes from a better calibration design and resolution; the evidence does not justify a claim that complexity or orientation-sensitive selection is necessary.

An important counter-result: confidence selection has lower rings loss(0.098917) than interpolation(0.100881). It lacks the same uniform half-turn safety property. On the evaluator-only half-posterior world, its rings loss rises from raw1.368141 to1.373493; interpolation lowers it to1.358839. In all three alternate half-turn-invariant worlds, the symmetric rules retain their structural justification. These worlds never influence fitting or selection.

<!-- pagebreak -->

## 2. Transfer to two independent invariant coordinates

The new support is X=(r1 cos(phi),r1 sin(phi),r2 cos(2phi),r2 sin(2phi)), with independent r1,r2 uniform on[0.4,1.4] and uniform phase. The group rotates the first coordinate pair bytheta and the second by2theta. The half-turn flips only the first pair.

The two pair norms separate orbits on this phase-coupled support. Total norm discards information. The posterior is0.5+0.4 tanh(4z1)tanh(4z2), zj=2(rj-0.9): it depends on both amplitudes. The quotient is genuinely two-dimensional here; on unrestricted ambient R4, relative phase would add another invariant.

The task law was frozen before a four-seed, three-recipe training pilot. PB4/500 beats PB8/500 and PB8/1000 by development expected loss and is then fixed for all16 fresh models. The model has144 parameters; no capacity increase or GPU is required for this transfer.

|Fresh mean metric|Raw|Interpolated grid|Full half-turn|Bayes benchmark|
|---|---|---|---|---|
|Expected logistic loss|0.700872|0.647930|0.629271|0.492907|
|Expected error|37.124%|35.002%|34.044%|22.656%|
|Mean predicted confidence|73.610%|70.923%|70.415%|Not comparable|
|Accuracy|62.876%|64.998%|65.956%|77.344%|

This confirms a useful extension, but the underlying models remain weakly calibrated. Raw mean loss exceeds the constant50%-probability predictor's log(2)=0.693147, despite accuracy above chance. The raw model's confidence exceeds its actual expected accuracy by10.734 percentage points; interpolation reduces that gap to5.925 points. Binned posterior calibration error falls from0.1081 to0.0640. These are descriptive diagnostics on held-out outcomes, not a fitted calibration policy.

The remaining Bayes gap is much larger than the available selection headroom. Training regularization and a separately validated probability-calibration procedure are higher priorities than a more elaborate selector. Positive temperature scaling alone would leave classification signs and error unchanged, so improved probability calibration and improved classification must be assessed separately.

The complete task checks, training pilot,64 fresh diagnostic rows,640 reliability-bin rows and1024 quotient-cell rows are retained. No seeds, failed runs or unfavorable recipes were dropped; all48 primary training runs completed.

<!-- pagebreak -->

## 3. Lower verification cost with a valid dependence argument

For a fixed half-turn-even pointwise score, exact top64-of256 selection is unchanged when any one input is flipped. Under a half-turn-invariant input law and posterior, expected logistic-loss gain equals expected selected Jensen gap. Its nonnegativity is structural; statistical verification quantifies a positive amount of improvement.

Verification evaluates both model views only on the selected-set union. For one policy,128 forward examples suffice per256-input batch. Two policies share overlapping work. This changes the admissible policy family; it is not an acceleration of the previous asymmetric policy's conditional-imbalance certificate.

Two concentration methods were frozen before fresh execution. Empirical Bernstein uses independent whole batches. Bounded differences uses the fact that replacing one input/tie pair changes a selected sum of weights in[0,c] by at mostc. It handles quota dependence explicitly. For two policies the bound is2c; the signed comparison uses the unclipped global Jensen bound Jmax for c, not the absolute-gain clipping levels. Both sharp sensitivity constants were independently proved and tested.

All48 models x3 claims x3 fixed looks x2 bound methods share a0.05 error budget(864 events). Taking the maximum of the two lower bounds is valid under this joint family. The original432-event outputs are retained as historical diagnostics; only the final864-event tables support this report. Clipping levels are fixed at0.25 for rings and0.5 for the harmonic tasks.

|Task|Mean bound at128|Primary512|Full2048|Mean modelwise retention|
|---|---|---|---|---|
{costtable}

All48 candidate bounds and all48 strong-histogram bounds are positive already at128 batches. A standalone policy costs16,384 /65,536 /262,144 forward examples at the three looks. At512, actual shared two-policy costs average70,438 /74,953 /81,201 on rings /one-invariant /two-invariant tasks. All three looks were collected; no early stopping saved work in this completed experiment.

For context, the earlier asymmetric certificate used524,288 ordinary forwards per model/mode plus expensive auxiliary construction. The new primary standalone count is8 times smaller; the128-batch count is32 times smaller. This is a different, restricted symmetric pipeline and a different quantitative bound. It is not a measured wall-clock or GPU speedup.

The signed comparison against the strong histogram remains uncertified:0/48 positive even at2048. Its global range bound is too loose for these small differences. Empirical superiority and an absolute safe-improvement guarantee must remain separate claims. Rings' signed bound also uses a known-distribution tail allowance; absolute clipped bounds do not require that tail calculation.

<!-- pagebreak -->

## What the literature changes about the research claim

Primary sources were searched before the new method and confirmation protocol were frozen. This focused review supports a narrower claim than a general new principle of safe augmentation.

- Chen, Dobriban and Lee, JMLR2020: orbit averaging, variance reduction and the connection to conditional averaging are established. Our exact-quota mask argument is a specialized application of symmetry and convexity. [Published paper](https://jmlr.org/papers/v21/20-163.html).
- Kim et al., NeurIPS2020, and Shanmugam et al., ICCV2021: per-instance augmentation choice and learned aggregation already exist. The present study's transparent, input-only gain surrogate and explicit query accounting should be compared with strong controls. [Loss-based augmentation](https://papers.neurips.cc/paper_files/paper/2020/file/2ba596643cbbbc20318224181fa46b28-Paper.pdf), [aggregation](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf).
- Jhawar and Wang, September2026 preprint: indistinguishable unlabeled worlds can require different actions; action identification and full-risk identification differ. Their Theorems1-2 overlap the generic obstruction. We make no first general impossibility or label-free selection claim. [Version1 full text](https://arxiv.org/html/2609.11235v1).
- Shanmugam et al., CVPR2025: augmentation with conformal coverage guarantees is prior art; its labeled policy/calibration setup and coverage target differ from our logistic-risk identity. [Author record](https://arxiv.org/abs/2505.22764).
- Dym and Gortler, FoCM2025(online2024): orbit-separating invariants matter. Our two-amplitude argument is elementary and support-specific, not a consequence proving general R4 sufficiency. [Published paper](https://doi.org/10.1007/s10208-024-09641-2).
- Maurer and Pontil2009 and McDiarmid1989 supply established concentration tools. Our study-specific step is the top-quota sensitivity calculation and honest accounting across all models, claims, looks and methods. [Empirical Bernstein](https://arxiv.org/abs/0907.3740), [bounded differences](https://doi.org/10.1017/CBO9781107359949.008).

Test-time augmentation keeps model parameters fixed while combining transformed views. Test-time adaptation changes deployment parameters or state. The2026 identifiability and monitoring papers concern adaptation; their information and dependence warnings inform this design, but their particular guarantees are not imported automatically. Schirmer et al.(NeurIPS2025) require a proxy-error relation for label-free monitoring; Han et al.(October2026 preprint) show failures of monitoring assumptions with dependent forecasting feedback. Our fresh inputs are IID and policies are frozen before verification.

The defensible contribution is a reproducible allocation/verification frontier under an exact symmetry and exact quota, with competitive controls, a genuine two-coordinate transfer and explicit negative results. A focused search is not an exhaustive novelty audit. Detailed source notes and theorem locations are in the literature, method, task and certification folders.

<!-- pagebreak -->

## Recommended next direction, in order

1. Improve the underlying two-invariant predictor and its probability calibration. Use a new, explicitly separate training/validation/calibration split; retain PB and ordinary-gradient controls, fixed capacity and all seeds. Select a bounded regularization/early-stopping grid on validation only, then freeze before fresh testing. Success should mean both lower raw loss and preserved selection benefit, with calibration/error reported separately. This addresses a much larger gap than the remaining less-than0.05% oracle-gain headroom.

2. Move the theoretical question from exact symmetry to controlled symmetry error. Under an invariant input law, a known bound on the difference between p(x) and p(Tx) gives a label-imbalance penalty to the Jensen gain identity. Specify how such an error budget is justified using independent labeled evidence or domain knowledge; it cannot be inferred from unlabeled inputs alone. Freeze a stress grid, report where the lower bound reaches zero, and compare against confidence and the strong symmetric histogram. This is a proposed follow-up, not a result established on real data here.

3. Use one task with a defensible real transformation and measure total cost. Reuse the strong histogram and nearest-grid controls. Include calibration, selection overhead, model inference and verification separately. Start with the current CPU workflow; use the existing5070Ti only if the chosen predictor is actually costly enough for model evaluations to dominate. Prior small-model GPU timings do not justify a new speed claim. Stop growing the selector if its extra improvement is below a predeclared practical threshold.

## Verification and reproducibility

All48 models,144 policy tables,48 performance sets and3,145,728 fresh performance inputs were retained. Independent reconstruction checks all384 metric rows(error at most4.45e-16), all masks and output relations(no discrepancies),240 committed files and all48 input/label/tie streams. Six100-step training replays are exact. Actual deployment ledgers contain98,304 records and35,389,440 forward examples. Calibration costs6,291,456 shared forward examples across48 models; legacy/nearest reuse is explicitly recorded.

The final certificate computation processes98,304 independent batches,25,165,824 coordinate inputs and14,497,086 actual shared forwards. Independent reconstruction verifies all864 individual bounds exactly, all48 deterministic range/tail calculations, and six128-batch input/model replays. These latter replays use113,282 additional audit forwards, reported separately. The original run took45.13 seconds including its audits on this machine; concurrent execution makes this descriptive, not a controlled runtime comparison.

Freezes precede training, calibration precedes performance and verification, and no fresh outcomes select methods or sample sizes. A proposed rings seed collision with an old pilot was caught and corrected to1161..1166 before any fresh state existed. A certificate approval-key alias was added before verification; it changed no frozen source or scientific choice.

Limitations: synthetic supports;16 model replications per task; fixed half-turn and25% quota; imperfect raw-model calibration; no real-data or GPU validation this cycle; no positive baseline-relative certificate; float64 numerical verification rather than a directed-rounding machine proof. The rings posterior helper overflows at extreme artificial radii15/40; all sampled/calibration checks are finite, and the label-free certificate never calls that helper. Stabilize it before reusing this API for extreme-radius studies.

The archive includes the report, primary-source review, all freezes and protocols, executable methods and audits, training states, calibration tables, predictions, raw ledgers, certificate batches, diagnostic tables and publication-quality figures. Prior studies are unchanged. Fresh reproduction is portable within this archive; rerunning historical development additionally requires the earlier study named in its scripts.
'''
def polish(part):
    part=re.sub(r'\b([A-Za-z]+)(?=\d)',lambda m:m[0] if m[0] in ('r','z','R','PB') else m[0]+' ',part)
    part=re.sub(r',(?=[A-Za-z])',', ',part)
    part=re.sub(r'(?<!\d),(?=\d)',', ',part)
    part=re.sub(r';(?=[A-Za-z0-9])','; ',part)
    part=re.sub(r':(?=[A-Za-z0-9])',': ',part)
    for old,new in [('mostc','most c'),('in[','in ['),('policy,128','policy, 128'),('al.(','al. ('),('error(' ,'error ('),('loss(','loss ('),('budget(','budget ('),('25% quota','25% quota')]:part=part.replace(old,new)
    return part
# Preserve exact source URLs while polishing ordinary report typography.
parts=re.split(r'(https?://[^\s)]+)',text)
text=''.join(p if p.startswith(('https://','http://')) else polish(p) for p in parts)
(ROOT/'REPORT.md').write_text(text,encoding='utf8')
print('Wrote report from audited outputs')
