"""Assemble the final report from the completed independently audited tables."""
from pathlib import Path
import csv,json,statistics
ROOT=Path(__file__).resolve().parent
read=lambda p:list(csv.DictReader((ROOT/p).open(encoding='utf8')))
cal={(r['family'],r['arm']):r for r in read('calibration/summary.csv')}
aug={(r['dataset'],r['arm'],r['pipeline']):r for r in read('augmentation/summary.csv')}
num=lambda r,k:float(r[k])
cal_table=[]
for family,label in [('legacy','Legacy training'),('selected','Selected training'),('extra_train','More training labels')]:
    r=cal[family,'raw_rawT'];f=cal[family,'full_finalT'];i=cal[family,'interp_finalT']
    cal_table.append(f"|{label}|{num(r,'expected_nll'):.6f}|{100*num(r,'expected_error'):.3f}%|{num(f,'expected_nll'):.6f}|{num(i,'expected_nll'):.6f}|")
aug_table=[]
for arm,label in [('noaug','No training flips'),('aug','Training flips')]:
    for pipeline,name in [('raw_Traw','Raw, calibrated'),('full_Tfull','Full, own calibrated')]:
        r=aug['holdout',arm,pipeline]
        aug_table.append(f"|{label}|{name}|{num(r,'ce'):.6f}|{100*num(r,'error'):.3f}%|")
serving=read('augmentation/serving_summary.csv');timing_table=[]
for mode,name in [('cpu','One-thread CPU'),('cuda_resident','GPU resident'),('cuda_transfer','GPU with transfers')]:
    rr=[r for r in serving if r['mode']==mode and r['arm']=='aug']
    med=lambda k:statistics.median(num(r,k) for r in rr)
    timing_table.append(f"|{name}|{med('raw_ms'):.3f}|{med('strongest_full_ms'):.3f}|{med('full_over_raw_latency'):.3f}x|")
shift=read('shift/certificate_summary.csv');shift_table=[]
for a in (0.,.01,.1,.6):
    rr={r['policy']:r for r in shift if int(r['n_batches'])==2048 and float(r['a'])==a}
    r=rr['symmetric_jensen'];s=rr['folded']
    shift_table.append(f"|{a:g}|{num(r,'eta'):.6f}|{num(r,'lower_epsilon0_mean'):.6f}|{num(s,'lower_epsilon0_mean'):.6f}|{int(r['positive_epsilon0'])+int(s['positive_epsilon0'])}/16|")
training=read('calibration/training.csv')
medtrain={f:statistics.median(float(r['training_seconds']) for r in training if r['family']==f) for f in ('legacy','selected','extra_train')}
ranges=json.loads((ROOT/'root_audit/CNN_RANGE_BOUNDS.json').read_text())['models']
Dlo=min(r['raw_span_D'] for r in ranges);Dhi=max(r['raw_span_D'] for r in ranges)
text=f'''# Spend labels on training; calibrate the final output

Completed 21 November 2026. All three requested follow-ups are complete, with independent reconstruction of results and a full replay of the new image holdout. Sources and methods were frozen before their new outcomes. No failed result was repaired through post-test tuning.

## What changed the research direction

**The strongest gain came from allocating more of the same label budget to training. Final-output calibration helped the synthetic pipelines. Learning reflection repaired the image failure, but bought only a small two-view accuracy gain. The broader theory is valid and remains too conservative for the tested nonzero input shifts.**

|Question|Completed result|Practical interpretation|
|---|---|---|
|Where should the label budget go?|At 4,352 labels, raw synthetic error falls from 36.773% to 26.989% when more labels go into training.|A larger training set is more promising than the previously selected validation-heavy allocation in this task.|
|Should calibration target the final output?|Selected-model full loss falls from 0.627230 to 0.614132; interpolation from 0.632169 to 0.626012.|Calibrate the actual deployed output; scalar calibration does not change classification decisions.|
|Can training make reflection useful?|Augmented image models improve from 9.437% raw error to 9.157% with full averaging.|A repaired transformation enables a small gain, with extra inference cost.|
|Can the theory cover natural asymmetry?|All 16 certificates fail at the smallest tested nonzero input shift, although measured gains remain positive.|The deterministic range penalty is loose; failure to certify is not proof of harm.|

The synthetic experiment uses 16 fresh paired replications, 48 base models, 144 scalar fits and 384 outcome rows. The image experiment uses six scratch-trained CNNs: three paired initializations with and without training flips. The shift experiment keeps eight historical models and both of their selectors fixed, then evaluates four newly sampled input laws.

The image holdout is **new to this round's fitting process, not pristine external data**. Its 10,000 images previously belonged to the older training set. The new models never train, validate or calibrate on them. The already observed official test is secondary descriptive evidence. All six models share one split, so three initializations do not constitute three independent datasets.

The next priority is a controlled label-versus-training-compute study, followed by one-pass preservation of the modest augmentation gain and an external evaluation. For theory, tighten transformation-specific bounds under explicit assumptions; a larger unlabeled sample alone will not repair the missing assumptions or the dominant range penalty.

<!-- pagebreak -->

## 1. Equal labels favor a larger training set

Every calibrated procedure uses exactly 4,352 unique fitting labels. Legacy uses 256 training + 4,096 calibration; selected uses 256 training + 2,048 checkpoint validation + 2,048 calibration; more-training uses 2,304 training + 2,048 calibration. Initial weights and the base 256 examples are paired. Architecture and label law remain unchanged.

|Procedure|Raw calibrated loss|Raw error|Full final-fit loss|Interp. final-fit loss|
|---|---|---|---|---|
{chr(10).join(cal_table)}

Raw inference costs one model evaluation per input, interpolation 1.25 and full averaging two. Each interpolation field is committed before temperature fitting, and its selection rule stays fixed across all scalar-fit arms. Temperature fits use independent sampled labels; the known synthetic posterior is evaluation information only.

![Equal fitting-label budgets, final-pipeline calibration, and classification error.](calibration/final_pipeline_results.png)

The two primary selected-model final-fit improvements are -0.013098 for full averaging, 98.75% interval [-0.016366,-0.010031], and -0.006157 for interpolation, [-0.008040,-0.004516]. Each improves on 16/16 seeds. Positive scaling leaves each pipeline's classes and mask unchanged: these are probability-quality gains, not temperature-caused accuracy gains.

More-training raw loss beats legacy raw loss by 0.094195, interval [-0.102746,-0.084668], on 16/16 seeds. Its error falls by 9.785 percentage points, a secondary outcome. Conversely, the previously selected allocation has worse raw loss than legacy by 0.005094, interval [0.001644,0.007989]. The four prespecified loss intervals use paired seed resampling and a nominal 95% family correction; they are approximate bootstrap intervals.

Equal labels do not mean equal computation. Median training times were {medtrain['legacy']:.2f}s legacy, {medtrain['selected']:.2f}s selected and {medtrain['extra_train']:.2f}s more-training, measured descriptively. The enlarged set has nine times as many training examples. Rate and damping were scaled consistently with the summed pullback metric; independent duplicate-data checks verified equivalent normalized updates. The 500-update extra-training result is not an equal-wall-clock comparison or a universal optimizer claim.

<!-- pagebreak -->

## 2. Training reflection repairs reliability, with a modest payoff

Each CNN arm uses the same 50,000 distinct labels: 40,000 training, 5,000 checkpoint validation and 5,000 calibration. Paired models start from identical weights and see identical epoch orders. One arm randomly reflects half its training inputs, replacing views rather than doubling calls. The 426,122-parameter architecture, Adam recipe and 12-epoch budget are unchanged. All six select epoch 12.

|Training|Prediction pipeline|Holdout loss|Holdout error|
|---|---|---|---|
{chr(10).join(aug_table)}

![Original-input quality and reflected-input reliability, on the same within-round holdout.](augmentation/augmentation_quality.png)

Training flips alone show no clear raw benefit: calibrated loss difference +0.000909, adjusted interval [-0.003506,0.005148]; error difference +0.027 percentage points, [-0.263,0.323]. Within augmented models, full averaging improves loss by 0.006484, interval [-0.008070,-0.004851], and error by 0.280 points, [-0.470,-0.093]. The four primary 98.75% image-bootstrap intervals condition on these six fitted models.

The earlier ankle-boot failure is substantially repaired. Mean reflected-image ankle-boot error changes from 100% without training flips to 3.033% with them; augmented full averaging has 2.867% error on this class. Independent replay queried every original and reflected holdout image through an independently defined CNN and reproduced all stored logits exactly.

Own-output calibration is not universally an additional win. In augmented models it changes full loss by only +0.000013 relative to raw-fitted temperature, with a secondary interval spanning zero. Without augmentation, it lowers full loss from 0.554614 to 0.516069 but leaves the harmful 18.280% error unchanged. The previously seen official test also shows a small within-augmented averaging benefit, while augmented raw loss worsens versus no-augmentation raw; those results remain secondary.

<!-- pagebreak -->

## 3. The image accuracy gain has a measurable serving cost

The following table focuses on the three **augmented** models, which provide the small positive full-averaging result. Entries are medians of model-specific median times for a batch of 256 images. Ratios are aggregated per model, so they need not equal ratios of the displayed median times.

|Serving mode|Raw ms / batch|Fastest full ms / batch|Full / raw latency|
|---|---|---|---|
{chr(10).join(timing_table)}

![Both training arms are retained; timings compare calibrated raw and the fastest measured full implementation.](augmentation/augmentation_serving.png)

Full averaging uses two network input examples per image. Both a fused batch and two sequential calls are measured, and the faster per-model full implementation is used. For augmented models, the median latency multiplier is 2.015 on one-thread CPU, 1.512 on resident GPU and 1.293 with transfers. Thus the 0.28-point error gain has a price; this experiment does not choose a deployment tradeoff for every application.

The RTX 5070 Ti is used directly. Serving runs are synchronized FP32, without TF32 or mixed precision, with three warmup blocks and seven measured blocks per model/mode/method; each block contains four validation batches. Other numerical work was paused during the benchmark. Timers include reflection, concatenation, network calls, averaging and scalar correction. Transfer mode includes input transfer and output return.

These are steady-serving measurements from normalized tensors. Decoding, normalization and model loading are excluded and were not individually timed; no complete cold-start claim follows. CPU uses one thread, not an optimized many-thread baseline. The six-model pooled ratios, also retained in the branch report, are 2.014 / 1.780 / 1.327; they differ from the augmented-only table by design. Differences between training arms' timings are not claimed as causal effects of augmentation.

Training plus checkpoint validation took 45.989 seconds across six models and calibration took 0.615 seconds, descriptive elapsed measurements. Costs were 2,880,000 training forward/backward examples, 120,000 validation forwards, 60,000 shared calibration forwards and 240,000 shared quality forwards. Serving used 645,120 measured and 276,480 warmup forwards. CPU/GPU checks found no class changes and maximum logit difference 5.73e-6 on the checked validation batches; quality aggregation uses float64, serving arithmetic FP32.

<!-- pagebreak -->

## 4. A multiclass extension exposes a loose guarantee

Let d=(z-zT)/2 for multiclass logits, J the log-sum-exp Jensen gap, and span(d)=max(d)-min(d). For a mask invariant to transforming each individual input with other records and keys fixed, the coordinatewise symmetrized loss gain is M[J-(p-pT) dot d/2]. A fixed quota alone is insufficient. If posterior TV is at most epsilon, input TV(P,T#P) at most eta, and span(d)<=D globally, independent moment bounds on the **same input law P** give

**Gain on P >= L_A - epsilon U_B/2 - D eta.**

Here A=E_P[M min(J,c)] and B=E_P[M span(d)]. The range ignores arbitrary common logit offsets. The theorem covers cross-entropy, not classification error. Eight fixed models, four laws, two policies, two moments, two bound methods and two fixed looks form 512 elementary events, jointly covered at 95%. The primary look is 2,048 independent batches; no mask-independence assumption is used.

|Phase tilt a|Known input TV|Interp. lower bound|Histogram lower bound|Positive lines|
|---|---|---|---|---|
{chr(10).join(shift_table)}

![Certificate failure and actual gain are distinct; the global deterministic penalty dominates.](shift/shift_frontier.png)

The table fixes posterior defect at zero. At a=.01, the known input-TV defect is .006366 and the mean penalty D eta is .182434. Yet independent known-density gain estimates are +.051992 / +.047172, positive for all eight models. The negative lower bound fails to certify benefit; it does not establish harm. At zero shift, conditional margins from those fixed moments are only about .00110-.00185; these are not additional tested shifts. More verification samples cannot remove the deterministic penalty. Strong adversarial posterior defects do cause harm and remain in the results.

The CNN pixel-cube interval calculation gives analytical D values of roughly {Dlo:,.0f}-{Dhi:,.0f}, evaluated in FP64 with numerical padding, not formal directed rounding. It supplies neither natural-image input-TV nor posterior-TV bounds. A separate reviewed indistinguishability proof shows why an unrestricted distribution-free TV upper confidence bound can be forced to be vacuous even at an invariant continuous law. Explicit law/regularity assumptions can escape that obstruction; a discriminator witness alone supplies a lower bound, not the required upper bound.

<!-- pagebreak -->

## Next direction, in priority order

1. **Optimize the label and training-compute allocation.** The most promising empirical result is the larger training set followed by calibration. Compare a small frozen grid of training/calibration splits at the same total label budget and also at equal total training work. Keep model capacity fixed, use new synthetic streams, and report probability loss and classification error separately. The present study establishes a useful allocation under its recipe, not a universal best split.

2. **Test whether the two-view gain can survive in one pass.** Use the successfully augmented two-view predictor as a teacher for a single-view student, with a plain augmented student as the equal-compute control. Freeze a target gain and latency tradeoff before evaluation. Validate the chosen procedure on a genuinely untouched external dataset or acquisition, since all current Fashion-MNIST images have historical exposure. This is a proposed experiment, not a completed result.

3. **Tighten the transformation-specific bound and state the assumptions that make it usable.** Replace a loose global range estimate only with an independently proved tighter bound, never an observed test maximum. Work first in a specified law family where input asymmetry is known or estimable under stated assumptions; use labeled validation for posterior relations. The present obstruction rules out a universal unlabeled solution over unrestricted laws, while leaving structured settings open.

## Independent checks and retained evidence

The calibration audit reconstructs all 384 risk rows, all 144 fits and calibration-pipeline outputs, 16 split/seed streams and checkpoint choices, and all four primary bootstrap intervals. It checks 98,304 deployment records totaling 36,962,304 forwards. Maximum risk discrepancy is 1.12e-16; mask and pipeline-output discrepancies are zero. This audit adds 396,288 forward examples. Field fitting used 3,145,728 and shared scalar fitting 262,144 forwards, separate from training and checkpoint selection.

The image audit reconstructs the 60,000 split IDs, all six checkpoint choices, 12 scalar fits, 60 risk rows, 120 classwise rows, 20 bootstrap intervals and 378 timing rows. Its complete holdout replay adds 120,000 forwards and matches logits exactly. The shift audit reconstructs all 512 elementary bounds and 2,560 stress rows, replays six 128-batch certificate prefixes, and adds 119,934 forwards. Primary shift verification used 10,266,050 shared forwards; stress used 4,194,304 plus 30,720 deployment-audit forwards. Costs are not merged with hypothetical standalone budgets.

Limitations remain: one synthetic task, only 16 fresh synthetic seed clusters, three paired image initializations on one historically exposed dataset split, unequal training compute, conditional bootstrap uncertainty, a conservative global range bound, and numerical rather than formal machine proof. No real-image symmetry certificate or deployment superiority is claimed.

Primary precedents include [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) on temperature calibration, [Hekler et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26735) on combining test-time augmentation and calibration, [Shanmugam et al.](https://arxiv.org/abs/2011.11156) on aggregation failures, and [Chen et al., JMLR 2020](https://jmlr.org/papers/v21/20-163.html) on group averaging and approximate invariance. [Fashion-MNIST](https://arxiv.org/abs/1708.07747) supplies the image benchmark. No first general calibration or robustness principle is claimed. Protocols, sources, models, predictions, labels, ledgers, figures, license and the complete proof appendix are retained.
'''
(ROOT/'REPORT.md').write_text(text,encoding='utf8')
print('Report assembled from completed audited results.')
