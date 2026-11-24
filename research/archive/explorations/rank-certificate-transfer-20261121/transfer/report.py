"""Readable transfer-task/pilot artifact; no model or selector tuning."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import task

def main():
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    for name,digest in freeze['file_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=list(csv.DictReader((ROOT/'pilot'/'results.csv').open()));means=[]
    for method,rate in (('gd',.1),('gd',.3),('pb',8),('pb',16)):
        selected=[r for r in rows if r['method']==method and float(r['rate'])==rate];means.append(dict(method=method,rate=rate,models=len(selected),conditional_risk=float(np.mean([float(r['conditional_risk']) for r in selected])),full_partner_risk=float(np.mean([float(r['full_partner_risk']) for r in selected])),training_loss=float(np.mean([float(r['training_loss']) for r in selected])),seconds=float(np.mean([float(r['seconds']) for r in selected]))))
    (ROOT/'PILOT_SUMMARY.json').write_text(json.dumps(dict(rows=means,training_choice='PB8 primary; GD.3 descriptivecontext',pilot_models=4,pilot_recipes=16,fresh_data_used=False),indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,ax=plt.subplots(1,3,figsize=(12,3.8),layout='constrained');t=np.linspace(0,2*np.pi,600);colors=['#146f83','#b87431']
    for r,color in zip((.6,1.2),colors):ax[0].plot(t,np.cos(t)+r*np.cos(2*t),color=color,label=f'Harmonic ratio r={r}')
    ax[0].plot(t,-np.cos(t)+1.2*np.cos(2*t),color=colors[1],ls='--',alpha=.7,label='Half-period partner, r=1.2');ax[0].set_xlabel('Time / phase coordinate');ax[0].set_ylabel('Waveform value');ax[0].set_title('Different shapes, with a true time-shift symmetry');ax[0].legend(fontsize=8)
    u=np.linspace(0,1,500);r=.4+u;ax[1].plot(r,30*u*(1-u)**4,color=colors[0],label='Negative class');ax[1].plot(r,30*u**4*(1-u),color=colors[1],label='Positive class');ax[1].set_xlabel('Second-harmonic amplitude r');ax[1].set_ylabel('Class-conditional density');ax[1].set_title('Overlapping classes; equal prior');ax[1].legend(fontsize=8)
    ax[2].plot(r,u**3/(u**3+(1-u)**3),color='#715a8b');ax[2].axhline(.5,ls='--',color='#888888',lw=.8);ax[2].set_xlabel('Second-harmonic amplitude r');ax[2].set_ylabel('P(positive class | input)');ax[2].set_ylim(-.02,1.02);ax[2].set_title('Known posterior; Bayes error 7/64');fig.savefig(ROOT/'task_geometry.png',dpi=180);plt.close(fig)
    table='\n'.join(f"| {m['method'].upper()} | {m['rate']:g} | {m['conditional_risk']:.4f} | {m['full_partner_risk']:.4f} | {m['training_loss']:.4f} | {m['seconds']:.3f} |" for m in means)
    text=f'''# Transfer to a two-harmonic waveform task

This task tests the frozen rank selector on a different symmetry representation: four waveform coefficients `(cos phi, sin phi, r cos 2phi, r sin 2phi)`. The class depends on harmonic shape through r; phase is uniform and irrelevant. A phase shift acts as `R_theta direct-sum R_(2theta)`. The selected half-period partner flips only the first two coordinates. Using the Euclidean antipode would apply the wrong transformation.

![Waveform task and label law](task_geometry.png)

The amplitude r lies in [.4,1.4]. Its two equally likely class laws are shifted Beta(2,5) and Beta(5,2), giving a known nonlinear posterior and Bayes classification error .109375. Training sets have exactly 128 examples of each class. Calibration and evaluation inputs follow the iid population mixture. The task changes ambient dimension, nonlinear orbit geometry, partner transformation, input law, and posterior law. It still has a scalar quotient that is monotone in the full input norm; this is not multidimensional-quotient or real-world validation.

## Fixed model and bounded pilot

The model is an ordinary two-block 4D residual network with eight hidden units per block and 144 parameters, trained on raw coefficients. It is not given r as an extra feature. Four independent pilot seeds evaluated exactly two rates for GD and two for pullback training, always 500 updates. All 16 fits and evaluation data are retained. The rate choice uses supervised pilot conditional risk; no selector hyperparameter is chosen using these labels.

| Optimizer | Rate | Mean conditional risk | Full partner risk | Training loss | Seconds/model |
|---|---:|---:|---:|---:|---:|
{table}

The frozen primary recipe is PB8 with damping .1 for 500 updates. GD.3 is retained as descriptive context and is visibly underfit. Equal update counts are not equal compute. This limited pilot and subsequent selector comparisons are not an optimizer-superiority benchmark.

## Frozen transfer and costs

The task, model, training recipe, and selector adaptation were frozen at **2026-11-21 10:11:28.091373 UTC**, before the sixteen new confirmation models were generated. `FREEZE.json` records all hashes. The rank family keeps tau 1, frequencies 1..4, the quarter-grid phase offset, 129 radial anchors, 512 angles, the same zero-moment hull optimization, and the same radius-only randomization. Only the model/input geometry and radial-law helpers change.

Calibration uses 66,048 model example forwards plus 131,072 coordinate-only radii. Every limited-view deployment evaluates 256 first views and exactly 64 correct partners, for 1.25 calls/input. The full two-view reference costs 2 calls/input. Independent validation retains confidence, invariant-folded, exact-even, unconstrained and other predeclared controls; it owns the new outcomes under `../validation`.

All eight chosen pilot states passed adapter checks: exact call counts and quotas, bitwise even-score invariance, invariance of even masks under flipping any subset of inputs, and direct-coordinate agreement with FFT action moments. The analytical population-scoreCDF agrees with independent split quadrature to at most 4.2e-14. The new radial law has polynomial CDF and first-moment primitives, so the evaluator can integrate the piecewise-linear score-location distribution analytically. Calibration still uses the frozen empirical-CDF method; the exact population formula is not secretly substituted into deployment.

## Files and reproduction

- [THEORY.md](THEORY.md): task, group, posterior, model, score-law proof, and geometry-specific bounds.
- [PROTOCOL.md](PROTOCOL.md): fixed seeds, scopes, costs, API, and baseline requirements.
- [FREEZE.json](FREEZE.json): immutable source and pilot hashes.
- `pilot/`: all 16 training recipes, raw data, traces, and results; `pilot_policy/`: eight calibrated pilot policies and checks.
- [CERTIFICATE_REVIEW.md](CERTIFICATE_REVIEW.md): independent read-only review of the certificate argument and source.
- `INDEPENDENT_CHECK_PROTOCOL.json` and `independent_checks/`: uniform-subset independent numerical checks of the fresh bounds.
- `../validation/`: the independent sixteen-model confirmation and measured deployment results.

From the archived study root, `python transfer/report.py` verifies frozen hashes and regenerates this report and figure. `python transfer/pilot.py` and `python transfer/checks.py` reproduce the bounded pilot and checks into fresh output directories; preserve the existing outputs before rerunning. NumPy and threadpoolctl are required, with matplotlib for reporting. The task/model and rank adapter are local and self-contained.
'''
    validation=ROOT.parent/'validation'
    if (validation/'summary.csv').exists() and (validation/'contrasts.csv').exists():
        fresh=[r for r in csv.DictReader((validation/'summary.csv').open()) if r['cohort']=='transfer' and r['outcome']=='expected_risk'];contrasts=[r for r in csv.DictReader((validation/'contrasts.csv').open()) if r['cohort']=='transfer' and r['method']=='pb' and r['policy']=='rank_balanced' and r['outcome']=='expected_risk']
        main={r['policy']:r for r in fresh if r['method']=='pb'};comparisons={r['reference']:r for r in contrasts};labels={'raw':'One view','full':'Full half-turn reference','confidence':'First-view confidence','folded':'Invariant folded baseline','rank_balanced':'Rank balanced at anchors','rank_even':'Exact-even rank control','rank_unbalanced':'Unconstrained rank','confidence_radial':'Confidence plus calibrated angle','full_radial':'Full calibrated-angle reference'}
        result_table='\n'.join(f"| {labels[name]} | {1 if name=='raw' else 2 if name.startswith('full') else 1.25:g} | {float(main[name]['mean']):.6f} |" for name in labels if name in main)
        comparison_table='\n'.join(f"| {labels[name]} | {float(comparisons[name]['delta_risk']):+.6f} | [{float(comparisons[name]['low98_75']):+.6f}, {float(comparisons[name]['high98_75']):+.6f}] | {comparisons[name]['wins']}/16 |" for name in ('confidence','rank_even','folded'))
        addition=f'''\n## Independent transfer confirmation\n\nOn the sixteen new PB models, the unchanged balanced rank family improves over confidence and the exact-even rank control. The invariant folded baseline remains better on every model. This is a transfer result for a fixed method, not a claim that it is the best allocation policy.\n\n| Policy | Actual calls/input | Mean conditional logistic risk |\n|---|---:|---:|\n{result_table}\n\nThe following paired differences are balanced rank risk minus the reference risk; negative values favor balanced rank. Intervals resample model seeds, not individual batches. The confidence and exact-even contrasts were primary; the folded comparison is a retained secondary control.\n\n| Reference | Risk difference | Prespecified98.75% paired seed-bootstrap interval | Balanced wins |\n|---|---:|---:|---:|\n{comparison_table}\n\nGD remains an underfit descriptive context; its model and compute budget are not matched to PB. The full references use twice as many model calls as one view and are separate-budget references. Original-task risk comparisons do not replace the all-invariant-posterior certificate or establish exact conditional balance for the learned policy. Independent validation owns raw per-model/per-batch risks, sampled-label results, invariant-world controls, calibration hashes, and deployment call ledgers in `../validation`. No fresh result changed the task, training recipe, score family, or calibration settings.\n'''
        text+=addition
        (ROOT/'FRESH_SUMMARY.json').write_text(json.dumps(dict(risks=fresh,contrasts=contrasts,validation_summary_sha256=hashlib.sha256((validation/'summary.csv').read_bytes()).hexdigest(),validation_contrasts_sha256=hashlib.sha256((validation/'contrasts.csv').read_bytes()).hexdigest(),selection_retrained_or_retuned=False),indent=2))
    if (ROOT/'INDEPENDENT_CHECK_SUMMARY.json').exists():
        audit=json.loads((ROOT/'INDEPENDENT_CHECK_SUMMARY.json').read_text())
        text+='\n## Independent certificate audit\n\nThe fixed six-model subset (seeds 1,8,16 in both tasks) passed all 222 high-resolution conditional-moment probes, 4,800 point-jet tests, 1,542 CDF probes, and 1,548 rank-marginal/centered-variable checks. No failed enclosure was omitted. [INDEPENDENT_CHECKS.md](INDEPENDENT_CHECKS.md) records the numerical scope, remaining angular sensitivity, exact call accounting, and raw artifacts. These checks supplement the reviewed real-arithmetic argument and do not claim a machine-checked floating-point proof.\n'
    (ROOT/'README.md').write_text(text,encoding='utf8');print(json.dumps(means,indent=2))

if __name__=='__main__':main()
