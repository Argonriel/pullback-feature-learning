"""Seed-clustered paired optimizer contrast analysis; no selection or refitting."""
import csv,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import study

def main():
    study.summarize()
    rows=study.csvin(ROOT/'method_contrasts.csv');groups=defaultdict(list)
    for r in rows:
        key=tuple(r[k] for k in ('task','stage','experiment','horizon','requested_drop','metric'));groups[key].append(r)
    rng=np.random.default_rng(27754100);output=[]
    for key,rr in groups.items():
        true=np.array([float(r['actual_pb_minus_gd']) for r in rr]);preds={'drop_clock':np.array([float(r['clock_pb_minus_gd']) for r in rr]),'realized_tangent':np.array([float(r['tangent_pb_minus_gd']) for r in rr]),'old_pooled':np.array([float(r['pooled_pb_minus_gd']) for r in rr]),'zero':np.zeros(len(rr))}
        idx=rng.integers(0,len(rr),(10000,len(rr)))
        for name,p in preds.items():
            err=abs(p-true);advzero=abs(true)-err;advold=abs(preds['old_pooled']-true)-err
            ci=np.quantile(np.column_stack([true,advzero,advold])[idx].mean(axis=1),[.025,.975],axis=0)
            output.append(dict(task=key[0],stage=key[1],experiment=key[2],horizon=key[3],requested_drop=key[4],metric=key[5],predictor=name,n_seed=len(rr),mae=err.mean(),rmse=np.sqrt(np.mean((p-true)**2)),correct_sign=int((p*true>0).sum()),actual_pb_better=int((true<0).sum()),actual_pb_minus_gd=true.mean(),actual_ci_lo=ci[0,0],actual_ci_hi=ci[1,0],seed_wins_zero=int((advzero>0).sum()),seed_wins_old=int((advold>0).sum()),advantage_zero=advzero.mean(),advantage_zero_lo=ci[0,1],advantage_zero_hi=ci[1,1],advantage_old=advold.mean(),advantage_old_lo=ci[0,2],advantage_old_hi=ci[1,2]))
    study.csvout(ROOT/'contrast_summary.csv',output)
    allrows=study.csvin(ROOT/'new'/'outcomes.csv');cov=defaultdict(list)
    for r in allrows:
        if r['metric']=='fit':cov[tuple(r[k] for k in ('task','stage','method','experiment','horizon','requested_drop'))].append(r)
    coverage=[]
    for key,rr in cov.items():
        coverage.append(dict(task=key[0],stage=key[1],method=key[2],experiment=key[3],horizon=key[4],requested_drop=key[5],total=len(rr),ok=sum(r['status']=='ok' for r in rr),norm_cap=sum(r['status']=='norm_cap' for r in rr),loss_floor=sum(r['status']=='loss_floor' for r in rr)))
    study.csvout(ROOT/'coverage_by_stratum.csv',coverage)
    # Exact shared-model identity and same-input paired-task audit.
    states=np.load(ROOT/'new'/'states.npz');shared=[]
    d2=ROOT.parent/'direction2'/'fresh_results'
    for seed in range(1,17):
        f=d2/f'{seed}_training_states.npz'
        if f.exists():
            b=np.load(f)
            shared.append(dict(seed=seed,training_x_equal=bool(np.array_equal(states[f'rings_{seed}_x'],b['x'])),training_y_equal=bool(np.array_equal(states[f'rings_{seed}_y'],b['y'])),initial_equal=bool(np.array_equal(states[f'rings_{seed}_initial'],b['initial'])),pb100_equal=bool(np.array_equal(states[f'rings_{seed}_trained'],b['prefix']))))
        assert np.array_equal(states[f'radial_{seed}_x'],states[f'angular_{seed}_x'])
        assert np.array_equal(states[f'radial_{seed}_initial'],states[f'angular_{seed}_initial'])
    study.jsonout(ROOT/'shared_states_check.json',dict(rows=shared,paired_synthetic_inputs_identical=True,paired_synthetic_initialization_identical=True))
    # Independent one-dimensional posterior law: uniform phase, same cos law.
    a=2*np.pi*(np.arange(1048576)+.5)/1048576;p=study.sigmoid(3*np.cos(a))
    entropy=-(p*np.log(p)+(1-p)*np.log1p(-p))
    study.jsonout(ROOT/'task_equivalence.json',dict(class_probability=float(p.mean()),bayes_loss=float(entropy.mean()),bayes_accuracy=float(np.maximum(p,1-p).mean()),proof='2*pi*u and (2*phi modulo2*pi) are both uniform phases, so their cosine posterior logits have the same arcsine distribution.',radial_symmetry='all rotations',angular_symmetry='rotation by pi; reflections also hold but are not used',angular_view_identity='two_view_value equals pi-symmetry nuisance exactly; not an independent third endpoint'))
    # Compact nonselective result table: all tasks/stages at both one-step sizes.
    lookup={tuple(r[k] for k in ('task','stage','experiment','horizon','requested_drop','metric','predictor')):r for r in output}
    lines=['# Actual-loss-matched predictive study','',
    'The frozen design tests whether pre-state population directional derivatives forecast the change in symmetry fitting loss, nuisance loss, and the value of a second true-symmetry view after GD and full pullback achieve the same actual training-loss decrease. The main contrasts compare pullback minus GD from an exactly shared model state.','',
    'Six exploratory seed pairs precede sixteen independent new pairs. Original rings have initial and PB100 states. Matched synthetic radial/angular tasks share every input and initialization, have identical posterior-value distributions, class probability 1/2, and Bayes difficulty. Their true symmetry groups differ. Angular averaging uses the valid pi-rotation subgroup {I,Rpi}, preserving its informative cos(2phi) label. Its endpoints are pi-subgroup fit and pi-antisymmetric nuisance; fit can retain reflection nuisance because the full D2 group is not averaged.','',
    '## Prospective boundary and evidence','',
    'All 16-seed updated states were computed with training data only. `new/predictions.csv`, root records and state bank were physically saved and SHA256 committed before `new/score_start.json`. `freeze.json` binds protocol, code and old outcomes before fresh prediction generation. Population evaluation is deterministic refined quadrature of the known data-generating law, not an independent iid sample. The fitting derivative uses the oracle posterior and is a mechanism diagnostic, not a deployable training-only estimator. The independent iid risk supplement is supplied by the root study.','',
    'The drop-clock predictor is fully pre-state: Qprime*delta/(-Lprime). The realized-tangent diagnostic uses the actual training-only root or endpoint displacement, so it is outcome-blind but not fully pre-update. The old pooled competitor is fit only on old outcomes, stratified by task/state/method/experiment/horizon/drop; empty strata use flagged same-task/state/method reachable-slope fallback. Zero predicts no change.','',
    '## Optimizer-contrast results','',
    'Each row includes only pairs where both roots reached the target. Sign counts compare the sign of the predicted pullback-minus-GD difference to the observed sign. MAE is in nats. Each seed appears once within a row; confidence intervals in the CSV bootstrap seeds, never treating repeated states or endpoints as independent.','',
    '|Task|State|Drop|Endpoint|Pairs|Clock sign|Clock MAE|Old-pool MAE|Zero MAE|PB lower outcome|','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for task in study.TASKS:
        for stage in study.CONFIG['stages']:
            for drop in ('1e-05','0.02'):
                for metric in study.METRICS:
                    k=(task,stage,'single','1',drop,metric)
                    r=lookup.get(k+('drop_clock',))
                    if not r:
                        lines.append(f'|{task}|{stage}|{drop}|{metric}|0|—|—|—|—|—|');continue
                    old=lookup[k+('old_pooled',)];zero=lookup[k+('zero',)]
                    lines.append(f"|{task}|{stage}|{drop}|{metric}|{r['n_seed']}|{r['correct_sign']}/{r['n_seed']}|{r['mae']:.3g}|{old['mae']:.3g}|{zero['mae']:.3g}|{r['actual_pb_better']}/{r['n_seed']}|")
    lines+=['','## Scope, failures, and numerical checks','',
    'A missing pair means unreachable within the prespecified direction-ray search and displacement norm cap4, not mathematically impossible. All failures remain in roots/predictions/outcomes and coverage tables. Initial rings pullback often cannot achieve the substantial drop under this cap. Synthetic radial GD300 is especially flat: its substantial targets cannot support a general transfer-success claim. The predeclared tiny and .005-step targets remain reported without outcome-driven retuning.','',
    'Multi-step horizons1,2,4,8 retain the initial-state prediction while recomputing each optimizer direction after each .005 actual loss drop; coverage and every horizon are in `summary.csv` and `contrast_summary.csv`. The horizon8 prediction concerns .04 cumulative loss reduction. These reports include failures and finite-change deterioration.','',
    'Local directional derivatives are independently central-difference checked; root residuals and all quadrature convergence diagnostics are retained. `shared_states_check.json` verifies shared ring models with direction2 exactly, and matched synthetic tasks use identical X and initial theta. Shared directions are not independent replications. The angular two-view value equals its pi nuisance by construction; it is intentionally reported as a redundant identity check.','',
    'An early old-cohort attempt produced NaNs from floating-point 0*log(0) in entropy before any scoring; its output remains in `old_attempt01_nonfinite_entropy`. Stable entropy fixed it before the frozen experiment. No failed scientific target was removed.','',
    '## Reproduction','',
    'Requirements: Python, NumPy, threadpoolctl. The frozen model core is local; the portable prior polar evaluator lives in `../legacy/rings-mechanism-20261120/` with workspace fallback. Use a fresh copied output directory, preserving existing outputs. Run `study.py generate --cohort old`, `study.py score --cohort old`, `study.py freeze`, `study.py generate --cohort new`, `study.py score --cohort new`, then `analyze.py`. Code intentionally rejects existing cohort directories and an already frozen protocol. All arrays, input seeds, endpoint hashes, root scales and raw rows are supplied.','']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(json.loads((ROOT/'coverage.json').read_text()),indent=2))
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):main()

