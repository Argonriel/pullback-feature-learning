"""Readable report and figures from completed, frozen certificate outputs."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
LABELS={'rings':'Gaussian-noisy rings','waveform':'One-amplitude waveform','two_invariants':'Two-amplitude waveform'}

def read(name):
    with (HERE/name).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def main():
    summary=read('final_summary.csv');costs=read('costs.csv');bounds=read('final_per_model_bounds.csv')
    cohort_order={name:index for index,name in enumerate(LABELS)}
    claim_order={'symmetric_jensen':0,'folded':1,'candidate_minus_folded':2}
    summary.sort(key=lambda row:(cohort_order[row['cohort']],claim_order[row['claim']],int(row['n_batches'])))
    completion=json.loads((HERE/'completion.json').read_text())
    lines=['# Independent-batch verification results','',
        'The frozen primary budget is 512 independent batches of 256 inputs per model. '
        'Each policy selects exactly 64. The lower bounds are simultaneous over all 48 models, '
        'both absolute-gain statements, the relative comparison, three batch-count looks, '
        'and both concentration methods (864 one-sided events; family error at most 0.05). Training and policy calibration '
        'were frozen before the independent verification stream.','',
        'The reported bound is the maximum of empirical Bernstein and bounded differences, '
        'both using the final joint error allocation. The earlier 432-event empirical-Bernstein '
        'outputs are retained as diagnostics and are not the primary result. These are bounds '
        'on expected logistic-loss reductions under the stated symmetry law. '
        'They are not classification-error guarantees. Structurally even selectors are already '
        'non-harmful by the theorem; a positive bound quantifies a guaranteed gain.','',
        '| Task | Statement at 512 batches | Positive / 16 | Mean lower bound | Model range |',
        '|---|---|---:|---:|---:|']
    for row in summary:
        if int(row['n_batches'])!=512:continue
        name={'symmetric_jensen':'Candidate absolute gain','folded':'Strong histogram absolute gain',
            'candidate_minus_folded':'Candidate gain minus strong histogram gain'}[row['claim']]
        lines.append(f"| {LABELS[row['cohort']]} | {name} | {row['positive']} | {float(row['mean_lower']):.7f} | [{float(row['min_lower']):.7f}, {float(row['max_lower']):.7f}] |")
    lines += ['', 'Negative statistical bounds are retained. The structural zero bound is available '
        'for each absolute gain separately; it cannot replace a negative relative bound. '
        'Absolute observations use the fixed clipped gap (0.25 for rings, 0.5 for both waveform tasks). '
        'Relative observations are untruncated on compact tasks; rings use a radius-three region '
        'plus a deterministic tail subtraction. Clipping the signed contrast was not used.','',
        '## Additive concentration comparison','',
        'Both columns below use the same final family allocation and exactly the same saved observations. '
        'The bounded-difference calculation adds no prediction calls. The primary budget remains 512; '
        '128 is a predeclared cost-curve look.','',
        '| Candidate, at 128 batches | Empirical Bernstein positive / 16 | Bounded differences positive / 16 | Mean EB lower bound | Mean bounded-difference lower bound |',
        '|---|---:|---:|---:|---:|']
    for row in summary:
        if int(row['n_batches'])!=128 or row['claim']!='symmetric_jensen':continue
        lines.append(f"| {LABELS[row['cohort']]} | {row['empirical_bernstein_positive']} | {row['bounded_difference_positive']} | {float(row['mean_empirical_bernstein_lower']):.7f} | {float(row['mean_bounded_difference_lower']):.7f} |")
    lines += ['',
        '## Cost and tightness','',
        '| Task | Shared actual forwards at 512, mean | Candidate lower bound / 2048 lower bound, mean | Candidate lower bound / empirical full gain, mean |',
        '|---|---:|---:|---:|']
    for row in summary:
        if int(row['n_batches'])!=512 or row['claim']!='symmetric_jensen':continue
        retention=row['mean_2048_lower_bound_retention'];recovery=row['mean_lower_to_empirical_full_gain']
        retention_text=f'{float(retention):.3f}' if retention else 'undefined'
        recovery_text=f'{float(recovery):.3f}' if recovery else 'undefined'
        lines.append(f"| {LABELS[row['cohort']]} | {float(row['mean_actual_shared_forwards']):,.1f} | {retention_text} ({row['valid_retention_denominators']} positive denominators) | {recovery_text} ({row['valid_empirical_gain_denominators']} positive denominators) |")
    lines += ['',
        'The last two columns are descriptive ratios calculated per model only when the denominator '
        'is positive. The final column uses the same policy\'s unclipped gain on the same 2048-batch '
        'stream; it does not mean averaging every input or an oracle policy, and is not an independent '
        'certified recovery denominator. All counts, unreached first-positive budgets, raw observations, '
        'and alternative fixed looks are retained in the accompanying tables.','',
        'A standalone absolute certificate at 512 batches requires exactly 65,536 network forwards '
        '(both views of 64 selected inputs per batch). The joint run evaluates both views only on '
        'the selected-set union and reuses them for both absolute and relative statements. '
        'Evaluating both views of all 256 inputs at the same 512 batches would cost 262,144 forwards '
        'and produce exactly the same selected-gap statistics. Coordinate inputs cost no network calls. '
        'These are offline verification counts, distinct from 320-forward deployment batches and '
        'the 65,536-forward calibration of each competing method.','',
        f"The completed joint 2048-batch run used {sum(int(r['actual_forward_examples']) for r in costs):,} actual network forwards across 48 models and {sum(int(r['coordinate_inputs']) for r in costs):,} coordinate inputs. "
        f"Its wall time was {completion['seconds']:.2f} seconds including invariance audits and bookkeeping. "
        'This is a descriptive run duration, not an isolated serving benchmark; no prefix runtime is '
        'estimated by dividing it.','',
        'The policy class changed from the earlier asymmetric rank construction. Exact symmetry removes '
        'the need for rank-CDF and interval-derivative certificates. This does not establish that the same '
        'asymmetric certificate was accelerated, or that clipping preserves its tightness.','',
        '## Audits and reproduction','',
        'The bounded-difference calculation uses the proved fact that replacing one input-key pair '
        'changes a selected clipped-weight sum by at most the clipping level. It accounts for '
        'within-batch dependence and does not treat selected contributions as IID. '
        'Each of the 98,304 complete verification batches has an exact-quota and actual-union-forward '
        'ledger. Full and coordinatewise mixed half-turn masks are checked bitwise. Algebraic antithetic '
        'identities, deterministic range bounds, seed provenance, source hashes, and policy hashes are '
        'saved. No verification label or hidden posterior is queried. The guarantees are mathematical '
        'real-arithmetic results; floating-point audits are not interval proofs of every numerical operation.','',
        'From a fresh reproduction directory with committed model/policy artifacts and root approval:',
        '', '```text', 'python certification/run.py run', 'python certification/run.py analyze',
        'python certification/final_compare.py fresh',
        'python certification/report.py', '```','',
        'The source freezes are `FREEZE.json` and `ADDENDUM_FREEZE.json`; the declarations '
        'are `PROTOCOL.md` and the superseding `ADDENDUM.md`; the derivation and '
        'primary references are in `LITERATURE_AND_PROOF.md` and `ADDENDUM.md`. The old development pilot is preserved '
        'under `development_old32`, including the 432-event recalculation and final 864-event comparison. Its historical rerun '
        'requires the old study path passed explicitly to `development.py`; fresh confirmation uses '
        'only the current portable study dependencies. Outputs are write-once, so rerun in a new '
        'output tree instead of overwriting an existing result.','']
    (HERE/'README.md').write_text('\n'.join(lines),encoding='utf8')
    fig,axes=plt.subplots(2,3,figsize=(12,6.6),constrained_layout=True)
    colors={'symmetric_jensen':'#176d9c','folded':'#bd6b20','candidate_minus_folded':'#8a345b'}
    for column,cohort in enumerate(LABELS):
        for name in colors:
            rows=[r for r in summary if r['cohort']==cohort and r['claim']==name]
            rows.sort(key=lambda r:int(r['n_batches']))
            target=axes[1,column] if name=='candidate_minus_folded' else axes[0,column]
            x=np.array([float(r['mean_actual_shared_forwards']) for r in rows])/1000
            legend={'symmetric_jensen':'Interpolated Jensen','folded':'Strong histogram',
                    'candidate_minus_folded':'Candidate minus histogram'}[name]
            target.plot(x,[float(r['mean_lower']) for r in rows],marker='o',color=colors[name],label=legend)
            target.fill_between(x,[float(r['min_lower']) for r in rows],[float(r['max_lower']) for r in rows],alpha=.10,color=colors[name])
            target.set_xscale('log',base=2);target.grid(alpha=.2);target.axhline(0,color='black',lw=.8)
            target.set_xticks(x,[f'{z:.0f}' for z in x])
        axes[0,column].set_title(LABELS[cohort]);axes[1,column].set_xlabel('Actual shared forwards (thousands)')
    axes[0,0].set_ylabel('Absolute gain lower bound');axes[1,0].set_ylabel('Relative gain lower bound')
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('Independent-batch certificates: mean and model range\nShading is variation across models, not an additional confidence interval',fontsize=12)
    fig.savefig(HERE/'certificate_frontier.png',dpi=180);fig.savefig(HERE/'certificate_frontier.svg')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for column,cohort in enumerate(LABELS):
        rows=sorted([r for r in summary if r['cohort']==cohort and r['claim']=='symmetric_jensen'],key=lambda r:int(r['n_batches']))
        x=np.array([float(r['mean_actual_shared_forwards']) for r in rows])/1000
        for key,label,color in [('mean_empirical_bernstein_lower','Empirical Bernstein','#176d9c'),
                                ('mean_bounded_difference_lower','Bounded differences','#267b4d')]:
            axes[column].plot(x,[float(r[key]) for r in rows],marker='o',label=label,color=color)
        axes[column].set_title(LABELS[cohort]);axes[column].set_xscale('log',base=2)
        axes[column].set_xticks(x,[f'{z:.0f}' for z in x]);axes[column].set_xlabel('Shared forwards (thousands)')
        axes[column].axhline(0,color='black',lw=.8);axes[column].grid(alpha=.2)
    axes[0].set_ylabel('Mean candidate gain lower bound');axes[0].legend(fontsize=8)
    fig.suptitle('Same observations and simultaneous 864-event allocation',fontsize=12)
    fig.savefig(HERE/'concentration_comparison.png',dpi=180);fig.savefig(HERE/'concentration_comparison.svg')
    plt.close(fig)

if __name__=='__main__':main()
