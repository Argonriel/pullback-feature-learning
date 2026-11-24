"""Postprocessing only: model-seed uncertainty is separate from certificate coverage."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,ci,co,jo,sha,now

def main():
    bounds=ci(HERE/'certificate_bounds.csv');curves=ci(HERE/'certificate_curves.csv')
    stress=ci(HERE/'stress_results.csv');clips=json.loads((HERE/'CLIP_COMMIT.json').read_text())['records']
    variants=['legacy_raw','pb_temp'];policies=['symmetric_jensen','folded']
    summaries=[];curve_summary=[];clip_summary=[];stress_summary=[];crossings=[]
    rng=np.random.default_rng(2129000);indices=rng.integers(0,16,(20000,16))
    for variant in variants:
        group=[r for r in clips if r['variant']==variant]
        clip_summary.append(dict(variant=variant,n=len(group),mean_c=float(np.mean([r['c'] for r in group])),
            min_c=float(min(r['c'] for r in group)),max_c=float(max(r['c'] for r in group)),
            mean_beta=float(np.mean([r['beta'] for r in group])),min_beta=float(min(r['beta'] for r in group)),
            max_beta=float(max(r['beta'] for r in group))))
        for name in policies:
            for n in (512,2048):
                group=[r for r in bounds if r['variant']==variant and r['policy']==name and int(r['n_batches'])==n]
                critical=np.array([float(r['epsilon_critical']) for r in group])
                summaries.append(dict(variant=variant,policy=name,n_batches=n,n_models=len(group),
                    positive_at_zero=sum(float(r['A_lower'])>0 for r in group),
                    mean_epsilon_critical=float(critical.mean()),median_epsilon_critical=float(np.median(critical)),
                    min_epsilon_critical=float(critical.min()),max_epsilon_critical=float(critical.max()),
                    mean_A_lower=float(np.mean([float(r['A_lower']) for r in group])),
                    mean_B_upper=float(np.mean([float(r['B_upper']) for r in group])),
                    mean_empirical_unclipped_gain=float(np.mean([float(r['empirical_unclipped_gain']) for r in group]))))
                for eps in (0.,.025,.05,.1,.2,.4,.8,1.):
                    g=[r for r in curves if r['variant']==variant and r['policy']==name and int(r['n_batches'])==n and float(r['epsilon'])==eps]
                    v=np.array([float(r['lower']) for r in g]);curve_summary.append(dict(variant=variant,policy=name,
                        n_batches=n,epsilon=eps,mean_lower=float(v.mean()),min_lower=float(v.min()),max_lower=float(v.max()),
                        positive=int((v>0).sum())))
        for world in ('cosine','adversarial'):
            for name in ('raw','full','symmetric_jensen','folded','confidence'):
                for eps in (0.,.025,.05,.1,.2,.4,.8,1.):
                    group=sorted([r for r in stress if r['variant']==variant and r['world']==world and r['policy']==name and float(r['epsilon'])==eps],key=lambda r:int(r['pair']))
                    row=dict(variant=variant,world=world,policy=name,epsilon=eps,n_models=len(group))
                    for metric in ('expected_nll','expected_error','gain','antithetic_nll','antithetic_error','antithetic_gain'):
                        values=np.array([float(r[metric]) for r in group]);boot=values[indices].mean(1)
                        row[metric+'_mean']=float(values.mean());row[metric+'_ci_low']=float(np.quantile(boot,.025));row[metric+'_ci_high']=float(np.quantile(boot,.975))
                    row['negative_gain_models']=sum(float(r['antithetic_gain'])<0 for r in group)
                    stress_summary.append(row)
                for pair in range(1,17):
                    eligible=[float(r['epsilon']) for r in stress if r['variant']==variant and r['world']==world and r['policy']==name and int(r['pair'])==pair and float(r['antithetic_gain'])<0]
                    crossings.append(dict(variant=variant,world=world,policy=name,pair=pair,
                        first_tested_negative_epsilon=min(eligible) if eligible else 'unreached'))
    co(HERE/'certificate_summary.csv',summaries);co(HERE/'certificate_curve_summary.csv',curve_summary)
    co(HERE/'clip_summary.csv',clip_summary);co(HERE/'stress_summary.csv',stress_summary);co(HERE/'stress_crossings.csv',crossings)
    figure(summaries,curve_summary,stress_summary)
    write_report(summaries,clip_summary,stress_summary)
    jo(HERE/'ANALYSIS_COMMIT.json',dict(timestamp_utc=now(),bootstrap_seed=2129000,bootstrap_repeats=20000,
        certificate_commit_sha256=sha(HERE/'CERTIFICATE_COMMIT.json'),stress_commit_sha256=sha(HERE/'STRESS_COMMIT.json'),
        outputs={name:sha(HERE/name) for name in ['certificate_summary.csv','certificate_curve_summary.csv','clip_summary.csv','stress_summary.csv','stress_crossings.csv']}))

def figure(summary,curves,stress):
    labels={'legacy_raw':'Legacy PB, raw','pb_temp':'Selected PB, temperature calibrated'}
    colors={'symmetric_jensen':'#186ba0','folded':'#b86d20','confidence':'#823a77','full':'#317b4b'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for column,variant in enumerate(labels):
        for row,world in enumerate(('cosine','adversarial')):
            ax=axes[row,column]
            for name in colors:
                g=sorted([r for r in stress if r['variant']==variant and r['world']==world and r['policy']==name],key=lambda r:r['epsilon'])
                ax.plot([r['epsilon'] for r in g],[r['antithetic_gain_mean'] for r in g],marker='o',ms=3,
                    label={'symmetric_jensen':'Interpolated Jensen','folded':'Strong histogram','confidence':'Confidence','full':'Full averaging'}[name],color=colors[name])
                if name in ('symmetric_jensen','folded'):
                    c=sorted([r for r in curves if r['variant']==variant and r['policy']==name and r['n_batches']==2048],key=lambda r:r['epsilon'])
                    ax.plot([r['epsilon'] for r in c],[r['mean_lower'] for r in c],ls='--',color=colors[name],lw=1.4)
            ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.2);ax.set_xlabel('Known posterior-defect upper bound epsilon')
            ax.set_title(labels[variant]+'\n'+('Fixed cosine world' if world=='cosine' else 'Model-dependent adversarial world'))
    axes[0,0].set_ylabel('Mean expected log-loss gain');axes[1,0].set_ylabel('Mean expected log-loss gain')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Solid: independent antithetic stress estimates; dashed: simultaneous robust lower bounds',fontsize=11)
    fig.savefig(HERE/'robustness_frontier.png',dpi=180);fig.savefig(HERE/'robustness_frontier.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4),constrained_layout=True)
    positions=[];values=[];names=[]
    raw=ci(HERE/'certificate_bounds.csv')
    for i,variant in enumerate(labels):
        for j,name in enumerate(('symmetric_jensen','folded')):
            g=[float(r['epsilon_critical']) for r in raw if r['variant']==variant and r['policy']==name and int(r['n_batches'])==2048]
            position=3*i+j;positions.append(position);values.append(g)
            short='Legacy PB\nraw' if variant=='legacy_raw' else 'Selected PB\n+ temperature'
            names.append(short+'\n'+('Jensen' if j==0 else 'Histogram'))
            ax.scatter(position+np.linspace(-.12,.12,16),g,s=22,color=colors[name],alpha=.8)
            ax.hlines(np.mean(g),position-.25,position+.25,color='black',lw=2)
    ax.set_xticks(positions,names,fontsize=8);ax.set_ylabel('Certified positive-gain epsilon limit');ax.grid(axis='y',alpha=.2)
    ax.set_ylim(bottom=0);ax.set_title('All sixteen models retained; black bars are means')
    fig.savefig(HERE/'certified_limits.png',dpi=180);fig.savefig(HERE/'certified_limits.svg');plt.close(fig)

def write_report(summary,clips,stress):
    text=['# Imperfect posterior symmetry: results','',
        'The input law remains exactly half-turn invariant. The guarantee below assumes a supplied uniform bound '
        'on the posterior difference, |p(x)-p(Tx)| <= epsilon. Synthetic construction supplies that bound; '
        'unlabeled input data do not establish it for real applications.','',
        'The exact gain identity is E[M(J-(p-pT)d/2)], with d=(h-hT)/2. The certificate is the affine line '
        'L_A-epsilon U_B/2. All 64 policy/predictor lines at both 512 and 2048 batches are simultaneously '
        'covered by a 0.05 family budget over 512 elementary bounds. Coverage holds for every epsilon in [0,1], '
        'not only the plotted values. Primary results use 2048 independent batches.','',
        '| Predictor | Selector | Positive at epsilon zero | Certified limit, mean | Minimum | Maximum |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['n_batches']==2048:text.append(f"| {r['variant']} | {r['policy']} | {r['positive_at_zero']}/16 | {r['mean_epsilon_critical']:.6f} | {r['min_epsilon_critical']:.6f} | {r['max_epsilon_critical']:.6f} |")
    text+=['','A positive guarantee holds strictly below an uncapped crossing; the line equals zero at the crossing. '
        'The limit is a sufficient tolerance bound, not an estimate of the actual point where augmentation becomes harmful. '
        'Neither the confidence selector nor approximately invariant input distributions are covered.','',
        '## Calibration-only clip and scaling','',
        '| Predictor | Mean clip | Clip range | Mean inverse temperature |',
        '|---|---:|---:|---:|']
    for r in clips:text.append(f"| {r['variant']} | {r['mean_c']:.6g} | [{r['min_c']:.6g}, {r['max_c']:.6g}] | {r['mean_beta']:.6g} |")
    text+=['','The clipping rule was fixed before fresh access and uses the 95th percentile of the independent '
        'calibration grid. All clip choices were committed before certificate sampling. The network bound scales '
        'both the skip coordinate and residual contribution by inverse temperature. Training and temperature fitting '
        'use labels; this study does not describe the complete pipeline as label-free. The fresh groups differ '
        'in fitted parameters as well as temperature, so their contrast does not isolate a causal temperature '
        'effect. The old fixed-beta diagnostics are the controlled scaling check.','',
        '## Independent stress tests','',
        'The fixed cosine world is a common population across models. The adversarial sign(d) world depends '
        'on each frozen predictor and is shared across its selectors; its cross-model averages describe sensitivity '
        'rather than a common population. Posterior probabilities remain valid and their uniform defect is bounded '
        'by epsilon at every tested setting. Both direct IID and antithetic expected losses and classification errors '
        'are preserved. The antithetic gain identity is checked numerically for both symmetric policies.','',
        '| Predictor | World | Selector | Gain at epsilon 0 | Gain at epsilon 1 | Negative models at epsilon 1 |',
        '|---|---|---|---:|---:|---:|']
    for variant in ('legacy_raw','pb_temp'):
        for world in ('cosine','adversarial'):
            for name in ('symmetric_jensen','folded','confidence','full'):
                zero=next(r for r in stress if r['variant']==variant and r['world']==world and r['policy']==name and r['epsilon']==0)
                last=next(r for r in stress if r['variant']==variant and r['world']==world and r['policy']==name and r['epsilon']==1)
                text.append(f"| {variant} | {world} | {name} | {zero['antithetic_gain_mean']:.7f} | {last['antithetic_gain_mean']:.7f} | {last['negative_gain_models']}/16 |")
    cert=ci(HERE/'certificate_costs.csv');evaluation=ci(HERE/'stress_costs.csv')
    text+=['','## Cost, audit and reproduction','',
        f"Certificate computation used {sum(int(r['forwards']) for r in cert):,} actual forwards on selected unions. "
        f"Independent stress evaluation used {sum(int(r['shared_research_forwards']) for r in evaluation):,} forwards, "
        f"plus {sum(int(r['extra_deployment_audit_forwards']) for r in evaluation):,} separately counted deployment-audit forwards. "
        'Full pair evaluation is charged as research cost; each limited deployment audit uses exactly 256 first '
        'and 64 second views. The original training and calibration costs are reported by the training branch.','',
        'Raw batch moment observations, masks/input hashes, clipping commit, exact calls, per-model limits, '
        'stress outputs, and first negative tested epsilon or unreached are preserved. Bootstrap intervals in '
        'stress_summary.csv are pointwise 95% model-seed descriptions, separate from simultaneous certificate coverage. '
        'Floating-point identity checks do not amount to interval-arithmetic verification of every operation.','',
        'After the training branch writes the committed model/policy manifest and root releases this branch, '
        'run these stages in a fresh output tree:', '', '```text',
        'python robustness/run.py prepare','python robustness/run.py certificate','python robustness/run.py stress',
        'python robustness/analyze.py','```','',
        'THEORY.md supplies the derivation and primary literature. PROTOCOL.md and FREEZE.json preserve the '
        'prospective design. The old-only development uses a separately supplied previous-study directory; '
        'fresh confirmation depends only on current study sources and committed training artifacts.','']
    (HERE/'README.md').write_text('\n'.join(text),encoding='utf8')

if __name__=='__main__':main()
