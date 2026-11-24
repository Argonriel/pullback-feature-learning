"""Summarize the fixed conditional-moment evaluations without changing policies."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import policy as policy_module

def load(path):return list(csv.DictReader(path.open()))
def main():
    original=load(ROOT/'diagnostics'/'summary.csv');final=load(ROOT/'diagnostics_refined'/'summary.csv');old={(int(r['seed']),r['policy'],r['pass_name']):r for r in original};new={(int(r['seed']),r['policy']):r for r in final};policies=['rank_balanced','rank_even','rank_unbalanced'];means={};refinements={}
    for name in policies:
        rows=[new[(i,name)] for i in range(1,17)];means[name]={k:float(np.mean([float(r[k]) for r in rows])) for k in ['C','centered_gain','b_mean','max_abs_b','numerical_width','numerical_low','numerical_high','formal_width_bound','mean_rate']}
        refinements[name]={k:float(max(abs(float(new[(i,name)][k])-float(old[(i,name,'main')][k])) for i in range(1,17))) for k in ['C','centered_gain','numerical_width','numerical_low','numerical_high','mean_rate']}
    lower_vs_even=np.array([float(new[(i,'rank_balanced')]['numerical_low'])-float(new[(i,'rank_even')]['C']) for i in range(1,17)])
    widths={name:np.array([float(new[(i,name)]['numerical_width']) for i in range(1,17)]) for name in policies};adv=np.array([float(new[(i,'rank_balanced')]['centered_gain'])-float(new[(i,'rank_even')]['C']) for i in range(1,17)])
    universal_caps=[]
    for seed in range(1,17):
        with np.load(ROOT.parent/'validation'/'policies'/f'{seed}_new.npz') as z:L,C=policy_module.network_constants(z['theta'])
        universal_caps.append(.5*min(np.sqrt(.89)+C,np.sqrt(.89)*L))
    checks=load(ROOT/'diagnostics_refined'/'checks.csv');summary=dict(models=16,means=means,rigorous_universal_width_caps=universal_caps,mean_rigorous_universal_width_cap=float(np.mean(universal_caps)),max_change_from_initial_main=refinements,numerical_balanced_lower_beats_even_count=int((lower_vs_even>0).sum()),numerical_balanced_lower_positive_count=sum(float(new[(i,'rank_balanced')]['numerical_low'])>0 for i in range(1,17)),max_integrated_rate_error=max(abs(float(r['mean_rate'])-.25) for r in final),mean_imbalance_reduction=1-means['rank_balanced']['numerical_width']/means['rank_unbalanced']['numerical_width'],max_CDF32_64_grid_difference=max(float(r['cdf32_64_max_difference']) for r in checks),max_marginal_interpolation_spot_error=max(float(r['spot_marginal_error']) for r in final),rank_q_interpolation_uniform_bound=float(checks[0]['rank_q_interpolation_error_bound']),total_evaluation_model_forwards=83886080+134217728,interval_scope='Unrestricted radial posteriors, individually extremized for each model; no fixed-prevalence constraint. Numerical quadrature, not an exact neural certificate.',formal_bound_scope='Uniform pointwise DKW/BV/radial bound with alpha=.05/16. Its stored Gauss integral is only a numerical estimate without a quadrature certificate; separately reported universal integrated cap is rigorous and vacuous.')
    (ROOT/'DIAGNOSTIC_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    b=means['rank_balanced'];e=means['rank_even'];u=means['rank_unbalanced'];delta=refinements['rank_balanced'];text=f'''# Conditional balance under global ranking

The new allocation family substantially reduces the ambiguity caused by unknown radial posteriors, but its learned balance is approximate. Across 16 new PB500 models, the refined numerical mean ambiguity width E|b(R)| is **{b['numerical_width']:.6g} nats**, versus **{u['numerical_width']:.6g}** for unconstrained ranking, a reduction of **{100*summary['mean_imbalance_reduction']:.2f}%**. The exact-even control has width zero by symmetry. The learned policy's mean maximum sampled |b(r)| is {b['max_abs_b']:.5g}; its zero calibration-anchor moments do not imply zero continuous conditional moments.

| Frozen method | Centered gain | Mean numerical lower gain | Mean numerical upper gain | Numerical integral of formal pointwise bound |
|---|---:|---:|---:|---:|
| Balanced at anchors | {b['centered_gain']:.7f} | {b['numerical_low']:.7f} | {b['numerical_high']:.7f} | {b['formal_width_bound']:.4g} |
| Exact even symmetry | {e['centered_gain']:.7f} | {e['numerical_low']:.7f} | {e['numerical_high']:.7f} | 0 |
| Unconstrained | {u['centered_gain']:.7f} | {u['numerical_low']:.7f} | {u['numerical_high']:.7f} | {u['formal_width_bound']:.4g} |

These are gains relative to one-view inference. Each numerical interval extremizes an unrestricted radial posterior separately for each model and the table averages those model-specific endpoints. It does not describe one shared extremizing posterior, does not fix prevalence, and is not an exact continuous neural-network certificate. For an individual model, the interval is [E C-E max(b,0), E C-E min(b,0)], with width E|b|.

The balanced numerical lower endpoint is positive for {summary['numerical_balanced_lower_positive_count']}/16 models, but exceeds the exact-even control for only {summary['numerical_balanced_lower_beats_even_count']}/16. Thus the small centered-gain and original-task advantage does not imply dominance under arbitrary radial posteriors. The simultaneous 95% DKW/BV/radial bound is valid **pointwise**, uniformly over radius. Its mean numerical integral is {b['formal_width_bound']:.4g} nats, far larger than the gains; that Gauss integral has no analytic quadrature certificate and is not itself a certified integrated upper bound. A separate rigorous deterministic integrated cap is available: because E[d|r]=0 and 0<=q<=1, |E[qd|r]|<=E[|d||r]/2. With E[R]<=sqrt(.89), E|b|<=.5 min(sqrt(.89)+C_h, sqrt(.89)L_h). This universal cap averages {summary['mean_rigorous_universal_width_cap']:.4g} nats and is also **vacuous**. Neither bound certifies the learned policy's small gain advantage. The exact-even guarantee is structural and does not need these loose bounds.

![Conditional ambiguity and gain tradeoff](conditional_balance.png)

## Numerical precision and fixed refinement

The initial diagnostic plan was physically recorded at 09:41:46.830295 UTC before this investigator opened any fresh policy or outcome. Main risk scoring had begun eight seconds earlier under the parent's authorization; no resulting outcome had been revealed to this investigator. The evaluator source was committed at 09:42:58.920609 UTC before the first diagnostic policy read. This chronology is explicit in `diagnostics/SOURCE_COMMIT.json`; it is not represented as a study-wide pre-scoring diagnostic freeze.

Initial grids used four Gauss nodes per radial anchor interval and 4,096 angles, with fixed radial, angular, and phase alternatives. Their non-negligible numerical sensitivity triggered one authorized refinement, uniformly applied to all 16 models: eight radial nodes, 8,192 angles, and 64 nodes per interval in the population score-CDF integration. `DIAGNOSTIC_REFINEMENT_FREEZE.json` and `diagnostics_refined/SOURCE_COMMIT.json` preserve the evaluation-only amendment. No policy, width, model, or winner changed. The initial outputs remain intact.

The largest per-model change from the initial main grid to the final refinement is {delta['numerical_width']:.3g} nats for the balanced ambiguity width, {delta['numerical_low']:.3g} for its lower endpoint, and {delta['centered_gain']:.3g} for its centered gain. The largest final integrated-rate discrepancy from the exact 1/4 identity is {summary['max_integrated_rate_error']:.3g}; this is quadrature error, not an exact-quota deployment failure. The score-CDF 32-to-64-node comparison differs by at most {summary['max_CDF32_64_grid_difference']:.3g} on the fixed score check grid. These refinement differences are sensitivity diagnostics, not rigorous continuous integration error bounds. No further adaptive refinement was performed; interpret rounded results at this precision.

The rank marginal uses the correct Q(F_population), including its action mixture and competition against the other 255 inputs. A 262,145-point table of the exact stable binomial polynomial accelerates evaluation with a global analytic interpolation-error bound {summary['rank_q_interpolation_uniform_bound']:.3g}; direct marginal spot checks differ by at most {summary['max_marginal_interpolation_spot_error']:.3g}. The exact-even moments are numerically near machine precision and are set to zero structurally only for their interval and bound. They are not silently zeroed in the raw moment arrays.

## Costs, raw files, and reproduction

The initial diagnostics use 83,886,080 model example forwards, and the uniform refinement adds 134,217,728, for **218,103,808 evaluation-only forwards**. These are separate from the frozen 66,048 calibration forwards per model and the actual 256+64 deployment budget. Coordinate/CDF/Q calculations do not evaluate the network. Stored orbit logits permit independent replay of moments without new model calls.

- `diagnostics/summary.csv` and `diagnostics/<seed>/`: all original grids, moments, orbit logits for the main grid, CDF comparisons, and provenance.
- `diagnostics_refined/summary.csv` and `diagnostics_refined/<seed>/`: the fixed refined results and orbit logits.
- `DIAGNOSTIC_SUMMARY.json`: full-precision aggregate values and sensitivities.
- `DIAGNOSTIC_FREEZE.json`, `DIAGNOSTIC_REFINEMENT_FREEZE.json`, and both source commits: transparent chronology and hashes.
- `../validation/`: independently measured original-task risks, baseline comparisons, exact call audits, and latency. This report does not replace that independent validation.

From the study root, `python selection/diagnostic_report.py` regenerates this report and figure from immutable raw arrays. `python selection/diagnose.py` and `python selection/diagnose_refinement.py` regenerate the numerical evaluations; both refuse to overwrite their output directories, so preserve existing outputs first. The complete original plus refined diagnostic computation takes about three minutes on the original CPU environment. NumPy, threadpoolctl, and the vendored model source are sufficient for evaluation; matplotlib is needed for the report figure.
'''
    (ROOT/'DIAGNOSTICS.md').write_text(text,encoding='utf8')
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False});fig,ax=plt.subplots(1,2,figsize=(10.6,4.3),layout='constrained');seed=np.arange(1,17)
    ax[0].scatter(seed,widths['rank_unbalanced'],c='#887098',label='Unconstrained ranking');ax[0].scatter(seed,widths['rank_balanced'],c='#156f83',label='Balanced at anchors');ax[0].set_yscale('log');ax[0].set_xlabel('Independent model seed');ax[0].set_ylabel('Numerical ambiguity width E|b(R)|');ax[0].set_title('Balance improves substantially, but is not exact');ax[0].set_xticks([1,4,8,12,16]);ax[0].legend(fontsize=9);ax[0].text(.98,.02,'Exact-even control: width = 0',transform=ax[0].transAxes,fontsize=9,ha='right')
    x=widths['rank_balanced']/2;upper=max(x.max(),adv.max())*1.1;ax[1].scatter(x,adv,c='#156f83');ax[1].plot([0,upper],[0,upper],c='#777777',linestyle='--',label='Numerical worst-case break-even');ax[1].axhline(0,color='#aaaaaa',linewidth=.7);ax[1].set_xlabel('Half the numerical posterior ambiguity width');ax[1].set_ylabel('Centered gain above exact-even control');ax[1].set_title('A small average gain does not imply dominance');ax[1].legend(fontsize=8);fig.savefig(ROOT/'conditional_balance.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
