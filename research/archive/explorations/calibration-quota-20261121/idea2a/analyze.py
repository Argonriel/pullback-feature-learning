"""Frozen-readout analysis; model-cluster uncertainty and calibration MC separate."""
import json
from pathlib import Path
import numpy as np
from study import HERE,MODELS,BUDGETS,CONFIG,sha,dump,csvin,csvout


def interval(values,indices):
    return np.quantile(np.asarray(values)[indices].mean(1),[.025,.975])


def analyze():
    out=HERE/'fresh';hist=np.load(HERE/'history/historical_curves.npz');defaultidx=hist['indices']
    curves=np.load(out/'population_gain_curves.npz')['gains']
    modelrows=csvin(out/'population_models.csv');decisions=csvin(out/'decisions.csv');iidrows=csvin(out/'iid_candidate_risks.csv')
    raw=np.empty((16,3));angular=np.empty_like(raw);random=np.empty_like(raw)
    for r in modelrows:
        pi=int(r['pair'])-1;mi=MODELS.index(r['model'])
        raw[pi,mi]=float(r['population_loss']);angular[pi,mi]=float(r['angular_excess']);random[pi,mi]=float(r['random_gain'])
    chosen=np.empty((16,3,2,6,20),dtype=int)
    for r in decisions:
        chosen[int(r['pair'])-1,MODELS.index(r['model']),('none','shrink16').index(r['rule']),BUDGETS.index(int(r['budget'])),int(r['repeat'])-1]=int(r['candidate_index'])-1
    risk=np.empty_like(chosen,dtype=float);sampled=np.empty_like(risk);expected=np.empty_like(risk)
    iidlookup={(int(r['pair']),r['model'],int(r['candidate_index'])):r for r in iidrows}
    perdraw=[];perseed=[];summary=[];mc=[];thresholds=[];baseline=[];threshold_draws=[]
    indices=np.random.default_rng(CONFIG['bootstrap_seed']).integers(0,16,(CONFIG['bootstrap_repeats'],16))
    for pi in range(16):
        for mi,model in enumerate(MODELS):
            for ri,rule in enumerate(('none','shrink16')):
                for bi,budget in enumerate(BUDGETS):
                    js=chosen[pi,mi,ri,bi]
                    risk[pi,mi,ri,bi]=raw[pi,mi]-curves[pi,mi,js]
                    sampled[pi,mi,ri,bi]=[float(iidlookup[pi+1,model,j+1]['sampled_risk']) for j in js]
                    expected[pi,mi,ri,bi]=[float(iidlookup[pi+1,model,j+1]['expected_risk']) for j in js]
                    for rep in range(20):
                        perdraw.append(dict(pair=pi+1,model=model,rule=rule,budget=budget,repeat=rep+1,
                            candidate_index=js[rep]+1,population_risk=risk[pi,mi,ri,bi,rep],
                            iid_sampled_risk=sampled[pi,mi,ri,bi,rep],iid_expected_risk=expected[pi,mi,ri,bi,rep]))
    for mi,model in enumerate(MODELS):
        default=raw[:,mi]-curves[:,mi,defaultidx[mi]]
        pi_risk=raw[:,mi]-curves[:,mi,-1]
        full=risk[:,mi,0,-1].mean(1)
        denominator=default-full
        db=denominator[indices].mean(1);positive=db>0
        dlo,dhi=interval(denominator,indices)
        for name,values,j in [('raw',raw[:,mi],0),('pi',pi_risk,64),('default',default,int(defaultidx[mi])+1),
            ('random_alpha',raw[:,mi]-random[:,mi],65),('full_none_65536',full,None),
            ('population_best_candidate_DIAGNOSTIC_ONLY',raw[:,mi]-curves[:,mi].max(1),None)]:
            lo,hi=interval(values,indices)
            baseline.append(dict(model=model,policy=name,mean_population_risk=values.mean(),risk_ci_low=lo,risk_ci_high=hi,
                angular_loss_removed=(raw[:,mi]-values).mean()/angular[:,mi].mean(),
                new_model_setup_forward_examples=65536 if name=='full_none_65536' else ('' if 'DIAGNOSTIC' in name else 0),
                iid_sampled_risk=float(np.mean([float(iidlookup[p+1,model,j]['sampled_risk']) for p in range(16)])) if j is not None else
                    (sampled[:,mi,0,-1].mean() if name=='full_none_65536' else ''),
                iid_expected_risk=float(np.mean([float(iidlookup[p+1,model,j]['expected_risk']) for p in range(16)])) if j is not None else
                    (expected[:,mi,0,-1].mean() if name=='full_none_65536' else '')))
        for ri,rule in enumerate(('none','shrink16')):
            policymean=risk[:,mi,ri].mean(-1)
            numerator=default[:,None]-policymean
            boot=numerator[indices].mean(1)
            ratios=np.full((len(indices),6),np.nan)
            ratios[positive]=boot[positive]/db[positive,None]
            reached=np.zeros(len(indices),dtype=bool);first=np.full(len(indices),-1,dtype=int)
            for bi,budget in enumerate(BUDGETS):
                hit=positive & ~reached & (ratios[:,bi]>=.9-1e-12)
                first[hit]=budget;reached|=hit
            empirical_ratio=numerator.mean(0)/denominator.mean() if denominator.mean()>0 else np.full(6,np.nan)
            emp_hit=np.flatnonzero(empirical_ratio>=.9-1e-12)
            finite_threshold=first[first>=0]
            thresholds.append(dict(model=model,rule=rule,full_increment_over_default=denominator.mean(),
                full_increment_ci_low=dlo,full_increment_ci_high=dhi,denominator_clearly_positive=int(dlo>0),
                first_tested_90_budget=int(BUDGETS[emp_hit[0]]) if len(emp_hit) else '',
                bootstrap_threshold_p025=float(np.quantile(finite_threshold,.025,method='nearest')) if len(finite_threshold) else '',
                bootstrap_threshold_median=float(np.median(finite_threshold)) if len(finite_threshold) else '',
                bootstrap_threshold_p975=float(np.quantile(finite_threshold,.975,method='nearest')) if len(finite_threshold) else '',
                denominator_nonpositive_probability=float(np.mean(~positive)),
                unattained_90_probability_given_positive=float(np.mean(first[positive]<0)) if positive.any() else '',
                subfull_unattained_probability_given_positive=float(np.mean((first[positive]<0)|(first[positive]>=65536))) if positive.any() else '',
                **{f'first_at_{b}_probability':float(np.mean(first==b)) for b in BUDGETS}))
            for draw in range(len(indices)):
                threshold_draws.append(dict(model=model,rule=rule,bootstrap_draw=draw,
                    denominator=db[draw],first_90_budget=first[draw],denominator_positive=int(positive[draw])))
            for bi,budget in enumerate(BUDGETS):
                v=policymean[:,bi];increment=default-v
                risklo,riskhi=interval(v,indices);gainlo,gainhi=interval(increment,indices)
                ratios_valid=ratios[positive,bi]
                ratio_ci=np.quantile(ratios_valid,[.025,.975]) if len(ratios_valid) else [np.nan,np.nan]
                wins=np.sum(v<default-1e-12);losses=np.sum(v>default+1e-12)
                sd=risk[:,mi,ri,bi].std(1,ddof=1)
                summary.append(dict(model=model,rule=rule,budget=budget,
                    mean_population_risk=v.mean(),risk_ci_low=risklo,risk_ci_high=riskhi,
                    incremental_gain_over_default=increment.mean(),gain_ci_low=gainlo,gain_ci_high=gainhi,
                    improvement_over_pi=float((pi_risk-v).mean()),
                    seed_wins_over_default=int(wins),seed_losses_over_default=int(losses),seed_ties_over_default=int(16-wins-losses),
                    full_increment_over_default=denominator.mean(),
                    recovery_ratio=empirical_ratio[bi],recovery_ci_low=ratio_ci[0],recovery_ci_high=ratio_ci[1],
                    recovery_ci_condition='positive bootstrap denominator only',
                    mean_regret_to_population_diagnostic=float((v-(raw[:,mi]-curves[:,mi].max(1))).mean()),
                    iid_sampled_risk=sampled[:,mi,ri,bi].mean(),iid_expected_risk=expected[:,mi,ri,bi].mean(),
                    draw_probability_worse_than_default=float(np.mean(risk[:,mi,ri,bi]>default[:,None]+1e-12))))
                mc.append(dict(model=model,rule=rule,budget=budget,
                    mean_within_model_calibration_sd=sd.mean(),median_within_model_calibration_sd=float(np.median(sd)),
                    max_within_model_calibration_sd=sd.max(),
                    cohort_mean_single_draw_calibration_sd=float(risk[:,mi,ri,bi].mean(0).std(ddof=1)),
                    calibration_mc_se_of_reported_20draw_mean=float(np.sqrt(np.sum(sd**2)/(16**2*20))),
                    between_model_se_of_20draw_means=float(v.std(ddof=1)/4)))
                for pi in range(16):
                    perseed.append(dict(pair=pi+1,model=model,rule=rule,budget=budget,
                        mean_population_risk=v[pi],calibration_sd=sd[pi],
                        default_population_risk=default[pi],full_population_risk=full[pi],
                        incremental_gain_over_default=increment[pi],
                        iid_sampled_risk=sampled[pi,mi,ri,bi].mean(),iid_expected_risk=expected[pi,mi,ri,bi].mean()))
    csvout(out/'per_draw_risks.csv',perdraw);csvout(out/'per_seed_means.csv',perseed)
    csvout(out/'summary.csv',summary);csvout(out/'calibration_monte_carlo.csv',mc)
    csvout(out/'baseline_summary.csv',baseline);csvout(out/'threshold_summary.csv',thresholds)
    csvout(out/'threshold_bootstrap_draws.csv',threshold_draws)
    np.savez_compressed(out/'risk_arrays.npz',population=risk,iid_sampled=sampled,iid_expected=expected,
        candidate_indices_zero_based=chosen)
    dump(out/'analysis_notes.json',dict(bootstrap='20,000 draws; 16 paired model clusters; 20 calibration repeats averaged within each cluster.',
        calibration_mc='Within-model replicate SD is reported separately; MonteCarloSE= sqrt(sum(seed_variance)/(16^2*20)).',
        threshold_minus_one='No threshold attained or nonpositive denominator; inspect denominator_positive.',
        full_calibration='Finite 512-radius no-shrinkage reference, not oracle. Ratios can exceed 1 and are not clipped.',
        historical='Old 16 defaults, separate by training method. 143.4375 degrees / 62.7022 percent is explicitly retrospective.',
        baseline_random='Uniform 128-element rotation group including identity; always 2 forward calls at deployment.'))
    return summary,thresholds


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,model,title in zip(axes,MODELS,('Gradient descent','Pullback','Pullback then GD')):
        for rule,label,color in [('none','New-model calibration','#1e6487'),('shrink16','Historical shrinkage (16 radii)','#c38039')]:
            r=[z for z in summary if z['model']==model and z['rule']==rule]
            y=np.array([z['incremental_gain_over_default'] for z in r])*1000
            low=np.array([z['gain_ci_low'] for z in r])*1000;high=np.array([z['gain_ci_high'] for z in r])*1000
            ax.plot(range(6),y,'o-',color=color,label=label,ms=4)
            ax.fill_between(range(6),low,high,color=color,alpha=.12)
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(6),['0','128','512','2,048','8,192','65,536'],rotation=35,ha='right')
        ax.set_title(title);ax.set_xlabel('New-model calibration forward evaluations')
    axes[0].set_ylabel('Risk reduction beyond historical default\n(milli-nats per input)')
    axes[1].legend(fontsize=8,frameon=False,loc='lower right')
    fig.suptitle('Same two deployment views; 16 fresh model pairs and 20 calibration draws',y=1.02)
    fig.tight_layout();fig.savefig(HERE/'fresh/calibration_budget_results.png',dpi=200,bbox_inches='tight')
    fig.savefig(HERE/'fresh/calibration_budget_results.svg',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),sharey=True)
    for ax,model,title in zip(axes,MODELS,('Gradient descent','Pullback','Pullback then GD')):
        for rule,label,color in [('none','No shrinkage','#1e6487'),('shrink16','Shrinkage16','#c38039')]:
            r=[z for z in summary if z['model']==model and z['rule']==rule]
            y=np.array([z['recovery_ratio'] for z in r])*100
            ax.plot(range(6),y,'o-',color=color,label=label,ms=4)
        ax.axhline(90,color='gray',ls='--',lw=.8);ax.axhline(100,color='gray',ls=':',lw=.7)
        ax.set_xticks(range(6),['0','128','512','2,048','8,192','65,536'],rotation=35,ha='right');ax.set_title(title)
        ax.set_xlabel('Calibration forward evaluations')
    axes[0].set_ylabel('Incremental gain recovered (%)\nrelative to finite 65,536-call reference')
    axes[-1].legend(frameon=False);fig.tight_layout();fig.savefig(HERE/'fresh/calibration_recovery.png',dpi=200);fig.savefig(HERE/'fresh/calibration_recovery.svg');plt.close(fig)


if __name__=='__main__':
    s,t=analyze();plots(s)
    print(t)
