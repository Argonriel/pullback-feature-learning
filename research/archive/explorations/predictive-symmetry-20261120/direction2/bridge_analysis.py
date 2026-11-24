"""Post-commit bridge reporting, with within-seed paired uncertainty."""
import csv,json
from pathlib import Path
import numpy as np
from experiment import csvout,dump

HERE=Path(__file__).resolve().parent


def report(folder):
    rows=list(csv.DictReader((folder/'risks.csv').open()))
    pred=list(csv.DictReader((folder/'predictions.csv').open()))
    policies=('stale','drop_clock_logits','matched_step_logits','first_order_gain','post_recalibrated')
    metrics=[];per_seed=[];rng=np.random.default_rng(629009)
    for method in ('gd','pullback'):
        seeds=sorted(set(int(r['pair']) for r in rows if r['model']==method))
        if not seeds:continue
        lookup={(int(r['pair']),r['policy']):r for r in rows if r['model']==method}
        indices=rng.integers(0,len(seeds),(20000,len(seeds)))
        staleerror=np.array([float(lookup[s,'stale']['predicted_gain'])-float(lookup[s,'stale']['gain']) for s in seeds])
        for policy in policies:
            chosen=[lookup[s,policy] for s in seeds]
            err=np.array([float(r['predicted_gain'])-float(r['gain']) for r in chosen])
            benefit=np.abs(staleerror)-np.abs(err)
            riskbenefit=np.array([float(lookup[s,'stale']['risk'])-float(lookup[s,policy]['risk']) for s in seeds])
            regret=np.array([float(lookup[s,policy]['risk'])-float(lookup[s,'test_oracle_global_DIAGNOSTIC_ONLY']['risk']) for s in seeds])
            changes=[abs(float(lookup[s,policy]['alpha_mean'])-float(lookup[s,'stale']['alpha_mean']))>1e-10 for s in seeds]
            metrics.append(dict(method=method,policy=policy,n=len(seeds),
                gain_prediction_mae=np.mean(abs(err)),gain_prediction_rmse=np.sqrt(np.mean(err*err)),
                mae_improvement_over_stale=benefit.mean(),
                mae_improvement_ci_low=np.quantile(benefit[indices].mean(1),.025),
                mae_improvement_ci_high=np.quantile(benefit[indices].mean(1),.975),
                risk_improvement_over_stale=riskbenefit.mean(),
                risk_improvement_ci_low=np.quantile(riskbenefit[indices].mean(1),.025),
                risk_improvement_ci_high=np.quantile(riskbenefit[indices].mean(1),.975),
                risk_wins=int((riskbenefit>1e-12).sum()),risk_losses=int((riskbenefit < -1e-12).sum()),
                risk_ties=int((abs(riskbenefit)<=1e-12).sum()),angle_changes=sum(changes),
                regret_to_population_diagnostic=regret.mean()))
            for i,s in enumerate(seeds):
                per_seed.append(dict(seed=s,method=method,policy=policy,
                    predicted_gain=float(lookup[s,policy]['predicted_gain']),actual_gain=float(lookup[s,policy]['gain']),
                    gain_error=err[i],risk_improvement_over_stale=riskbenefit[i],angle_change=int(changes[i])))
    csvout(folder/'forecast_quality.csv',metrics);csvout(folder/'forecast_quality_by_seed.csv',per_seed)
    timing={}
    for field in ('forecast_seconds','recalibration_seconds'):
        vals=[float(r[field]) for r in pred if r['policy']=='stale']
        timing[field+'_median']=float(np.median(vals))
    dump(folder/'cost_summary.json',dict(**timing,
        one_branch_prestate_forward_examples=65536,one_branch_prestate_JVP_examples=65536,
        post_recalibration_additional_forward_examples=65536,
        all_deployed_view_policies_model_calls_per_input=2,
        cost_caveat='forecast_seconds times all forecast variants plus their artifact serialization; JVP is counted separately and is not a plain forward evaluation.'))


if __name__=='__main__':
    for folder in (HERE/'bridge_old',HERE/'bridge_new',
        HERE/'late_bridge_results/bridge_old',HERE/'late_bridge_results_selected/bridge_old',
        HERE/'late_bridge_results_selected/bridge_new'):
        if (folder/'risks.csv').exists():report(folder)
