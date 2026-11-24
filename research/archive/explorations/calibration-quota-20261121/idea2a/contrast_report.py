"""Descriptive paired baseline/shrinkage contrasts; no new policy selection."""
import numpy as np
from study import HERE,MODELS,BUDGETS,csvin,csvout

out=HERE/'fresh';g=np.load(out/'population_gain_curves.npz')['gains']
prior=np.load(HERE/'history/historical_curves.npz')['indices']
risk=np.load(out/'risk_arrays.npz')['population'];rows=csvin(out/'population_models.csv')
indices=np.random.default_rng(729002).integers(0,16,(20000,16))
contrasts=[];shrink=[]
for mi,model in enumerate(MODELS):
    raw=np.array([float(next(r['population_loss'] for r in rows if r['model']==model and int(r['pair'])==p)) for p in range(1,17)])
    random=np.array([float(next(r['random_gain'] for r in rows if r['model']==model and int(r['pair'])==p)) for p in range(1,17)])
    values={'default':raw-g[:,mi,prior[mi]],'pi':raw-g[:,mi,-1],
        'full':risk[:,mi,0,-1].mean(1),'random':raw-random,
        'population_DIAGNOSTIC':raw-g[:,mi].max(1)}
    for policy,reference in [('default','pi'),('full','pi'),('full','default'),('random','pi'),('full','population_DIAGNOSTIC')]:
        diff=values[reference]-values[policy];lo,hi=np.quantile(diff[indices].mean(1),[.025,.975])
        contrasts.append(dict(model=model,policy=policy,reference=reference,
            paired_improvement=diff.mean(),ci_low=lo,ci_high=hi,
            seed_wins=int((diff>1e-12).sum()),seed_losses=int((diff < -1e-12).sum()),seed_ties=int((abs(diff)<=1e-12).sum())))
    for bi,budget in enumerate(BUDGETS):
        diff=risk[:,mi,0,bi].mean(1)-risk[:,mi,1,bi].mean(1)
        lo,hi=np.quantile(diff[indices].mean(1),[.025,.975])
        shrink.append(dict(model=model,budget=budget,shrink16_improvement_over_none=diff.mean(),
            ci_low=lo,ci_high=hi,seed_wins=int((diff>1e-12).sum()),seed_losses=int((diff < -1e-12).sum()),
            seed_ties=int((abs(diff)<=1e-12).sum())))
csvout(out/'baseline_paired_contrasts.csv',contrasts);csvout(out/'shrinkage_paired_contrasts.csv',shrink)
print(contrasts)
