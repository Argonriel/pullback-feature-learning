"""Why a squared-spectrum proxy can rank well but mispredict log-loss gain."""
from pathlib import Path
import numpy as np
from experiment import load_selection,csvout
from rotation_selectors import phi

HERE=Path(__file__).resolve().parent
rows=[]
for path in sorted((HERE/'fresh_results').glob('*_selector.npz')):
    pair,model=path.stem.replace('_selector','').split('_')
    cal=load_selection(path);h=cal.h;mean=h.mean(1);delta=h-mean[:,None]
    sig=np.exp(-np.logaddexp(0,-mean));q=sig*(1-sig)
    energy=cal.power.mean(0);weighted=(q[:,None]*cal.power).mean(0)
    angular=float(np.mean(phi(h)-phi(mean[:,None])))
    exactpi=float(np.mean(phi(h)-phi(.5*(h+np.roll(h,64,1)))))
    rows.append(dict(pair=int(pair),model=model,
        odd_fraction_unweighted=energy[::2].sum()/energy.sum(),
        odd_fraction_curvature_weighted=weighted[::2].sum()/weighted.sum(),
        exact_pi_fraction_calibration=exactpi/angular,
        calibrated_angular_log_loss=angular,
        quadratic_angular_approximation=.5*weighted.sum(),
        squared_logit_tail_fraction_after_K8=energy[8:].sum()/energy.sum(),
        curvature_weighted_tail_fraction_after_K8=weighted[8:].sum()/weighted.sum(),
        curvature_weighted_K8_tail_gain_bound=.5*weighted[8:].sum(),
        angular_logit_RMS=float(np.sqrt(np.mean(delta**2))),
        worst_case_cubic_gain_error_bound=float(np.mean(abs(delta)**3)/(18*np.sqrt(3)))))
csvout(HERE/'fresh_results/spectrum_diagnostics.csv',rows)
summary=[]
for model in ('gd','pb','hybrid'):
    q=[r for r in rows if r['model']==model]
    summary.append(dict(model=model,**{k:float(np.mean([r[k] for r in q])) for k in rows[0] if k not in ('pair','model')}))
csvout(HERE/'fresh_results/spectrum_diagnostics_summary.csv',summary)
print(summary)
