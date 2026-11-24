"""Summarize deterministic bounds and retain development/confirmation separation."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def read(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def main():
    output={};rows=[]
    for cohort in ['old','fresh']:
        data=read(ROOT/cohort/'summary.csv');control=read(ROOT/cohort/'parity_control.csv');checks=json.loads((ROOT/cohort/'numerical_checks.json').read_text())
        vector=lambda k:np.array([float(r[k]) for r in data])
        old=vector('old_integrated_bound');new=vector('integrated_bound');observed=vector('observed_integrated_abs_b');odd=np.array([float(r['global_odd_integrated_bound']) for r in control])
        result=dict(n_models=len(data),mean_old_integrated_bound=float(old.mean()),mean_global_odd_integrated_bound=float(odd.mean()),mean_local_integrated_bound=float(new.mean()),mean_observed_integrated_abs_b=float(observed.mean()),ratio_of_local_to_old_means=float(new.mean()/old.mean()),old_to_local_mean_factor=float(old.mean()/new.mean()),odd_global_to_local_mean_factor=float(odd.mean()/new.mean()),local_to_observed_mean_factor=float(new.mean()/observed.mean()),primary_target_pass=bool(new.mean()<=.01*old.mean()),all_models_tighter_than_old=bool(np.all(new<old)),all_models_tighter_than_global_odd=bool(np.all(new<odd)),mean_uniform_local_bound=float(vector('uniform_bound').mean()),max_uniform_local_bound=float(vector('uniform_bound').max()),mean_angular_integral=float(vector('angular_integral').mean()),mean_radial_integral=float(vector('radial_integral').mean()),mean_endpoint_integral=float(vector('endpoint_integral').mean()),mean_setup_seconds=float((vector('point_seconds')+vector('interval_seconds')).mean()),interval_jet_propagations=int(vector('interval_jet_propagations').sum()),ordinary_point_forwards=int(vector('exact_point_forwards').sum()),diagnostic_moment_forwards=int(vector('diagnostic_moment_point_forwards').sum()+vector('spotcheck_moment_point_forwards').sum()),maximum_moment_bound_violation=float(vector('maximum_moment_bound_violation').max()),finite_difference_diagnostic_pass_count=sum(r['finite_difference_diagnostic_pass']=='True' for r in data),maximum_jet_enclosure_violation=max(r['maximum_jet_enclosure_violation'] for r in checks),independent_confirmation=cohort=='fresh')
        result.update(derivative_spotcheck_interval_jets_per_model=128,derivative_spotcheck_analytic_point_jets_per_model=512,finite_difference_ordinary_point_forwards_per_model=3200)
        output[cohort]=result
        for d,c in zip(data,control):rows.append(dict(**d,**{k:v for k,v in c.items() if k!='seed'}))
    (ROOT/'analysis_summary.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    with (ROOT/'combined_summary.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    data=[r for r in rows if r['cohort']=='fresh'];x=np.arange(1,17)
    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    for key,label,color in [('old_integrated_bound','Previous global bound','#999999'),('global_odd_integrated_bound','Global bound, odd harmonics only','#a8794a'),('integrated_bound','Local deterministic bound','#087f72'),('observed_integrated_abs_b','Numerical actual moment','#3768a6')]:
        ax[0].plot(x,[float(r[key]) for r in data],marker='o',ms=3,label=label,color=color)
    ax[0].set_yscale('log');ax[0].set_xlabel('Fresh model');ax[0].set_ylabel('Input-mass integrated absolute imbalance');ax[0].set_xticks([1,4,8,12,16]);ax[0].grid(alpha=.2);ax[0].legend(frameon=False,fontsize=8)
    d=np.load(ROOT/'fresh/1_diagnostics.npz');b=np.load(ROOT/'fresh/1_local.npz');r=d['r'];local=np.maximum(abs(d['pointwise_lower']),abs(d['pointwise_upper']));old=d['old_angular_bound']+d['old_radial_bound'];obs=abs(d['observed_moment'])
    for y,label,color in [(old,'Previous global bound','#999999'),(local,'Local signed-envelope bound','#087f72'),(obs,'Numerical actual moment','#3768a6')]:
        ax[1].plot(r,np.where(y>1e-16,y,np.nan),color=color,label=label,lw=1.3)
    ax[1].set_yscale('log');ax[1].set_ylim(1e-8,max(old)*1.5);ax[1].set_xlabel('Continuous radius');ax[1].set_ylabel('Absolute conditional imbalance');ax[1].set_title('Prespecified example: fresh model 1');ax[1].grid(alpha=.2);ax[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Local derivative enclosures make conditional-balance bounds informative',fontsize=13)
    fig.text(.01,.015,'16 independent fresh models. Numerical moments are diagnostics, not the guarantee. Zero values are omitted on logarithmic axes.\nReal-arithmetic theorem; float64 implementation is numerically checked, not formally directed-rounded.',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(ROOT/'local_bounds.png',dpi=180);fig.savefig(ROOT/'local_bounds.pdf');plt.close(fig)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
