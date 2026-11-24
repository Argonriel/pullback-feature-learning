"""Summaries and figures for frozen rotation experiments; no policy fitting."""
import csv
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core
from rotation_selectors import deploy, rotate, phi
from experiment import csvout, dump, load_selection

HERE=Path(__file__).resolve().parent
MAIN=('raw','pi','random_alpha','energy_K8_global','fourier_K8_global','exact_global','exact_radial')


def read(path):
    return list(csv.DictReader(path.open(encoding='utf8')))


def ci(values,indices):
    v=np.asarray(values)
    return np.quantile(v[indices].mean(1),[.025,.975]).tolist()


def summarize(folder):
    rows=read(folder/'risks.csv')
    lookup={(int(r['pair']),r['model'],r['policy']):r for r in rows}
    pairs=sorted(set(int(r['pair']) for r in rows))
    indices=np.random.default_rng(619001).integers(0,len(pairs),(20000,len(pairs)))
    summary=[]; contrasts=[]; failures=[]
    for model in ('gd','pb','hybrid'):
        for policy in list(dict.fromkeys(r['policy'] for r in rows)):
            risk=np.array([float(lookup[p,model,policy]['risk']) for p in pairs])
            gain=np.array([float(lookup[p,model,policy]['gain']) for p in pairs])
            angular=np.array([float(lookup[p,model,policy]['angular']) for p in pairs])
            ref=np.array([float(lookup[p,model,'pi']['risk']) for p in pairs])
            prediction=np.array([float(lookup[p,model,policy]['predicted_gain']) for p in pairs])
            frac_samples=gain[indices].mean(1)/angular[indices].mean(1)
            summary.append(dict(model=model,policy=policy,n=len(pairs),risk=risk.mean(),
                risk_ci_low=ci(risk,indices)[0],risk_ci_high=ci(risk,indices)[1],gain=gain.mean(),
                angular_loss_removed=gain.mean()/angular.mean(),
                fraction_ci_low=np.quantile(frac_samples,.025),fraction_ci_high=np.quantile(frac_samples,.975),
                improvement_over_pi=(ref-risk).mean(),
                improvement_ci_low=ci(ref-risk,indices)[0],improvement_ci_high=ci(ref-risk,indices)[1],
                wins_over_pi=int((risk<ref-1e-12).sum()),
                predicted_gain=np.mean(prediction),mean_prediction_bias=np.mean(prediction-gain),
                prediction_rmse=np.sqrt(np.mean((prediction-gain)**2))))
        for p in pairs:
            exact=float(lookup[p,model,'exact_global']['risk'])
            for policy in ('energy_K8_global','fourier_K4_global','fourier_K8_global','fourier_K16_global','fourier_K64_global'):
                row=lookup[p,model,policy]
                failures.append(dict(pair=p,model=model,policy=policy,
                    excess_risk_over_exact=float(row['risk'])-exact,
                    gain_prediction=float(row['predicted_gain']),gain_actual=float(row['gain']),
                    gain_prediction_error=float(row['predicted_gain'])-float(row['gain'])))
        for policy,reference in [('exact_global','pi'),('exact_radial','exact_global'),
            ('fourier_K8_global','exact_global'),('energy_K8_global','exact_global'),
            ('exact_global','test_oracle_global_DIAGNOSTIC_ONLY')]:
            difference=np.array([float(lookup[p,model,reference]['risk'])-float(lookup[p,model,policy]['risk']) for p in pairs])
            contrasts.append(dict(model=model,policy=policy,reference=reference,
                improvement=difference.mean(),ci_low=ci(difference,indices)[0],ci_high=ci(difference,indices)[1],
                wins=int((difference>1e-12).sum()),ties=int((abs(difference)<=1e-12).sum()),n=len(pairs)))
    csvout(folder/'summary.csv',summary); csvout(folder/'paired_contrasts.csv',contrasts)
    csvout(folder/'proxy_failures.csv',failures)
    return summary


def benchmark(folder):
    bank=np.load(folder/'models.npz')
    rng=np.random.default_rng(618001)
    x=rng.normal(size=(64,256,2))
    rows=[]
    for model in ('gd','pb','hybrid'):
        for pair in (1,2,3,4):
            key=f'{pair}_{model}'; theta=bank[key]
            cal=load_selection(folder/(key+'_selector.npz'))
            for rep in range(8):
                for policy in rng.permutation(['raw','pi','random_alpha','fourier_K8_global','exact_global','exact_radial']):
                    start=perf_counter_ns()
                    for block in x:
                        if policy=='raw': core.representation(theta,block)
                        elif policy=='pi':
                            .5*(core.representation(theta,block)[:,0]+core.representation(theta,-block)[:,0])
                        elif policy=='random_alpha':
                            .5*(core.representation(theta,block)[:,0]+core.representation(theta,
                                rotate(block,rng.integers(0,128,len(block))*2*np.pi/128))[:,0])
                        else: deploy(theta,block,cal,policy)
                    if rep:
                        rows.append(dict(pair=pair,model=model,policy=policy,repeat=rep,
                            milliseconds_per_256=(perf_counter_ns()-start)/len(x)/1e6,
                            network_example_evaluations=256 if policy=='raw' else 512))
    csvout(folder/'deployment_timing.csv',rows)


def figures(folder,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),sharey=True)
    policies=['pi','random_alpha','energy_K8_global','fourier_K8_global','exact_global','exact_radial']
    labels=['Half-turn','Random','Energy K8','Curvature K8','Exact global','Exact radial']
    colors=['#9ba7b4','#c7ced6','#c88c55','#699db7','#226b85','#193b55']
    for ax,model,title in zip(axes,('gd','pb','hybrid'),('Gradient descent','Pullback','Pullback → GD')):
        q=[next(r for r in summary if r['model']==model and r['policy']==p) for p in policies]
        val=np.array([r['angular_loss_removed'] for r in q])*100
        err=np.array([[r['angular_loss_removed']-r['fraction_ci_low'],r['fraction_ci_high']-r['angular_loss_removed']] for r in q]).T*100
        ax.bar(np.arange(len(q)),val,color=colors,yerr=err,capsize=2)
        ax.set_xticks(np.arange(len(q)),labels,rotation=40,ha='right')
        ax.set_title(title); ax.set_ylim(0,100); ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Angular log loss removed (%)')
    fig.suptitle('Same two model calls; angles calibrated without labels (16 fresh pairs)',y=1.02)
    fig.tight_layout(); fig.savefig(folder/'rotation_results.png',dpi=200,bbox_inches='tight')
    fig.savefig(folder/'rotation_results.svg',bbox_inches='tight'); plt.close(fig)
    # Full curves: exact Jensen vs the Taylor approximation, including failures.
    curves=read(folder/'population_gain_curves.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.2))
    for ax,model in zip(axes,('gd','pb','hybrid')):
        pair=1
        cal=load_selection(folder/f'{pair}_{model}_selector.npz')
        pop=np.array([float(r['gain']) for r in curves if r['model']==model and int(r['pair'])==pair])
        ang=np.arange(1,65)*360/128
        for name,label,color in [('fourier_K8_global','Curvature K8','#699db7'),('exact_global','Exact calibration','#226b85')]:
            ax.plot(ang,cal.gain_curves[cal.names.index(name)],label=label,color=color)
        ax.plot(ang,pop,'--',label='Independent population',color='#282f35',lw=1.2)
        ax.set_title(model.upper()+', fresh pair 1'); ax.set_xlabel('Second-view rotation (degrees)')
    axes[0].set_ylabel('Predicted / actual risk reduction'); axes[-1].legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(folder/'calibration_curves.png',dpi=200); fig.savefig(folder/'calibration_curves.svg'); plt.close(fig)
    fail=read(folder/'proxy_failures.csv')
    fig,ax=plt.subplots(figsize=(4.6,3.5))
    for model,color in zip(('gd','pb','hybrid'),('#c88c55','#226b85','#193b55')):
        rr=[r for r in fail if r['model']==model and r['policy']=='fourier_K8_global']
        ax.scatter([float(r['gain_actual']) for r in rr],[float(r['gain_prediction']) for r in rr],label=model,color=color,s=22)
    lim=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([0,lim],[0,lim],':',color='gray')
    ax.set_xlabel('Actual selected-angle gain');ax.set_ylabel('Fourier K8 predicted gain');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(folder/'proxy_gain_calibration.png',dpi=200);fig.savefig(folder/'proxy_gain_calibration.svg');plt.close(fig)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        summarize(HERE/'exploration')
        folder=HERE/'fresh_results'; summary=summarize(folder)
        benchmark(folder); figures(folder,summary)
        print(json.dumps([{k:v for k,v in r.items() if k in ('model','policy','risk','angular_loss_removed','improvement_over_pi','wins_over_pi','mean_prediction_bias')} for r in summary if r['policy'] in MAIN],indent=2))
