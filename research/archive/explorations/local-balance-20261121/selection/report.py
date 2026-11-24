"""Post-freeze reporting of retained old development only; no policy fitting."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
    frozen=json.loads((ROOT/'FREEZE.json').read_text())
    for name,digest in frozen['file_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=list(csv.DictReader((ROOT/'development_v2'/'scores.csv').open()));policies=('rank_balanced','rank_even','rank_unbalanced');taus=(.25,1.,4.);means={};profiles=[];odd=[]
    for name in policies:
        means[name]={str(t):float(np.mean([float(r['centered_gain']) for r in rows if r['policy']==name and float(r['tau'])==t])) for t in taus}
    for basepath in sorted((ROOT/'development_v2'/'bases').glob('*.npz')):
        base=dict(np.load(basepath));table=dict(np.load(ROOT/'development_v2'/'actions'/f'{basepath.stem}_tau1.npz'));profiles.append(table['rate_profile']);freq=table['rank_balanced_frequency'];mix=table['rank_balanced_mix'];mass=mix*(freq[:,0]%2)+(1-mix)*(freq[:,1]%2);odd.append(float(np.mean(np.interp(base['cdf_cal_radii'],base['radii'],mass))))
    profiles=np.array(profiles);radii=base['radii'];output=dict(old_models=64,rows=len(rows),mean_centered_gain=means,selected_tau=1.,mean_balanced_odd_action_mass=float(np.mean(odd)),all_failures=0,interpretation='Development-only input-based centered logistic gain; not fresh confirmation, not original-posterior risk.',freeze_source_sha256=frozen['source_sha256'])
    (ROOT/'DEVELOPMENT_SUMMARY.json').write_text(json.dumps(output,indent=2))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');colors=['#1b6f82','#bf7e31','#7e668c'];labels=['Balanced at calibration anchors','Exact even-symmetry control','Unconstrained control']
    for name,color,label in zip(policies,colors,labels):axes[0].plot(taus,[means[name][str(t)] for t in taus],marker='o',label=label,color=color)
    axes[0].set_xscale('log',base=4);axes[0].set_xticks(taus,labels=['0.25','1 (chosen)','4']);axes[0].set_xlabel('Score width multiplier');axes[0].set_ylabel('Mean centered logistic gain (nats)');axes[0].set_title('64 old models: label-free choice');axes[0].legend(fontsize=8,loc='best')
    lo,mid,hi=np.quantile(profiles,[.1,.5,.9],axis=0);axes[1].fill_between(radii,lo,hi,color=colors[0],alpha=.18,label='10–90% across old models');axes[1].plot(radii,mid,color=colors[0],label='Median global-rank allocation');axes[1].axhline(.25,color='#777777',linestyle='--',label='Earlier group-of-four allocation');axes[1].set_xlim(0,2);axes[1].set_ylim(0,1.02);axes[1].set_xlabel('Radius');axes[1].set_ylabel('Conditional mean inclusion probability');axes[1].set_title('The new family reallocates across radii');axes[1].legend(fontsize=8,loc='best')
    fig.savefig(ROOT/'development.png',dpi=180);plt.close(fig);print(json.dumps(output,indent=2))

if __name__=='__main__':main()
