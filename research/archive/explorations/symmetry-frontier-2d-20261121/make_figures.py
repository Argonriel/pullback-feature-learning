"""Scientific figures from frozen fresh outcomes; no policy tuning."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'validation/summary.csv').open()))
contrasts=list(csv.DictReader((ROOT/'validation/contrasts.csv').open()))
cohorts=['rings','waveform','two_invariants'];titles=['Noisy rings','One-invariant harmonic','Two-invariant harmonic']
colors=['#9AA9AD','#BC8B4B','#2B8C91','#163D58']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
for ax,c,title in zip(axes,cohorts,titles):
    names=['raw','folded','symmetric_jensen','full']
    y=[float(next(r for r in rows if r['cohort']==c and r['policy']==name)['expected_loss']) for name in names]
    ax.bar(range(4),y,color=colors,width=.68)
    ax.set_xticks(range(4),['Raw\n1.00','Strong hist.\n1.25','Grid\n1.25','Full\n2.00'])
    for j,v in enumerate(y):ax.text(j,v+.012*max(y),f'{v:.4f}',ha='center',va='bottom',fontsize=9)
    ax.set_ylim(0,max(y)*1.17);ax.set_title(title);ax.set_xlabel('Policy / forward examples per input')
axes[0].set_ylabel('Expected logistic loss (lower is better)')
fig.savefig(ROOT/'validation/fresh_loss.png',dpi=190);fig.savefig(ROOT/'validation/fresh_loss.svg');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(11,2.6),layout='constrained')
for ax,c,title in zip(axes,cohorts,titles):
    r=next(r for r in contrasts if r['cohort']==c and r['primary']=='True')
    mean,lo,hi=[float(r[k]) for k in ('mean','lower','upper')]
    ax.axvline(0,color='#77888D',ls='--',lw=1)
    ax.errorbar(mean,0,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color='#117D84')
    ax.set_yticks([]);ax.set_ylim(-1,1);ax.set_title(title)
    ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3));ax.set_xlabel('Grid minus strong histogram loss')
    ax.text(.04,.88,f"{r['wins']}/16 modelwise wins",transform=ax.transAxes)
fig.savefig(ROOT/'validation/primary_contrasts.png',dpi=190);fig.savefig(ROOT/'validation/primary_contrasts.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(10.5,2.9),layout='constrained')
for j,(name,label,color) in enumerate([('folded_legacy','Legacy histogram','#B6C0C2'),('folded','Strong histogram','#BC8B4B'),('grid_nearest','Nearest grid','#48788F'),('symmetric_jensen','Interpolated grid','#087F83')]):
    regret=[100*(1-float(next(r for r in rows if r['cohort']==c and r['policy']==name)['oracle_fraction'])) for c in cohorts]
    ax.bar(np.arange(3)+(j-1.5)*.19,regret,width=.18,label=label,color=color)
ax.set_xticks(range(3),titles);ax.set_yscale('log');ax.set_ylabel('Mean missed oracle gain (%)\nlog scale; lower is better')
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.22),frameon=False)
fig.savefig(ROOT/'validation/oracle_headroom.png',dpi=190);fig.savefig(ROOT/'validation/oracle_headroom.svg');plt.close(fig)
