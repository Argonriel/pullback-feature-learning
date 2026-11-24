"""Publication-style paired contrasts from the frozen independent cohort."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
with (ROOT/'validation/contrasts.csv').open(newline='',encoding='utf8') as f:
    data={r['baseline']:r for r in csv.DictReader(f) if r['policy']=='rank_balanced' and r['outcome']=='expected_risk'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'width_ratios':[1.15,1]})
sets=[(['group_balanced','confidence','confidence_radial'],['Previous grouped rule','Global confidence','Confidence + radius angle']),
      (['rank_even','folded'],['New exact-even control','Legacy folded gate'])]
for ax,(names,labels) in zip(axes,sets):
    for i,name in enumerate(names):
        row=data[name];primary=row['primary']=='True';scale=1000
        mean=float(row['delta_risk'])*scale
        lo=float(row['low97_5'] if primary else row['low95'])*scale
        hi=float(row['high97_5'] if primary else row['high95'])*scale
        color='#147D83' if mean<0 else '#B76442'
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=4,linewidth=2,markersize=7)
    ax.axvline(0,color='#6C7A80',linestyle='--',linewidth=1)
    ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    ax.set_xlabel('Risk difference (× 10⁻³ nats/input)')
axes[0].set_title('Main performance comparisons',loc='left',weight='bold')
axes[1].set_title('Closer invariant controls (zoomed scale)',loc='left',weight='bold')
fig.suptitle('New rank-balanced rule minus comparator: negative favors the new rule',fontsize=12,y=1.02)
fig.tight_layout(w_pad=2)
fig.savefig(ROOT/'paired_contrasts.png',dpi=200,bbox_inches='tight')
fig.savefig(ROOT/'paired_contrasts.pdf',bbox_inches='tight')
