"""Plot the completed fixed-network quality and measured timing results."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
with (HERE/'optimized'/'summary.csv').open() as f:rows=list(csv.DictReader(f))
widths=[64,256,1024];x=np.arange(3)
fig,(ax,bx)=plt.subplots(1,2,figsize=(10.8,4.2))
for name,label,color in [('raw','Raw','#7A8696'),('full_fused','Full half-turn','#262F3D'),
                         ('folded_fused','Folded, exact quota','#007F82'),
                         ('confidence_radial','Confidence + radius angles','#7654A7')]:
    means=[float(next(r['mean_expected_risk'] for r in rows if int(r['width'])==w and r['condition']=='resident' and r['implementation']=='optimized' and r['policy']==name)) for w in widths]
    ax.plot(x,means,'o-',label=label,color=color,lw=1.6)
ax.set_xticks(x,[str(w) for w in widths]);ax.set_xlabel('Hidden-layer width');ax.set_ylabel('Expected logistic risk')
ax.set_title('CUDA-scored prediction quality');ax.legend(frameon=False,fontsize=8)
for condition,label,color,offset in [('resident','GPU resident','#007F82',-.07),('transfer','GPU with transfers','#C77720',0),('cpu','One-thread CPU','#7654A7',.07)]:
    subset=[next(r for r in rows if int(r['width'])==w and r['condition']==condition and r['implementation']=='optimized' and r['policy']=='folded_fused') for w in widths]
    mean=np.array([float(r['mean_speedup']) for r in subset]);lo=np.array([float(r['minimum_speedup']) for r in subset]);hi=np.array([float(r['maximum_speedup']) for r in subset])
    bx.errorbar(x+offset,mean,yerr=np.vstack((mean-lo,hi-mean)),fmt='o-',color=color,label=label,capsize=3,lw=1.6)
bx.axhline(1,color='#777777',ls='--',lw=1);bx.set_ylim(0,1.7)
bx.set_xticks(x,[str(w) for w in widths]);bx.set_xlabel('Hidden-layer width');bx.set_ylabel('Full-reference time / folded-policy time')
bx.set_title('Folded policy: >1 is faster');bx.legend(frameon=False,fontsize=8)
for a in (ax,bx):
    a.spines['top'].set_visible(False);a.spines['right'].set_visible(False);a.grid(alpha=.18);a.set_axisbelow(True)
fig.suptitle('Fewer forward examples did not produce a GPU latency gain',fontsize=12)
fig.text(.5,.015,'Three trained models per width. Timing points: mean model ratios; bars: model range. Strongest equivalent full reference, including fused calls.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.93))
fig.savefig(HERE/'cost_quality.png',dpi=200);fig.savefig(HERE/'cost_quality.svg');plt.close(fig)
