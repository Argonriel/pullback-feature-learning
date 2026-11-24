"""Render the completed independent confirmation without changing its analysis."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
with (HERE/'summary.csv').open() as f:
    summary={r['policy']:r for r in csv.DictReader(f) if r['outcome']=='expected_risk'}
with (HERE/'contrasts.csv').open() as f:
    contrasts=[r for r in csv.DictReader(f) if r['primary']=='True']
names=['raw','group_balanced','rank_balanced','rank_even','folded','confidence','confidence_radial','full']
labels=['Raw','Previous grouped balanced','New rank balanced','New rank even','Folded gate','Confidence + half-turn','Confidence + radius angles','Full half-turn']
colors=['#7A8696','#7A8696','#007F82','#3A9DA0','#7A8696','#C77720','#6C51A5','#7A8696']
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,4.6),gridspec_kw={'width_ratios':[1.4,1]})
y=np.arange(len(names))[::-1]
for i,name in enumerate(names):
    r=summary[name];mean=float(r['mean']);lo=float(r['low95']);hi=float(r['high95'])
    ax.errorbar(mean,y[i],xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color=colors[i],capsize=3,lw=1.6)
ax.set_yticks(y,labels);ax.set_xlabel('Expected logistic risk');ax.set_title('Absolute risk: 16 fresh PB models')
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
for j,r in enumerate(contrasts):
    delta=float(r['delta_risk'])*1000;lo=float(r['low97_5'])*1000;hi=float(r['high97_5'])*1000
    bx.errorbar(delta,1-j,xerr=np.array([[delta-lo],[hi-delta]]),fmt='o',color='#007F82',capsize=4,lw=1.8)
bx.axvline(0,color='#777777',lw=1,ls='--')
bx.set_yticks([1,0],['Balanced - confidence','Balanced - even'])
bx.set_xlabel('Paired logistic-risk difference × 1,000\nNegative favors balanced')
bx.set_title('Prespecified paired contrasts');bx.set_ylim(-.7,1.7);bx.grid(axis='x',alpha=.2)
for a in (ax,bx):
    a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
fig.suptitle('Richer exact-quota allocation improves its even control, but trails confidence',fontsize=12)
fig.text(.5,.01,'Left: 95% model-bootstrap intervals. Right: 97.5% intervals for two primary contrasts. All limited policies select exactly 64 of 256.',ha='center',fontsize=8.5)
fig.tight_layout(rect=(0,.07,1,.94))
fig.savefig(HERE/'confirmation.png',dpi=200)
fig.savefig(HERE/'confirmation.svg')
plt.close(fig)
