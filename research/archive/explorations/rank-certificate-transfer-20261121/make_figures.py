"""Publication-style figures from retained model-level summaries."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def rows(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':'#16333C','text.color':'#16333C',
    'xtick.color':'#52656B','ytick.color':'#52656B','savefig.facecolor':'white'})

def quality():
    data=rows(ROOT/'validation/summary.csv')
    lookup={(r['cohort'],r['method'],r['policy'],r['outcome']):float(r['mean']) for r in data}
    names=['rank_balanced','rank_even','folded','confidence','confidence_radial']
    labels=['Learned balance','Even rank','Folded symmetric','Confidence','Confidence + angle']
    colors=['#0F6C73','#73A9A9','#254C68','#D98143','#B05232']
    fig,axes=plt.subplots(2,2,figsize=(10.8,6.1),layout='constrained')
    for col,cohort in enumerate(('rings','transfer')):
        for row,outcome in enumerate(('expected_risk','expected_error')):
            ax=axes[row,col];base=lookup[cohort,'pb','raw',outcome]
            values=[100*(1-lookup[cohort,'pb',name,outcome]/base) for name in names]
            y=np.arange(len(names));ax.barh(y,values,color=colors,height=.68)
            ax.set_yticks(y,labels if col==0 else []);ax.invert_yaxis();ax.axvline(0,color='#52656B',lw=.8)
            ax.set_xlim(0,max(values)*1.22)
            for yy,value in zip(y,values):ax.text(value+.25,yy,f'{value:.1f}%',va='center',fontsize=9)
            ax.set_xlabel('Relative loss reduction (%)' if row==0 else 'Relative classification-error reduction (%)')
            ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            if row==0:ax.set_title('Original rings' if cohort=='rings' else '4D harmonic amplitude',fontweight='bold')
    fig.suptitle('Same extra-view budget; improvement over the same raw model',fontsize=13,fontweight='bold')
    fig.savefig(ROOT/'quality_comparison.png',dpi=180);fig.savefig(ROOT/'quality_comparison.pdf');plt.close(fig)

def certificates():
    paths=[ROOT/'root_audit/certificates/per_model.csv',ROOT/'root_audit/certificates_empirical/per_model.csv']
    if not all(p.exists() for p in paths):return
    data=[rows(p) for p in paths]
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.6),sharex=True,layout='constrained')
    for ax,cohort in zip(axes,('rings','transfer')):
        for group,color,label,offset in zip(data,('#0F6C73','#D98143'),('Known radial law','Independent empirical law'),(-.13,.13)):
            v=sorted([r for r in group if r['cohort']==cohort],key=lambda r:int(r['seed']))
            ax.scatter(np.arange(1,17)+offset,[float(r['gain_lower']) for r in v],s=24,c=color,label=label)
        ax.axhline(0,c='#A33F3F',ls='--',lw=1);ax.set_xticks([1,4,8,12,16]);ax.set_xlabel('Independent trained-model seed')
        ax.set_title('Original rings' if cohort=='rings' else '4D harmonic amplitude',fontweight='bold');ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Conservative gain lower bound (nats/input)');axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('All-posterior certificates: each color is a separate 95% family',fontsize=13,fontweight='bold')
    fig.savefig(ROOT/'certificate_results.png',dpi=180);fig.savefig(ROOT/'certificate_results.pdf');plt.close(fig)

if __name__=='__main__':quality();certificates()
