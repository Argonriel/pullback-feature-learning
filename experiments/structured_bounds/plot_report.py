"""Print-size primary result figure; reads saved outputs and runs no experiment."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    with (HERE/'certificate_curves.csv').open(newline='',encoding='utf-8') as stream:
        rows=[r for r in csv.DictReader(stream)
              if int(r['n_batches'])==2048 and float(r['a'])==.01 and float(r['epsilon'])==0]
    rows.sort(key=lambda r:(r['policy']!='symmetric_jensen',int(r['pair'])))
    assert len(rows)==16
    plt.rcParams.update({'font.size':9.5,'axes.titlesize':10.5,'axes.labelsize':9.5,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(7.1,2.8))
    x=np.arange(16)
    for key,color,marker,offset,label in [
        ('known_lower','#087c80','o',-.19,'Known a'),
        ('estimated_lower','#cb8418','^',0,'Estimated a + CI'),
        ('old_lower','#b7484c','x',.19,'Old global TV')]:
        ax.scatter(x+offset,[float(row[key]) for row in rows],s=23,linewidths=1.1,
                   marker=marker,color=color,label=label,zorder=3)
    ax.axhline(0,color='#62676a',linewidth=.85,zorder=2)
    ax.axvline(7.5,color='#d1d5d8',linewidth=.65,zorder=1)
    ax.set_xticks(x,[f"{'I' if row['policy']=='symmetric_jensen' else 'F'}{row['pair']}" for row in rows])
    ax.set_yticks([-.15,-.10,-.05,0,.05],['−.15','−.10','−.05','0','.05'])
    ax.set(xlim=(-.65,15.65),ylim=(-.18,.09),ylabel='Loss-gain lower bound (nats)',
           xlabel='Fixed model-policy pair  ·  I: interpolation, F: folded',
           title='Small shift: 16/16 structured bounds are positive; old bound 0/16')
    ax.legend(loc='upper center',ncol=3,frameon=False,columnspacing=1.7,
              handletextpad=.45,borderaxespad=.2)
    fig.text(.99,.015,'a=.01, posterior TV=0; 2,048 batches; simultaneous 95% bounds',
             ha='right',va='bottom',fontsize=8.5,color='#485156')
    fig.subplots_adjust(left=.105,right=.99,bottom=.235,top=.865)
    fig.savefig(HERE/'structured_report.png',dpi=220)
    plt.close(fig)
    print(HERE/'structured_report.png')

if __name__=='__main__':main()
