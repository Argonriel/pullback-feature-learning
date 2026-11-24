"""Print-sized overview from committed results only; no experiment rerun."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,ci
from math_utils import LAWS

def main():
    curves=ci(HERE/'certificate_curves.csv')
    rows=[r for r in curves if int(r['n_batches'])==2048]
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.1,2.8),gridspec_kw={'width_ratios':[1.1,1.]})
    positions=np.array([0,1,2,3,4,5,6.5])
    counts=[]
    for law in LAWS:
        values=[float(r['estimated_lower']) for r in rows if r['law']==law['name'] and r['scenario']=='e0']
        counts.append(sum(v>0 for v in values))
    ax.bar(positions[:6],counts[:6],color='#087c80',width=.68)
    ax.bar(positions[6:],counts[6:],color='#f4d6b6',edgecolor='#9a5e25',hatch='///',width=.68)
    ax.axvline(5.75,color='#b69b81',ls='--',lw=.8)
    for xx,count in zip(positions,counts):ax.text(xx,count+.3,str(count),ha='center',va='bottom',fontsize=9)
    ax.set_xticks(positions,['Unif.','First','Third','Mixed','Resid.','Strong','Invalid'],rotation=50,ha='right')
    ax.set_yticks([0,4,8,12,16]);ax.set_ylim(0,19)
    ax.set(title='Assumed posterior defect = 0',ylabel='Positive values / 16',xlabel='Last bar: invalid formula, no certificate')
    audits=[('covered_third','label0'),('covered_third','label1'),('covered_residual','label0'),('covered_residual','label1')]
    for k,(law,scenario) in enumerate(audits):
        values=np.array([float(r['estimated_lower']) for r in rows if r['law']==law and r['scenario']==scenario])
        values.sort();count=int(np.sum(values>0))
        bx.scatter(k+np.linspace(-.15,.15,len(values)),values,s=12,color=np.where(values>0,'#765395','#b7484c'),zorder=3)
        bx.plot([k-.23,k+.23],[values.mean(),values.mean()],color='#222',lw=1.2,zorder=4)
        bx.text(k,1.01,f'{count}/16',transform=bx.get_xaxis_transform(),ha='center',va='bottom',fontsize=9)
    bx.axhline(0,color='#777',lw=.8);bx.set_xlim(-.5,3.5)
    bx.set_xticks(range(4),['Third\n0','Third\n.1','Residual\n0','Residual\n.1'])
    bx.set(title='Independent labels + coefficients',ylabel='Gain lower bound',xlabel='Input law / true cell defect')
    bx.set_title('Independent labels + coefficients',pad=22)
    fig.subplots_adjust(left=.08,right=.99,bottom=.29,top=.78,wspace=.38)
    fig.savefig(HERE/'residual_overview.png',dpi=240)
    fig.savefig(HERE/'residual_overview.svg')
    plt.close(fig)
    print('Wrote residual_overview.png and residual_overview.svg')

if __name__=='__main__':main()
