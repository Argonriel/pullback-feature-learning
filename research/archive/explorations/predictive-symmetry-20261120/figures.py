"""Figures from the frozen independent seed records; no model selection."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#24383C',
    'text.color':'#24383C','axes.edgecolor':'#96A8AA','savefig.facecolor':'white',
    'figure.facecolor':'white','axes.grid':True,'grid.alpha':.15})
TEAL='#087F8C';ORANGE='#C9672C';GRAY='#6B7B80';BLUE='#4568AC'
def read(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def prediction():
    rows=read(ROOT/'direction1'/'method_contrasts.csv')
    summary=read(ROOT/'direction1'/'contrast_summary.csv')
    metrics=['fit','nuisance','view_value'];titles=['Radial fitting excess','Angular nuisance excess','Value of one random view']
    fig,axs=plt.subplots(2,3,figsize=(12,7.8),layout='constrained')
    for col,(m,title) in enumerate(zip(metrics,titles)):
        rr=[r for r in rows if r['task']=='rings' and r['stage']=='trained' and r['experiment']=='single' and r['requested_drop']=='0.02' and r['metric']==m]
        actual=np.array([float(r['actual_pb_minus_gd']) for r in rr]);pred=np.array([float(r['clock_pb_minus_gd']) for r in rr])
        ax=axs[0,col];bounds=np.r_[actual,pred];lo,hi=bounds.min(),bounds.max();pad=(hi-lo)*.12
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color=GRAY,lw=1,ls='--')
        ax.scatter(actual,pred,c=TEAL,s=36,alpha=.85,edgecolors='white',linewidth=.5)
        ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),title=title,xlabel='Observed PB - GD change (nats)',ylabel='Pre-state prediction (nats)')
        ax.ticklabel_format(axis='both',style='sci',scilimits=(-2,2))
        ax.text(.04,.96,f'One step, drop 0.02\n{len(rr)}/16 pairs reached target\nSign: {np.sum(actual*pred>0)}/{len(rr)}',transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        ax=axs[1,col]
        for predictor,label,color,marker in [('drop_clock','Initial derivative',TEAL,'o'),('old_pooled','Old-seed pooled baseline',ORANGE,'s')]:
            ss=sorted([r for r in summary if r['task']=='rings' and r['stage']=='trained' and r['experiment']=='path' and r['metric']==m and r['predictor']==predictor],key=lambda r:int(r['horizon']))
            ax.plot([int(r['horizon']) for r in ss],[float(r['mae']) for r in ss],color=color,marker=marker,label=label)
        ax.set(xlabel='Updates of 0.005 actual loss drop',ylabel='MAE of PB - GD contrast (nats)',xticks=[1,2,4,8],title='Initial forecast loses accuracy')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        ax.text(.04,.96,'16/16 pairs at every horizon',transform=ax.transAxes,va='top',fontsize=9)
    axs[1,0].legend(loc='center left',fontsize=8,frameon=False)
    fig.suptitle('Same PB100 starting state: useful local forecasts, limited horizon',fontsize=15,fontweight='bold')
    save(fig,'loss_matched_prediction')

def views():
    rows=read(ROOT/'direction2'/'fresh_results'/'summary.csv')
    fig,axs=plt.subplots(1,3,figsize=(13.2,4.2),layout='constrained')
    a=np.linspace(0,np.pi,257);k=np.arange(1,9)
    im=axs[0].imshow(np.cos(k[:,None]*a[None,:]/2)**2,aspect='auto',extent=[0,180,8.5,.5],vmin=0,vmax=1,cmap='viridis')
    axs[0].set(title='A half-turn preserves even modes',xlabel='Second-view rotation (degrees)',ylabel='Angular Fourier mode k',xticks=[0,45,90,135,180],yticks=k)
    axs[0].grid(False);fig.colorbar(im,ax=axs[0],fraction=.035,pad=.02,label='Residual mode power')
    policies=['pi','random_alpha','fourier_K8_global','exact_global','exact_radial']
    labels=['180°','Random','Fourier K=8','Exact global','Exact by radius']
    pb={r['policy']:r for r in rows if r['model']=='pb'}
    vals=np.array([float(pb[p]['angular_loss_removed']) for p in policies]);lo=np.array([float(pb[p]['fraction_ci_low']) for p in policies]);hi=np.array([float(pb[p]['fraction_ci_high']) for p in policies])
    axs[1].barh(np.arange(5),vals,color=[GRAY,'#ACB7BB',BLUE,TEAL,'#175C56'],height=.65)
    axs[1].errorbar(vals,np.arange(5),xerr=[vals-lo,hi-vals],fmt='none',color='#27383C',capsize=3,lw=1)
    axs[1].set(yticks=np.arange(5),yticklabels=labels,xlim=(0,1),xlabel='Fraction of angular loss removed',title='Pullback: calibrated views improve')
    axs[1].invert_yaxis();axs[1].xaxis.set_major_formatter(PercentFormatter(1))
    for i,v in enumerate(vals):axs[1].text(.98,i,f'{v:.1%}',ha='right',va='center',fontsize=9)
    xs=np.arange(3)
    for p,color,offset,label in [('exact_global',TEAL,-.12,'Exact global'),('exact_radial',ORANGE,.12,'Exact by radius')]:
        ss=[next(r for r in rows if r['model']==m and r['policy']==p) for m in ['gd','pb','hybrid']]
        v=np.array([float(r['improvement_over_pi']) for r in ss]);low=np.array([float(r['improvement_ci_low']) for r in ss]);high=np.array([float(r['improvement_ci_high']) for r in ss])
        axs[2].errorbar(xs+offset,v,yerr=[v-low,high-v],fmt='o',color=color,capsize=4,label=label)
    axs[2].axhline(0,color=GRAY,lw=1)
    axs[2].set(xticks=xs,xticklabels=['GD','Pullback','PB then GD'],ylabel='Risk reduction beyond 180° (nats)',title='Paired independent-seed intervals')
    axs[2].legend(frameon=False,fontsize=9)
    fig.suptitle('16 fresh seed pairs, frozen unlabeled calibration, two calls per input',fontsize=14,fontweight='bold')
    save(fig,'fourier_rotation_results')

def gain_curves():
    curves=read(ROOT/'direction2'/'fresh_results'/'population_gain_curves.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,model in zip(axs,['gd','pb','hybrid']):
        exact=[];proxy=[];actual=[]
        for pair in range(1,17):
            d=np.load(ROOT/'direction2'/'fresh_results'/f'{pair}_{model}_selector.npz')
            names=list(d['names']);exact.append(d['gain_curves'][names.index('exact_global')]);proxy.append(d['gain_curves'][names.index('fourier_K8_global')])
            rr=sorted([r for r in curves if r['model']==model and int(r['pair'])==pair],key=lambda r:float(r['angle']));actual.append([float(r['gain']) for r in rr])
        angles=np.arange(1,65)*180/64
        ax.plot(angles,np.mean(actual,axis=0),color=TEAL,label='Independent population gain')
        ax.plot(angles,np.mean(exact,axis=0),color=ORANGE,ls='--',label='Exact unlabeled calibration')
        ax.plot(angles,np.mean(proxy,axis=0),color=BLUE,ls=':',label='Curvature-weighted Fourier proxy')
        ax.set(title={'gd':'Gradient descent','pb':'Pullback','hybrid':'Pullback then GD'}[model],xlabel='Second-view rotation (degrees)',ylabel='Mean gain (nats)',xticks=[0,45,90,135,180])
    axs[0].legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('Good angle rankings do not guarantee accurate gain magnitudes',fontweight='bold',fontsize=14)
    save(fig,'calibration_gain_curves')

if __name__=='__main__':prediction();views();gain_curves()
