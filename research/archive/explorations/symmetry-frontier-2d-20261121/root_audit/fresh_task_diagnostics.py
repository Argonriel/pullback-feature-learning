"""Descriptive two-invariant reliability diagnostics; never fits a policy."""
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
NAMES=('raw','folded','symmetric_jensen','full')
COLORS={'raw':'#af4c35','folded':'#c28b26','symmetric_jensen':'#176b94','full':'#348054'}
LABELS={'raw':'Raw (1 call)','folded':'Folded (1.25 calls)','symmetric_jensen':'Interpolated (1.25 calls)','full':'Full half-turn (2 calls)'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_write(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
def now():return datetime.now(timezone.utc).isoformat()

def run():
    complete=ROOT/'validation/score_complete.json';assert complete.exists()
    out=HERE/'fresh_task_diagnostics';out.mkdir(exist_ok=False)
    spec=dict(timestamp_utc=now(),purpose='post-confirmation descriptive task-quality diagnosis; no fitting or retuning',
      status='specified after parent released aggregate NLL/error outcomes; before this script read saved per-input predictions',
      policies=NAMES,models=list(range(1,17)),probability_bin_edges=np.linspace(0,1,11).tolist(),
      quotient_bins_per_axis=4,metrics='conditional logloss, conditional classification error, expected Brier loss, squared posterior error, fixed-bin oracle-posterior ECE, top-label confidence and accuracy; all16 retained',
      no_temperature_fit=True,no_new_model_calls=True,no_new_inferential_claim=True,
      script_sha256=sha(Path(__file__)),score_complete_sha256=sha(complete))
    (out/'PROTOCOL.json').write_text(json.dumps(spec,indent=2)+'\n')
    commit=json.loads(complete.read_text());rows=[];bins=[];cells=[];input_hashes={}
    reliability={name:np.zeros((10,4)) for name in NAMES}
    for pair in range(1,17):
        path=ROOT/f'validation/predictions/two_invariants_{pair}.npz'
        key=path.relative_to(ROOT/'validation').as_posix();assert sha(path)==commit['manifest'][key]
        input_hashes[key]=sha(path);data=np.load(path);p=data['p'];y=(data['y']+1)/2;x=data['x']
        r=np.stack([np.linalg.norm(x[:,:2],axis=1),np.linalg.norm(x[:,2:],axis=1)],axis=1)
        cell=np.minimum(np.maximum(((r-.4)*4).astype(int),0),3);cell=4*cell[:,0]+cell[:,1]
        bayes=-p*np.log(p)-(1-p)*np.log1p(-p)
        for name in NAMES:
            idx=list(data['names']).index(name);h=data['outputs'][idx];q=np.exp(-np.logaddexp(0,-h))
            nll=np.logaddexp(0,h)-p*h;error=np.where(h>=0,1-p,p)
            brier=(q-p)**2+p*(1-p);prob_bin=np.minimum((q*10).astype(int),9)
            total_cal=0.;predconf=np.maximum(q,1-q);acc=1-error
            for b in range(10):
                mask=prob_bin==b;n=int(mask.sum())
                if n:
                    meanq=float(q[mask].mean());meanp=float(p[mask].mean());meany=float(y[mask].mean())
                    total_cal+=n*abs(meanq-meanp)
                    reliability[name][b]+=np.array([n,q[mask].sum(),p[mask].sum(),y[mask].sum()])
                    bins.append(dict(pair=pair,policy=name,bin=b,left=b/10,right=(b+1)/10,count=n,predicted_probability=meanq,
                                     true_probability=meanp,sampled_positive_fraction=meany,calibration_gap=meanq-meanp))
                else:bins.append(dict(pair=pair,policy=name,bin=b,left=b/10,right=(b+1)/10,count=0,predicted_probability='',true_probability='',sampled_positive_fraction='',calibration_gap=''))
            row=dict(pair=pair,policy=name,n=len(p),expected_nll=float(nll.mean()),bayes_nll=float(bayes.mean()),
              excess_nll=float((nll-bayes).mean()),expected_error=float(error.mean()),bayes_error=float(np.minimum(p,1-p).mean()),
              expected_brier=float(brier.mean()),bayes_brier=float(np.mean(p*(1-p))),squared_posterior_error=float(np.mean((q-p)**2)),
              posterior_binned_ece=total_cal/len(p),mean_top_label_confidence=float(predconf.mean()),expected_accuracy=float(acc.mean()),
              overconfidence_gap=float((predconf-acc).mean()),sampled_nll=float(np.mean(np.logaddexp(0,h)-y*h)),
              sampled_accuracy=float(np.mean((h>=0)==(y==1))),mean_absolute_logit=float(np.mean(abs(h))))
            rows.append(row)
            for c in range(16):
                mask=cell==c
                cells.append(dict(pair=pair,policy=name,axis1_bin=c//4,axis2_bin=c%4,count=int(mask.sum()),
                    expected_nll=float(nll[mask].mean()),expected_error=float(error[mask].mean()),posterior_mean=float(p[mask].mean()),
                    predicted_probability_mean=float(q[mask].mean())))
    csv_write(out/'per_model.csv',rows);csv_write(out/'reliability_bins.csv',bins);csv_write(out/'quotient_cells.csv',cells)
    summaries=[]
    numeric=[key for key in rows[0] if key not in ('pair','policy','n')]
    for name in NAMES:
        selected=[r for r in rows if r['policy']==name]
        result=dict(policy=name,models=len(selected))
        for metric in numeric:
            values=np.array([r[metric] for r in selected]);result[metric+'_mean']=float(values.mean())
            result[metric+'_min']=float(values.min());result[metric+'_max']=float(values.max())
        summaries.append(result)
    csv_write(out/'summary.csv',summaries)
    pooled=[]
    for name in NAMES:
        for b,(n,sq,sp,sy) in enumerate(reliability[name]):
            pooled.append(dict(policy=name,bin=b,count=int(n),predicted_probability=sq/n,true_probability=sp/n,sampled_positive_fraction=sy/n))
    csv_write(out/'pooled_reliability.csv',pooled)
    fig,axs=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    for name in NAMES:
        v=reliability[name];valid=v[:,0]>0
        axs[0].plot(v[valid,1]/v[valid,0],v[valid,2]/v[valid,0],'-o',ms=3,color=COLORS[name],label=LABELS[name])
    axs[0].plot([0,1],[0,1],'--',color='0.5',lw=1);axs[0].set(xlabel='Mean predicted probability',ylabel='Mean known posterior',title='Reliability: fixed probability bins',xlim=(0,1),ylim=(0,1));axs[0].legend(fontsize=7)
    for j,(metric,title) in enumerate([('expected_nll','Expected log loss by model'),('expected_error','Classification error by model')],start=1):
        for pair in range(1,17):
            vals=[next(r[metric] for r in rows if r['pair']==pair and r['policy']==name) for name in NAMES]
            axs[j].plot(range(4),vals,color='0.8',lw=.7,zorder=1)
        axs[j].plot(range(4),[s[metric+'_mean'] for s in summaries],'o-',color='#174b78',lw=2,label='16-model mean')
        bayes_metric='bayes_nll_mean' if metric=='expected_nll' else 'bayes_error_mean'
        axs[j].axhline(summaries[0][bayes_metric],color='#348054',ls=':',lw=1.5,label='Bayes')
        axs[j].axhline(np.log(2) if metric=='expected_nll' else .5,color='0.5',ls='--',lw=1,label='Constant predictor')
        axs[j].set(xticks=range(4),xticklabels=['Raw','Folded','Interpolated','Full'],title=title)
        axs[j].tick_params(axis='x',rotation=25);axs[j].grid(alpha=.15)
    axs[1].legend(fontsize=7)
    fig.savefig(out/'fresh_task_reliability.png',dpi=180);fig.savefig(out/'fresh_task_reliability.pdf');plt.close(fig)
    report='''# Fresh two-invariant task diagnostics

All 16 prespecified fresh models are included. This is a descriptive post-confirmation analysis of saved predictions. The parent had already released aggregate log-loss and classification-error results when these reliability diagnostics were specified. No model, temperature, threshold, policy, or calibration parameter was fit, and no new model calls were made. The original model and policy freezes are untouched.

| Policy | Calls/input | Expected log loss | Expected classification error | Expected Brier loss | Mean confidence | Expected accuracy | Confidence minus accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for name,s in zip(NAMES,summaries):
        calls=1 if name=='raw' else 2 if name=='full' else 1.25
        report+=f"| {name} | {calls:g} | {s['expected_nll_mean']:.6f} | {s['expected_error_mean']:.6f} | {s['expected_brier_mean']:.6f} | {s['mean_top_label_confidence_mean']:.6f} | {s['expected_accuracy_mean']:.6f} | {s['overconfidence_gap_mean']:.6f} |\n"
    raw=summaries[0];new=summaries[2];full=summaries[3]
    report+=f'''
The raw model learns real class signal: its mean expected accuracy is {raw['expected_accuracy_mean']:.2%}, compared with 50% for the constant classifier. Its mean log loss is {raw['expected_nll_mean']:.6f}, compared with log(2)=0.693147; learning useful decisions is therefore not evidence of well-calibrated probabilities. Its mean top-label confidence is {raw['mean_top_label_confidence_mean']:.2%}, exceeding expected accuracy by {raw['overconfidence_gap_mean']:.2%} (percentage-point difference). The fixed-bin reliability curves independently show the direction of probability calibration errors rather than inferring calibration solely from log loss.

Interpolation reduces mean expected log loss to {new['expected_nll_mean']:.6f} and classification error to {new['expected_error_mean']:.6f}. The full half-turn reference reaches {full['expected_nll_mean']:.6f} and {full['expected_error_mean']:.6f}, at twice the raw call count. These policies still remain far from the task's Bayes log loss 0.4929065 and Bayes classification error 0.2265627. The Brier excess, equal to mean squared difference between predicted and true probabilities, is {raw['squared_posterior_error_mean']:.6f} for raw, {new['squared_posterior_error_mean']:.6f} for interpolation, and {full['squared_posterior_error_mean']:.6f} for full averaging. That excess mixes probability calibration and lost predictive information; it is not a pure calibration statistic.

The fixed-bin posterior ECE averages are {raw['posterior_binned_ece_mean']:.6f}, {new['posterior_binned_ece_mean']:.6f}, and {full['posterior_binned_ece_mean']:.6f} respectively. This is the weighted absolute binwise difference between predicted probability and the known posterior, computed separately per model in ten fixed equal-width bins. It removes sampled-label noise but remains a coarse binned diagnostic, not an exact conditional-calibration norm. The plot pools all equal-size test samples for readability; seed-specific bins and counts are retained. No confidence interval or new superiority claim is inferred from these descriptive summaries.

A grounded next direction is to improve training regularization or probability calibration on a separate development/validation split, then freeze any change and repeat the same independent comparison. The current pilot already showed that lower training loss at larger rate or more steps increased population log loss. That observation and the fresh overconfidence diagnostics support investigating calibration and generalization, rather than claiming the task is solved or applying a correction fitted on this test set. The persistent Bayes classification gap also shows that a probability-only correction would not by itself solve all task error.

The data are synthetic, bounded, and phase-coupled; the quotient genuinely has two coordinates on this support. This is not unrestricted ambient-four-dimensional or real-data validation. The fitted model sees raw coordinates. Bayes probabilities enter these diagnostics only as evaluator information, never as deployment inputs.

Artifacts are in `root_audit/fresh_task_diagnostics/`: `PROTOCOL.json`, all 64 model/policy metric rows, 640 reliability-bin rows including empty bins, 1,024 two-dimensional quotient-cell rows, pooled curves, summary table, and PNG/PDF figures. All 16 saved prediction files were verified against the immutable main scoring manifest. Reproduce into a new output folder with `python root_audit/fresh_task_diagnostics.py`; the script refuses to overwrite its existing directory. It performs no model evaluations.
'''
    (ROOT/'task2d/FRESH_DIAGNOSTICS.md').write_text(report)
    (out/'COMPLETE.json').write_text(json.dumps(dict(timestamp_utc=now(),models=16,policies=4,rows=len(rows),bin_rows=len(bins),cell_rows=len(cells),additional_model_forwards=0,input_hashes=input_hashes),indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':run()
