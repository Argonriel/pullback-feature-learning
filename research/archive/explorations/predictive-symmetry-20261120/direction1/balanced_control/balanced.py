"""Frozen exactly balanced antithetic training sensitivity on an independent seed cohort."""
import sys,json,argparse,hashlib,csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import study as s
s.CONFIG['new_synthetic_train_base']=27760100
s.CONFIG['new_synthetic_init_base']=27761100
CONFIG=dict(seeds=list(range(1,17)),tasks=['radial','angular'],stage='initial',drops=[1e-5,.02],train_seed_base=27760100,init_seed_base=27761100,fd_parameter_norms=[.001,.0005],fd_selected='smaller_step_no_adaptation',root_cap=4.,old_baseline='original_iid_old6_out_of_design',training_sampling='128_antithetic_pairs_exact_label_balance',bootstrap_seed=27762100)
def balanced_dataset(task,seed,cohort):
    assert cohort=='new' and task in ('radial','angular')
    ds=27760100+seed;ini=27761100+seed
    rng=np.random.default_rng(ds);u0=rng.random(128);a0=rng.uniform(0,2*np.pi,128);q0=rng.random(128)
    u=np.concatenate([u0,(u0+.5)%1]);a=np.concatenate([a0,(a0+np.pi/2)%(2*np.pi)]);q=np.concatenate([q0,1-q0])
    radius=.3+1.2*u;x=np.column_stack([radius*np.cos(a),radius*np.sin(a)])
    posterior=s.sigmoid(3*np.cos(2*np.pi*u) if task=='radial' else 3*np.cos(2*a))
    y=np.where(q<posterior,1.,-1.)
    assert np.array_equal(y[128:],-y[:128]) and np.sum(y==1)==128
    return x,y,s.core.initialize_parameters(ini),ds,ini
s.dataset=balanced_dataset

def freeze():
    if (ROOT/'freeze.json').exists():raise FileExistsError('Already frozen')
    protocol='# Frozen exact finite-label balance sensitivity\n\nPurpose: prior matched tasks had identical population class probability but different realized finite training class counts. This control exactly matches finite class fraction while preserving each observation marginally under the same input/posterior laws. It does not replace any previous result.\n\nUse sixteen new seed pairs: training27760101..27760116 and initialization27761101..27761116. Draw128 iid triples(u,phi,q), each uniform on[0,1]×[0,2pi]×[0,1]. Add the companion((u+.5) mod1,(phi+pi/2) mod2pi,1-q). Both posterior logits3cos(2piu) and3cos(2phi) change sign under this companion, so their Bernoulli labels are exactly opposite within each pair. Each task has exactly128 positive and128 negative training labels. Inputs and initial parameters are exactly shared across tasks. Each point has the correct population marginal; the256 observations are correlated within128 antithetic pairs, and cannot be treated as256 independent observations.\n\nInitial state only. Same GD/PB directions, damping.1, root cap4, targets1e-5 and.02. Same predeclared first-order and second-order loss-clock formulas from the curvature follow-up, with central second derivatives at parameter perturbation norms.001 and.0005 and the smaller step always used. All unreachable targets and numerical discrepancies retained. Predictions and state hashes physically committed before endpoint scoring. No predictor or target selection on these outcomes.\n\nPrimary: radial initial fitting PB-minus-GD contrast at.02, testing whether prior finite-drop failure and curvature remedy persist under exact finite class balance. Report both tasks, all components and tiny controls. Angular outcomes use the valid pi subgroup, not all rotations. Compare first-order, second-order, zero and realized-step tangent. Original-old6 pooled slopes remain solely as an explicitly out-of-design comparator trained on iid, not antithetically balanced data. No balanced-old calibration or fit to new outcomes.\n\nSeed bootstrap confidence intervals use independent datasets as clusters; dependence within training pairs is retained, not resampled as independent points. This is a bounded sensitivity experiment and not a new source of adaptively selected winners.\n'
    (ROOT/'PROTOCOL.md').write_text(protocol,encoding='utf8')
    s.jsonout(ROOT/'freeze.json',dict(timestamp_utc=s.now(),config=CONFIG,script_sha256=s.sha(Path(__file__).read_bytes()),protocol_sha256=s.sha((ROOT/'PROTOCOL.md').read_bytes()),parent_script_sha256=s.sha((ROOT.parent/'study.py').read_bytes()),old_outcomes_sha256=s.sha((ROOT.parent/'old'/'outcomes.csv').read_bytes())))
def generate():
    fr=json.loads((ROOT/'freeze.json').read_text());assert fr['script_sha256']==s.sha(Path(__file__).read_bytes())
    if (ROOT/'predictions.csv').exists():raise FileExistsError('Already generated')
    predictions=[];roots=[];bank={};fd=[];base,fallback=s.baseline()
    for seed in CONFIG['seeds']:
        for task in CONFIG['tasks']:
            x,y,t,ds,ini=s.dataset(task,seed,'new');prekey=f'{task}_{seed}_initial';bank[prekey]=t;bank[f'{task}_{seed}_x']=x;bank[f'{task}_{seed}_y']=y
            pre,gq,_=s.measure(t,task,*s.CONFIG['derivative_grid'],True);train0=s.loss(t,x,y)
            for method in s.METHODS:
                d=s.core.direction(t,x,y,method,.1)[0];slope=-float(s.core.parameter_gradient(t,x,y)@d);w=d/slope;a=gq@w
                estimates=[]
                for parnorm in CONFIG['fd_parameter_norms']:
                    eps=parnorm/np.linalg.norm(w)
                    plus=s.measure(t+eps*w,task)[0];minus=s.measure(t-eps*w,task)[0]
                    c=(plus-2*pre+minus)/eps**2;b=(s.loss(t+eps*w,x,y)-2*train0+s.loss(t-eps*w,x,y))/eps**2
                    estimates.append((c,b,eps))
                c,b,eps=estimates[-1];c0,b0,_=estimates[0]
                fd.append(dict(seed=seed,task=task,method=method,w_norm=float(np.linalg.norm(w)),epsilon=eps,loss_second=b,loss_second_coarse=b0,loss_relative_disagreement=abs(b-b0)/max(abs(b),1e-12),fit_second=c[0],fit_second_coarse=c0[0],nuisance_second=c[1],nuisance_second_coarse=c0[1],value_second=c[2],value_second_coarse=c0[2],outcome_relative_disagreement=float(np.max(abs(c-c0)/np.maximum(abs(c),1e-12)))))
                for drop in CONFIG['drops']:
                    end,info=s.match(t,x,y,d,drop);rowid=f'{prekey}_{method}_{drop:g}';endkey=rowid+'_end'
                    roots.append(dict(seed=seed,task=task,method=method,**info))
                    if end is not None:bank[endkey]=end
                    for j,metric in enumerate(s.METRICS):
                        key=(task,'initial',method,'single','1',str(drop),metric)
                        bs,bn=base.get(key,fallback[(task,'initial',method,metric)])
                        predictions.append(dict(row_id=rowid,seed=seed,task=task,method=method,requested_drop=drop,metric=metric,status=info['status'],data_seed=ds,initialization_seed=ini,n_positive=int(np.sum(y==1)),n_negative=int(np.sum(y==-1)),pre_key=prekey,end_key=endkey,pre_sha256=s.sha(t.tobytes()),end_sha256=s.sha(end.tobytes()) if end is not None else '',pre_value=pre[j],train_slope=slope,q1=a[j],q2=c[j],loss2=b,drop_clock=drop*a[j],second_order=drop*a[j]+.5*drop**2*(c[j]+a[j]*b),realized_tangent=float(gq[j]@(end-t)) if end is not None else '',old_pooled=drop*bs,old_baseline_n=bn,old_baseline_design='iid_out_of_design',zero=0.,actual_drop=info['actual_drop'],alpha=info['alpha'],residual=info['residual']))
        print(f'followup prediction seed{seed}',flush=True)
    s.csvout(ROOT/'predictions.csv',predictions);s.csvout(ROOT/'roots.csv',roots);s.csvout(ROOT/'finite_differences.csv',fd);np.savez_compressed(ROOT/'states.npz',**bank)
    s.jsonout(ROOT/'prediction_commit.json',dict(timestamp_utc=s.now(),predictions_sha256=s.sha((ROOT/'predictions.csv').read_bytes()),state_sha256=s.sha((ROOT/'states.npz').read_bytes()),n_rows=len(predictions),n_root=len(roots)))
def score():
    commit=json.loads((ROOT/'prediction_commit.json').read_text());assert commit['predictions_sha256']==s.sha((ROOT/'predictions.csv').read_bytes())
    if (ROOT/'outcomes.csv').exists():raise FileExistsError('Already scored')
    s.jsonout(ROOT/'score_start.json',dict(timestamp_utc=s.now(),prediction_commit=commit))
    rows=s.csvin(ROOT/'predictions.csv');bank=np.load(ROOT/'states.npz');cache={};quad=[]
    for r in rows:
        if r['status']=='ok':
            for key in (r['pre_key'],r['end_key']):
                if key not in cache:
                    v,meta=s.refined(bank[key],r['task']);cache[key]=v;quad.append(dict(key=key,task=r['task'],**meta))
            j=s.METRICS.index(r['metric']);r['actual_change']=cache[r['end_key']][j]-cache[r['pre_key']][j]
        else:r['actual_change']=''
    s.csvout(ROOT/'outcomes.csv',rows);s.csvout(ROOT/'quadrature.csv',quad)
    s.jsonout(ROOT/'score_complete.json',dict(timestamp_utc=s.now(),outcome_sha256=s.sha((ROOT/'outcomes.csv').read_bytes()),n_states=len(cache),nonconverged=sum(1-r['converged'] for r in quad)))
def summarize():
    rows=s.csvin(ROOT/'outcomes.csv');okay=[r for r in rows if r['status']=='ok'];ind={tuple(r[k] for k in ('seed','task','requested_drop','metric','method')):r for r in okay};groups=defaultdict(list);contrasts=[]
    preds=('drop_clock','second_order','realized_tangent','old_pooled','zero')
    for k,gd in ind.items():
        if k[-1]!='gd':continue
        pb=ind.get(k[:-1]+('pullback',))
        if pb:
            r=dict(seed=k[0],task=k[1],requested_drop=k[2],metric=k[3],actual=float(pb['actual_change'])-float(gd['actual_change']))
            for p in preds:r[p]=float(pb[p])-float(gd[p])
            contrasts.append(r);groups[k[1:-1]].append(r)
    s.csvout(ROOT/'method_contrasts.csv',contrasts);summary=[];rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    for k,rr in groups.items():
        actual=np.array([r['actual'] for r in rr]);idx=rng.integers(0,len(rr),(10000,len(rr)));e1=abs(np.array([r['drop_clock'] for r in rr])-actual);eold=abs(np.array([r['old_pooled'] for r in rr])-actual)
        for p in preds:
            pr=np.array([r[p] for r in rr]);e=abs(pr-actual);adv=e1-e;ci=np.quantile(adv[idx].mean(axis=1),[.025,.975]);ado=eold-e;cio=np.quantile(ado[idx].mean(axis=1),[.025,.975]);summary.append(dict(task=k[0],requested_drop=k[1],metric=k[2],predictor=p,n_seed=len(rr),mae=e.mean(),rmse=np.sqrt(np.mean((pr-actual)**2)),correct_sign=int((pr*actual>0).sum()),seed_wins_first=int((adv>0).sum()),advantage_first=adv.mean(),advantage_first_lo=ci[0],advantage_first_hi=ci[1],advantage_old=ado.mean(),advantage_old_lo=cio[0],advantage_old_hi=cio[1]))
    s.csvout(ROOT/'contrast_summary.csv',summary)
    fd=s.csvin(ROOT/'finite_differences.csv');coverage=dict(n_root=len(rows)//3,successful_root=len(okay)//3,failed_norm_cap=sum(r['status']=='norm_cap' for r in rows)//3,n_paired_metric_contrast=len(contrasts),max_fd_loss_disagreement=max(float(r['loss_relative_disagreement']) for r in fd),max_fd_outcome_disagreement=max(float(r['outcome_relative_disagreement']) for r in fd))
    s.jsonout(ROOT/'coverage.json',coverage)
    lines=['# Exact finite-label balance control','',
    'This sensitivity control uses sixteen new datasets with exactly128 positive and128 negative training labels in each task. Shared inputs and initialization are preserved. Each dataset contains128 independent antithetic pairs, so its256 observations are correlated within pairs. Every observation has the original correct input/posterior marginal. The first/second-order formulas, targets and root cap are unchanged. A separate freeze and physical prediction commit precede endpoint scoring. Original iid-old6 pooled slopes are an out-of-design comparator, explicitly labeled in raw rows.','',
    '## Paired optimizer-contrast accuracy','',
    '|Task|Drop|Metric|Pairs|First MAE|Second MAE|Old iid MAE (out of design)|Second sign|Second-minus-first benefit95%CI|','|---|---:|---|---:|---:|---:|---:|---:|---|']
    ix={tuple(r[k] for k in ('task','requested_drop','metric','predictor')):r for r in summary}
    for task in CONFIG['tasks']:
        for drop in ('1e-05','0.02'):
            for m in s.METRICS:
                k=(task,drop,m);r=ix[k+('second_order',)];first=ix[k+('drop_clock',)];old=ix[k+('old_pooled',)]
                lines.append(f"|{task}|{drop}|{m}|{r['n_seed']}|{first['mae']:.4g}|{r['mae']:.4g}|{old['mae']:.4g}|{r['correct_sign']}/{r['n_seed']}|[{r['advantage_first_lo']:.4g},{r['advantage_first_hi']:.4g}]|")
    lines+=['',f"Coverage: {coverage['successful_root']}/{coverage['n_root']} roots reached target, {coverage['failed_norm_cap']} retained cap failures. Maximum coarse/fine second-derivative relative discrepancies were {coverage['max_fd_loss_disagreement']:.4g} for training loss and {coverage['max_fd_outcome_disagreement']:.4g} for outcomes. No discrepancy-triggered tuning or outcome-driven refit was performed.",'',
    'The primary balance-control endpoint is radial initial fitting contrast at.02. This is a small prospective validation, conditional on paired root reachability. Angular fitting retains informative angle structure; its nuisance and view-value endpoints are identical pi-subgroup quantities. The second-order model is population-informed, and cannot be interpreted as a training-only deployable predictor.','',
    'Reproduce in an empty copy with the parent frozen study and old6 data: balanced.py freeze; balanced.py generate; balanced.py score; balanced.py summarize. Raw finite differences, states, roots, predictions, outcomes, hashes and bootstrap summaries are supplied.']
    (ROOT/'BALANCED_CONTROL.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(coverage),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','generate','score','summarize']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.action]()
