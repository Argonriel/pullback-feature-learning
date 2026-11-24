"""Frozen two-stage label-allocation frontier selection and confirmation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits
import model

SOURCE=Path(__file__).resolve().parent
OUT=SOURCE.parents[2]/'tmp/frontier-replication-residuals-20261121/allocation'
NS=(2304,3072,3584,3840)
BUDGET=207260412000
PROCEDURES=[dict(name=f'train_{n}',n=n,calibration=4352-n,validation=0,steps=BUDGET//(179045*n+2001144)) for n in NS]
BASELINE='train_2304'
ARMS=(('raw_uncal','raw','none'),('raw_T','raw','raw'),('full_uncal','full','none'),('full_rawT','full','raw'),('full_T','full','full'))
CONFIG=dict(development_pairs=8,confirmation_pairs=16,development_seed_offset=0,confirmation_seed_offset=50,
 total_fit_labels=4352,evaluation_n=65536,batch=256,procedures=PROCEDURES,arms=ARMS,nominal_training_budget=BUDGET,
 flop_slope=179045,flop_constant=2001144,checkpoint_fractions=[.25,.5,.75,1.],
 seeds=dict(training=2411000,initialization=2412000,calibration=2414000,inputs=2416000,labels=2417000,bootstrap=2419050),
 training_increments=[2304,768,512,256],increment_seed_offsets=[0,100,200,300],shuffle_seed_base=2411800,
 selection_metric='sampled_nll',selection_arm='raw_T',tie_tolerance=1e-12,tie_rule='smallest training n within tolerance of minimum',
 bootstrap_draws=20000,primary_contrasts=2,baseline_wins='retain baseline; fresh absolute confirmation; identical contrast zero by construction, no improvement test',
 timing=dict(development_pair=1,warmup_blocks=1,measured_blocks=3,order_seed=2419100,scope='complete training including diagnostic checkpoints plus raw/full calibration; excludes input generation/loading, output I/O, test scoring and verification'))

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jo(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def cw(path,rows):
    with Path(path).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def read_csv(path):return list(csv.DictReader(Path(path).open(newline='')))
def manifest(paths):return {p.relative_to(OUT).as_posix():sha(p) for p in paths}
def commit(path,paths,**extra):jo(path,dict(utc=now(),manifest=manifest(paths),**extra))
def verify_commit(path):
    record=json.loads((OUT/path).read_text())
    for p,h in record.get('manifest',{}).items():assert sha(OUT/p)==h,p
    parent=Path(path).parent
    for key,target in [('source_freeze_sha256',OUT/'SOURCE_FREEZE.json'),('model_commit_sha256',OUT/parent/'MODEL_COMMIT.json'),('fit_commit_sha256',OUT/parent/'FIT_COMMIT.json'),('selection_commit_sha256',OUT/'SELECTION_COMMIT.json')]:
        if key in record and record[key] is not None:assert record[key]==sha(target),key
    return record
def source_files():return [SOURCE/p for p in ('runner.py','model.py','preflight.py','timing.py')]+sorted(p for p in (SOURCE/'reference').glob('*') if p.is_file())
def freeze():
    assert not (OUT/'development').exists() and not (OUT/'confirmation').exists()
    jo(OUT/'SOURCE_FREEZE.json',dict(utc=now(),config=CONFIG,source_root=str(SOURCE),source_hashes={p.relative_to(SOURCE).as_posix():sha(p) for p in source_files()},
      protocol_sha256=sha(OUT/'PROTOCOL.md'),preflight_sha256=sha(OUT/'PREFLIGHT.json'),fresh_streams_accessed=False))
def verify():
    frozen=json.loads((OUT/'SOURCE_FREEZE.json').read_text());assert frozen['config']==json.loads(json.dumps(CONFIG))
    for p,h in frozen['source_hashes'].items():assert sha(SOURCE/p)==h,p
    assert sha(OUT/'PROTOCOL.md')==frozen['protocol_sha256']
    approval=json.loads((OUT/'ROOT_APPROVAL.json').read_text());assert approval['fresh_authorized'] and approval['source_freeze_sha256']==sha(OUT/'SOURCE_FREEZE.json')
def selection():
    record=verify_commit('SELECTION_COMMIT.json');assert record['development_score_commit_sha256']==sha(OUT/'development/SCORE_COMPLETE.json')
    return record
def procedures(phase):
    if phase=='development':return PROCEDURES
    winner=selection()['selected_procedure']
    start=OUT/'confirmation/TRAIN_START.json'
    if start.exists():assert json.loads(start.read_text())['selection_commit_sha256']==sha(OUT/'SELECTION_COMMIT.json')
    return [p for p in PROCEDURES if p['name'] in (BASELINE,winner)]
def phase_config(phase):
    assert phase in ('development','confirmation')
    return (8,0) if phase=='development' else (16,50)
def checkpoints(steps):return sorted(set(max(1,int(steps*f)) for f in (.25,.5,.75,1.)))
def work(n,steps):
    return dict(updates_executed=steps,training_example_updates=n*steps,standalone_representation_forward_examples=n*steps,jacobian_forward_examples=n*steps,
      jacobian_rows=4*n*steps,jacobian_scalar_entries=4*n*144*steps,loss_derivative_examples=n*steps,separate_reverse_mode_backward_calls=0,
      jacobian_transpose_vector_products=steps,gram_matrices=steps,lu_144_solves=steps,nominal_arithmetic_flops=(179045*n+2001144)*steps,
      tanh_scalar_evaluations=32*n*steps,exp_scalar_evaluations=n*steps,logaddexp_scalar_evaluations=n*steps)

def train(phase):
    verify();ps=procedures(phase);count,offset=phase_config(phase);folder=OUT/phase;folder.mkdir();(folder/'states').mkdir();(folder/'splits').mkdir()
    jo(folder/'TRAIN_START.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'SOURCE_FREEZE.json'),selection_commit_sha256=sha(OUT/'SELECTION_COMMIT.json') if phase=='confirmation' else None))
    rows=[];traces=[];paths=[folder/'TRAIN_START.json'];maxn=max(p['n'] for p in ps)
    for pair in range(1,count+1):
        sid=pair+offset;chunks=[];ys=[];sampling=[];accumulated=0
        for increment,inc_offset in zip(CONFIG['training_increments'],CONFIG['increment_seed_offsets']):
            x,y,meta=model.task.sample_train_with_metadata(2411000+sid+inc_offset,increment);chunks.append(x);ys.append(y);sampling.append(meta);accumulated+=increment
            if accumulated==maxn:break
        assert accumulated==maxn
        initial=model.task.initialize(2412000+sid);cx,cy=model.labeled_inputs(2414000+sid,2048)
        split=dict(initial=initial,calibration_x=cx,calibration_y=cy);states={}
        for proc in ps:
            n=proc['n'];take=NS.index(n)+1;x=np.concatenate(chunks[:take]);y=np.concatenate(ys[:take]);order=np.random.default_rng(2411800+sid).permutation(n);x=x[order];y=y[order]
            assert len(y)==n and int(np.sum(y==1))==n//2
            split[f'x_{n}']=x;split[f'y_{n}']=y;theta=initial.copy();marks=checkpoints(proc['steps']);start=time.perf_counter()
            for step in range(1,proc['steps']+1):
                theta+=(4*n/256)*model.task.direction(theta,x,y,method='pb',damping=.1*n/256)
                if not np.isfinite(theta).all():raise FloatingPointError(f'nonfinite {phase} {pair} {n} {step}')
                if step in marks:
                    states[f'checkpoint_{n}_{step}']=theta.copy();traces.append(dict(phase=phase,pair=pair,seed_id=sid,procedure=proc['name'],step=step,training_nll=model.task.loss(theta,x,y),seconds_descriptive=time.perf_counter()-start))
            states[proc['name']]=theta.copy()
            rows.append(dict(phase=phase,pair=pair,seed_id=sid,procedure=proc['name'],n_train=n,n_validation=0,n_calibration=proc['calibration'],unique_fit_labels=4352,
              rate=4*n/256,damping=.1*n/256,**work(n,proc['steps']),training_diagnostic_forward_examples=n*len(marks),validation_forward_examples=0,
              nominal_training_budget=BUDGET,unused_nominal_budget=BUDGET-(179045*n+2001144)*proc['steps'],training_nll=traces[-1]['training_nll'],seconds_descriptive=time.perf_counter()-start))
        path=folder/'states'/f'{pair}.npz';np.savez_compressed(path,**states);paths.append(path)
        path=folder/'splits'/f'{pair}.npz';np.savez_compressed(path,**split);paths.append(path)
        path=folder/'splits'/f'{pair}_sampling.json';jo(path,dict(phase=phase,pair=pair,seed_id=sid,metadata=sampling,accepted_training_union=maxn,calibration_label_union=2048,unique_fitting_label_union=maxn+2048));paths.append(path)
        print('Trained',phase,pair,flush=True)
    cw(folder/'training.csv',rows);cw(folder/'training_trace.csv',traces);paths += [folder/'training.csv',folder/'training_trace.csv']
    commit(folder/'MODEL_COMMIT.json',paths,models=count*len(ps),source_freeze_sha256=sha(OUT/'SOURCE_FREEZE.json'))

def calibrate(phase):
    verify();verify_commit(f'{phase}/MODEL_COMMIT.json');ps=procedures(phase);count,offset=phase_config(phase);folder=OUT/phase;(folder/'temperatures').mkdir();rows=[];paths=[]
    for pair in range(1,count+1):
        states=np.load(folder/'states'/f'{pair}.npz');split=np.load(folder/'splits'/f'{pair}.npz')
        for proc in ps:
            name=proc['name'];n=proc['calibration'];x=split['calibration_x'][:n];y=split['calibration_y'][:n]
            raw=model.task.predict(states[name],x);partner=model.task.predict(states[name],model.task.partner(x));full=.5*(raw+partner);fits={}
            for pipeline,h in [('raw',raw),('full',full)]:
                fit=model.temperature(h,y);assert fit['calibrated_nll']<=fit['raw_nll']+1e-12;fits[pipeline]=fit
                rows.append(dict(phase=phase,pair=pair,seed_id=pair+offset,procedure=name,pipeline=pipeline,unique_fit_labels=4352,standalone_calibration_forwards=n*(1 if pipeline=='raw' else 2),actual_shared_calibration_forwards=2*n,
                  derivative_evaluations=fit['iterations']+3,objective_evaluations=2,derivative_label_evaluations=(fit['iterations']+3)*n,objective_label_evaluations=2*n,**fit))
            path=folder/'temperatures'/f'{pair}_{name}.json';jo(path,dict(phase=phase,pair=pair,procedure=name,fits=fits,model_file_sha256=sha(folder/'states'/f'{pair}.npz')));paths.append(path)
            path=folder/'temperatures'/f'{pair}_{name}_outputs.npz';np.savez_compressed(path,raw=raw,partner=partner,full=full,labels=y);paths.append(path)
        print('Calibrated',phase,pair,flush=True)
    cw(folder/'temperature_fits.csv',rows);paths.append(folder/'temperature_fits.csv')
    commit(folder/'FIT_COMMIT.json',paths,fits=len(rows),evaluation_streams_accessed=False,model_commit_sha256=sha(folder/'MODEL_COMMIT.json'))

def score(phase):
    verify();verify_commit(f'{phase}/MODEL_COMMIT.json');verify_commit(f'{phase}/FIT_COMMIT.json');ps=procedures(phase);count,offset=phase_config(phase);folder=OUT/phase;(folder/'predictions').mkdir()
    jo(folder/'SCORE_START.json',dict(utc=now(),fit_commit_sha256=sha(folder/'FIT_COMMIT.json'),selection_commit_sha256=sha(OUT/'SELECTION_COMMIT.json') if phase=='confirmation' else None))
    rows=[];bins=[];costs=[];paths=[]
    for pair in range(1,count+1):
        sid=pair+offset;x=model.task.sample_inputs(2416000+sid,65536);p=model.task.posterior(x);y=np.where(np.random.default_rng(2417000+sid).random(len(x))<p,1.,-1.);states=np.load(folder/'states'/f'{pair}.npz')
        saved=dict(x=x,y=y)
        if phase=='confirmation':saved['p']=p
        else:del p # Development metrics/selection only consume sampled labels.
        for proc in ps:
            name=proc['name'];theta=states[name];fits=json.loads((folder/'temperatures'/f'{pair}_{name}.json').read_text())['fits']
            for arm,pipeline,temp in ARMS:
                beta=1. if temp=='none' else fits[temp]['beta'];outputs=[];forwards=0;calls=0
                for begin in range(0,len(x),256):
                    xx=x[begin:begin+256];h=model.task.predict(theta,xx);forwards+=len(xx);calls+=1
                    if pipeline=='full':h=.5*(h+model.task.predict(theta,model.task.partner(xx)));forwards+=len(xx);calls+=1
                    outputs.append(beta*h)
                h=np.concatenate(outputs);q=np.exp(-np.logaddexp(0,-h));target=(y==1).astype(float)
                if temp=='none':saved[name+'_'+pipeline]=h
                else:assert np.array_equal(saved[name+'_'+pipeline]>=0,h>=0)
                row=dict(phase=phase,pair=pair,seed_id=sid,procedure=name,n_train=proc['n'],n_validation=0,n_calibration=0 if temp=='none' else proc['calibration'],
                  actual_unique_fit_labels=proc['n']+(0 if temp=='none' else proc['calibration']),arm=arm,pipeline=pipeline,beta=beta,
                  sampled_nll=float(np.logaddexp(0,-y*h).mean()),sampled_error=float(np.mean((h>=0)!=(y==1))),sampled_brier=float(np.mean((q-target)**2)),inference_forwards=forwards,inference_invocations=calls)
                if phase=='confirmation':
                    row.update(expected_nll=float(np.mean(np.logaddexp(0,h)-p*h)),expected_error=float(np.mean(np.where(h>=0,1-p,p))),expected_brier=float(np.mean((q-p)**2+p*(1-p))),
                      bayes_nll=float(np.mean(-p*np.log(p)-(1-p)*np.log1p(-p))),bayes_error=float(np.minimum(p,1-p).mean()))
                ids=np.minimum((q*10).astype(int),9);ece=0.
                for b in range(10):
                    keep=ids==b;n=int(keep.sum());pred=float(q[keep].mean()) if n else None;truth=float((p if phase=='confirmation' else target)[keep].mean()) if n else None
                    if n:ece+=n*abs(pred-truth)
                    bins.append(dict(phase=phase,pair=pair,procedure=name,arm=arm,bin=b,count=n,predicted_probability=pred,target_probability=truth,target_type='posterior' if phase=='confirmation' else 'sampled_label'))
                row['posterior_binned_ece' if phase=='confirmation' else 'sampled_binned_ece']=ece/len(x);rows.append(row)
                costs.append(dict(phase=phase,pair=pair,procedure=name,arm=arm,batches=256,call_pattern='256' if pipeline=='raw' else '256|256',inference_forward_examples=forwards,inference_invocations=calls))
        path=folder/'predictions'/f'{pair}.npz';np.savez_compressed(path,**saved);paths.append(path);print('Scored',phase,pair,flush=True)
    assert len(rows)==count*len(ps)*5
    cw(folder/'per_model.csv',rows);cw(folder/'reliability_bins.csv',bins);cw(folder/'deployment_ledger.csv',costs);paths += [folder/'per_model.csv',folder/'reliability_bins.csv',folder/'deployment_ledger.csv']
    commit(folder/'SCORE_COMPLETE.json',paths,rows=len(rows),actual_evaluation_label_union=count*65536,actual_inference_forwards=sum(r['inference_forward_examples'] for r in costs),fit_commit_sha256=sha(folder/'FIT_COMMIT.json'))

def summarize(phase):
    rows=read_csv(OUT/phase/'per_model.csv');metrics=['sampled_nll','sampled_error','sampled_brier']+(['expected_nll','expected_error','expected_brier','posterior_binned_ece'] if phase=='confirmation' else ['sampled_binned_ece']);result=[]
    for proc in procedures(phase):
        for arm,_,_ in ARMS:
            values=[r for r in rows if r['procedure']==proc['name'] and r['arm']==arm]
            result.append(dict(phase=phase,procedure=proc['name'],n_train=proc['n'],n_calibration=proc['calibration'],updates=proc['steps'],arm=arm,pairs=len(values),**{m:float(np.mean([float(r[m]) for r in values])) for m in metrics}))
    return result
def select():
    verify();verify_commit('development/SCORE_COMPLETE.json');assert not (OUT/'confirmation').exists()
    summary=summarize('development');cw(OUT/'development/summary.csv',summary)
    candidates=[r for r in summary if r['arm']=='raw_T'];minimum=min(r['sampled_nll'] for r in candidates);winner=min([r for r in candidates if r['sampled_nll']<=minimum+1e-12],key=lambda r:r['n_train'])
    jo(OUT/'SELECTION_COMMIT.json',dict(utc=now(),selected_procedure=winner['procedure'],selected_training_n=winner['n_train'],selected_development_sampled_nll=winner['sampled_nll'],selection_metric='mean sampled-label calibrated raw NLL over 8 paired seeds',
      candidate_scores=[dict(procedure=r['procedure'],n_train=r['n_train'],sampled_nll=r['sampled_nll']) for r in candidates],baseline_selected=winner['procedure']==BASELINE,
      confirmation_streams_accessed=False,development_score_commit_sha256=sha(OUT/'development/SCORE_COMPLETE.json'),development_selection_label_union=8*65536,
      manifest=manifest([OUT/'development/summary.csv']),winner_bias_warning='Development minimum is selected and optimistic; only independent confirmation supports inference.'))
    print(json.dumps(dict(selected=winner['procedure'],development_sampled_nll=winner['sampled_nll'],baseline_selected=winner['procedure']==BASELINE)))
def analyze():
    verify();record=selection();verify_commit('confirmation/SCORE_COMPLETE.json');summary=summarize('confirmation');cw(OUT/'confirmation/summary.csv',summary)
    rows=read_csv(OUT/'confirmation/per_model.csv');winner=record['selected_procedure'];rng=np.random.default_rng(2419050);draw=rng.integers(0,16,(20000,16));contrasts=[];paired=[]
    def vals(name,arm,metric):return np.array([float(r[metric]) for r in sorted([r for r in rows if r['procedure']==name and r['arm']==arm],key=lambda r:int(r['pair']))])
    for arm in ('raw_T','full_T'):
        for metric in ('expected_nll','expected_error','sampled_nll','sampled_error','expected_brier'):
            diff=vals(winner,arm,metric)-vals(BASELINE,arm,metric);primary=metric=='expected_nll';alpha=.05/2 if primary else .05
            lo,hi=np.quantile(diff[draw].mean(1),[alpha/2,1-alpha/2]) if winner!=BASELINE else ('','')
            contrasts.append(dict(contrast='selected_minus_baseline',selected_procedure=winner,arm=arm,metric=metric,primary=primary,mean_difference=float(diff.mean()),lower=lo,upper=hi,
              alpha=alpha if winner!=BASELINE else '',wins=int(np.sum(diff<0)),ties=int(np.sum(diff==0)),inference_status='paired bootstrap' if winner!=BASELINE else 'identical procedure; no improvement test'))
            paired += [dict(pair=i+1,arm=arm,metric=metric,difference=float(d)) for i,d in enumerate(diff)]
    cw(OUT/'confirmation/contrasts.csv',contrasts);cw(OUT/'confirmation/paired_contrasts.csv',paired)
    commit(OUT/'ANALYSIS_COMPLETE.json',[OUT/'confirmation/summary.csv',OUT/'confirmation/contrasts.csv',OUT/'confirmation/paired_contrasts.csv'],selection_commit_sha256=sha(OUT/'SELECTION_COMMIT.json'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','calibrate','score','select','analyze']);parser.add_argument('--phase',choices=['development','confirmation']);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.out:OUT=args.out.resolve()
    try:
        with threadpool_limits(limits=1):
            if args.stage in ('train','calibrate','score'):
                assert args.phase is not None;globals()[args.stage](args.phase)
            else:globals()[args.stage]()
    except Exception:
        if OUT.exists():
            failure=OUT/f'FAILED_{args.stage}_{args.phase}_{time.time_ns()}.txt';failure.write_text(traceback.format_exc(),encoding='utf8')
        raise
