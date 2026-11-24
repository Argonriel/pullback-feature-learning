"""Prospective label allocation and explicitly accounted PB training-work study."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, time, sys
import numpy as np
from threadpoolctl import threadpool_limits
import model

SOURCE=Path(__file__).resolve().parent
OUT=Path(__file__).resolve().parents[3]/'tmp/budget-distillation-structured-bounds-20261121/allocation'
NS=(256,768,1280,2304)
MAX_STEPS={256:4500,768:1485,1280:896,2304:500}
FLOP_STEPS={n:(500*(179045*2304+2001144))//(179045*n+2001144) for n in NS}
PROCEDURES=[dict(name=f'fixed_{n}',n=n,step=500,validation=0,calibration=4352-n,regime='fixed_steps') for n in NS]
PROCEDURES += [dict(name=f'flop_{n}',n=n,step=FLOP_STEPS[n],validation=0,calibration=4352-n,regime='nominal_flops') for n in NS[:-1]]
PROCEDURES += [dict(name='examples_256',n=256,step=4500,validation=0,calibration=4096,regime='example_jacobian_work'),dict(name='validation_1280',n=1280,step=896,validation=1024,calibration=2048,regime='validation')]
ARMS=(('raw_uncal','raw','none'),('raw_T','raw','raw'),('full_uncal','full','none'),('full_rawT','full','raw'),('full_T','full','full'))
CONFIG=dict(pairs=16,total_labels=4352,test_n=65536,batch=256,n_parameters=144,representation_dimension=4,procedures=PROCEDURES,arms=ARMS,
 seeds=dict(training=2311000,initialization=2312000,validation=2313000,calibration=2314000,test=2316000,test_labels=2317000,bootstrap=2319000),
 training_increment_seed_offsets=[0,100,200,300],training_increment_sizes=[256,512,512,1024],training_shuffle_offset=800,
 validation_checkpoints=[100,250,500,896],bootstrap_draws=20000,primary_count=8,flops_per_example=179045,flops_constant=2001144)

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jo(p,value):
    with Path(p).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def cw(p,rows):
    with Path(p).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def manifest(paths):return {str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in paths}
def source_files():return sorted([SOURCE/'runner.py',SOURCE/'model.py',SOURCE/'preflight.py']+list((SOURCE/'reference').glob('*')))
def freeze():
    assert not (OUT/'states').exists()
    jo(OUT/'SOURCE_FREEZE.json',dict(utc=now(),config=CONFIG,source_root=str(SOURCE),source_hashes={str(p.relative_to(SOURCE)).replace('\\','/'):sha(p) for p in source_files() if p.is_file()},
        protocol_sha256=sha(OUT/'PROTOCOL.md'),preflight_sha256=sha(OUT/'PREFLIGHT.json'),fresh_streams_accessed=False))
def verify():
    record=json.loads((OUT/'SOURCE_FREEZE.json').read_text())
    assert record['config']==json.loads(json.dumps(CONFIG))
    for path,h in record['source_hashes'].items():assert sha(SOURCE/path)==h,path
    assert record['protocol_sha256']==sha(OUT/'PROTOCOL.md')
    approval=json.loads((OUT/'ROOT_APPROVAL.json').read_text())
    assert approval['fresh_authorized'] and approval['source_freeze_sha256']==sha(OUT/'SOURCE_FREEZE.json')
def verify_commit(name):
    for path,h in json.loads((OUT/name).read_text())['manifest'].items():assert sha(OUT/path)==h,path
def work(n,updates):
    return dict(updates_executed=updates,training_example_updates=n*updates,standalone_representation_forward_examples=n*updates,
      jacobian_forward_examples=n*updates,jacobian_rows=4*n*updates,jacobian_scalar_entries=4*n*144*updates,
      loss_derivative_examples=n*updates,gradient_jacobian_transpose_vector_products=updates,separate_reverse_mode_backward_calls=0,
      gram_matrices=updates,lu_solves_144=updates,nominal_arithmetic_flops=(179045*n+2001144)*updates,
      tanh_scalar_evaluations=32*n*updates,exp_scalar_evaluations=n*updates,logaddexp_scalar_evaluations=n*updates)

def train():
    verify();(OUT/'states').mkdir();(OUT/'splits').mkdir()
    jo(OUT/'TRAIN_START.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'SOURCE_FREEZE.json')))
    rows=[];trace=[];actual=[];paths=[]
    for pair in range(1,17):
        chunks=[];ys=[];split={};sampled=[]
        for offset,size in zip(CONFIG['training_increment_seed_offsets'],CONFIG['training_increment_sizes']):
            x,y,meta=model.task.sample_train_with_metadata(2311000+offset+pair,size);chunks.append(x);ys.append(y);sampled.append(meta)
        initial=model.task.initialize(2312000+pair)
        xv,yv=model.labeled_inputs(2313000+pair,1024);xc,yc=model.labeled_inputs(2314000+pair,4096)
        split.update(initial=initial,validation_x=xv,validation_y=yv,calibration_x=xc,calibration_y=yc)
        all_states={};times={};checkpoints={};nlls={}
        for i,n in enumerate(NS):
            x=np.concatenate(chunks[:i+1]);y=np.concatenate(ys[:i+1]);order=np.random.default_rng(2311800+pair).permutation(n);x=x[order];y=y[order]
            assert len(y)==n and np.sum(y==1)==n//2
            split[f'x_{n}']=x;split[f'y_{n}']=y
            needed={500,FLOP_STEPS[n],MAX_STEPS[n]}
            if n==1280:needed.update(CONFIG['validation_checkpoints'])
            theta=initial.copy();checkpoints[n]={};times[n]={};nlls[n]={};start=time.perf_counter()
            for step in range(1,MAX_STEPS[n]+1):
                theta+=(4*n/256)*model.task.direction(theta,x,y,method='pb',damping=.1*n/256)
                if not np.isfinite(theta).all():raise FloatingPointError(f'nonfinite {pair} {n} {step}')
                if step in needed:
                    checkpoints[n][step]=theta.copy();times[n][step]=time.perf_counter()-start
                    nlls[n][step]=model.task.loss(theta,x,y)
                    trace.append(dict(pair=pair,n_train=n,step=step,training_nll=nlls[n][step],elapsed_seconds=times[n][step]))
            actual.append(dict(pair=pair,n_train=n,**work(n,MAX_STEPS[n]),training_diagnostic_forward_examples=n*len(needed),elapsed_seconds=time.perf_counter()-start))
        validation_losses={s:model.task.loss(checkpoints[1280][s],xv,yv) for s in CONFIG['validation_checkpoints']}
        selected=min(validation_losses,key=lambda s:(validation_losses[s],s))
        for proc in PROCEDURES:
            name=proc['name'];n=proc['n'];step=selected if proc['validation'] else proc['step'];all_states[name]=checkpoints[n][step]
            rows.append(dict(pair=pair,procedure=name,regime=proc['regime'],n_train=n,n_validation=proc['validation'],n_calibration=proc['calibration'],unique_fit_labels=n+proc['validation']+proc['calibration'],
                rate=4*n/256,damping=.1*n/256,selected_step=step,**work(n,proc['step']),
                validation_forward_examples=4*proc['validation'],standalone_training_diagnostic_forward_examples=0,
                training_nll=nlls[n][step],training_seconds_descriptive=times[n][proc['step']],selected_validation_nll=validation_losses[step] if proc['validation'] else '',
                nominal_work_budget=work(2304,500)['nominal_arithmetic_flops'],nominal_work_slack=work(2304,500)['nominal_arithmetic_flops']-work(n,proc['step'])['nominal_arithmetic_flops']))
        for n in NS:
            for step,theta in checkpoints[n].items():all_states[f'checkpoint_{n}_{step}']=theta
        all_states['validation_steps']=np.array(CONFIG['validation_checkpoints']);all_states['validation_losses']=np.array(list(validation_losses.values()));all_states['validation_selected']=np.array(selected)
        p=OUT/'states'/f'{pair}.npz';np.savez_compressed(p,**all_states);paths.append(p)
        p=OUT/'splits'/f'{pair}.npz';np.savez_compressed(p,**split);paths.append(p)
        p=OUT/'splits'/f'{pair}_sampling.json';jo(p,dict(pair=pair,metadata=sampled,accepted_training_labels=2304,experimental_union_fit_labels=7424));paths.append(p)
        print('Trained pair',pair,flush=True)
    cw(OUT/'training.csv',rows);cw(OUT/'training_trace.csv',trace);cw(OUT/'actual_training_work.csv',actual)
    paths.extend([OUT/'training.csv',OUT/'training_trace.csv',OUT/'actual_training_work.csv'])
    jo(OUT/'MODEL_COMMIT.json',dict(utc=now(),models=144,manifest=manifest(paths),source_freeze_sha256=sha(OUT/'SOURCE_FREEZE.json')))

def calibrate():
    verify();verify_commit('MODEL_COMMIT.json');(OUT/'temperatures').mkdir();rows=[];paths=[]
    for pair in range(1,17):
        states=np.load(OUT/'states'/f'{pair}.npz');split=np.load(OUT/'splits'/f'{pair}.npz')
        for proc in PROCEDURES:
            name=proc['name'];n=proc['calibration'];x=split['calibration_x'][:n];y=split['calibration_y'][:n];theta=states[name]
            raw=model.task.predict(theta,x);partner=model.task.predict(theta,model.task.partner(x));full=.5*(raw+partner);fits={}
            for pipeline,h in [('raw',raw),('full',full)]:
                fit=model.temperature(h,y);fits[pipeline]=fit;assert fit['calibrated_nll']<=fit['raw_nll']+1e-12
                rows.append(dict(pair=pair,procedure=name,pipeline=pipeline,unique_total_fit_labels=4352,n_calibration=n,standalone_forward_examples=n*(1 if pipeline=='raw' else 2),shared_actual_forward_examples=2*n,**fit))
            p=OUT/'temperatures'/f'{pair}_{name}.json';jo(p,dict(pair=pair,procedure=name,fits=fits,model_sha256=sha(OUT/'states'/f'{pair}.npz')));paths.append(p)
            p=OUT/'temperatures'/f'{pair}_{name}_outputs.npz';np.savez_compressed(p,raw=raw,partner=partner,full=full,labels=y);paths.append(p)
        print('Calibrated pair',pair,flush=True)
    cw(OUT/'temperature_fits.csv',rows);paths.append(OUT/'temperature_fits.csv')
    jo(OUT/'FIT_COMMIT.json',dict(utc=now(),fits=len(rows),test_streams_accessed=False,manifest=manifest(paths),model_commit_sha256=sha(OUT/'MODEL_COMMIT.json')))

def score():
    verify();verify_commit('MODEL_COMMIT.json');verify_commit('FIT_COMMIT.json');(OUT/'predictions').mkdir()
    jo(OUT/'SCORE_START.json',dict(utc=now(),fit_commit_sha256=sha(OUT/'FIT_COMMIT.json')))
    rows=[];bins=[];costs=[];paths=[]
    for pair in range(1,17):
        x=model.task.sample_inputs(2316000+pair,CONFIG['test_n']);p=model.task.posterior(x);y=np.where(np.random.default_rng(2317000+pair).random(len(p))<p,1.,-1.)
        states=np.load(OUT/'states'/f'{pair}.npz');saved={}
        for proc in PROCEDURES:
            name=proc['name'];fits=json.loads((OUT/'temperatures'/f'{pair}_{name}.json').read_text())['fits'];theta=states[name]
            for arm,pipeline,temperature in ARMS:
                beta=1. if temperature=='none' else fits[temperature]['beta'];output=[];calls=0;forwards=0;start=time.perf_counter()
                for begin in range(0,len(x),256):
                    xx=x[begin:begin+256];h=model.task.predict(theta,xx);calls+=1;forwards+=len(xx)
                    if pipeline=='full':h=.5*(h+model.task.predict(theta,model.task.partner(xx)));calls+=1;forwards+=len(xx)
                    output.append(beta*h)
                h=np.concatenate(output);q=np.exp(-np.logaddexp(0,-h));err=np.where(h>=0,1-p,p)
                if temperature=='none':saved[name+'_'+pipeline]=h
                else:assert np.array_equal(saved[name+'_'+pipeline]>=0,h>=0)
                ece=0.;ids=np.minimum((q*10).astype(int),9)
                for b in range(10):
                    keep=ids==b;count=int(keep.sum());pred=float(q[keep].mean()) if count else None;truth=float(p[keep].mean()) if count else None
                    if count:ece+=count*abs(pred-truth)
                    bins.append(dict(pair=pair,procedure=name,arm=arm,bin=b,count=count,predicted_probability=pred,true_probability=truth))
                rows.append(dict(pair=pair,procedure=name,regime=proc['regime'],arm=arm,pipeline=pipeline,beta=beta,n_train=proc['n'],n_validation=proc['validation'],n_calibration=0 if temperature=='none' else proc['calibration'],
                  actual_unique_fit_labels=proc['n']+proc['validation']+(0 if temperature=='none' else proc['calibration']),
                  expected_nll=float(np.mean(np.logaddexp(0,h)-p*h)),expected_error=float(err.mean()),sampled_nll=float(np.mean(np.logaddexp(0,-y*h))),sampled_error=float(np.mean((h>=0)!=(y==1))),
                  expected_brier=float(np.mean((q-p)**2+p*(1-p))),posterior_binned_ece=ece/len(p),bayes_nll=float(np.mean(-p*np.log(p)-(1-p)*np.log1p(-p))),bayes_error=float(np.minimum(p,1-p).mean()),
                  inference_forwards=forwards,inference_invocations=calls,seconds_descriptive=time.perf_counter()-start))
                costs.append(dict(pair=pair,procedure=name,arm=arm,batches=len(x)//256,call_pattern='256' if pipeline=='raw' else '256|256',inference_forward_examples=forwards,inference_invocations=calls))
        path=OUT/'predictions'/f'{pair}.npz';np.savez_compressed(path,x=x,p=p,y=y,**saved);paths.append(path);print('Scored pair',pair,flush=True)
    assert len(rows)==720
    cw(OUT/'per_model.csv',rows);cw(OUT/'reliability_bins.csv',bins);cw(OUT/'deployment_ledger.csv',costs)
    paths.extend([OUT/'per_model.csv',OUT/'reliability_bins.csv',OUT/'deployment_ledger.csv'])
    jo(OUT/'SCORE_COMPLETE.json',dict(utc=now(),rows=len(rows),manifest=manifest(paths),actual_inference_forward_examples=sum(r['inference_forward_examples'] for r in costs),fit_commit_sha256=sha(OUT/'FIT_COMMIT.json')))

def analyze():
    verify();verify_commit('SCORE_COMPLETE.json')
    rows=list(csv.DictReader((OUT/'per_model.csv').open(newline='')));metrics=('expected_nll','expected_error','sampled_nll','sampled_error','expected_brier','posterior_binned_ece')
    summary=[]
    for proc in PROCEDURES:
        for arm,_,_ in ARMS:
            rs=[r for r in rows if r['procedure']==proc['name'] and r['arm']==arm]
            summary.append(dict(procedure=proc['name'],regime=proc['regime'],n_train=proc['n'],updates_executed=proc['step'],arm=arm,models=len(rs),**{key:float(np.mean([float(r[key]) for r in rs])) for key in metrics}))
    cw(OUT/'summary.csv',summary)
    definitions=[]
    for arm in ('raw_T','full_T'):
        for first in ('flop_256','flop_768','flop_1280'):
            definitions.append((first+'_vs_baseline_'+arm,first,arm,'fixed_2304',arm,True))
        definitions.append(('validation_vs_flop1280_'+arm,'validation_1280',arm,'flop_1280',arm,True))
        for first,second in [('fixed_768','fixed_256'),('fixed_1280','fixed_768'),('fixed_2304','fixed_1280'),('examples_256','fixed_2304'),('flop_256','fixed_256')]:
            definitions.append((first+'_vs_'+second+'_'+arm,first,arm,second,arm,False))
    for proc in PROCEDURES:definitions.append((proc['name']+'_full_own_vs_rawT',proc['name'],'full_T',proc['name'],'full_rawT',False))
    rng=np.random.default_rng(2319000);draw=rng.integers(0,16,(20000,16));contrasts=[];paired=[]
    def vals(proc,arm,key):return np.array([float(r[key]) for r in sorted([r for r in rows if r['procedure']==proc and r['arm']==arm],key=lambda r:int(r['pair']))])
    for label,fa,aa,fb,ab,primary in definitions:
        for key in ('expected_nll','expected_error'):
            d=vals(fa,aa,key)-vals(fb,ab,key);is_primary=primary and key=='expected_nll';alpha=.05/8 if is_primary else .05;ci=np.quantile(d[draw].mean(1),[alpha/2,1-alpha/2])
            contrasts.append(dict(contrast=label,metric=key,primary=is_primary,mean_difference=float(d.mean()),lower=float(ci[0]),upper=float(ci[1]),alpha=alpha,wins=int(np.sum(d<0)),ties=int(np.sum(d==0))))
            paired.extend([dict(contrast=label,metric=key,pair=i+1,difference=float(v)) for i,v in enumerate(d)])
    cw(OUT/'contrasts.csv',contrasts);cw(OUT/'paired_contrasts.csv',paired)
    jo(OUT/'ANALYSIS_COMPLETE.json',dict(utc=now(),summary_sha256=sha(OUT/'summary.csv'),contrasts_sha256=sha(OUT/'contrasts.csv')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','calibrate','score','analyze']);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.out:OUT=args.out.resolve()
    with threadpool_limits(limits=1):globals()[args.stage]()
