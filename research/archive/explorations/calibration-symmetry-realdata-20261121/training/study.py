"""Frozen fresh training, independent temperature fitting, and measured selectors."""
from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,json,sys,time,traceback
from datetime import datetime,timezone
import numpy as np
from threadpoolctl import threadpool_limits
import model
from adapters import get_adapter,load_variant,policy_paths,FAMILIES,VARIANTS,FAMILY_FOR_VARIANT
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('frozen_prior_selector',HERE/'reference/baseline.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
POLICIES=('raw','full','folded','symmetric_jensen')
N=65536;BATCH=256
CONFIG=dict(replications=16,train_n=256,validation_n=2048,temperature_n=2048,test_n=N,
  seed_bases=dict(train=2101000,init=2102000,validation=2103000,temperature=2104000,selector=2105000,test=2106000,ties=2107000,labels=2108000),
  variants=VARIANTS,policies=POLICIES,grid=[16,16,256],folded=[16,16,16],bootstrap_seed=2109001,bootstrap_draws=20000,primary_alpha=.05/4)
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def jo(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def cw(path,rows):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)

def freeze():
    assert (HERE/'TRAINING_CHOICE.json').exists() and not (HERE/'states').exists()
    paths=[HERE/p for p in ('model.py','pilot.py','adapters.py','study.py','PROTOCOL.md','FRESH_PROTOCOL.md','LITERATURE.md','PILOT_FREEZE.json','TRAINING_CHOICE.json','smoke.py','SMOKE.json')]
    paths+=list((HERE/'reference').glob('*'))
    jo(HERE/'FRESH_FREEZE.json',dict(timestamp_utc=now(),config=CONFIG,choice=json.loads((HERE/'TRAINING_CHOICE.json').read_text())['choices'],
       sources={p.relative_to(HERE).as_posix():sha(p) for p in paths if p.is_file()},fresh_generated=False))
def verify():
    frozen=json.loads((HERE/'FRESH_FREEZE.json').read_text());assert frozen['config']==json.loads(json.dumps(CONFIG))
    for p,h in frozen['sources'].items():assert sha(HERE/p)==h,p
    approval=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
    assert approval['fresh_authorized'] and approval['fresh_freeze_sha256']==sha(HERE/'FRESH_FREEZE.json')
    return frozen

def train():
    frozen=verify();choice=frozen['choice']
    for folder in ('states','splits','temperatures'):(HERE/folder).mkdir(exist_ok=False)
    jo(HERE/'TRAINING_START.json',dict(timestamp_utc=now(),freeze_sha256=sha(HERE/'FRESH_FREEZE.json')))
    rows=[];trace_rows=[];manifest={}
    for pair in range(1,17):
      start=time.perf_counter()
      try:
        x,y=model.task.sample_train(2101000+pair);initial=model.task.initialize(2102000+pair)
        xv,yv=model.labeled_inputs(2103000+pair,2048);xc,yc=model.labeled_inputs(2104000+pair,2048)
        assert x.shape==(256,4) and np.sum(y==1)==128
        arrays=dict(x=x,y=y,initial=initial)
        legacy,legacy_info=model.task.train(initial,x,y,steps=500,method='pb',rate=4.,damping=.1)
        arrays['legacy_pb']=legacy
        temp_results={};selected_steps={'legacy_pb':500};training_seconds={'legacy_pb':legacy_info['seconds']}
        for optimizer,family in [('pb','selected_pb'),('gd','selected_gd')]:
            states,trace,index,seconds=model.trajectory(initial,x,y,xv,yv,**choice[optimizer])
            arrays[family]=states[index];arrays[family+'_checkpoints']=states;arrays[family+'_selected_index']=np.array(index)
            selected_steps[family]=trace[index]['step'];training_seconds[family]=seconds
            trace_rows.extend([dict(pair=pair,family=family,selected=r['step']==trace[index]['step'],**r) for r in trace])
        for family in FAMILIES:
            theta=arrays[family];h=model.task.predict(theta,xc);result=model.temperature(h,yc);temp_results[family]=result
            assert result['calibrated_nll']<=result['raw_nll']+1e-12
            arrays[family+'_calibration_logits']=h
            rows.append(dict(pair=pair,family=family,selected_step=selected_steps[family],training_seconds=training_seconds[family],
                training_nll=model.task.loss(theta,x,y),validation_nll=model.task.loss(theta,xv,yv),
                temperature=result['temperature'],beta=result['beta'],temperature_status=result['status'],
                temperature_raw_nll=result['raw_nll'],temperature_calibrated_nll=result['calibrated_nll'],temperature_forward_examples=2048,
                validation_selection_forward_examples=0 if family=='legacy_pb' else 4*2048,additional_validation_diagnostic_forwards=2048))
        state=HERE/'states'/f'{pair}.npz';np.savez_compressed(state,**arrays)
        split=HERE/'splits'/f'{pair}.npz';np.savez_compressed(split,validation_x=xv,validation_y=yv,temperature_x=xc,temperature_y=yc)
        temp=HERE/'temperatures'/f'{pair}.json';jo(temp,dict(timestamp_utc=now(),pair=pair,families=temp_results,label_seed=2104000+pair))
        for path in (state,split,temp):manifest[path.relative_to(HERE).as_posix()]=sha(path)
        print('Trained/calibrated',pair,'seconds',time.perf_counter()-start,flush=True)
      except Exception:
        (HERE/f'FAILURE_training_{pair}.txt').write_text(traceback.format_exc());raise
    cw(HERE/'training.csv',rows);cw(HERE/'validation_checkpoints.csv',trace_rows)
    jo(HERE/'MODEL_COMMIT.json',dict(timestamp_utc=now(),models=48,variants=96,manifest=manifest,freeze_sha256=sha(HERE/'FRESH_FREEZE.json')))

def scaled_grid(base,beta):
    table={k:v for k,v in base.items()};h=beta*base['h'];paired=np.roll(h,h.shape[-1]//2,axis=-1)
    gain=policy.jensen(h,paired);half=h.shape[-1]//2
    table.update(h=h,folded_jensen=.5*(gain[...,:half]+gain[...,half:]),beta=np.array(beta),actual_incremental_forward_examples=np.array(0))
    return table

def calibrate():
    verify();assert (HERE/'MODEL_COMMIT.json').exists();(HERE/'policies').mkdir(exist_ok=False)
    rows=[];manifest={};robustness_models=[]
    for pair in range(1,17):
        x=model.task.sample_inputs(2105000+pair,32768)
        for family,raw_variant,temp_variant in [('legacy_pb','legacy_raw','legacy_temp'),('selected_pb','pb_raw','pb_temp'),('selected_gd','gd_raw','gd_temp')]:
            adapter,theta,_=load_variant(pair,raw_variant);original=adapter.predict;calls=[]
            def counted(th,z):calls.append(len(z));return original(th,z)
            adapter.predict=counted;start=time.perf_counter();basegrid=policy.fit_grid(adapter,theta);grid_calls=calls.copy();calls.clear()
            assert sum(grid_calls)==65536
            h=adapter.predict(theta,x);ht=adapter.predict(theta,adapter.partner(x));assert calls==[32768,32768]
            for variant in (raw_variant,temp_variant):
                a,_,metadata=load_variant(pair,variant);beta=metadata['beta']
                grid=scaled_grid(basegrid,beta);grid['actual_incremental_forward_examples']=np.array(65536 if variant==raw_variant else 0)
                folded=policy.fit_folded(a,theta,x,config=(16,16,16),prepared=(beta*h,beta*ht))
                folded['beta']=np.array(beta);folded['actual_incremental_forward_examples']=np.array(65536 if variant==raw_variant else 0)
                for name,table in [('grid',grid),('folded',folded)]:
                    path=policy_paths(pair,variant)[name];np.savez_compressed(path,**table);manifest[path.relative_to(HERE).as_posix()]=sha(path)
                if variant in ('legacy_raw','pb_temp'):
                    state_path=HERE/'states'/f'{pair}.npz';paths=policy_paths(pair,variant)
                    robustness_models.append(dict(pair=pair,variant=variant,theta_path=state_path.relative_to(ROOT).as_posix(),theta_key=family,beta=beta,
                      grid_path=paths['grid'].relative_to(ROOT).as_posix(),folded_path=paths['folded'].relative_to(ROOT).as_posix(),
                      theta_file_sha256=sha(state_path),grid_sha256=sha(paths['grid']),folded_sha256=sha(paths['folded'])))
                rows.append(dict(pair=pair,family=family,variant=variant,beta=beta,standalone_grid_forwards=65536,standalone_folded_forwards=65536,
                  actual_incremental_grid_forwards=65536 if variant==raw_variant else 0,actual_incremental_folded_forwards=65536 if variant==raw_variant else 0,
                  base_family_elapsed_seconds=time.perf_counter()-start,calibration_input_sha256=ash(x)))
        print('Selector calibration',pair,flush=True)
    cw(HERE/'selector_calibration.csv',rows)
    jo(HERE/'POLICY_COMMIT.json',dict(timestamp_utc=now(),policies=len(manifest),manifest=manifest,
      actual_calibration_forwards=16*3*2*65536,standalone_equivalent_forwards=16*6*2*65536,model_commit_sha256=sha(HERE/'MODEL_COMMIT.json')))
    jo(HERE/'robustness_inputs.json',dict(timestamp_utc=now(),models=robustness_models,policy_commit_sha256=sha(HERE/'POLICY_COMMIT.json'),
       task_source_sha256=sha(HERE/'reference/task.py'),selector_source_sha256=sha(HERE/'reference/baseline.py')))

def deploy(adapter,theta,x,ties,name,tables):
    calls=[]
    def predict(z):calls.append(len(z));return adapter.predict(theta,z)
    if name in ('folded','symmetric_jensen'):
        table=tables['folded' if name=='folded' else 'grid'];h,mask=policy.deploy(adapter,theta,x,table,ties,name,predict)
    else:
        h=predict(x);mask=np.zeros(len(x),bool)
        if name=='full':mask[:]=True;h=.5*(h+predict(adapter.partner(x)))
        elif name!='raw':raise ValueError(name)
    expected=[256] if name=='raw' else [256,256] if name=='full' else [256,64]
    assert calls==expected and mask.sum()==(0 if name=='raw' else 256 if name=='full' else 64)
    return h,mask,calls

def score():
    verify();assert (HERE/'POLICY_COMMIT.json').exists();(HERE/'predictions').mkdir(exist_ok=False)
    jo(HERE/'SCORE_START.json',dict(timestamp_utc=now(),policy_commit_sha256=sha(HERE/'POLICY_COMMIT.json')))
    rows=[];bins=[];manifest={};forwards=0;ledger_count=0
    with gzip.open(HERE/'deployment_ledger.csv.gz','xt',newline='') as f:
      writer=csv.DictWriter(f,['pair','variant','policy','batch','selected','calls','forward_examples']);writer.writeheader()
      for pair in range(1,17):
        x=model.task.sample_inputs(2106000+pair,N);p=model.task.posterior(x)
        ties=np.random.default_rng(2107000+pair).random(N);y=np.where(np.random.default_rng(2108000+pair).random(N)<p,1.,-1.)
        outputs=np.empty((6,4,N));masks=np.zeros((6,4,N),bool)
        for vi,variant in enumerate(VARIANTS):
            adapter,theta,metadata=load_variant(pair,variant);tables={k:dict(np.load(v)) for k,v in policy_paths(pair,variant).items()}
            for pi,name in enumerate(POLICIES):
                start=time.perf_counter();cost=0
                for begin in range(0,N,BATCH):
                    h,mask,calls=deploy(adapter,theta,x[begin:begin+BATCH],ties[begin:begin+BATCH],name,tables)
                    outputs[vi,pi,begin:begin+BATCH]=h;masks[vi,pi,begin:begin+BATCH]=mask;cost+=sum(calls);ledger_count+=1
                    writer.writerow(dict(pair=pair,variant=variant,policy=name,batch=begin//BATCH,selected=int(mask.sum()),calls='|'.join(map(str,calls)),forward_examples=sum(calls)))
                forwards+=cost;h=outputs[vi,pi];q=np.exp(-np.logaddexp(0,-h));nll=np.logaddexp(0,h)-p*h;error=np.where(h>=0,1-p,p)
                index=np.minimum((q*10).astype(int),9);ece=0.
                for b in range(10):
                    keep=index==b;n=int(keep.sum());pred=float(q[keep].mean()) if n else None;truth=float(p[keep].mean()) if n else None
                    if n:ece+=n*abs(pred-truth)
                    bins.append(dict(pair=pair,variant=variant,policy=name,bin=b,count=n,predicted_probability=pred,true_probability=truth))
                rows.append(dict(pair=pair,variant=variant,family=metadata['family'],policy=name,beta=metadata['beta'],expected_nll=float(nll.mean()),
                  expected_error=float(error.mean()),sampled_nll=float(np.logaddexp(0,-y*h).mean()),sampled_error=float(np.mean((h>=0)!=(y==1))),
                  expected_brier=float(np.mean((q-p)**2+p*(1-p))),squared_posterior_error=float(np.mean((q-p)**2)),
                  confidence=float(np.maximum(q,1-q).mean()),accuracy=float(1-error.mean()),posterior_binned_ece=ece/N,
                  bayes_nll=float(np.mean(-p*np.log(p)-(1-p)*np.log1p(-p))),bayes_error=float(np.minimum(p,1-p).mean()),
                  forward_examples=cost,seconds=time.perf_counter()-start))
            if variant.endswith('_temp'):
                assert np.array_equal(outputs[vi,0]>=0,outputs[vi-1,0]>=0),'Positive temperature changed raw signs'
                assert np.array_equal(outputs[vi,1]>=0,outputs[vi-1,1]>=0),'Positive temperature changed full signs'
        path=HERE/'predictions'/f'{pair}.npz';np.savez_compressed(path,x=x,p=p,y=y,ties=ties,variants=np.array(VARIANTS),policies=np.array(POLICIES),outputs=outputs,masks=np.packbits(masks,axis=-1))
        manifest[path.relative_to(HERE).as_posix()]=sha(path);print('Scored',pair,flush=True)
    assert len(rows)==384 and ledger_count==98304 and forwards==34603008,(len(rows),ledger_count,forwards)
    cw(HERE/'per_model.csv',rows);cw(HERE/'reliability_bins.csv',bins)
    jo(HERE/'SCORE_COMPLETE.json',dict(timestamp_utc=now(),rows=len(rows),ledger_rows=ledger_count,actual_forwards=forwards,manifest=manifest,policy_commit_sha256=sha(HERE/'POLICY_COMMIT.json')))

def analyze():
    verify()
    with (HERE/'per_model.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    metrics=['expected_nll','expected_error','sampled_nll','sampled_error','expected_brier','squared_posterior_error','confidence','accuracy','posterior_binned_ece']
    summary=[]
    for variant in VARIANTS:
      for policy_name in POLICIES:
        group=[r for r in rows if r['variant']==variant and r['policy']==policy_name];assert len(group)==16
        summary.append(dict(variant=variant,policy=policy_name,models=16,**{metric:float(np.mean([float(r[metric]) for r in group])) for metric in metrics}))
    cw(HERE/'summary.csv',summary)
    definitions=[('primary_total','pb_temp','raw','legacy_raw','raw',True),('primary_preserved_gain','pb_temp','symmetric_jensen','pb_temp','raw',True),
      ('strong_comparator','pb_temp','symmetric_jensen','pb_temp','folded',False),
      ('training_only','pb_raw','raw','legacy_raw','raw',True),('legacy_temperature','legacy_temp','raw','legacy_raw','raw',True),
      ('selected_temperature','pb_temp','raw','pb_raw','raw',False),('gd_context','gd_temp','raw','pb_temp','raw',False)]
    rng=np.random.default_rng(2109001);draw=rng.integers(0,16,(20000,16));contrasts=[];paired=[]
    def values(variant,name,metric):
        group=sorted([r for r in rows if r['variant']==variant and r['policy']==name],key=lambda r:int(r['pair']))
        return np.array([float(r[metric]) for r in group])
    for label,a,ap,b,bp,primary in definitions:
      for metric in ('expected_nll','expected_error'):
        delta=values(a,ap,metric)-values(b,bp,metric);alpha=.05/4 if primary and metric=='expected_nll' else .05
        interval=np.quantile(delta[draw].mean(1),[alpha/2,1-alpha/2])
        contrasts.append(dict(contrast=label,metric=metric,primary=primary and metric=='expected_nll',a_variant=a,a_policy=ap,b_variant=b,b_policy=bp,
            mean_difference=float(delta.mean()),lower=float(interval[0]),upper=float(interval[1]),alpha=alpha,wins=int(np.sum(delta<0)),ties=int(np.sum(delta==0))))
        paired.extend([dict(contrast=label,metric=metric,pair=i+1,difference=float(v)) for i,v in enumerate(delta)])
    cw(HERE/'contrasts.csv',contrasts);cw(HERE/'paired_contrasts.csv',paired)
    jo(HERE/'ANALYSIS_COMPLETE.json',dict(timestamp_utc=now(),summary_sha256=sha(HERE/'summary.csv'),contrasts_sha256=sha(HERE/'contrasts.csv')))
    print(json.dumps(contrasts,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','calibrate','score','analyze']);args=parser.parse_args()
    with threadpool_limits(limits=1):globals()[args.stage]()
