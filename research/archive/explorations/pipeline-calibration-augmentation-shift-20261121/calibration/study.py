"""Equal-label-budget prospective confirmation with fixed-pipeline temperatures."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,csv,gzip,hashlib,importlib.util,json,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits
import model
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pipeline_calibration_fixed_selector',HERE/'reference/baseline.py')
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
FAMILIES=('legacy','selected','extra_train')
ARMS=(('raw_uncal','raw','none'),('raw_rawT','raw','raw'),('full_uncal','full','none'),('full_rawT','full','raw'),('full_finalT','full','full'),
      ('interp_uncal','interp','none'),('interp_rawT','interp','raw'),('interp_finalT','interp','interp'))
N=65536;BATCH=256
CONFIG=dict(pairs=16,families=FAMILIES,arms=ARMS,n_test=N,total_labels=4352,base_train_n=256,extra_train_n=2048,validation_n=2048,
  calibration_n=dict(legacy=4096,selected=2048,extra_train=2048),grid=[16,16,256],
  seeds=dict(train=2151000,extra=2151500,shuffle=2151600,init=2152000,validation=2153000,calibration=2154000,test=2156000,ties=2157000,calibration_ties=2157500,labels=2158000),
  bootstrap_seed=2159000,bootstrap_draws=20000,primary_alpha=.05/4)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def jo(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def cw(p,rows):
    with Path(p).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def adapter():
    t=model.task
    return SimpleNamespace(name='two_invariants',dimension=4,quotient_dim=2,predict=t.predict,partner=t.partner,quotient=t.quotient,
       coordinates=t.coordinates,angle=t.angle,support_lo=np.array([.4,.4]),support_hi=np.array([1.4,1.4]))
def freeze():
    assert not (HERE/'states').exists()
    paths=[HERE/p for p in ('study.py','model.py','PROTOCOL.md','preflight.py','PREFLIGHT.json')]+[p for p in (HERE/'reference').glob('*') if p.is_file()]
    jo(HERE/'SOURCE_FREEZE.json',dict(utc=now(),config=CONFIG,source_hashes={p.relative_to(HERE).as_posix():sha(p) for p in paths},fresh_accessed=False))
def verify():
    f=json.loads((HERE/'SOURCE_FREEZE.json').read_text());assert f['config']==json.loads(json.dumps(CONFIG))
    for p,h in f['source_hashes'].items():assert sha(HERE/p)==h,p
    a=json.loads((HERE/'ROOT_APPROVAL.json').read_text());assert a['fresh_authorized'] and a['source_freeze_sha256']==sha(HERE/'SOURCE_FREEZE.json')
def verify_commit(name):
    record=json.loads((HERE/name).read_text())
    for p,h in record['manifest'].items():assert sha(HERE/p)==h,p
def train():
    verify()
    for name in ('states','splits'):(HERE/name).mkdir(exist_ok=False)
    jo(HERE/'TRAIN_START.json',dict(utc=now(),source_freeze_sha256=sha(HERE/'SOURCE_FREEZE.json')))
    rows=[];traces=[];manifest={}
    for pair in range(1,17):
      try:
        x,y=model.task.sample_train(2151000+pair);xe,ye=model.task.sample_train(2151500+pair,2048)
        permutation=np.random.default_rng(2151600+pair).permutation(2304)
        xx=np.concatenate([x,xe])[permutation];yy=np.concatenate([y,ye])[permutation]
        assert np.sum(y==1)==128 and np.sum(ye==1)==1024 and np.sum(yy==1)==1152
        initial=model.task.initialize(2152000+pair);xv,yv=model.labeled_inputs(2153000+pair,2048);xc,yc=model.labeled_inputs(2154000+pair,4096)
        ties=np.random.default_rng(2157500+pair).random(4096)
        arrays=dict(x=x,y=y,extra_x=xe,extra_y=ye,combined_permutation=permutation,initial=initial)
        for family,tx,ty,rate,damping in [('legacy',x,y,4.,.1),('extra_train',xx,yy,36.,.9)]:
            theta,metadata=model.task.train(initial,tx,ty,steps=500,method='pb',rate=rate,damping=damping)
            arrays[family]=theta
            rows.append(dict(pair=pair,family=family,n_train=len(ty),n_validation_used=0,n_temperature=CONFIG['calibration_n'][family],
               unique_labels=4352,validation_selection_forward_examples=0,rate=rate,damping=damping,selected_step=500,updates_executed=500,training_seconds=metadata['seconds'],training_nll=model.task.loss(theta,tx,ty)))
            traces.extend([dict(pair=pair,family=family,step=int(s),training_nll=float(v),validation_nll='',selected=int(s)==500) for s,v in metadata['loss_trace']])
        states,trace,index,seconds=model.trajectory(initial,x,y,xv,yv,optimizer='pb',rate=2.,lam=0.,damping=.1)
        arrays['selected']=states[index];arrays['selected_checkpoints']=states;arrays['selected_index']=np.array(index)
        rows.append(dict(pair=pair,family='selected',n_train=256,n_validation_used=2048,n_temperature=2048,unique_labels=4352,
              validation_selection_forward_examples=8192,rate=2.,damping=.1,selected_step=trace[index]['step'],updates_executed=1000,training_seconds=seconds,training_nll=trace[index]['training_nll']))
        traces.extend([dict(pair=pair,family='selected',selected=r['step']==trace[index]['step'],**r) for r in trace])
        p=HERE/'states'/f'{pair}.npz';np.savez_compressed(p,**arrays);manifest[p.relative_to(HERE).as_posix()]=sha(p)
        p=HERE/'splits'/f'{pair}.npz';np.savez_compressed(p,validation_x=xv,validation_y=yv,calibration_x=xc,calibration_y=yc,calibration_ties=ties);manifest[p.relative_to(HERE).as_posix()]=sha(p)
        print('Trained',pair,flush=True)
      except Exception:
        (HERE/f'FAILED_TRAIN_{pair}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    cw(HERE/'training.csv',rows);cw(HERE/'training_trace.csv',traces)
    jo(HERE/'MODEL_COMMIT.json',dict(utc=now(),models=48,manifest=manifest,source_freeze_sha256=sha(HERE/'SOURCE_FREEZE.json')))
def fields():
    verify();assert (HERE/'MODEL_COMMIT.json').exists();(HERE/'fields').mkdir(exist_ok=False)
    verify_commit('MODEL_COMMIT.json')
    rows=[];manifest={}
    for pair in range(1,17):
        states=np.load(HERE/'states'/f'{pair}.npz')
        for family in FAMILIES:
            a=adapter();original=a.predict;calls=[]
            def counted(theta,x):calls.append(len(x));return original(theta,x)
            a.predict=counted;start=time.perf_counter();table=selector.fit_grid(a,states[family]);assert sum(calls)==65536
            path=HERE/'fields'/f'{pair}_{family}.npz';np.savez_compressed(path,**table);manifest[path.relative_to(HERE).as_posix()]=sha(path)
            rows.append(dict(pair=pair,family=family,actual_forwards=sum(calls),invocations=len(calls),seconds=time.perf_counter()-start))
        print('Fixed fields',pair,flush=True)
    cw(HERE/'field_costs.csv',rows);jo(HERE/'FIELD_COMMIT.json',dict(utc=now(),actual_forwards=3145728,manifest=manifest,temperatures_fitted=False))
def temperatures():
    verify();assert (HERE/'FIELD_COMMIT.json').exists();(HERE/'temperatures').mkdir(exist_ok=False)
    verify_commit('MODEL_COMMIT.json');verify_commit('FIELD_COMMIT.json')
    rows=[];manifest={};total=0
    for pair in range(1,17):
        states=np.load(HERE/'states'/f'{pair}.npz');split=np.load(HERE/'splits'/f'{pair}.npz')
        for family in FAMILIES:
            n=CONFIG['calibration_n'][family];x=split['calibration_x'][:n];y=split['calibration_y'][:n];ties=split['calibration_ties'][:n]
            table=dict(np.load(HERE/'fields'/f'{pair}_{family}.npz'));theta=states[family];a=adapter();calls=[]
            first=a.predict(theta,x);calls.append(n);second=a.predict(theta,a.partner(x));calls.append(n);total+=sum(calls)
            mask=selector.select(a,x,table,ties);assert np.all(mask.reshape(-1,256).sum(1)==64)
            full=.5*(first+second);output=dict(raw=first,full=full,interp=np.where(mask,full,first));fits={}
            for pipeline,h in output.items():
                fit=model.temperature(h,y);assert fit['calibrated_nll']<=fit['raw_nll']+1e-12;fits[pipeline]=fit
                rows.append(dict(pair=pair,family=family,pipeline=pipeline,n_labels=n,unique_total_labels=4352,
                    beta=fit['beta'],temperature=fit['temperature'],fit_status=fit['status'],uncalibrated_fit_nll=fit['raw_nll'],calibrated_fit_nll=fit['calibrated_nll'],
                    standalone_fit_forward_examples=int(n*{'raw':1,'full':2,'interp':1.25}[pipeline]),shared_family_forwards=2*n))
            path=HERE/'temperatures'/f'{pair}_{family}.json';jo(path,dict(utc=now(),pair=pair,family=family,n_labels=n,fits=fits,
              field_sha256=sha(HERE/'fields'/f'{pair}_{family}.npz'),calibration_labels_sha256=ash(y),calibration_ties_sha256=ash(ties)))
            manifest[path.relative_to(HERE).as_posix()]=sha(path)
            path=HERE/'temperatures'/f'{pair}_{family}_outputs.npz';np.savez_compressed(path,**output,partner=second,mask=mask,labels=y);manifest[path.relative_to(HERE).as_posix()]=sha(path)
        print('Temperatures',pair,flush=True)
    assert total==262144;cw(HERE/'temperature_fits.csv',rows)
    jo(HERE/'TEMPERATURE_COMMIT.json',dict(utc=now(),fits=144,actual_shared_forwards=total,manifest=manifest,field_commit_sha256=sha(HERE/'FIELD_COMMIT.json')))
def deploy(theta,x,ties,table,pipeline,beta):
    a=adapter();calls=[]
    def predict(z):calls.append(len(z));return a.predict(theta,z)
    if pipeline=='interp':h,mask=selector.deploy(a,theta,x,table,ties,'symmetric_jensen',predict)
    else:
        h=predict(x);mask=np.full(len(x),pipeline=='full',dtype=bool)
        if pipeline=='full':h=.5*(h+predict(a.partner(x)))
        elif pipeline!='raw':raise ValueError(pipeline)
    assert calls==({'raw':[256],'full':[256,256],'interp':[256,64]}[pipeline])
    return beta*h,mask,calls
def score():
    verify();assert (HERE/'TEMPERATURE_COMMIT.json').exists();(HERE/'predictions').mkdir(exist_ok=False)
    verify_commit('MODEL_COMMIT.json');verify_commit('FIELD_COMMIT.json');verify_commit('TEMPERATURE_COMMIT.json')
    jo(HERE/'SCORE_START.json',dict(utc=now(),temperature_commit_sha256=sha(HERE/'TEMPERATURE_COMMIT.json')))
    rows=[];bins=[];manifest={};total=0;records=0
    with gzip.open(HERE/'deployment_ledger.csv.gz','xt',newline='') as f:
      writer=csv.DictWriter(f,['pair','family','arm','batch','selected','calls','forwards']);writer.writeheader()
      for pair in range(1,17):
        x=model.task.sample_inputs(2156000+pair,N);p=model.task.posterior(x);ties=np.random.default_rng(2157000+pair).random(N)
        y=np.where(np.random.default_rng(2158000+pair).random(N)<p,1.,-1.);states=np.load(HERE/'states'/f'{pair}.npz')
        outputs=np.empty((3,8,N));masks=np.zeros((3,8,N),bool)
        for fi,family in enumerate(FAMILIES):
            table=dict(np.load(HERE/'fields'/f'{pair}_{family}.npz'));fit=json.loads((HERE/'temperatures'/f'{pair}_{family}.json').read_text())['fits']
            for ai,(arm,pipeline,calibration) in enumerate(ARMS):
                beta=1. if calibration=='none' else fit[calibration]['beta'];start=time.perf_counter();cost=0
                for begin in range(0,N,256):
                    h,mask,calls=deploy(states[family],x[begin:begin+256],ties[begin:begin+256],table,pipeline,beta)
                    outputs[fi,ai,begin:begin+256]=h;masks[fi,ai,begin:begin+256]=mask;cost+=sum(calls);records+=1
                    writer.writerow(dict(pair=pair,family=family,arm=arm,batch=begin//256,selected=int(mask.sum()),calls='|'.join(map(str,calls)),forwards=sum(calls)))
                total+=cost;h=outputs[fi,ai];q=np.exp(-np.logaddexp(0,-h));error=np.where(h>=0,1-p,p);nll=np.logaddexp(0,h)-p*h
                ids=np.minimum((q*10).astype(int),9);ece=0.
                for b in range(10):
                    keep=ids==b;n=int(keep.sum());pred=float(q[keep].mean()) if n else None;truth=float(p[keep].mean()) if n else None
                    if n:ece+=n*abs(pred-truth)
                    bins.append(dict(pair=pair,family=family,arm=arm,bin=b,count=n,predicted_probability=pred,true_probability=truth))
                rows.append(dict(pair=pair,family=family,arm=arm,pipeline=pipeline,temperature_fit=calibration,beta=beta,
                    unique_labels=(256 if family=='legacy' else 2304) if calibration=='none' else 4352,
                    expected_nll=float(nll.mean()),expected_error=float(error.mean()),sampled_nll=float(np.logaddexp(0,-y*h).mean()),sampled_error=float(np.mean((h>=0)!=(y==1))),
                    expected_brier=float(np.mean((q-p)**2+p*(1-p))),posterior_binned_ece=ece/N,confidence=float(np.maximum(q,1-q).mean()),accuracy=float(1-error.mean()),
                    bayes_nll=float(np.mean(-p*np.log(p)-(1-p)*np.log1p(-p))),bayes_error=float(np.minimum(p,1-p).mean()),actual_forwards=cost,seconds=time.perf_counter()-start))
                first={'raw':0,'full':2,'interp':5}[pipeline]
                assert np.array_equal(outputs[fi,first]>=0,h>=0),'Positive scaling changed fixed-pipeline class decisions'
                assert np.array_equal(masks[fi,first],masks[fi,ai]),'Calibration changed fixed masks'
        path=HERE/'predictions'/f'{pair}.npz';np.savez_compressed(path,x=x,p=p,y=y,ties=ties,families=FAMILIES,arms=np.array([a[0] for a in ARMS]),outputs=outputs,masks=np.packbits(masks,axis=-1));manifest[path.relative_to(HERE).as_posix()]=sha(path)
        print('Scored',pair,flush=True)
    assert len(rows)==384 and records==98304 and total==36962304,(len(rows),records,total)
    cw(HERE/'per_model.csv',rows);cw(HERE/'reliability_bins.csv',bins)
    jo(HERE/'SCORE_COMPLETE.json',dict(utc=now(),rows=len(rows),batch_records=records,actual_forwards=total,manifest=manifest,temperature_commit_sha256=sha(HERE/'TEMPERATURE_COMMIT.json')))
def analyze():
    verify()
    with (HERE/'per_model.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    metrics=('expected_nll','expected_error','sampled_nll','sampled_error','expected_brier','posterior_binned_ece','confidence','accuracy')
    summary=[]
    for family in FAMILIES:
      for arm,_,_ in ARMS:
        rs=[r for r in rows if r['family']==family and r['arm']==arm];assert len(rs)==16
        summary.append(dict(family=family,arm=arm,models=16,**{key:float(np.mean([float(r[key]) for r in rs])) for key in metrics}))
    cw(HERE/'summary.csv',summary)
    definitions=[('selected_full_final_vs_rawT','selected','full_finalT','selected','full_rawT'),('selected_interp_final_vs_rawT','selected','interp_finalT','selected','interp_rawT'),
                 ('extra_training_vs_legacy_rawT','extra_train','raw_rawT','legacy','raw_rawT'),('selected_vs_legacy_rawT','selected','raw_rawT','legacy','raw_rawT')]
    rng=np.random.default_rng(2159000);draw=rng.integers(0,16,(20000,16));contrasts=[];paired=[]
    def vals(fam,arm,key):return np.array([float(r[key]) for r in sorted([r for r in rows if r['family']==fam and r['arm']==arm],key=lambda r:int(r['pair']))])
    for label,fa,aa,fb,ab in definitions:
      for key in ('expected_nll','expected_error'):
        d=vals(fa,aa,key)-vals(fb,ab,key);alpha=.05/4 if key=='expected_nll' else .05
        ci=np.quantile(d[draw].mean(1),[alpha/2,1-alpha/2]);contrasts.append(dict(contrast=label,metric=key,primary=key=='expected_nll',mean_difference=float(d.mean()),lower=float(ci[0]),upper=float(ci[1]),alpha=alpha,wins=int(np.sum(d<0)),ties=int(np.sum(d==0))))
        paired.extend([dict(contrast=label,metric=key,pair=i+1,difference=float(v)) for i,v in enumerate(d)])
    cw(HERE/'contrasts.csv',contrasts);cw(HERE/'paired_contrasts.csv',paired)
    jo(HERE/'ANALYSIS_COMPLETE.json',dict(utc=now(),summary_sha256=sha(HERE/'summary.csv'),contrasts_sha256=sha(HERE/'contrasts.csv')))
    print(json.dumps(contrasts,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','fields','temperatures','score','analyze']);args=parser.parse_args()
    with threadpool_limits(limits=1):globals()[args.stage]()
