"""Frozen independent validation on rings and a two-harmonic group action."""
from pathlib import Path
import argparse,csv,datetime,gzip,hashlib,json,sys,time
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from adapters import get_adapter,nll,worlds,fit_folded,fit_angles,deploy

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.dont_write_bytecode=True
COHORTS={'rings':dict(train=1001000,init=1002000,rank=1003000,folded=1004000,
    test=1005000,ties=1006000,actions=1007000,labels=1008000),
    'transfer':dict(train=1041000,init=1042000,rank=1043000,folded=1044000,
    test=1045000,ties=1046000,actions=1047000,labels=1048000)}
POLICIES=['raw','full','confidence','folded','rank_balanced','rank_even',
          'rank_unbalanced','confidence_radial','full_radial']
CONFIG=dict(cohorts=COHORTS,n_pairs=16,test_n=65536,batch=256,quota=64,tau=1.,
    rank_forwards=66048,rank_coordinate_inputs=131072,folded_inputs=33024,
    angle_quotients=64,angle_nodes=128,angle_candidate_indices=list(range(1,65)),
    angle_bins=16,minimum_independent_quotients=4,bootstrap_seed=1009000,
    bootstrap_repeats=20000,primary_methods=['pb'],secondary_transfer_methods=['gd'],
    primary_references=['confidence','rank_even'],primary_family_size=4,
    ordinary_interval=.95,family_adjusted_interval=.9875,
    secondary_worlds=['half','increasing','decreasing','threshold'],no_retuning=True)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(v):return hashlib.sha256(v).hexdigest()
def jo(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def co(path,rows):
    with path.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def ci(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def freeze():
    files=[Path(__file__),HERE/'adapters.py',HERE/'PROTOCOL.md',
        ROOT/'reference'/'rank_policy.py',ROOT/'source'/'pullback_feature_learning'/'core.py',
        ROOT/'legacy'/'budgeted-rotation-final'/'generator.py',ROOT/'transfer'/'task.py',
        ROOT/'transfer'/'rank_policy.py',ROOT/'transfer'/'FREEZE.json',ROOT/'transfer'/'TRAINING_CHOICE.json']
    training={name:get_adapter(name).training for name in COHORTS}
    jo(HERE/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,policies=POLICIES,
        training=training,secondary_policy_names=['raw','full'],
        source_hashes={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in files},
        fresh_state_directory_absent=not(HERE/'states').exists(),fresh_outcomes_accessed=False))

def verify(approval=False):
    frozen=json.loads((HERE/'freeze.json').read_text(encoding='utf8'))
    assert frozen['config']==CONFIG and frozen['policies']==POLICIES
    for path,digest in frozen['source_hashes'].items():assert sha((ROOT/path).read_bytes())==digest,path
    if approval:
        auth=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
        assert auth['fresh_training_authorized'] is True
        assert auth['validation_freeze_sha256']==sha((HERE/'freeze.json').read_bytes())
    return frozen

def train():
    frozen=verify(approval=True);folder=HERE/'states';folder.mkdir(exist_ok=False)
    records=[];failures=[]
    jo(HERE/'training_start.json',dict(timestamp_utc=now(),approval_sha256=sha((HERE/'ROOT_APPROVAL.json').read_bytes())))
    for cohort,seeds in COHORTS.items():
        adapter=get_adapter(cohort)
        assert adapter.training==frozen['training'][cohort]
        for pair in range(1,17):
            x,y=adapter.sample_train(seeds['train']+pair);initial=adapter.initialize(seeds['init']+pair)
            assert x.shape==(256,adapter.dimension) and initial.shape==(adapter.parameters,)
            assert np.sum(y==1)==128 and np.sum(y==-1)==128
            states=dict(x=x,y=y,initial=initial);temporary=[]
            for method in adapter.methods:
                theta=initial.copy();cfg=adapter.training[method];trace=[];start=time.perf_counter()
                try:
                    for step in range(1,cfg['steps']+1):
                        theta+=cfg['rate']*adapter.direction(theta,x,y,method,cfg['damping'])
                        if not np.isfinite(theta).all():raise FloatingPointError(f'Nonfinite state at step {step}')
                        if step==100:states[method+'100']=theta.copy()
                        if step%100==0:trace.append((step,adapter.loss(theta,x,y)))
                    states[method]=theta;states[method+'_loss_trace']=np.array(trace)
                    temporary.append(dict(cohort=cohort,pair=pair,method=method,
                        train_seed=seeds['train']+pair,init_seed=seeds['init']+pair,
                        parameters=adapter.parameters,steps=cfg['steps'],rate=cfg['rate'],
                        damping=cfg['damping'],training_seconds=time.perf_counter()-start,
                        final_training_loss=trace[-1][1],x_sha256=sha(x.tobytes()),
                        y_sha256=sha(y.tobytes()),initial_sha256=sha(initial.tobytes()),
                        state_sha256=sha(theta.tobytes())))
                except Exception as error:
                    failure=dict(cohort=cohort,pair=pair,method=method,step=step,
                        exception=type(error).__name__,message=str(error))
                    failures.append(failure)
                    np.savez_compressed(folder/f'{cohort}_{pair}_{method}_failure.npz',**states,last_theta=theta)
                    jo(folder/f'{cohort}_{pair}_{method}_failure.json',failure)
                    raise
            path=folder/f'{cohort}_{pair}.npz';np.savez_compressed(path,**states)
            for record in temporary:record['state_file_sha256']=sha(path.read_bytes())
            records.extend(temporary);print(f'Trained {cohort} pair {pair}/16',flush=True)
    co(HERE/'training.csv',records)
    jo(HERE/'training_commit.json',dict(timestamp_utc=now(),failures=failures,n_records=len(records),
        files={p.name:sha(p.read_bytes()) for p in folder.glob('*.npz')},
        freeze_sha256=sha((HERE/'freeze.json').read_bytes()),
        approval_sha256=sha((HERE/'ROOT_APPROVAL.json').read_bytes())))

def measured_fit(adapter,function):
    original=adapter.network.representation;calls=[]
    def counted(theta,x):calls.append(len(x));return original(theta,x)
    adapter.network.representation=counted;start=time.perf_counter()
    try:result=function()
    finally:
        elapsed=time.perf_counter()-start;adapter.network.representation=original
    return result,elapsed,calls

def calibrate():
    verify();training=json.loads((HERE/'training_commit.json').read_text())
    assert training['n_records']==48 and not training['failures']
    folder=HERE/'policies';folder.mkdir(exist_ok=False);records=[]
    jo(HERE/'calibration_start.json',dict(timestamp_utc=now(),training_commit_sha256=sha((HERE/'training_commit.json').read_bytes())))
    for cohort,seeds in COHORTS.items():
        adapter=get_adapter(cohort)
        for pair in range(1,17):
            path=HERE/'states'/f'{cohort}_{pair}.npz'
            assert sha(path.read_bytes())==training['files'][path.name]
            theta=np.load(path)['pb']
            try:
                table,elapsed,calls=measured_fit(adapter,lambda:adapter.rank.calibrate(theta,seeds['rank']+pair,tau=1.))
                assert sum(calls)==66048
                np.savez_compressed(folder/f'{cohort}_{pair}_pb_rank.npz',**table)
                records.append(dict(cohort=cohort,pair=pair,method='pb',kind='rank',
                    forward_examples=sum(calls),forward_invocations=len(calls),
                    independent_network_inputs=0,coordinate_only_inputs=131072,
                    calibration_seed=seeds['rank']+pair,seconds=elapsed))
                x=adapter.sample_inputs(seeds['folded']+pair,33024)
                table,elapsed,calls=measured_fit(adapter,lambda:fit_folded(adapter,theta,x))
                assert calls==[33024,33024]
                np.savez_compressed(folder/f'{cohort}_{pair}_pb_folded.npz',**table)
                records.append(dict(cohort=cohort,pair=pair,method='pb',kind='folded',
                    forward_examples=sum(calls),forward_invocations=len(calls),
                    independent_network_inputs=33024,coordinate_only_inputs=0,
                    calibration_seed=seeds['folded']+pair,seconds=elapsed))
                r=adapter.quotient(adapter.sample_inputs(seeds['rank']+100+pair,64))
                phase=np.random.default_rng(seeds['rank']+200+pair).uniform(0,2*np.pi,64)
                table,elapsed,calls=measured_fit(adapter,lambda:fit_angles(adapter,theta,r,phase))
                assert calls==[8192]
                np.savez_compressed(folder/f'{cohort}_{pair}_pb_angles.npz',**table)
                records.append(dict(cohort=cohort,pair=pair,method='pb',kind='angles',
                    forward_examples=sum(calls),forward_invocations=len(calls),
                    independent_network_inputs=64,coordinate_only_inputs=0,
                    calibration_seed=seeds['rank']+100+pair,seconds=elapsed))
            except Exception as error:
                jo(HERE/f'calibration_failure_{cohort}_{pair}.json',dict(cohort=cohort,pair=pair,
                    exception=type(error).__name__,message=str(error)))
                raise
            print(f'Calibrated {cohort} pair {pair}/16',flush=True)
    co(HERE/'calibration.csv',records)
    jo(HERE/'policy_commit.json',dict(timestamp_utc=now(),test_outcomes_accessed=False,
        manifest={p.relative_to(HERE).as_posix():sha(p.read_bytes()) for p in folder.glob('*.npz')},
        calibration_sha256=sha((HERE/'calibration.csv').read_bytes()),
        training_commit_sha256=sha((HERE/'training_commit.json').read_bytes())))

def load_tables(cohort,pair):
    return {kind:dict(np.load(HERE/'policies'/f'{cohort}_{pair}_pb_{kind}.npz'))
            for kind in ('rank','folded','angles')}

def score():
    verify();commit=json.loads((HERE/'policy_commit.json').read_text())
    for path,digest in commit['manifest'].items():assert sha((HERE/path).read_bytes())==digest,path
    jo(HERE/'score_start.json',dict(timestamp_utc=now(),policy_commit_sha256=sha((HERE/'policy_commit.json').read_bytes())))
    folder=HERE/'predictions';folder.mkdir(exist_ok=False)
    rows=[];audit=[];batches=[];samples=[]
    for cohort,seeds in COHORTS.items():
        adapter=get_adapter(cohort)
        for pair in range(1,17):
            state=np.load(HERE/'states'/f'{cohort}_{pair}.npz');tables=load_tables(cohort,pair)
            x=adapter.sample_inputs(seeds['test']+pair,65536);p=adapter.posterior(x)
            ties=np.random.default_rng(seeds['ties']+pair).random(len(x))
            actions=np.random.default_rng(seeds['actions']+pair).random(len(x))
            y=(np.random.default_rng(seeds['labels']+pair).random(len(x))<p).astype(float)
            alternate=worlds(adapter,x)
            samples.append(dict(cohort=cohort,pair=pair,test_seed=seeds['test']+pair,
                ties_seed=seeds['ties']+pair,actions_seed=seeds['actions']+pair,label_seed=seeds['labels']+pair,
                x_sha256=sha(x.tobytes()),ties_sha256=sha(ties.tobytes()),actions_sha256=sha(actions.tobytes()),
                labels_sha256=sha(y.tobytes()),n_test=len(x),n_positive_labels=int(y.sum())))
            for method in adapter.methods:
                theta=state[method];names=POLICIES if method=='pb' else ['raw','full'];outputs=[];masks=[]
                for name in names:
                    result=np.empty(len(x));selected=np.zeros(len(x),dtype=bool);start=time.perf_counter()
                    for batch in range(256):
                        sl=slice(batch*256,(batch+1)*256);calls=[]
                        def counted(z):calls.append(len(z));return adapter.predict(theta,z)
                        out,mask=deploy(adapter,theta,x[sl],name,tables,ties[sl],actions[sl],counted)
                        expected=[256] if name=='raw' else ([256,256] if name in ('full','full_radial') else [256,64])
                        assert calls==expected,(cohort,pair,method,name,batch,calls)
                        assert np.isfinite(out).all()
                        result[sl]=out;selected[sl]=mask
                        audit.append(dict(cohort=cohort,pair=pair,method=method,policy=name,batch=batch,
                            calls=json.dumps(calls),selected=int(mask.sum()),mask_sha256=sha(np.packbits(mask).tobytes())))
                        batches.append(dict(cohort=cohort,pair=pair,method=method,policy=name,batch=batch,
                            expected_risk=float(nll(out,p[sl]).mean()),sampled_label_risk=float(nll(out,y[sl]).mean())))
                    row=dict(cohort=cohort,pair=pair,method=method,policy=name,
                        expected_risk=float(nll(result,p).mean()),sampled_label_risk=float(nll(result,y).mean()),
                        expected_error=float(np.where(result>=0,1-p,p).mean()),
                        sampled_error=float(((result>=0)!=y).mean()),
                        first_forwards=len(x),second_forwards=int(selected.sum()),
                        calls_per_input=(len(x)+selected.sum())/len(x),scoring_seconds=time.perf_counter()-start,
                        output_sha256=sha(result.tobytes()),mask_sha256=sha(np.packbits(selected).tobytes()))
                    for world,posterior in alternate.items():row[world+'_risk']=float(nll(result,posterior).mean())
                    rows.append(row);outputs.append(result);masks.append(np.packbits(selected))
                np.savez_compressed(folder/f'{cohort}_{pair}_{method}.npz',names=np.array(names),
                    outputs=np.array(outputs),masks=np.array(masks))
            print(f'Scored {cohort} pair {pair}/16',flush=True)
    co(HERE/'per_model.csv',rows);co(HERE/'samples.csv',samples)
    for filename,records in [('deployment_audit.csv.gz',audit),('per_batch.csv.gz',batches)]:
        with gzip.open(HERE/filename,'xt',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)
    jo(HERE/'score_complete.json',dict(timestamp_utc=now(),n_primary_models=32,n_context_models=16,
        n_model_policy_rows=len(rows),n_batch_audits=len(audit),all_call_quota_checks_pass=True,
        policy_commit_sha256=sha((HERE/'policy_commit.json').read_bytes()),
        outcomes_sha256=sha((HERE/'per_model.csv').read_bytes()),threadpools=threadpool_info()))

def analyze():
    verify();rows=ci(HERE/'per_model.csv');rng=np.random.default_rng(1009000)
    draws={cohort:rng.integers(0,16,(20000,16)) for cohort in COHORTS}
    summaries=[];contrasts=[]
    fields=['expected_risk','sampled_label_risk','expected_error','sampled_error',
            'half_risk','increasing_risk','decreasing_risk','threshold_risk']
    for cohort in COHORTS:
        draw=draws[cohort]
        for method in get_adapter(cohort).methods:
            names=POLICIES if method=='pb' else ['raw','full']
            index={(r['policy'],int(r['pair'])):r for r in rows if r['cohort']==cohort and r['method']==method}
            comparisons=([('rank_balanced',ref) for ref in ['confidence','rank_even','folded','rank_unbalanced','raw','full','confidence_radial']]+[('confidence_radial','confidence')]) if method=='pb' else [('full','raw')]
            for field in fields:
                values={name:np.array([float(index[name,pair][field]) for pair in range(1,17)]) for name in names}
                for name,v in values.items():
                    boot=v[draw].mean(axis=1)
                    summaries.append(dict(cohort=cohort,method=method,policy=name,outcome=field,n_models=16,
                        mean=float(v.mean()),model_sd=float(v.std(ddof=1)),
                        low95=float(np.quantile(boot,.025)),high95=float(np.quantile(boot,.975))))
                for candidate,reference in comparisons:
                    delta=values[candidate]-values[reference];boot=delta[draw].mean(axis=1)
                    primary=method=='pb' and field=='expected_risk' and candidate=='rank_balanced' and reference in ('confidence','rank_even')
                    contrasts.append(dict(cohort=cohort,method=method,policy=candidate,reference=reference,
                        outcome=field,primary=primary,n_models=16,delta_risk=float(delta.mean()),
                        wins=int((delta<0).sum()),low95=float(np.quantile(boot,.025)),high95=float(np.quantile(boot,.975)),
                        low98_75=float(np.quantile(boot,.00625)),high98_75=float(np.quantile(boot,.99375))))
    co(HERE/'summary.csv',summaries);co(HERE/'contrasts.csv',contrasts)
    jo(HERE/'analysis_complete.json',dict(timestamp_utc=now(),primary_family_size=4,
        bootstrap_unit='paired trained-model seed within cohort',bootstrap_repeats=20000,
        secondary_gd_context_not_an_additional_selector_cohort=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','calibrate','score','analyze'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
