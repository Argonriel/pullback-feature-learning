"""Frozen paired augmentation study; all artifact text and data are English."""
from pathlib import Path
from datetime import datetime, timezone
import csv, gzip, hashlib, json, struct, time, traceback
import numpy as np
from model import CNN, runtime, predict_chunks, load_model
import torch

HERE = Path(__file__).resolve().parent
CONFIG = dict(pairs=[1,2,3], arms=['noaug','aug'], split_seed=2161000,
    init_base=2162000, order_base=2163000, augmentation_base=2164000,
    epochs=12, validation_epochs=[3,6,9,12], batch=256, learning_rate=.001,
    weight_decay=.0001, beta_interval=[.25,4.], beta_iterations=80,
    bootstrap_seed=2168000, bootstrap_replicates=10000, timing_seed=2169000,
    timing_warmups=3, timing_repeats=7, timing_batches=4)
FILES={'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d',
       'train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
       't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
       't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}
PIPELINES=['raw','raw_Traw','full','full_Traw','full_Tfull']

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest_array(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write_json(path, obj):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2)
def write_csv(path, rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def check_freeze():
    freeze=json.loads((HERE/'SOURCE_FREEZE.json').read_text())
    assert freeze['config']==CONFIG
    for name,digest in freeze['hashes'].items():assert sha(HERE/name)==digest,name
def approval():
    check_freeze();p=HERE.parent/'ROOT_APPROVAL.json'
    assert p.exists(),'Parent scientific release is required before training or outcome access.'
    return json.loads(p.read_text())
def labels_only(prefix):
    z=gzip.decompress((HERE/'data'/f'{prefix}-labels-idx1-ubyte.gz').read_bytes())
    magic,n=struct.unpack('>2I',z[:8]);assert magic==2049
    a=np.frombuffer(z,np.uint8,offset=8).astype(np.int64);assert len(a)==n
    return a
def arrays(prefix):
    approval()
    z=gzip.decompress((HERE/'data'/f'{prefix}-images-idx3-ubyte.gz').read_bytes())
    magic,n,h,w=struct.unpack('>4I',z[:16]);assert (magic,h,w)==(2051,28,28)
    x=np.frombuffer(z,np.uint8,offset=16).reshape(n,1,28,28).copy()
    y=labels_only(prefix);assert len(x)==len(y)
    return x,y

def freeze():
    labels=labels_only('train');prior=dict(np.load(HERE/'prior_splits.npz'))
    rng=np.random.default_rng(CONFIG['split_seed']);held=[]
    for c in range(10):
        ids=prior['train'][labels[prior['train']]==c];assert len(ids)==4000
        held.append(rng.permutation(ids)[:1000])
    held=rng.permutation(np.concatenate(held))
    keep=prior['train'][~np.isin(prior['train'],held)]
    train=rng.permutation(np.concatenate([keep,prior['selector_fit'],prior['selector_validation']]))
    split=dict(train=train,validation=prior['validation'],temperature=prior['temperature'],holdout=held)
    all_ids=np.concatenate(list(split.values()));assert len(all_ids)==len(np.unique(all_ids))==60000
    rows=[]
    for name,ids in split.items():
        counts=np.bincount(labels[ids],minlength=10)
        assert np.all(counts==dict(train=4000,validation=500,temperature=500,holdout=1000)[name])
        rows.append(dict(split=name,count=len(ids),class_counts=counts.tolist(),indices_sha256=digest_array(ids)))
    np.savez_compressed(HERE/'splits.npz',**split)
    data_hashes={}
    for file,expected in FILES.items():
        assert hashlib.md5((HERE/'data'/file).read_bytes()).hexdigest()==expected
        data_hashes['data/'+file]=sha(HERE/'data'/file)
    write_json(HERE/'SPLIT_AUDIT.json',dict(utc=now(),rows=rows,unique_count=60000,
        disjoint=True,holdout_subset_previous_training=True,previous_study_exposure=True,
        previous_weights_reused=False,labels_used_per_arm=50000,images_decoded=False))
    names=['model.py','study.py','PROTOCOL.md','prior_splits.npz','splits.npz','SPLIT_AUDIT.json']
    write_json(HERE/'SOURCE_FREEZE.json',dict(utc=now(),config=CONFIG,
        hashes={**{n:sha(HERE/n) for n in names},**data_hashes},new_training_started=False))

def temp_fit(z,y):
    z=np.asarray(z,np.float64);zy=z[np.arange(len(y)),y]
    def grad(beta):
        t=beta*z;t-=t.max(1,keepdims=True);p=np.exp(t);p/=p.sum(1,keepdims=True)
        return float(np.mean(np.sum(p*z,axis=1)-zy))
    lo,hi=CONFIG['beta_interval'];glo,ghi=grad(lo),grad(hi)
    if glo>=0:beta=lo;status='lower_endpoint'
    elif ghi<=0:beta=hi;status='upper_endpoint'
    else:
        for _ in range(CONFIG['beta_iterations']):
            mid=(lo+hi)/2
            if grad(mid)>0:hi=mid
            else:lo=mid
        beta=(lo+hi)/2;status='interior'
    return dict(beta=float(beta),temperature=float(1/beta),derivative=grad(beta),status=status)
def losses(z,y):
    z=np.asarray(z,np.float64);m=z.max(1)
    return m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]
def state_digest(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train():
    approval();runtime();assert torch.cuda.is_available()
    images,y=arrays('train');split=dict(np.load(HERE/'splits.npz'))
    out=HERE/'models';out.mkdir(exist_ok=False)
    all_x=torch.as_tensor(images,device='cuda',dtype=torch.float32)/255
    all_y=torch.as_tensor(y,device='cuda')
    ids={k:torch.as_tensor(v,device='cuda') for k,v in split.items()}
    write_json(HERE/'TRAINING_START.json',dict(utc=now(),freeze=sha(HERE/'SOURCE_FREEZE.json'),
        approval=sha(HERE.parent/'ROOT_APPROVAL.json'),torch_version=torch.__version__,
        device=torch.cuda.get_device_name(0),threads=1,parameters=sum(p.numel() for p in CNN().parameters())))
    records=[];traces=[];temps=[];manifest={};pair_checks=[]
    for pair in CONFIG['pairs']:
        reference_init=None;reference_orders=None
        for arm in CONFIG['arms']:
            tag=f'{pair}_{arm}';torch.manual_seed(CONFIG['init_base']+pair)
            torch.cuda.manual_seed_all(CONFIG['init_base']+pair)
            net=CNN().cuda();initial_hash=state_digest(net)
            if reference_init is None:reference_init=initial_hash
            assert reference_init==initial_hash
            optimizer=torch.optim.Adam(net.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
            rng=np.random.default_rng(CONFIG['order_base']+pair)
            aug_rng=np.random.default_rng(CONFIG['augmentation_base']+pair)
            order_hashes=[];coin_hashes=[];flip_count=0;best=float('inf');best_epoch=None;local_trace=[]
            torch.cuda.synchronize();start=time.perf_counter()
            try:
                for epoch in range(1,CONFIG['epochs']+1):
                    net.train();order=rng.permutation(len(ids['train']));order_hashes.append(digest_array(order))
                    coins=aug_rng.random(len(order))<.5;coin_hashes.append(digest_array(coins))
                    total=0.
                    for begin in range(0,len(order),CONFIG['batch']):
                        ii=ids['train'][torch.as_tensor(order[begin:begin+256],device='cuda')]
                        xb=all_x[ii]
                        if arm=='aug':
                            chosen=torch.as_tensor(coins[begin:begin+256],device='cuda')
                            xb=xb.clone();xb[chosen]=xb[chosen].flip(-1)
                            flip_count+=int(coins[begin:begin+256].sum())
                        optimizer.zero_grad(set_to_none=True)
                        z=net(xb);loss=torch.nn.functional.cross_entropy(z,all_y[ii]);loss.backward();optimizer.step()
                        total+=float(loss.detach())*len(ii)
                    row=dict(pair=pair,arm=arm,epoch=epoch,training_ce=total/40000,validation_ce='',validation_error='')
                    if epoch in CONFIG['validation_epochs']:
                        net.eval();z=predict_chunks(net,all_x[ids['validation']])
                        val=float(torch.nn.functional.cross_entropy(z,all_y[ids['validation']]))
                        row['validation_ce']=val;row['validation_error']=float((z.argmax(1)!=all_y[ids['validation']]).float().mean())
                        p=out/f'{tag}_epoch{epoch}.pt'
                        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},pair=pair,arm=arm,epoch=epoch),p)
                        manifest[p.name]=sha(p)
                        if val<best:best=val;best_epoch=epoch
                    local_trace.append(row)
                    print(tag,epoch,round(total/40000,5),row['validation_ce'],flush=True)
                torch.cuda.synchronize();train_seconds=time.perf_counter()-start
                if reference_orders is None:reference_orders=order_hashes
                assert reference_orders==order_hashes
                payload=torch.load(out/f'{tag}_epoch{best_epoch}.pt',map_location='cpu',weights_only=True)
                selected=out/f'{tag}.pt';torch.save(payload,selected);manifest[selected.name]=sha(selected)
                net.load_state_dict(payload['state_dict']);net.eval()
                torch.cuda.synchronize();cal_start=time.perf_counter()
                cx=all_x[ids['temperature']]
                h=predict_chunks(net,cx).cpu().numpy();ht=predict_chunks(net,cx.flip(-1)).cpu().numpy()
                m=(h.astype(np.float64)+ht.astype(np.float64))/2
                raw=temp_fit(h,y[split['temperature']]);full=temp_fit(m,y[split['temperature']])
                cal_seconds=time.perf_counter()-cal_start
                cp=out/f'{tag}_calibration.npz'
                np.savez_compressed(cp,raw=h,reflected=ht,labels=y[split['temperature']],indices=split['temperature'],
                    beta_raw=raw['beta'],beta_full=full['beta'])
                manifest[cp.name]=sha(cp)
                temps.append(dict(pair=pair,arm=arm,raw=raw,full=full))
                pair_checks.append(dict(pair=pair,arm=arm,initial_sha256=initial_hash,
                    epoch_order_sha256=order_hashes,augmentation_coin_sha256=coin_hashes,
                    actual_flipped_examples=flip_count,validation_selected_epoch=best_epoch))
                records.append(dict(pair=pair,arm=arm,best_epoch=best_epoch,validation_ce=best,
                    beta_raw=raw['beta'],beta_full=full['beta'],train_seconds=train_seconds,
                    calibration_seconds=cal_seconds,training_forward_backward_examples=480000,
                    validation_forward_examples=20000,calibration_forward_examples=10000))
                traces.extend(local_trace)
            except Exception as exc:
                torch.save(dict(state_dict=net.state_dict(),pair=pair,arm=arm,epoch=epoch),out/f'{tag}_FAILED.pt')
                write_json(out/f'{tag}_FAILED.json',dict(utc=now(),exception=repr(exc),traceback=traceback.format_exc(),trace=local_trace,replaced=False))
                raise
    write_csv(HERE/'training.csv',records);write_csv(HERE/'training_trace.csv',traces)
    write_json(HERE/'temperatures.json',dict(utc=now(),rows=temps))
    write_json(HERE/'PAIRING_AUDIT.json',dict(utc=now(),rows=pair_checks,initializations_and_orders_identical_within_pair=True))
    write_json(HERE/'MODEL_COMMIT.json',dict(utc=now(),manifest=manifest,freeze=sha(HERE/'SOURCE_FREEZE.json'),
        temperature_sha256=sha(HERE/'temperatures.json'),pairing_sha256=sha(HERE/'PAIRING_AUDIT.json'),outcomes_accessed=False))

def models():
    commit=json.loads((HERE/'MODEL_COMMIT.json').read_text());check_freeze()
    for name,h in commit['manifest'].items():assert sha(HERE/'models'/name)==h
    assert sha(HERE/'temperatures.json')==commit['temperature_sha256']
    return json.loads((HERE/'temperatures.json').read_text())['rows']
def outputs(h,ht,row):
    h=np.asarray(h,np.float64);ht=np.asarray(ht,np.float64);m=(h+ht)/2
    return dict(raw=h,raw_Traw=h*row['raw']['beta'],full=m,
        full_Traw=m*row['raw']['beta'],full_Tfull=m*row['full']['beta'])

def score():
    approval();runtime();rows=models();out=HERE/'predictions';out.mkdir(exist_ok=False)
    write_json(HERE/'SCORE_START.json',dict(utc=now(),model_commit=sha(HERE/'MODEL_COMMIT.json')))
    quality=[];classwise=[];diagnostics=[];calls=[];manifest={}
    for dataset in ['holdout','official_test']:
        if dataset=='holdout':
            x,y=arrays('train');ii=np.load(HERE/'splits.npz')['holdout'];x=x[ii];y=y[ii]
        else:x,y=arrays('t10k');ii=np.arange(len(y))
        assert len(y)==10000
        xt=torch.as_tensor(x,device='cuda',dtype=torch.float32)/255
        for row in rows:
            pair,arm=row['pair'],row['arm'];tag=f'{dataset}_{pair}_{arm}'
            net=load_model(HERE/'models'/f'{pair}_{arm}.pt','cuda')
            h=predict_chunks(net,xt).cpu().numpy();ht=predict_chunks(net,xt.flip(-1)).cpu().numpy()
            p=out/f'{tag}.npz';np.savez_compressed(p,raw=h,reflected=ht,labels=y,indices=ii)
            manifest[p.name]=sha(p)
            variants=outputs(h,ht,row)
            for name,z in variants.items():
                ce=losses(z,y);err=z.argmax(1)!=y
                rz=(ht*row['raw']['beta']) if name=='raw_Traw' else ht
                pair_ce=.5*(ce+losses(rz,y)) if name in ['raw','raw_Traw'] else ce
                quality.append(dict(dataset=dataset,pair=pair,arm=arm,pipeline=name,ce=float(ce.mean()),
                    error=float(err.mean()),pair_closed_ce=float(pair_ce.mean()),standalone_forward_examples=10000 if name.startswith('raw') else 20000))
            for c in range(10):
                take=y==c
                classwise.append(dict(dataset=dataset,pair=pair,arm=arm,class_id=c,n=int(take.sum()),
                    original_error=float((h.argmax(1)[take]!=y[take]).mean()),
                    reflected_error=float((ht.argmax(1)[take]!=y[take]).mean()),
                    full_error=float((variants['full'].argmax(1)[take]!=y[take]).mean())))
            diagnostics.append(dict(dataset=dataset,pair=pair,arm=arm,
                flip_agreement=float((h.argmax(1)==ht.argmax(1)).mean()),
                reflected_raw_ce=float(losses(ht,y).mean()),reflected_calibrated_ce=float(losses(ht*row['raw']['beta'],y).mean()),
                reflected_error=float((ht.argmax(1)!=y).mean()),
                raw_argmax_temperature_changes=int(np.sum(variants['raw'].argmax(1)!=variants['raw_Traw'].argmax(1))),
                full_argmax_temperature_changes=int(np.sum(variants['full'].argmax(1)!=variants['full_Tfull'].argmax(1)))))
            calls.append(dict(dataset=dataset,pair=pair,arm=arm,original_calls=10000,reflected_calls=10000,total_actual_forward_examples=20000))
    write_csv(HERE/'quality.csv',quality);write_csv(HERE/'classwise.csv',classwise)
    write_csv(HERE/'diagnostics.csv',diagnostics);write_csv(HERE/'evaluation_calls.csv',calls)
    write_json(HERE/'SCORE_COMPLETE.json',dict(utc=now(),manifest=manifest,actual_forward_examples=sum(r['total_actual_forward_examples'] for r in calls),
        model_commit=sha(HERE/'MODEL_COMMIT.json'),freeze=sha(HERE/'SOURCE_FREEZE.json')))

@torch.inference_mode()
def serve(net,x,method,br,bf,device,transfer):
    if transfer:x=x.to(device)
    if method=='raw':z=net(x)*br
    elif method=='full_fused':
        both=net(torch.cat([x,x.flip(-1)],0));z=((both[:len(x)]+both[len(x):])*.5)*bf
    elif method=='full_sequential':z=((net(x)+net(x.flip(-1)))*.5)*bf
    else:raise ValueError(method)
    return z.cpu() if transfer else z

def benchmark():
    approval();runtime();rows=models();assert (HERE/'SCORE_COMPLETE.json').exists()
    x,y=arrays('train');ii=np.load(HERE/'splits.npz')['validation'][:1024]
    x=torch.as_tensor(x[ii],dtype=torch.float32)/255
    write_json(HERE/'TIMING_START.json',dict(utc=now(),validation_indices_sha256=digest_array(ii),
        source_freeze=sha(HERE/'SOURCE_FREEZE.json'),timing_scope='normalized tensors to final calibrated logits'))
    times=[];calls=[];checks=[]
    for mi,row in enumerate(rows):
        pair,arm=row['pair'],row['arm'];br=row['raw']['beta'];bf=row['full']['beta'];reference={}
        for mode in ['cpu','cuda_resident','cuda_transfer']:
            device='cpu' if mode=='cpu' else 'cuda';transfer=mode=='cuda_transfer'
            net=load_model(HERE/'models'/f'{pair}_{arm}.pt',device)
            xb=x.to(device) if mode=='cuda_resident' else x
            rng=np.random.default_rng(CONFIG['timing_seed']+mi+100*['cpu','cuda_resident','cuda_transfer'].index(mode))
            for repeat in range(-CONFIG['timing_warmups'],CONFIG['timing_repeats']):
                for method in rng.permutation(['raw','full_fused','full_sequential']):
                    if device=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter();last=None
                    for j in range(CONFIG['timing_batches']):
                        last=serve(net,xb[j*256:(j+1)*256],method,br,bf,device,transfer)
                    if device=='cuda':torch.cuda.synchronize()
                    elapsed=(time.perf_counter()-start)*1000/CONFIG['timing_batches']
                    if repeat>=0:times.append(dict(pair=pair,arm=arm,mode=mode,method=method,repeat=repeat,ms_per_batch=elapsed))
                    calls.append(dict(pair=pair,arm=arm,mode=mode,method=method,repeat=repeat,warmup=repeat<0,
                        actual_forward_examples=1024*(1 if method=='raw' else 2)))
                    if repeat==0:
                        value=last.detach().cpu().numpy()
                        if mode=='cpu':reference[method]=value
                        else:checks.append(dict(pair=pair,arm=arm,mode=mode,method=method,
                            max_logit_difference=float(np.max(np.abs(value-reference[method]))),
                            argmax_changes=int(np.sum(value.argmax(1)!=reference[method].argmax(1)))))
            print('Timing complete',pair,arm,mode,flush=True)
    write_csv(HERE/'latency.csv',times);write_csv(HERE/'timing_calls.csv',calls);write_csv(HERE/'cpu_cuda_equivalence.csv',checks)
    write_json(HERE/'TIMING_COMPLETE.json',dict(utc=now(),recorded_rows=len(times),
        recorded_forward_examples=sum(r['actual_forward_examples'] for r in calls if not r['warmup']),
        warmup_forward_examples=sum(r['actual_forward_examples'] for r in calls if r['warmup'])))

def analyze():
    check_freeze();modelrows=models();summary=[];contrasts=[];bootstrap_rows=[]
    q=read_csv(HERE/'quality.csv')
    for dataset in ['holdout','official_test']:
        data={}
        for row in modelrows:
            pair,arm=row['pair'],row['arm'];p=np.load(HERE/'predictions'/f'{dataset}_{pair}_{arm}.npz');y=p['labels']
            for name,z in outputs(p['raw'],p['reflected'],row).items():data[pair,arm,name]=(losses(z,y),(z.argmax(1)!=y).astype(float))
        for arm in CONFIG['arms']:
            for name in PIPELINES:
                rr=[r for r in q if r['dataset']==dataset and r['arm']==arm and r['pipeline']==name]
                summary.append(dict(dataset=dataset,arm=arm,pipeline=name,
                    ce=float(np.mean([float(r['ce']) for r in rr])),error=float(np.mean([float(r['error']) for r in rr])),
                    pair_closed_ce=float(np.mean([float(r['pair_closed_ce']) for r in rr]))))
        specs=[('augmentation_raw',('aug','raw_Traw'),('noaug','raw_Traw')),
               ('augmented_full_vs_raw',('aug','full_Tfull'),('aug','raw_Traw')),
               ('noaug_full_vs_raw',('noaug','full_Tfull'),('noaug','raw_Traw')),
               ('aug_full_own_vs_raw_temperature',('aug','full_Tfull'),('aug','full_Traw')),
               ('noaug_full_own_vs_raw_temperature',('noaug','full_Tfull'),('noaug','full_Traw'))]
        vectors=[];metadata=[]
        for label,a,b in specs:
            for metric_index,metric in enumerate(['ce','error']):
                diff=np.mean([data[p,*a][metric_index]-data[p,*b][metric_index] for p in CONFIG['pairs']],axis=0)
                vectors.append(diff);metadata.append((label,metric,a,b))
        matrix=np.array(vectors);rng=np.random.default_rng(CONFIG['bootstrap_seed']);boot=[]
        # Each resample is one common image-index multiset for every model and pipeline.
        for _ in range(CONFIG['bootstrap_replicates']//20):
            counts=rng.multinomial(10000,np.full(10000,.0001),size=20)
            boot.append((counts@matrix.T)/10000)
        boot=np.concatenate(boot)
        np.savez_compressed(HERE/f'{dataset}_bootstrap.npz',values=boot,contrast_names=np.array([a[0]+'_'+a[1] for a in metadata]))
        for j,(label,metric,a,b) in enumerate(metadata):
            primary=dataset=='holdout' and j<4;alpha=.05/4 if primary else .05
            low,high=np.quantile(boot[:,j],[alpha/2,1-alpha/2])
            contrasts.append(dict(dataset=dataset,contrast=label,metric=metric,mean=float(matrix[j].mean()),
                lower=float(low),upper=float(high),nominal_interval=1-alpha,primary=primary,
                estimand='image resampling conditional on these six fitted models'))
    write_csv(HERE/'summary.csv',summary);write_csv(HERE/'contrasts.csv',contrasts)
    if (HERE/'latency.csv').exists():
        t=read_csv(HERE/'latency.csv');speed=[]
        for pair in CONFIG['pairs']:
            for arm in CONFIG['arms']:
                for mode in ['cpu','cuda_resident','cuda_transfer']:
                    med={m:float(np.median([float(r['ms_per_batch']) for r in t if int(r['pair'])==pair and r['arm']==arm and r['mode']==mode and r['method']==m])) for m in ['raw','full_fused','full_sequential']}
                    best=min(['full_fused','full_sequential'],key=lambda m:med[m])
                    speed.append(dict(pair=pair,arm=arm,mode=mode,raw_ms=med['raw'],full_fused_ms=med['full_fused'],full_sequential_ms=med['full_sequential'],
                        strongest_full=best,strongest_full_ms=med[best],full_over_raw_latency=med[best]/med['raw']))
        write_csv(HERE/'serving_summary.csv',speed)
    write_json(HERE/'ANALYSIS_COMPLETE.json',dict(utc=now(),primary_intervals=4,bootstrap_unit='original image',
        model_replicates=3,independent_training_datasets=1,freeze=sha(HERE/'SOURCE_FREEZE.json')))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','train','score','benchmark','analyze'])
    globals()[p.parse_args().stage]()
