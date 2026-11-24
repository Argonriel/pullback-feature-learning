"""Frozen offline two-view distillation experiment. See PROTOCOL.md."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime, timezone
import csv, gzip, hashlib, json, math, time, traceback
import numpy as np
from model import CNN, runtime, predict_chunks, load_model
import torch
import torch.nn.functional as fn

HERE=Path(__file__).resolve().parent
CONFIG=dict(pairs=[1,2,3,4,5,6],init_base=2422000,order_base=2423000,flip_base=2424000,
    batch=256,epochs=12,lr=.001,weight_decay=.0001,distill_temperature=2.,alpha=.5,
    beta_interval=[.25,4.],beta_iterations=80,bootstrap_seed=2428000,
    bootstrap_replicates=20000,timing_seed=2429000)
F=2*(28*28*16*9+14*14*32*16*9+7*7*64*32*9+3136*128+128*10)
G=3*F-2*28*28*16*9
CONTROL_EXAMPLES=(960000*G+110000*F)//G
ARMS=['teacher','student','control']
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def adigest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def jwrite(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,indent=2)
def cwrite(p,rows):
    with Path(p).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def cread(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def check():
    obj=json.loads((HERE/'SOURCE_FREEZE.json').read_text());assert obj['config']==CONFIG
    for p,h in obj['hashes'].items():assert sha(HERE/p)==h,p
def approve():
    check();release=HERE.parent/'ROOT_APPROVAL.json';assert release.exists(),'Scientific release required.'
    assert json.loads(release.read_text())['distillation_freeze_sha256']==sha(HERE/'SOURCE_FREEZE.json'),'Release must match this source freeze.'
def fashion():
    x=np.frombuffer(gzip.decompress((HERE/'data'/'train-images-idx3-ubyte.gz').read_bytes()),np.uint8,offset=16).reshape(-1,1,28,28).copy()
    y=np.frombuffer(gzip.decompress((HERE/'data'/'train-labels-idx1-ubyte.gz').read_bytes()),np.uint8,offset=8).astype(np.int64)
    return x,y
def lossvec(z,y):
    z=np.asarray(z,np.float64);m=z.max(1)
    return m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]
def fit(z,y):
    z=np.asarray(z,np.float64);zy=z[np.arange(len(y)),y]
    def grad(b):
        a=b*z;a-=a.max(1,keepdims=True);p=np.exp(a);p/=p.sum(1,keepdims=True)
        return float(np.mean((p*z).sum(1)-zy))
    lo,hi=CONFIG['beta_interval'];gl,gh=grad(lo),grad(hi)
    if gl>=0:b=lo;status='lower_endpoint'
    elif gh<=0:b=hi;status='upper_endpoint'
    else:
        for _ in range(CONFIG['beta_iterations']):
            mid=(lo+hi)/2
            if grad(mid)>0:hi=mid
            else:lo=mid
        b=(lo+hi)/2;status='interior'
    return dict(beta=b,status=status,derivative=grad(b),ce=float(lossvec(z*b,y).mean()))
def statehash(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def freeze():
    paths=['PROTOCOL.md','study.py','model.py','prepare_zappos.py','inference.py','splits.npz',
        'external/POPULATION_FREEZE.json','external/population_manifest.json','external/population.npz',
        'external/meta-data.csv','external/DOWNLOAD.json','external/ACQUISITION_METADATA.json','external/readme.txt','external/index.html']
    paths+=['data/'+p.name for p in (HERE/'data').iterdir() if p.is_file()]
    paths+=[str(p.relative_to(HERE)).replace('\\','/') for folder in ['external/selected_raw','legacy_external'] for p in (HERE/folder).rglob('*') if p.is_file()]
    jwrite(HERE/'SOURCE_FREEZE.json',dict(utc=now(),config=CONFIG,hashes={p:sha(HERE/p) for p in paths},
        forward_matrix_flops=F,training_matrix_flops=G,control_training_examples=CONTROL_EXAMPLES,
        distillation_total_matrix_flops=960000*G+135000*F,
        control_total_matrix_flops=CONTROL_EXAMPLES*G+25000*F,
        new_models_trained=False,new_outcomes_scored=False))
    print('Frozen',F,G,CONTROL_EXAMPLES,flush=True)

def train():
    approve();runtime();assert torch.cuda.is_available()
    images,y=fashion();splits=dict(np.load(HERE/'splits.npz'))
    x=torch.as_tensor(images,device='cuda',dtype=torch.float32)/255
    yy=torch.as_tensor(y,device='cuda');trainids=torch.as_tensor(splits['train'],device='cuda')
    valids=torch.as_tensor(splits['validation'],device='cuda');calids=torch.as_tensor(splits['temperature'],device='cuda')
    out=HERE/'models';out.mkdir(exist_ok=False)
    jwrite(HERE/'TRAINING_START.json',dict(utc=now(),torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(0),
        source_freeze=sha(HERE/'SOURCE_FREEZE.json'),parameters=sum(p.numel() for p in CNN().parameters()),
        approval_sha256=sha(HERE.parent/'ROOT_APPROVAL.json')))
    records=[];traces=[];fits=[];pairings=[];manifest={}
    for pair in CONFIG['pairs']:
        q=None;ref=None;ref_orders=None;ref_coins=None
        for arm in ARMS:
            torch.cuda.synchronize();arm_start=time.perf_counter()
            tag=f'{pair}_{arm}';torch.manual_seed(CONFIG['init_base']+pair);torch.cuda.manual_seed_all(CONFIG['init_base']+pair)
            net=CNN().cuda();initial=statehash(net)
            if ref is None:ref=initial
            assert ref==initial
            optim=torch.optim.Adam(net.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
            rng=np.random.default_rng(CONFIG['order_base']+pair);coins_rng=np.random.default_rng(CONFIG['flip_base']+pair)
            budget=CONTROL_EXAMPLES if arm=='control' else 480000
            marks=[math.ceil(budget*k/4) for k in range(1,5)]
            seen=steps=epoch=flips=0;best=float('inf');best_ck=None;selection=0;orders=[];coins_hash=[]
            checkpoint_seconds=0.;validation_seconds=0.;train_start=time.perf_counter();torch.cuda.synchronize()
            try:
                while seen<budget:
                    epoch+=1;net.train();order=rng.permutation(40000);coins=coins_rng.random(40000)<.5
                    orders.append(adigest(order));coins_hash.append(adigest(coins));epoch_loss=0.;epoch_n=0
                    for begin in range(0,40000,CONFIG['batch']):
                        n=min(CONFIG['batch'],40000-begin,budget-seen)
                        if n<=0:break
                        loc=torch.as_tensor(order[begin:begin+n],device='cuda');ii=trainids[loc]
                        xb=x[ii].clone();flip=torch.as_tensor(coins[begin:begin+n],device='cuda');xb[flip]=xb[flip].flip(-1)
                        flips+=int(coins[begin:begin+n].sum());optim.zero_grad(set_to_none=True);z=net(xb)
                        loss=fn.cross_entropy(z,yy[ii])
                        if arm=='student':
                            t=CONFIG['distill_temperature'];loss=(1-CONFIG['alpha'])*loss+CONFIG['alpha']*t*t*fn.kl_div(fn.log_softmax(z/t,dim=1),q[loc],reduction='batchmean')
                        loss.backward();optim.step();steps+=1;seen+=n;epoch_n+=n;epoch_loss+=float(loss.detach())*n
                        if selection<4 and seen>=marks[selection]:
                            torch.cuda.synchronize();vt=time.perf_counter();net.eval()
                            vz=predict_chunks(net,x[valids]);v=float(fn.cross_entropy(vz,yy[valids]));ve=float((vz.argmax(1)!=yy[valids]).float().mean())
                            torch.cuda.synchronize();validation_seconds+=time.perf_counter()-vt
                            ct=time.perf_counter();cp=out/f'{tag}_checkpoint{selection+1}.pt'
                            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},pair=pair,arm=arm,epoch=epoch,training_examples=seen,steps=steps),cp)
                            manifest[cp.name]=sha(cp);checkpoint_seconds+=time.perf_counter()-ct
                            if v<best:best=v;best_ck=cp.name
                            traces.append(dict(pair=pair,arm=arm,checkpoint=selection+1,epoch=epoch,steps=steps,training_examples=seen,validation_ce=v,validation_error=ve))
                            print('CHECKPOINT',tag,selection+1,seen,v,flush=True);selection+=1;net.train()
                    print('TRAIN',tag,epoch,seen,epoch_loss/epoch_n,flush=True)
                torch.cuda.synchronize();training_total=time.perf_counter()-train_start
                assert selection==4
                if ref_orders is None:ref_orders=orders;ref_coins=coins_hash
                assert orders[:12]==ref_orders and coins_hash[:12]==ref_coins
                payload=torch.load(out/best_ck,map_location='cpu',weights_only=True);selected=out/f'{tag}.pt';torch.save(payload,selected);manifest[selected.name]=sha(selected)
                net.load_state_dict(payload['state_dict']);net.eval()
                ts=time.perf_counter();h=predict_chunks(net,x[calids]).cpu().numpy();ht=None
                if arm=='teacher':ht=predict_chunks(net,x[calids].flip(-1)).cpu().numpy()
                torch.cuda.synchronize();cal_forward_seconds=time.perf_counter()-ts
                ts=time.perf_counter();rawfit=fit(h,y[splits['temperature']]);fullfit=fit((h.astype(np.float64)+ht)/2,y[splits['temperature']]) if arm=='teacher' else None
                scalar_seconds=time.perf_counter()-ts
                p=out/f'{tag}_calibration.npz';np.savez_compressed(p,raw=h,reflected=ht if ht is not None else np.empty((0,10)),labels=y[splits['temperature']],indices=splits['temperature']);manifest[p.name]=sha(p)
                target_seconds=0.
                if arm=='teacher':
                    ts=time.perf_counter();a=predict_chunks(net,x[trainids]);b=predict_chunks(net,x[trainids].flip(-1));q=torch.softmax((a+b)*.5/CONFIG['distill_temperature'],dim=1).detach()
                    torch.cuda.synchronize();target_seconds=time.perf_counter()-ts
                    p=out/f'{pair}_targets.npz';np.savez_compressed(p,raw=a.cpu().numpy(),reflected=b.cpu().numpy(),targets=q.cpu().numpy(),indices=splits['train']);manifest[p.name]=sha(p)
                fits.append(dict(pair=pair,arm=arm,raw=rawfit,full=fullfit))
                pairings.append(dict(pair=pair,arm=arm,initial_sha256=initial,epoch_orders=orders,flip_coins=coins_hash,flipped_examples=flips))
                records.append(dict(pair=pair,arm=arm,training_examples=seen,updates=steps,selected_examples=payload['training_examples'],selected_checkpoint=best_ck,
                    fitting_pipeline_seconds=time.perf_counter()-arm_start,training_total_seconds=training_total,validation_seconds=validation_seconds,checkpoint_seconds=checkpoint_seconds,
                    training_only_seconds=training_total-validation_seconds-checkpoint_seconds,calibration_forward_seconds=cal_forward_seconds,
                    scalar_fit_seconds=scalar_seconds,target_forward_softmax_seconds=target_seconds,validation_forward_examples=20000,
                    calibration_forward_examples=10000 if arm=='teacher' else 5000,target_forward_examples=80000 if arm=='teacher' else 0,
                    scalar_fits=2 if arm=='teacher' else 1,matrix_flops=seen*G+(20000+(90000 if arm=='teacher' else 5000))*F))
            except Exception as exc:
                torch.save(dict(state_dict=net.state_dict(),epoch=epoch,seen=seen),out/f'{tag}_FAILED.pt')
                jwrite(out/f'{tag}_FAILED.json',dict(exception=repr(exc),traceback=traceback.format_exc(),replaced=False));raise
    cwrite(HERE/'training.csv',records);cwrite(HERE/'training_trace.csv',traces)
    jwrite(HERE/'temperatures.json',fits);jwrite(HERE/'PAIRING_AUDIT.json',pairings)
    jwrite(HERE/'MODEL_COMMIT.json',dict(utc=now(),manifest=manifest,temperatures_sha256=sha(HERE/'temperatures.json'),
        source_freeze=sha(HERE/'SOURCE_FREEZE.json'),outcomes_accessed=False))

def committed():
    approve();obj=json.loads((HERE/'MODEL_COMMIT.json').read_text())
    assert sha(HERE/'temperatures.json')==obj['temperatures_sha256']
    for p,h in obj['manifest'].items():assert sha(HERE/'models'/p)==h
    return json.loads((HERE/'temperatures.json').read_text())

def score():
    runtime();rows=committed();out=HERE/'predictions';out.mkdir(exist_ok=False)
    jwrite(HERE/'SCORE_START.json',dict(utc=now(),model_commit=sha(HERE/'MODEL_COMMIT.json')))
    quality=[];classes=[];manifest={};calls=[]
    for dataset in ['holdout','myntra_seen','zappos_new']:
        if dataset=='holdout':
            a,y=fashion();ii=np.load(HERE/'splits.npz')['holdout'];a=a[ii];y=y[ii]
        else:
            folder='external' if dataset=='zappos_new' else 'legacy_external'
            p=np.load(HERE/folder/'population.npz');a=p['images'];y=p['labels'];ii=p['ids']
        groups=p['groups'] if dataset=='zappos_new' else ii
        xt=torch.as_tensor(a,device='cuda',dtype=torch.float32)/255
        for row in rows:
            pair,arm=row['pair'],row['arm'];net=load_model(HERE/'models'/f'{pair}_{arm}.pt','cuda')
            h=predict_chunks(net,xt).cpu().numpy();ht=predict_chunks(net,xt.flip(-1)).cpu().numpy() if arm=='teacher' else np.empty((0,10))
            p=out/f'{dataset}_{pair}_{arm}.npz';np.savez_compressed(p,raw=h,reflected=ht,labels=y,indices=ii,groups=groups);manifest[p.name]=sha(p)
            h=h.astype(np.float64)
            variants={'B':h*row['raw']['beta'],'T':(h.astype(np.float64)+ht)*.5*row['full']['beta']} if arm=='teacher' else {('S' if arm=='student' else 'C'):h*row['raw']['beta']}
            for method,z in variants.items():
                ce=lossvec(z,y);er=z.argmax(1)!=y
                quality.append(dict(dataset=dataset,pair=pair,method=method,ce=float(ce.mean()),error=float(er.mean()),n=len(y)))
                for c in np.unique(y):
                    mask=y==c;classes.append(dict(dataset=dataset,pair=pair,method=method,label=int(c),n=int(mask.sum()),ce=float(ce[mask].mean()),error=float(er[mask].mean())))
            calls.append(dict(dataset=dataset,pair=pair,arm=arm,forward_examples=len(y)*(2 if arm=='teacher' else 1)))
    cwrite(HERE/'quality.csv',quality);cwrite(HERE/'classwise.csv',classes);cwrite(HERE/'evaluation_calls.csv',calls)
    jwrite(HERE/'SCORE_COMPLETE.json',dict(utc=now(),manifest=manifest))

def analyze():
    check();committed()
    from inference import main
    main()

@torch.inference_mode()
def serve(net,x,method,beta,device,transfer):
    if transfer:x=x.to(device)
    if method=='raw':z=net(x)*beta
    elif method=='full_fused':
        zz=net(torch.cat([x,x.flip(-1)],dim=0));z=(zz[:len(x)]+zz[len(x):])*(.5*beta)
    else:z=(net(x)+net(x.flip(-1)))*(.5*beta)
    return z.cpu() if transfer else z

def benchmark():
    runtime();rows=committed();assert (HERE/'SCORE_COMPLETE.json').exists();assert (HERE/'TIMING_RELEASE.json').exists()
    a,_=fashion();ii=np.load(HERE/'splits.npz')['validation'][:1024];x=torch.as_tensor(a[ii],dtype=torch.float32)/255
    jwrite(HERE/'TIMING_START.json',dict(utc=now(),release_sha256=sha(HERE/'TIMING_RELEASE.json'),indices_sha256=adigest(ii)))
    times=[];checks=[];calls=[]
    for pair in CONFIG['pairs']:
        entries={r['arm']:r for r in rows if r['pair']==pair};ref={}
        for mi,mode in enumerate(['cpu','cuda_resident','cuda_transfer']):
            device='cpu' if mode=='cpu' else 'cuda';transfer=mode=='cuda_transfer'
            nets={arm:load_model(HERE/'models'/f'{pair}_{arm}.pt',device) for arm in ARMS}
            xx=x.to(device) if mode=='cuda_resident' else x
            methods={'B':('teacher','raw'),'S':('student','raw'),'C':('control','raw'),'T_fused':('teacher','full_fused'),'T_sequential':('teacher','full_sequential')}
            rng=np.random.default_rng(CONFIG['timing_seed']+pair+mi*100)
            for rep in range(-3,7):
                for m in rng.permutation(list(methods)):
                    arm,method=methods[m];beta=entries[arm]['full' if m.startswith('T_') else 'raw']['beta']
                    if device=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter()
                    for j in range(4):last=serve(nets[arm],xx[j*256:(j+1)*256],method,beta,device,transfer)
                    if device=='cuda':torch.cuda.synchronize()
                    elapsed=(time.perf_counter()-start)*250
                    if rep>=0:times.append(dict(pair=pair,mode=mode,method=m,repeat=rep,ms_per_batch=elapsed))
                    calls.append(dict(pair=pair,mode=mode,method=m,repeat=rep,warmup=rep<0,forward_examples=1024*(2 if m.startswith('T_') else 1)))
                    if rep==0:
                        v=last.detach().cpu().numpy()
                        if mode=='cpu':ref[m]=v
                        else:checks.append(dict(pair=pair,mode=mode,method=m,max_logit_difference=float(np.max(np.abs(v-ref[m]))),argmax_changes=int(np.sum(v.argmax(1)!=ref[m].argmax(1)))))
            print('TIMED',pair,mode,flush=True)
    cwrite(HERE/'latency.csv',times);cwrite(HERE/'cpu_cuda_equivalence.csv',checks);cwrite(HERE/'timing_calls.csv',calls)
    summary=[]
    for pair in CONFIG['pairs']:
        for mode in ['cpu','cuda_resident','cuda_transfer']:
            med={m:float(np.median([r['ms_per_batch'] for r in times if r['pair']==pair and r['mode']==mode and r['method']==m])) for m in ['B','S','C','T_fused','T_sequential']}
            best=min(med['T_fused'],med['T_sequential']);summary.append(dict(pair=pair,mode=mode,**med,best_full_ms=best,full_over_student=best/med['S']))
    cwrite(HERE/'serving_summary.csv',summary);jwrite(HERE/'TIMING_COMPLETE.json',dict(utc=now(),rows=len(times)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','train','score','analyze','benchmark']);globals()[p.parse_args().stage]()
