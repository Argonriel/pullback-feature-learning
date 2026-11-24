"""Frozen selector calibration, guarded test evaluation, and fair serving timing."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,time
import numpy as np
import torch
from data import HERE,arrays,sha
from model import runtime,load_model,predict_chunks
import policy
CONFIG=dict(seeds=[1,2,3],batch=256,test_n=10000,test_order_seed=2136000,
    selector_tie_base=2134000,test_tie_base=2135000,timing_base=2137000,
    bootstrap_seed=2138000,bootstrap_draws=5000,warmups=3,repeats=7,timing_batches=4,
    modes=['cpu','cuda_resident','cuda_transfer'],dtype='float32',cpu_threads=1,
    ridge_alphas=list(policy.RIDGE_ALPHAS),knn_k=list(policy.NEIGHBORS),hist_bins=list(policy.HIST_BINS),
    practical_speedup=1.10,practical_loss_tolerance=.01)
def now():return datetime.now(timezone.utc).isoformat()
def jo(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def co(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def freeze():
    names=['study.py','policy.py','PROTOCOL.md','TRAINING_FREEZE.json','data.py','model.py','train.py']
    jo(HERE/'SOURCE_FREEZE.json',dict(utc=now(),config=CONFIG,source_hashes={n:sha(HERE/n) for n in names},test_accessed=False))
def verify():
    f=json.loads((HERE/'SOURCE_FREEZE.json').read_text());assert f['config']==CONFIG
    for n,h in f['source_hashes'].items():assert sha(HERE/n)==h,n
    return f
def temperatures():return {r['seed']:r['temperature'] for r in json.loads((HERE/'temperatures.json').read_text())['rows']}
def call_mask(indices,n,name):
    mask=np.zeros(n,bool)
    if indices is None:mask[:]=name.startswith('full_')
    else:mask[indices.detach().cpu().numpy()]=True
    return mask

@torch.inference_mode()
def calibrate():
    verify();runtime();assert not (HERE/'TEST_APPROVAL.json').exists()
    images,_=arrays('train');split=dict(np.load(HERE/'splits.npz'));out=HERE/'policies';out.mkdir(exist_ok=False)
    ts=temperatures();rows=[];costs=[];choices={};manifest={}
    for seed in CONFIG['seeds']:
        net=load_model(HERE/'models'/f'{seed}.pt','cuda');t=ts[seed]
        x=torch.as_tensor(images[np.r_[split['selector_fit'],split['selector_validation']]],device='cuda',dtype=torch.float32)/255
        torch.cuda.synchronize();start=time.perf_counter()
        first=predict_chunks(net,x)/t;second=predict_chunks(net,torch.flip(x,[-1]))/t
        g=policy.gap(first,second);f=policy.features(x)
        torch.cuda.synchronize();forward_seconds=time.perf_counter()-start
        start=time.perf_counter();table=policy.fit_tables(f[:8000].cpu().numpy(),g[:8000].cpu().numpy());fitting_seconds=time.perf_counter()-start
        path=out/f'{seed}.npz';np.savez_compressed(path,**table);manifest[path.name]=sha(path)
        tab=policy.device_tables(table,'cuda');v=x[8000:];ties=torch.as_tensor(np.random.default_rng(2134000+seed).random(2000),device='cuda')
        local=[];start=time.perf_counter()
        for method,settings in [('ridge',policy.RIDGE_ALPHAS),('histogram',policy.HIST_BINS),('knn',policy.NEIGHBORS)]:
            for value in settings:
                choice=dict(ridge_alpha=float(value) if method=='ridge' else .01,hist_bins=int(value) if method=='histogram' else 8,knn_k=int(value) if method=='knn' else 1)
                total=0.;count=0
                for begin in range(0,2000,256):
                    end=min(begin+256,2000);sc=policy.score(v[begin:end],tab,method,choice)
                    ids=policy.top_quarter(sc,ties[begin:end]);total+=float(g[8000+begin:8000+end][ids].sum());count+=len(ids)
                row=dict(seed=seed,method=method,value=value,selected_jensen=total/2000,selected_count=count)
                local.append(row);rows.append(row)
        chosen={m:max([r for r in local if r['method']==m],key=lambda r:r['selected_jensen']) for m in ('ridge','histogram','knn')}
        strong=max(('histogram','knn'),key=lambda m:chosen[m]['selected_jensen'])
        choices[str(seed)]=dict(ridge_alpha=chosen['ridge']['value'],hist_bins=int(chosen['histogram']['value']),knn_k=int(chosen['knn']['value']),strong_control=strong,
            validation_selected_jensen={m:chosen[m]['selected_jensen'] for m in chosen})
        torch.cuda.synchronize();selection_seconds=time.perf_counter()-start
        np.savez_compressed(out/f'{seed}_calibration.npz',first=first.cpu().numpy(),partner=second.cpu().numpy(),features=f.cpu().numpy(),gaps=g.cpu().numpy(),ties=ties.cpu().numpy())
        manifest[f'{seed}_calibration.npz']=sha(out/f'{seed}_calibration.npz')
        costs.append(dict(seed=seed,fit_forward_examples=16000,validation_forward_examples=4000,forward_seconds=forward_seconds,
            fitting_seconds=fitting_seconds,hyperparameter_selection_seconds=selection_seconds,fit_coordinate_images=8000,validation_coordinate_images=2000))
        print('Calibrated selectors',seed,choices[str(seed)],flush=True)
    co(HERE/'selector_candidates.csv',rows);co(HERE/'selector_costs.csv',costs)
    jo(HERE/'choices.json',dict(utc=now(),choices=choices))
    jo(HERE/'POLICY_COMMIT.json',dict(utc=now(),manifest=manifest,choices_sha256=sha(HERE/'choices.json'),
        model_commit_sha256=sha(HERE/'MODEL_COMMIT.json'),source_freeze_sha256=sha(HERE/'SOURCE_FREEZE.json'),test_accessed=False))

def verify_test_release():
    verify();a=json.loads((HERE/'TEST_APPROVAL.json').read_text());assert a['test_access_authorized']
    assert a['policy_commit_sha256']==sha(HERE/'POLICY_COMMIT.json')
    commit=json.loads((HERE/'POLICY_COMMIT.json').read_text())
    assert commit['choices_sha256']==sha(HERE/'choices.json')
    for n,h in commit['manifest'].items():assert sha(HERE/'policies'/n)==h

@torch.inference_mode()
def score():
    verify_test_release();runtime();images,labels=arrays('test')
    order=np.random.default_rng(2136000).permutation(len(images));images=images[order];labels=labels[order]
    ts=temperatures();choices=json.loads((HERE/'choices.json').read_text())['choices'];outdir=HERE/'predictions';outdir.mkdir(exist_ok=False)
    jo(HERE/'TEST_START.json',dict(utc=now(),source_freeze_sha256=sha(HERE/'SOURCE_FREEZE.json'),policy_commit_sha256=sha(HERE/'POLICY_COMMIT.json')))
    rows=[];ledger=[];checks=[];manifest={}
    for seed in CONFIG['seeds']:
      for device in ('cuda','cpu'):
        start=time.perf_counter();net=load_model(HERE/'models'/f'{seed}.pt',device)
        tab=policy.device_tables(dict(np.load(HERE/'policies'/f'{seed}.npz')),device);choice=choices[str(seed)]
        x=torch.as_tensor(images,device=device,dtype=torch.float32)/255
        ties=torch.as_tensor(np.random.default_rng(2135000+seed).random(len(x)),device=device)
        logits=np.empty((len(policy.NAMES),len(x),10),np.float32);masks=np.zeros((len(policy.NAMES),len(x)),bool)
        for j,name in enumerate(policy.NAMES):
          for begin in range(0,len(x),256):
            end=min(begin+256,len(x));calls=[]
            z,ids=policy.deploy(net,x[begin:end],ties[begin:end],tab,choice,ts[seed],name,calls)
            assert calls==policy.expected_calls(name,end-begin)
            mask=call_mask(ids,end-begin,name)
            assert mask.sum()==(0 if name=='raw' else end-begin if name.startswith('full_') else (end-begin)//4)
            logits[j,begin:end]=z.cpu().numpy();masks[j,begin:end]=mask
            ledger.append(dict(seed=seed,device=device,policy=name,batch=begin//256,images=end-begin,selected=int(mask.sum()),calls='|'.join(map(str,calls)),forward_examples=sum(calls)))
        # Pair-closed diagnostic reuses the paid all-view reference, not deployment access.
        first=torch.as_tensor(logits[0],device=device);other=2*torch.as_tensor(logits[1],device=device)-first
        avg=.5*(first+other);g=policy.gap(first,other).cpu().numpy();flipped=np.empty_like(logits);flip_masks=np.zeros_like(masks)
        for j,name in enumerate(policy.NAMES):
          for begin in range(0,len(x),256):
            end=min(begin+256,len(x));n=end-begin
            if name=='raw':mask=np.zeros(n,bool)
            elif name.startswith('full_'):mask=np.ones(n,bool)
            elif name=='confidence':
                mask=call_mask(policy.top_quarter(-torch.softmax(other[begin:end],1).max(1).values,ties[begin:end]),n,name)
            elif name=='oracle_jensen':mask=masks[j,begin:end]
            else:
                ids=policy.top_quarter(policy.score(torch.flip(x[begin:end],[-1]),tab,name,choice),ties[begin:end])
                mask=call_mask(ids,n,name);assert np.array_equal(mask,masks[j,begin:end])
            z=other[begin:end].clone();ids=np.flatnonzero(mask);z[ids]=avg[begin:end][ids]
            flipped[j,begin:end]=z.cpu().numpy();flip_masks[j,begin:end]=mask
        y=torch.as_tensor(labels);losses=np.empty((len(policy.NAMES),len(x)));paired=np.empty_like(losses)
        for j,name in enumerate(policy.NAMES):
            a=torch.as_tensor(logits[j]);b=torch.as_tensor(flipped[j]);loss=torch.nn.functional.cross_entropy(a,y,reduction='none').numpy();opposite=torch.nn.functional.cross_entropy(b,y,reduction='none').numpy()
            losses[j]=loss;paired[j]=.5*(loss+opposite)
            selected=(masks[j]*g).mean();oracle=(masks[-1]*g).mean()
            identity=float(np.max(abs((paired[0]-paired[j])-masks[j]*g))) if name in ('ridge','histogram','knn','oracle_jensen','raw','full_fused','full_sequential') else None
            if identity is not None:assert identity<3e-5,(seed,device,name,identity)
            rows.append(dict(seed=seed,device=device,policy=name,original_loss=float(loss.mean()),original_error=float(np.mean(logits[j].argmax(1)!=labels)),
                paired_loss=float(paired[j].mean()),paired_error=float(.5*(np.mean(logits[j].argmax(1)!=labels)+np.mean(flipped[j].argmax(1)!=labels))),
                selected_jensen=float(selected),oracle_fraction=float(selected/oracle),strong_control=choice['strong_control'],
                actual_forward_examples=sum(r['forward_examples'] for r in ledger if r['seed']==seed and r['device']==device and r['policy']==name),
                paired_identity_max_error='' if identity is None else identity))
        path=outdir/f'{seed}_{device}.npz';np.savez_compressed(path,original_indices=order,labels=labels,logits=logits,flipped_logits=flipped,masks=np.packbits(masks,axis=1),
            flip_masks=np.packbits(flip_masks,axis=1),losses=losses,paired_losses=paired,gaps=g,names=np.array(policy.NAMES))
        manifest[path.name]=sha(path)
        checks.append(dict(seed=seed,device=device,seconds=time.perf_counter()-start,all_even_flip_masks_equal=True,all_quotas_exact=True))
        print('Scored official test',seed,device,flush=True)
    co(HERE/'quality.csv',rows);co(HERE/'deployment_ledger.csv',ledger);co(HERE/'score_checks.csv',checks)
    jo(HERE/'TEST_COMPLETE.json',dict(utc=now(),manifest=manifest,images_per_model=10000,models=3,rows=len(rows),actual_forward_examples=sum(r['forward_examples'] for r in ledger),
        test_partners_are_evaluation_reuse=True))

@torch.inference_mode()
def benchmark():
    verify_test_release();runtime();images,_=arrays('train');split=dict(np.load(HERE/'splits.npz'))
    bx=torch.as_tensor(images[split['selector_validation'][:1024]],dtype=torch.float32)/255
    ts=temperatures();choices=json.loads((HERE/'choices.json').read_text())['choices'];rows=[];setup=[];counts=[]
    assert (HERE/'TEST_COMPLETE.json').exists()
    jo(HERE/'TIMING_START.json',dict(utc=now(),source_freeze_sha256=sha(HERE/'SOURCE_FREEZE.json'),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),cpu_threads=1))
    for seed in CONFIG['seeds']:
        tables=dict(np.load(HERE/'policies'/f'{seed}.npz'));choice=choices[str(seed)]
        keys=torch.as_tensor(np.random.default_rng(2139000+seed).random(len(bx)))
        nets={};tabs={}
        for device in ('cpu','cuda'):
            t0=time.perf_counter();nets[device]=load_model(HERE/'models'/f'{seed}.pt',device);tabs[device]=policy.device_tables(tables,device)
            if device=='cuda':torch.cuda.synchronize()
            setup.append(dict(seed=seed,device=device,model_and_table_loading_seconds=time.perf_counter()-t0,table_bytes=sum(v.nbytes for v in tables.values())))
        gx=bx.to('cuda');gk=keys.to('cuda');torch.cuda.synchronize()
        rng=np.random.default_rng(2137000+seed)
        for mode in CONFIG['modes']:
          for repeat in range(-3,7):
            for rank,index in enumerate(rng.permutation(len(policy.NAMES))):
                name=policy.NAMES[index];needs_keys=name not in ('raw','full_fused','full_sequential')
                if mode!='cpu':torch.cuda.synchronize()
                start=time.perf_counter_ns();invocations=[]
                for begin in range(0,1024,256):
                    sl=slice(begin,begin+256)
                    if mode=='cpu':x=bx[sl];tie=keys[sl] if needs_keys else None;device='cpu'
                    elif mode=='cuda_resident':x=gx[sl];tie=gk[sl] if needs_keys else None;device='cuda'
                    else:x=bx[sl].to('cuda');tie=keys[sl].to('cuda') if needs_keys else None;device='cuda'
                    z,ids=policy.deploy(nets[device],x,tie,tabs[device],choice,ts[seed],name,invocations)
                    if mode=='cuda_transfer':z=z.cpu()
                if mode!='cpu':torch.cuda.synchronize()
                elapsed=(time.perf_counter_ns()-start)/1e6/4
                assert invocations==policy.expected_calls(name,256)*4
                if repeat>=0:
                    rows.append(dict(seed=seed,mode=mode,policy=name,repeat=repeat,order=rank,ms_per_batch=elapsed,forward_examples=sum(invocations),batches=4))
                elif repeat==-3:
                    counts.append(dict(seed=seed,mode=mode,policy=name,calls='|'.join(map(str,invocations)),forward_examples=sum(invocations)))
            print('Timed',seed,mode,'block',repeat,flush=True)
    co(HERE/'latency.csv',rows);co(HERE/'loading_costs.csv',setup);co(HERE/'timing_calls.csv',counts)
    jo(HERE/'TIMING_COMPLETE.json',dict(utc=now(),rows=len(rows),latency_sha256=sha(HERE/'latency.csv')))

def analyze():
    verify_test_release();quality=read_csv(HERE/'quality.csv');latency=read_csv(HERE/'latency.csv');choices=json.loads((HERE/'choices.json').read_text())['choices']
    summary=[];cost=[];equivalence=[]
    for device in ('cuda','cpu'):
      for name in policy.NAMES:
        r=[v for v in quality if v['device']==device and v['policy']==name]
        summary.append(dict(device=device,policy=name,models=3,**{k:float(np.mean([float(v[k]) for v in r])) for k in ('original_loss','original_error','paired_loss','paired_error','selected_jensen','oracle_fraction')}))
    for seed in (1,2,3):
        a=dict(np.load(HERE/'predictions'/f'{seed}_cpu.npz'));b=dict(np.load(HERE/'predictions'/f'{seed}_cuda.npz'))
        for j,name in enumerate(policy.NAMES):
            ma=np.unpackbits(a['masks'],axis=1)[j,:10000];mb=np.unpackbits(b['masks'],axis=1)[j,:10000]
            equivalence.append(dict(seed=seed,policy=name,max_logit_difference=float(np.max(abs(a['logits'][j]-b['logits'][j]))),
                changed_masks=int(np.sum(ma!=mb)),changed_classes=int(np.sum(a['logits'][j].argmax(1)!=b['logits'][j].argmax(1))),
                cpu_minus_cuda_loss=float(a['losses'][j].mean()-b['losses'][j].mean())))
        for mode in CONFIG['modes']:
            med={name:float(np.median([float(r['ms_per_batch']) for r in latency if int(r['seed'])==seed and r['mode']==mode and r['policy']==name])) for name in policy.NAMES}
            full=min(('full_fused','full_sequential'),key=med.get)
            device='cpu' if mode=='cpu' else 'cuda'
            risk={name:float(next(r['original_loss'] for r in quality if int(r['seed'])==seed and r['device']==device and r['policy']==name)) for name in policy.NAMES}
            for name in policy.NAMES:
                cost.append(dict(seed=seed,mode=mode,policy=name,median_ms=med[name],strong_full=full,strong_full_ms=med[full],speedup_vs_full=med[full]/med[name],
                    original_loss=risk[name],excess_loss_vs_full=risk[name]-risk[full],practical_pass=med[full]/med[name]>=1.1 and risk[name]-risk[full]<=.01))
    contrasts=[];sizes=np.r_[np.full(39,256),16]
    for device in ('cuda','cpu'):
      for world,key in [('original','losses'),('pair_closed','paired_losses')]:
        for reference in ('strong_control','full_fused','confidence'):
            diffs=[]
            for seed in (1,2,3):
                d=np.load(HERE/'predictions'/f'{seed}_{device}.npz');names=list(d['names']);ref=choices[str(seed)]['strong_control'] if reference=='strong_control' else reference
                diffs.append(d[key][names.index('ridge')]-d[key][names.index(ref)])
            delta=np.mean(diffs,axis=0);sums=np.array([delta[j:j+256].sum() for j in range(0,10000,256)])
            draws=np.random.default_rng(2138000).integers(0,40,(5000,40));boot=sums[draws].sum(1)/sizes[draws].sum(1)
            lo,hi=np.quantile(boot,[.025,.975]);contrasts.append(dict(device=device,world=world,contrast='ridge_minus_'+reference,mean=float(delta.mean()),lower=float(lo),upper=float(hi),clusters=40,models_fixed=3))
    co(HERE/'summary.csv',summary);co(HERE/'cost_quality.csv',cost);co(HERE/'cpu_cuda_equivalence.csv',equivalence);co(HERE/'contrasts.csv',contrasts)
    jo(HERE/'ANALYSIS_COMPLETE.json',dict(utc=now(),files={p:sha(HERE/p) for p in ('summary.csv','cost_quality.csv','cpu_cuda_equivalence.csv','contrasts.csv')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','calibrate','score','benchmark','analyze']);args=p.parse_args()
    globals()[args.stage]()
