"""Real trained MLP inference: exact view quotas and synchronized GPU costs."""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.util, json, os, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent; STUDY=HERE.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(STUDY/'source'))
sys.path.insert(0,str(STUDY/'legacy'/'budgeted-rotation-final'))
from gate import Gate
spec=importlib.util.spec_from_file_location('gpu_prior_angles',STUDY/'reference'/'experiment.py')
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
CONFIG=dict(widths=[64,256,1024],seeds=[1,2,3],train_n=1024,train_steps=1000,
    train_seed_base=971000,init_seed_base=972000,gate_seed_base=973000,
    angle_radius_seed_base=974000,angle_phase_seed_base=975000,test_seed_base=976000,
    tie_seed_base=977000,label_seed_base=978000,timing_seed_base=978100,
    train_lr=.001,test_n=16384,batch=256,quota=64,calibration_forwards=8192,
    angle_radii=64,angle_nodes=128,warmups=3,repeats=7,timing_batches=16,
    hidden_layers=2,activation='tanh',dtype='float32',tf32=False,autocast=False,
    cpu_threads=1,optimizer='Adam',calibration_tables_resident=True)
POLICIES=['raw','full_sequential','full_fused','confidence_pi','folded_sequential',
          'folded_fused','confidence_radial']

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(v):return hashlib.sha256(v).hexdigest()
def jo(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def co(path,rows):
    with path.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def ci(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()
    return torch

def model(torch,width,device):
    return torch.nn.Sequential(torch.nn.Linear(2,width),torch.nn.Tanh(),
        torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,1)).to(device)

def load_model(torch,width,seed,device):
    result=model(torch,width,device)
    checkpoint=torch.load(HERE/'models'/f'w{width}_s{seed}.pt',map_location=device,weights_only=True)
    result.load_state_dict(checkpoint['state_dict']);result.eval()
    for p in result.parameters():p.requires_grad_(False)
    return result

def freeze():
    files=[Path(__file__),HERE/'PROTOCOL.md',STUDY/'reference'/'experiment.py',
        STUDY/'legacy'/'budgeted-rotation-final'/'gate.py',
        STUDY/'source'/'pullback_feature_learning'/'core.py']
    jo(HERE/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,policies=POLICIES,
        source_hashes={str(p.relative_to(STUDY)):sha(p.read_bytes()) for p in files},
        gpu_training_started=False,conditional_go_ahead='Root explicitly activated GPU branch after old-model local-bound progress.'))

def verify():
    frozen=json.loads((HERE/'freeze.json').read_text())
    assert frozen['config']==CONFIG and frozen['policies']==POLICIES
    for path,digest in frozen['source_hashes'].items():assert sha((STUDY/path).read_bytes())==digest,path
    return frozen

def training_data(seed):
    rng=np.random.default_rng(971000+seed);r=rng.choice([.5,1.2],1024)
    a=rng.uniform(0,2*np.pi,1024)
    x=r[:,None]*np.column_stack((np.cos(a),np.sin(a)))+rng.normal(0,.15,(1024,2))
    return x.astype(np.float32),(r==1.2).astype(np.float32)

def train():
    verify();torch=runtime();folder=HERE/'models';folder.mkdir(exist_ok=False)
    device=torch.device('cuda');records=[];failures=[]
    for width in CONFIG['widths']:
        for seed in CONFIG['seeds']:
            x,y=training_data(seed)
            torch.manual_seed(972000+seed);torch.cuda.manual_seed_all(972000+seed)
            net=model(torch,width,device);net.train()
            xt=torch.from_numpy(x).to(device);yt=torch.from_numpy(y).to(device)
            optimizer=torch.optim.Adam(net.parameters(),lr=.001)
            torch.cuda.synchronize();start=time.perf_counter();losses=[]
            try:
                for step in range(1000):
                    optimizer.zero_grad(set_to_none=True)
                    logits=net(xt).flatten()
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yt)
                    loss.backward();optimizer.step()
                    if step in (0,9,99,499,999):losses.append((step+1,float(loss.detach().cpu())))
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                state={k:v.detach().cpu() for k,v in net.state_dict().items()}
                assert all(torch.isfinite(v).all() for v in state.values())
                path=folder/f'w{width}_s{seed}.pt'
                torch.save(dict(state_dict=state,width=width,seed=seed),path)
                np.savez_compressed(folder/f'w{width}_s{seed}_training.npz',x=x,y=y,losses=np.array(losses))
                records.append(dict(width=width,seed=seed,parameters=sum(p.numel() for p in net.parameters()),
                    train_seed=971000+seed,init_seed=972000+seed,training_seconds=elapsed,
                    final_recorded_training_loss=losses[-1][1],x_sha256=sha(x.tobytes()),
                    y_sha256=sha(y.tobytes()),weights_sha256=sha(path.read_bytes())))
                print(f'Trained GPU MLP width {width}, seed {seed}',flush=True)
            except Exception as e:
                failure=dict(width=width,seed=seed,error=type(e).__name__,message=str(e))
                failures.append(failure);jo(folder/f'w{width}_s{seed}_failure.json',failure)
                raise
    co(HERE/'training.csv',records)
    jo(HERE/'training_commit.json',dict(timestamp_utc=now(),failures=failures,
        files={p.name:sha(p.read_bytes()) for p in folder.iterdir()},
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(),torch_config=torch.__config__.show(),
        freeze_sha256=sha((HERE/'freeze.json').read_bytes())))

def calibrate():
    verify();torch=runtime();folder=HERE/'policies';folder.mkdir(exist_ok=False)
    records=[]
    for width in CONFIG['widths']:
        for seed in CONFIG['seeds']:
            net=load_model(torch,width,seed,'cuda');calls=[]
            def predict(x):
                calls.append(len(x))
                with torch.inference_mode():
                    return net(torch.as_tensor(x,dtype=torch.float32,device='cuda')).flatten().cpu().numpy().astype(np.float64)
            x=ref.sample(973000+seed,4096)
            torch.cuda.synchronize();start=time.perf_counter()
            gate=Gate.fit(x,predict);torch.cuda.synchronize();gate_seconds=time.perf_counter()-start
            assert calls==[4096,4096]
            records.append(dict(width=width,seed=seed,kind='folded',forward_examples=sum(calls),
                independent_inputs=4096,calibration_seconds=gate_seconds))
            r=np.linalg.norm(ref.sample(974000+seed,64),axis=1)
            phase=np.random.default_rng(975000+seed).uniform(0,2*np.pi,64)
            angle=phase[:,None]+np.arange(128)[None,:]*2*np.pi/128
            orbit=(r[:,None,None]*np.stack((np.cos(angle),np.sin(angle)),axis=-1)).reshape(-1,2)
            calls.clear();torch.cuda.synchronize();start=time.perf_counter()
            h=predict(orbit).reshape(64,128)
            angles=ref.fit(h,r,phase,'pb');torch.cuda.synchronize();angle_seconds=time.perf_counter()-start
            assert calls==[8192]
            records.append(dict(width=width,seed=seed,kind='radial_angle',forward_examples=sum(calls),
                independent_inputs=64,calibration_seconds=angle_seconds))
            np.savez_compressed(folder/f'w{width}_s{seed}.npz',gate_edges=gate.edges,
                gate_radius=gate.radius_scores,gate_folded=gate.folded_scores,
                angle_edges=angles['edges'],angle_indices=angles['radial_index'],
                angle_h=h,angle_radii=r,angle_phases=phase,angle_curves=angles['curves'])
            print(f'Calibrated GPU MLP width {width}, seed {seed}',flush=True)
    co(HERE/'calibration.csv',records)
    jo(HERE/'policy_commit.json',dict(timestamp_utc=now(),test_outcomes_accessed=False,
        files={p.name:sha(p.read_bytes()) for p in folder.glob('*.npz')},
        training_commit_sha256=sha((HERE/'training_commit.json').read_bytes()),
        calibration_sha256=sha((HERE/'calibration.csv').read_bytes())))

def tables(torch,width,seed,device):
    data=dict(np.load(HERE/'policies'/f'w{width}_s{seed}.npz'))
    return {name:torch.as_tensor(data[name],device=device,dtype=(torch.int64 if name=='angle_indices' else torch.float32))
            for name in ('gate_edges','gate_folded','angle_edges','angle_indices')}

def allocate(torch,score,tie):
    by_tie=torch.argsort(tie,stable=True)
    ranks=torch.argsort(score[by_tie],descending=True,stable=True)
    chosen=by_tie[ranks[:64]]
    mask=torch.zeros(len(score),dtype=torch.bool,device=score.device);mask[chosen]=True
    return mask

def folded_score(torch,x,tab):
    radius=torch.linalg.vector_norm(x,dim=1)
    b=torch.bucketize(radius,tab['gate_edges'],right=False)
    flip=(x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0))
    canonical=torch.where(flip[:,None],-x,x)
    angle=torch.remainder(torch.atan2(canonical[:,1],canonical[:,0]),np.pi)
    ab=torch.clamp((angle*8/np.pi).to(torch.int64),max=7)
    return tab['gate_folded'][8*b+ab]

def deploy(torch,net,x,tie,tab,name,calls=None):
    def predict(z):
        if calls is not None:calls.append(len(z))
        return net(z).flatten()
    n=len(x)
    if name=='full_fused':
        both=predict(torch.cat((x,-x),dim=0))
        return .5*(both[:n]+both[n:]),torch.ones(n,dtype=torch.bool,device=x.device)
    if name=='folded_fused':
        mask=allocate(torch,folded_score(torch,x,tab),tie)
        both=predict(torch.cat((x,-x[mask]),dim=0));out=both[:n].clone()
        out[mask]=.5*(out[mask]+both[n:])
        return out,mask
    first=predict(x)
    if name=='raw':return first,torch.zeros(n,dtype=torch.bool,device=x.device)
    if name=='full_sequential':
        return .5*(first+predict(-x)),torch.ones(n,dtype=torch.bool,device=x.device)
    mask=allocate(torch,folded_score(torch,x,tab) if name=='folded_sequential' else -torch.abs(first),tie)
    if name=='confidence_radial':
        b=torch.bucketize(torch.linalg.vector_norm(x[mask],dim=1),tab['angle_edges'],right=True)
        angle=tab['angle_indices'][b].to(torch.float32)*(2*np.pi/128)
        c=torch.cos(angle);s=torch.sin(angle);xx=x[mask]
        second=torch.stack((c*xx[:,0]-s*xx[:,1],s*xx[:,0]+c*xx[:,1]),dim=1)
    else:second=-x[mask]
    out=first.clone();out[mask]=.5*(first[mask]+predict(second))
    return out,mask

def expected_calls(name):
    if name=='raw':return [256]
    if name=='full_fused':return [512]
    if name=='folded_fused':return [320]
    if name=='full_sequential':return [256,256]
    return [256,64]

def test_data(seed,n=16384):
    x=ref.sample(976000+seed,n)
    tie=np.random.default_rng(977000+seed).random(n)
    p=ref.posterior(x)
    y=(np.random.default_rng(978000+seed).random(n)<p).astype(float)
    return x,tie,p,y

def score():
    verify();torch=runtime();commit=json.loads((HERE/'policy_commit.json').read_text())
    for name,digest in commit['files'].items():assert sha((HERE/'policies'/name).read_bytes())==digest
    jo(HERE/'score_start.json',dict(timestamp_utc=now(),policy_commit_sha256=sha((HERE/'policy_commit.json').read_bytes())))
    folder=HERE/'predictions';folder.mkdir(exist_ok=False);rows=[];audits=[];equivalence=[]
    for width in CONFIG['widths']:
        for seed in CONFIG['seeds']:
            net=load_model(torch,width,seed,'cuda');tab=tables(torch,width,seed,'cuda')
            x,tie,p,y=test_data(seed);outputs=[];masks=[]
            xt=torch.as_tensor(x,dtype=torch.float32,device='cuda');tt=torch.as_tensor(tie,dtype=torch.float64,device='cuda')
            with torch.inference_mode():
                for name in POLICIES:
                    output=np.empty(len(x),dtype=np.float32);maskout=np.zeros(len(x),dtype=bool)
                    torch.cuda.synchronize();start=time.perf_counter()
                    for batch in range(len(x)//256):
                        sl=slice(batch*256,(batch+1)*256);calls=[]
                        out,mask=deploy(torch,net,xt[sl],tt[sl],tab,name,calls)
                        output[sl]=out.cpu().numpy();maskout[sl]=mask.cpu().numpy()
                        assert calls==expected_calls(name)
                        wanted=0 if name=='raw' else (256 if name.startswith('full_') else 64)
                        assert int(mask.sum())==wanted
                        audits.append(dict(width=width,seed=seed,policy=name,batch=batch,
                            calls=json.dumps(calls),selected=wanted))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    assert np.isfinite(output).all()
                    rows.append(dict(width=width,seed=seed,policy=name,
                        expected_risk=float(ref.nll(output.astype(float),p).mean()),
                        sampled_label_risk=float(ref.nll(output.astype(float),y).mean()),
                        expected_error=float(np.where(output>=0,1-p,p).mean()),
                        sampled_error=float(((output>=0)!=y).mean()),
                        second_forwards=int(maskout.sum()),forward_examples=int(len(x)+maskout.sum()),
                        scoring_seconds=elapsed,output_sha256=sha(output.tobytes()),
                        mask_sha256=sha(np.packbits(maskout).tobytes())))
                    outputs.append(output);masks.append(maskout)
            for sequential,fused in [('full_sequential','full_fused'),('folded_sequential','folded_fused')]:
                i=POLICIES.index(sequential);j=POLICIES.index(fused)
                error=float(np.max(abs(outputs[i]-outputs[j])));same=bool(np.array_equal(masks[i],masks[j]))
                assert same and error<1e-4,(width,seed,sequential,error)
                equivalence.append(dict(width=width,seed=seed,sequential=sequential,fused=fused,
                    same_mask=same,max_logit_error=error))
            np.savez_compressed(folder/f'w{width}_s{seed}.npz',x=x,ties=tie,posterior=p,labels=y,
                names=np.array(POLICIES),outputs=np.array(outputs),masks=np.packbits(np.array(masks),axis=1))
            print(f'Scored GPU MLP width {width}, seed {seed}',flush=True)
    co(HERE/'per_model.csv',rows);co(HERE/'deployment_audit.csv',audits);co(HERE/'fusion_equivalence.csv',equivalence)
    jo(HERE/'score_complete.json',dict(timestamp_utc=now(),n_models=9,n_results=len(rows),
        all_call_quotas_pass=True,max_fusion_logit_error=max(r['max_logit_error'] for r in equivalence),
        per_model_sha256=sha((HERE/'per_model.csv').read_bytes())))

def timing():
    verify();torch=runtime();assert (HERE/'score_complete.json').exists()
    approval=json.loads((HERE/'TIMING_APPROVAL.json').read_text())
    assert approval['main_cpu_jobs_finished'] is True
    rows=[];cpu_checks=[]
    for width in CONFIG['widths']:
        for seed in CONFIG['seeds']:
            x,tie,_,_=test_data(seed,4096)
            cpu_x=torch.as_tensor(x,dtype=torch.float32);cpu_tie=torch.as_tensor(tie,dtype=torch.float64)
            gpu_x=cpu_x.to('cuda');gpu_tie=cpu_tie.to('cuda')
            gpu_net=load_model(torch,width,seed,'cuda');cpu_net=load_model(torch,width,seed,'cpu')
            gpu_tab=tables(torch,width,seed,'cuda');cpu_tab=tables(torch,width,seed,'cpu')
            rng=np.random.default_rng(978100+seed+width)
            with torch.inference_mode():
                for name in POLICIES:
                    gh,gm=deploy(torch,gpu_net,gpu_x[:256],gpu_tie[:256],gpu_tab,name)
                    ch,cm=deploy(torch,cpu_net,cpu_x[:256],cpu_tie[:256],cpu_tab,name)
                    cpu_checks.append(dict(width=width,seed=seed,policy=name,
                        max_logit_error=float(np.max(abs(gh.cpu().numpy()-ch.numpy()))),
                        mask_differences=int(np.sum(gm.cpu().numpy()!=cm.numpy()))))
                for condition in ('resident','transfer','cpu'):
                    for repeat in range(3+7):
                        for name in rng.permutation(POLICIES):
                            if condition!='cpu':
                                torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                                begin.record()
                            started=time.perf_counter_ns()
                            for batch in range(16):
                                sl=slice(256*batch,256*(batch+1))
                                if condition=='resident':
                                    out,_=deploy(torch,gpu_net,gpu_x[sl],gpu_tie[sl],gpu_tab,str(name))
                                elif condition=='transfer':
                                    bx=cpu_x[sl].to('cuda');bt=cpu_tie[sl].to('cuda')
                                    out,_=deploy(torch,gpu_net,bx,bt,gpu_tab,str(name));out=out.cpu()
                                else:out,_=deploy(torch,cpu_net,cpu_x[sl],cpu_tie[sl],cpu_tab,str(name))
                            if condition!='cpu':
                                end.record();torch.cuda.synchronize();event_ms=begin.elapsed_time(end)/16
                            else:event_ms=''
                            wall_ms=(time.perf_counter_ns()-started)/1e6/16
                            if repeat>=3:
                                rows.append(dict(width=width,seed=seed,condition=condition,policy=str(name),
                                    repeat=repeat-3,wall_ms_per_batch=wall_ms,cuda_event_ms_per_batch=event_ms))
            print(f'Timed MLP width {width}, seed {seed}',flush=True)
    co(HERE/'latency.csv',rows);co(HERE/'cpu_gpu_agreement.csv',cpu_checks)
    jo(HERE/'timing_complete.json',dict(timestamp_utc=now(),n_records=len(rows),
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,
        timing_sha256=sha((HERE/'latency.csv').read_bytes()),
        timing_approval_sha256=sha((HERE/'TIMING_APPROVAL.json').read_bytes()),
        main_cpu_jobs_completed_before_start=True))

def analyze():
    verify();risks=ci(HERE/'per_model.csv');timings=ci(HERE/'latency.csv');cal=ci(HERE/'calibration.csv')
    rows=[]
    for width in CONFIG['widths']:
        for seed in CONFIG['seeds']:
            risk={z['policy']:float(z['expected_risk']) for z in risks if int(z['width'])==width and int(z['seed'])==seed}
            for condition in ('resident','transfer','cpu'):
                med={name:float(np.median([float(z['wall_ms_per_batch']) for z in timings
                    if int(z['width'])==width and int(z['seed'])==seed and z['condition']==condition and z['policy']==name])) for name in POLICIES}
                full=min(med['full_sequential'],med['full_fused'])
                for name in POLICIES:
                    kind='folded' if name.startswith('folded') else ('radial_angle' if name=='confidence_radial' else None)
                    cost=next(float(z['calibration_seconds']) for z in cal if int(z['width'])==width and int(z['seed'])==seed and z['kind']==kind) if kind else 0.
                    forwards=8192 if kind else 0;denominator=512-sum(expected_calls(name))
                    saving=full-med[name]
                    rows.append(dict(width=width,seed=seed,condition=condition,policy=name,
                        median_wall_ms=med[name],fastest_full_wall_ms=full,
                        speedup_vs_fastest_full=full/med[name],expected_risk=risk[name],
                        risk_delta_vs_raw=risk[name]-risk['raw'],risk_delta_vs_full=risk[name]-risk['full_fused'],
                        calibration_seconds=cost,calibration_forwards=forwards,
                        latency_break_even_batches=(int(np.ceil(1000*cost/saving)) if saving>0 else ''),
                        forward_break_even_batches=(int(np.ceil(forwards/denominator)) if denominator>0 else '')))
    co(HERE/'cost_quality.csv',rows)
    summary=[]
    for width in CONFIG['widths']:
        for condition in ('resident','transfer','cpu'):
            for name in POLICIES:
                selected=[r for r in rows if r['width']==width and r['condition']==condition and r['policy']==name]
                summary.append(dict(width=width,condition=condition,policy=name,n_models=3,
                    mean_median_wall_ms=float(np.mean([r['median_wall_ms'] for r in selected])),
                    minimum_model_speedup=float(min(r['speedup_vs_fastest_full'] for r in selected)),
                    mean_model_speedup=float(np.mean([r['speedup_vs_fastest_full'] for r in selected])),
                    maximum_model_speedup=float(max(r['speedup_vs_fastest_full'] for r in selected)),
                    models_faster_than_full=sum(r['speedup_vs_fastest_full']>1 for r in selected),
                    mean_expected_risk=float(np.mean([r['expected_risk'] for r in selected])),
                    mean_risk_delta_vs_raw=float(np.mean([r['risk_delta_vs_raw'] for r in selected])),
                    mean_risk_delta_vs_full=float(np.mean([r['risk_delta_vs_full'] for r in selected]))))
    co(HERE/'summary.csv',summary)
    jo(HERE/'analysis_complete.json',dict(timestamp_utc=now(),n_trained_models=9,
        descriptive_models_per_width=3,timing_repeats_are_not_model_replications=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','calibrate','score','time','analyze'])
    args=parser.parse_args();globals()['timing' if args.stage=='time' else args.stage]()
