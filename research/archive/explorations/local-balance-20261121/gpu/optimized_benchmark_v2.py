"""Audit and time original and fixed-index implementations in common blocks."""
from pathlib import Path
import argparse, json, time
import numpy as np
import benchmark as base
import optimized_serving as serving
import optimized_benchmark as prior

HERE=Path(__file__).resolve().parent;OUT=HERE/'optimized'

def freeze():
    prior.verify()
    files=[Path(__file__),HERE/'TRANSFER_TIMING_AMENDMENT.md',HERE/'optimized_benchmark.py',
           OUT/'freeze.json',OUT/'equivalence_commit.json']
    base.jo(OUT/'freeze_v2.json',dict(timestamp_utc=base.now(),timing_started=False,
        source_hashes={str(p.relative_to(HERE)):base.sha(p.read_bytes()) for p in files},
        reason='Transfer only needed tie keys; preserve original frozen runner.'))

def verify():
    prior.verify()
    for p,digest in json.loads((OUT/'freeze_v2.json').read_text())['source_hashes'].items():
        assert base.sha((HERE/p).read_bytes())==digest,p


def audit():
    verify();torch=base.runtime();records=[]
    for width in base.CONFIG['widths']:
        for seed in base.CONFIG['seeds']:
            x,tie,_,_=base.test_data(seed)
            for device in ('cuda','cpu'):
                net=base.load_model(torch,width,seed,device)
                tab=serving.prepare_tables(torch,base.tables(torch,width,seed,device))
                xt=torch.as_tensor(x,dtype=torch.float32,device=device)
                tt=torch.as_tensor(tie,dtype=torch.float64,device=device)
                boundary_ties=0
                with torch.inference_mode():
                    for batch in range(len(x)//256):
                        sl=slice(batch*256,(batch+1)*256)
                        first=net(xt[sl]).flatten()
                        rank=torch.topk(-torch.abs(first),65,sorted=True).values
                        boundary_ties+=int(rank[63]==rank[64])
                    for name in base.POLICIES:
                        max_error=0.;mask_errors=0
                        for batch in range(len(x)//256):
                            sl=slice(batch*256,(batch+1)*256);calls=[]
                            old,oldmask=base.deploy(torch,net,xt[sl],tt[sl],tab,name)
                            new,indices=serving.deploy(torch,net,xt[sl],tt[sl],tab,name,calls)
                            assert calls==base.expected_calls(name)
                            wanted=0 if name=='raw' else (256 if name.startswith('full_') else 64)
                            if indices is None:
                                mask=np.full(256,wanted==256,dtype=bool)
                            else:
                                assert indices.shape==(64,)
                                mask=np.zeros(256,dtype=bool);mask[indices.cpu().numpy()]=True
                            mask_errors+=int(np.sum(mask!=oldmask.cpu().numpy()))
                            max_error=max(max_error,float(torch.max(torch.abs(old-new))))
                        records.append(dict(width=width,seed=seed,device=device,policy=name,
                            exact_confidence_tie_order=True,batches_with_confidence_boundary_ties=boundary_ties,
                            n_batches=len(x)//256,mask_differences=mask_errors,max_logit_error=max_error))
                        if mask_errors or max_error>=1e-4:
                            base.co(OUT/'failed_equivalence.csv',records)
                            raise AssertionError(records[-1])
            print(f'Audited fixed-index serving width {width}, seed {seed}',flush=True)
    base.co(OUT/'equivalence.csv',records)
    base.jo(OUT/'equivalence_commit.json',dict(timestamp_utc=base.now(),
        n_audited_batches=64,n_test_inputs_per_model=16384,
        all_selected_sets_identical=True,max_logit_error=max(r['max_logit_error'] for r in records),
        audit_sha256=base.sha((OUT/'equivalence.csv').read_bytes())))

def timing():
    verify();torch=base.runtime()
    approval=json.loads((HERE/'TIMING_APPROVAL.json').read_text())
    assert approval['main_cpu_jobs_finished'] is True
    assert (OUT/'equivalence_commit.json').exists();rows=[]
    for width in base.CONFIG['widths']:
        for seed in base.CONFIG['seeds']:
            x,tie,_,_=base.test_data(seed);x=x[:4096];tie=tie[:4096]
            cpu_x=torch.as_tensor(x,dtype=torch.float32);cpu_tie=torch.as_tensor(tie,dtype=torch.float64)
            gpu_x=cpu_x.to('cuda');gpu_tie=cpu_tie.to('cuda')
            gpu_net=base.load_model(torch,width,seed,'cuda');cpu_net=base.load_model(torch,width,seed,'cpu')
            gpu_tab=serving.prepare_tables(torch,base.tables(torch,width,seed,'cuda'))
            cpu_tab=serving.prepare_tables(torch,base.tables(torch,width,seed,'cpu'))
            rng=np.random.default_rng(978100+seed+width)
            arms=[(impl,name) for impl in ('original','optimized') for name in base.POLICIES]
            with torch.inference_mode():
                for condition in ('resident','transfer','cpu'):
                    for repeat in range(10):
                        for index in rng.permutation(len(arms)):
                            implementation,name=arms[index]
                            def run(net,bx,bt,tab):
                                if implementation=='original':return base.deploy(torch,net,bx,bt,tab,name)
                                return serving.deploy(torch,net,bx,bt,tab,name)
                            if condition!='cpu':
                                torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                            started=time.perf_counter_ns()
                            for batch in range(16):
                                sl=slice(batch*256,(batch+1)*256)
                                if condition=='resident':out,_=run(gpu_net,gpu_x[sl],None if name in ('raw','full_sequential','full_fused') else gpu_tie[sl],gpu_tab)
                                elif condition=='transfer':
                                    bx=cpu_x[sl].to('cuda');bt=None if name in ('raw','full_sequential','full_fused') else cpu_tie[sl].to('cuda')
                                    out,_=run(gpu_net,bx,bt,gpu_tab);out=out.cpu()
                                else:out,_=run(cpu_net,cpu_x[sl],None if name in ('raw','full_sequential','full_fused') else cpu_tie[sl],cpu_tab)
                            if condition!='cpu':
                                end.record();torch.cuda.synchronize();event_ms=begin.elapsed_time(end)/16
                            else:event_ms=''
                            wall_ms=(time.perf_counter_ns()-started)/1e6/16
                            if repeat>=3:
                                rows.append(dict(width=width,seed=seed,condition=condition,
                                    implementation=implementation,policy=name,repeat=repeat-3,
                                    wall_ms_per_batch=wall_ms,cuda_event_ms_per_batch=event_ms))
            print(f'Timed both serving implementations width {width}, seed {seed}',flush=True)
    base.co(OUT/'latency.csv',rows)
    base.jo(OUT/'timing_complete.json',dict(timestamp_utc=base.now(),n_records=len(rows),
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,
        timing_approval_sha256=base.sha((HERE/'TIMING_APPROVAL.json').read_bytes()),
        transfer_policy='Only needed input and tie tensors are transferred.',
        runner_freeze_sha256=base.sha((OUT/'freeze_v2.json').read_bytes()),
        equivalence_commit_sha256=base.sha((OUT/'equivalence_commit.json').read_bytes())))

def analyze():
    verify();risks=base.ci(HERE/'per_model.csv');latency=base.ci(OUT/'latency.csv');cal=base.ci(HERE/'calibration.csv')
    rows=[]
    for width in base.CONFIG['widths']:
        for seed in base.CONFIG['seeds']:
            risk={z['policy']:float(z['expected_risk']) for z in risks if int(z['width'])==width and int(z['seed'])==seed}
            for condition in ('resident','transfer','cpu'):
                med={(implementation,name):float(np.median([float(z['wall_ms_per_batch']) for z in latency
                    if int(z['width'])==width and int(z['seed'])==seed and z['condition']==condition
                    and z['implementation']==implementation and z['policy']==name]))
                    for implementation in ('original','optimized') for name in base.POLICIES}
                full=min(value for (implementation,name),value in med.items() if name in ('full_sequential','full_fused'))
                for (implementation,name),value in med.items():
                    kind='folded' if name.startswith('folded') else ('radial_angle' if name=='confidence_radial' else None)
                    cost=next(float(z['calibration_seconds']) for z in cal if int(z['width'])==width and int(z['seed'])==seed and z['kind']==kind) if kind else 0.
                    forward_cost=8192 if kind else 0;denom=512-sum(base.expected_calls(name));saving=full-value
                    rows.append(dict(width=width,seed=seed,condition=condition,implementation=implementation,
                        policy=name,median_wall_ms=value,fastest_equivalent_full_ms=full,
                        speedup_vs_fastest_full=full/value,expected_risk=risk[name],
                        risk_delta_vs_raw=risk[name]-risk['raw'],risk_delta_vs_full=risk[name]-risk['full_fused'],
                        calibration_seconds=cost,calibration_forwards=forward_cost,
                        latency_break_even_batches=(int(np.ceil(1000*cost/saving)) if saving>0 else ''),
                        forward_break_even_batches=(int(np.ceil(forward_cost/denom)) if denom>0 else '')))
    base.co(OUT/'cost_quality.csv',rows);summary=[]
    for width in base.CONFIG['widths']:
        for condition in ('resident','transfer','cpu'):
            for implementation in ('original','optimized'):
                for name in base.POLICIES:
                    selected=[r for r in rows if r['width']==width and r['condition']==condition and r['implementation']==implementation and r['policy']==name]
                    summary.append(dict(width=width,condition=condition,implementation=implementation,policy=name,n_models=3,
                        mean_median_wall_ms=float(np.mean([r['median_wall_ms'] for r in selected])),
                        mean_speedup=float(np.mean([r['speedup_vs_fastest_full'] for r in selected])),
                        minimum_speedup=float(min(r['speedup_vs_fastest_full'] for r in selected)),
                        maximum_speedup=float(max(r['speedup_vs_fastest_full'] for r in selected)),
                        models_faster_than_full=sum(r['speedup_vs_fastest_full']>1 for r in selected),
                        mean_expected_risk=float(np.mean([r['expected_risk'] for r in selected])),
                        mean_risk_delta_vs_raw=float(np.mean([r['risk_delta_vs_raw'] for r in selected])),
                        mean_risk_delta_vs_full=float(np.mean([r['risk_delta_vs_full'] for r in selected]))))
    base.co(OUT/'summary.csv',summary)
    base.jo(OUT/'analysis_complete.json',dict(timestamp_utc=base.now(),n_models=9,
        descriptive_models_per_width=3,comparison='fastest equivalent full from both implementations'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','time','analyze'])
    args=parser.parse_args();globals()['timing' if args.stage=='time' else args.stage]()
