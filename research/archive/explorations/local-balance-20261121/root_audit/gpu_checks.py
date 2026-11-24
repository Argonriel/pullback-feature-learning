"""Independent CSV/NPZ GPU audit; no model inference or benchmark is rerun."""
from pathlib import Path
import csv,json,hashlib,math,collections
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;GPU=ROOT/'gpu';OPT=GPU/'optimized'

def read(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sample(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);a=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0,.15,(n,2))
def close(a,b,tol=1e-12):assert abs(float(a)-float(b))<tol,(a,b)

def main():
    initial=json.loads((GPU/'freeze.json').read_text());cfg=initial['config'];policies=initial['policies'];widths=cfg['widths'];seeds=cfg['seeds']
    source_count=0
    for root,path in [(ROOT,GPU/'freeze.json'),(GPU,OPT/'freeze.json'),(GPU,OPT/'freeze_v2.json')]:
        f=json.loads(path.read_text())
        for rel,digest in f['source_hashes'].items():assert sha(root/rel)==digest,rel;source_count+=1
    training_commit=json.loads((GPU/'training_commit.json').read_text());policy_commit=json.loads((GPU/'policy_commit.json').read_text());score_start=json.loads((GPU/'score_start.json').read_text());score_complete=json.loads((GPU/'score_complete.json').read_text());equiv_commit=json.loads((OPT/'equivalence_commit.json').read_text());timing_complete=json.loads((OPT/'timing_complete.json').read_text())
    for folder,commit in [('models',training_commit),('policies',policy_commit)]:
        for name,digest in commit['files'].items():assert sha(GPU/folder/name)==digest
    assert sha(GPU/'per_model.csv')==score_complete['per_model_sha256']
    assert sha(OPT/'equivalence.csv')==equiv_commit['audit_sha256']
    assert initial['timestamp_utc']<training_commit['timestamp_utc']<policy_commit['timestamp_utc']<score_start['timestamp_utc']<score_complete['timestamp_utc']
    assert json.loads((OPT/'freeze_v2.json').read_text())['timestamp_utc']<timing_complete['timestamp_utc']
    assert sha(OPT/'freeze_v2.json')==timing_complete['runner_freeze_sha256']
    assert sha(GPU/'TIMING_APPROVAL.json')==timing_complete['timing_approval_sha256']
    riskrows=read(GPU/'per_model.csv');risk_index={(int(r['width']),int(r['seed']),r['policy']):r for r in riskrows};assert len(risk_index)==63
    max_risk_error=0.;max_p_error=0.;quality=[];test_hashes=collections.defaultdict(set);training_hashes=collections.defaultdict(set)
    callsets={'raw':[256],'full_sequential':[256,256],'full_fused':[512],'confidence_pi':[256,64],'folded_sequential':[256,64],'folded_fused':[320],'confidence_radial':[256,64]}
    for width in widths:
        for seed in seeds:
            d=dict(np.load(GPU/f'predictions/w{width}_s{seed}.npz'));x=d['x'];n=len(x);assert n==16384 and np.array_equal(x,sample(976000+seed,n))
            rr=np.linalg.norm(x,axis=1);logodds=(.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(1.2*rr/.15**2))-np.log(np.i0(.5*rr/.15**2));p=np.exp(-np.logaddexp(0,-logodds));max_p_error=max(max_p_error,float(abs(p-d['posterior']).max()));assert np.array_equal(p,d['posterior'])
            assert np.array_equal(d['ties'],np.random.default_rng(977000+seed).random(n));y=(np.random.default_rng(978000+seed).random(n)<p).astype(float);assert np.array_equal(y,d['labels'])
            test_hashes[seed].add(hashlib.sha256(x.tobytes()).hexdigest());train=dict(np.load(GPU/f'models/w{width}_s{seed}_training.npz'));training_hashes[seed].add(hashlib.sha256(train['x'].tobytes()).hexdigest())
            masks=np.unpackbits(d['masks'],axis=1,count=n).astype(bool)
            for j,name in enumerate(d['names'].tolist()):
                row=risk_index[width,seed,name];h=d['outputs'][j].astype(float);risk=float((np.logaddexp(0,h)-p*h).mean());labelrisk=float((np.logaddexp(0,h)-y*h).mean());err=abs(risk-float(row['expected_risk']));max_risk_error=max(max_risk_error,err);assert err<1e-12;close(labelrisk,row['sampled_label_risk']);close(np.where(h>=0,1-p,p).mean(),row['expected_error']);close(((h>=0)!=y).mean(),row['sampled_error'])
                wanted=0 if name=='raw' else 256 if name.startswith('full_') else 64;assert np.all(masks[j].reshape(-1,256).sum(1)==wanted)
                assert hashlib.sha256(d['outputs'][j].tobytes()).hexdigest()==row['output_sha256'];assert hashlib.sha256(d['masks'][j].tobytes()).hexdigest()==row['mask_sha256']
                assert int(row['forward_examples'])==n+int(masks[j].sum())
            for a,b in [('full_sequential','full_fused'),('folded_sequential','folded_fused')]:
                ia=policies.index(a);ib=policies.index(b);assert np.array_equal(masks[ia],masks[ib]);assert np.max(abs(d['outputs'][ia]-d['outputs'][ib]))<1e-4
    assert all(len(v)==1 for v in test_hashes.values()) and all(len(v)==1 for v in training_hashes.values())
    assert len({next(iter(v)) for v in test_hashes.values()})==3
    audit=read(GPU/'deployment_audit.csv');assert len(audit)==4032
    for row in audit:
        assert json.loads(row['calls'])==callsets[row['policy']]
        assert int(row['selected'])==(0 if row['policy']=='raw' else 256 if row['policy'].startswith('full_') else 64)
    equiv=read(OPT/'equivalence.csv');assert len(equiv)==126 and all(int(r['mask_differences'])==0 and float(r['max_logit_error'])==0 and int(r['n_batches'])==64 for r in equiv)
    fusion=read(GPU/'fusion_equivalence.csv');assert len(fusion)==18 and all(r['same_mask']=='True' and float(r['max_logit_error'])<1e-4 for r in fusion)
    latency=read(OPT/'latency.csv');assert len(latency)==2646
    groups=collections.defaultdict(list)
    for row in latency:groups[int(row['width']),int(row['seed']),row['condition'],row['implementation'],row['policy']].append(row)
    assert len(groups)==378
    for key,v in groups.items():assert len(v)==7 and {int(r['repeat']) for r in v}==set(range(7)) and all(float(r['wall_ms_per_batch'])>0 for r in v)
    median={k:float(np.median([float(r['wall_ms_per_batch']) for r in v])) for k,v in groups.items()}
    cal={(int(r['width']),int(r['seed']),r['kind']):r for r in read(GPU/'calibration.csv')};cost=read(OPT/'cost_quality.csv');assert len(cost)==378;computed=[];max_speed_error=0.;max_latency_error=0.
    for row in cost:
        width=int(row['width']);seed=int(row['seed']);condition=row['condition'];impl=row['implementation'];name=row['policy'];key=width,seed,condition,impl,name;actual=median[key]
        full=min(median[width,seed,condition,i,f] for i in ['original','optimized'] for f in ['full_sequential','full_fused']);speed=full/actual;max_speed_error=max(max_speed_error,abs(speed-float(row['speedup_vs_fastest_full'])));max_latency_error=max(max_latency_error,abs(actual-float(row['median_wall_ms'])));close(full,row['fastest_equivalent_full_ms']);close(speed,row['speedup_vs_fastest_full'])
        kind='folded' if name.startswith('folded') else 'radial_angle' if name=='confidence_radial' else None;seconds=float(cal[width,seed,kind]['calibration_seconds']) if kind else 0.;forwards=8192 if kind else 0;saving=full-actual
        latency_be=str(math.ceil(1000*seconds/saving)) if saving>0 else '';forward_be=str(math.ceil(forwards/(512-sum(callsets[name])))) if sum(callsets[name])<512 else ''
        assert row['latency_break_even_batches']==latency_be and row['forward_break_even_batches']==forward_be
        risk=float(risk_index[width,seed,name]['expected_risk']);raw=float(risk_index[width,seed,'raw']['expected_risk']);frisk=float(risk_index[width,seed,'full_fused']['expected_risk']);close(risk,row['expected_risk']);close(risk-raw,row['risk_delta_vs_raw']);close(risk-frisk,row['risk_delta_vs_full'])
        computed.append(dict(width=width,seed=seed,condition=condition,implementation=impl,policy=name,median_wall_ms=actual,full_ms=full,speedup=speed,risk=risk,risk_raw_delta=risk-raw,risk_full_delta=risk-frisk,latency_break_even_batches=latency_be,forward_break_even_batches=forward_be))
    summaries=read(OPT/'summary.csv');assert len(summaries)==126;max_summary_error=0.
    for row in summaries:
        selected=[r for r in computed if r['width']==int(row['width']) and r['condition']==row['condition'] and r['implementation']==row['implementation'] and r['policy']==row['policy']];assert len(selected)==3
        fields={'mean_median_wall_ms':np.mean([r['median_wall_ms'] for r in selected]),'mean_speedup':np.mean([r['speedup'] for r in selected]),'minimum_speedup':min(r['speedup'] for r in selected),'maximum_speedup':max(r['speedup'] for r in selected),'mean_expected_risk':np.mean([r['risk'] for r in selected]),'mean_risk_delta_vs_raw':np.mean([r['risk_raw_delta'] for r in selected]),'mean_risk_delta_vs_full':np.mean([r['risk_full_delta'] for r in selected])}
        for k,v in fields.items():max_summary_error=max(max_summary_error,abs(v-float(row[k])));close(v,row[k])
        assert int(row['n_models'])==3 and int(row['models_faster_than_full'])==sum(r['speedup']>1 for r in selected)
    limited=[r for r in computed if r['policy'] not in ['raw','full_sequential','full_fused']];gpu_limited=[r for r in limited if r['condition'] in ['resident','transfer']];assert all(r['speedup']<1 and r['latency_break_even_batches']=='' for r in gpu_limited)
    headline=[]
    for width in widths:
        for condition in ['resident','transfer','cpu']:
            chosen=[r for r in computed if r['width']==width and r['condition']==condition and r['implementation']=='optimized' and r['policy']=='folded_fused']
            headline.append(dict(width=width,condition=condition,mean_speedup=float(np.mean([r['speedup'] for r in chosen])),min_speedup=min(r['speedup'] for r in chosen),max_speedup=max(r['speedup'] for r in chosen),faster_models=sum(r['speedup']>1 for r in chosen),calibration_only_break_even_batches=[r['latency_break_even_batches'] for r in chosen]))
        means={name:float(np.mean([float(risk_index[width,seed,name]['expected_risk']) for seed in seeds])) for name in policies}
        quality.append(dict(width=width,**means,folded_fraction_of_full_mean_gain=(means['raw']-means['folded_fused'])/(means['raw']-means['full_fused'])))
    result=dict(status='PASS with interpretation qualifications',frozen_source_hashes_checked=source_count,model_weight_and_policy_hashes_pass=True,quality_rows_recomputed=63,max_expected_risk_recomputation_error=max_risk_error,posterior_recomputation_error=max_p_error,raw_call_records_checked=len(audit),equivalence_rows=len(equiv),equivalence_policy_device_batches=sum(int(r['n_batches']) for r in equiv),equivalence_max_logit_error=0.,fusion_checks=18,timing_rows=len(latency),median_groups=len(median),maximum_median_error=max_latency_error,maximum_speedup_error=max_speed_error,maximum_summary_error=max_summary_error,break_even_rows_checked=len(cost),gpu_limited_policy_comparisons=len(gpu_limited),gpu_limited_faster_count=sum(r['speedup']>1 for r in gpu_limited),folded_fused_headlines=headline,cuda_quality=quality,models_per_width=3,independent_training_data_draws=3,paired_training_and_test_data_across_widths=True,distinct_from_main='Two-hidden-layer FP32 MLPs, Adam1000 on1024 examples; not the main80-parameter PB500 models.',interpretation=['Fastest equivalent full baseline is the minimum of both full policies in both implementations within each model and timing condition.','The v2 source transfers no tie tensor for raw or full policies; needed ties are retained for limited policies.','Quality values are CUDA-scored and attached to CPU timing rows for the same frozen weights; no cross-device risk equivalence is inferred.','Latency break-even is calibration-only, excluding one-time rank preparation, table/model loading and generation of supplied random tie keys.','Seven retained timing repeats are not model replications; only three paired training-data seeds per width support descriptive conclusions.','GPU results do not establish PB scaling or an all-hidden-posterior certificate for the larger networks.'],raw_sha256={str(p.relative_to(GPU)):sha(p) for p in [OPT/'latency.csv',OPT/'cost_quality.csv',OPT/'summary.csv',GPU/'per_model.csv',GPU/'deployment_audit.csv',OPT/'equivalence.csv']})
    (ROOT/'root_audit/gpu_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ['raw_sha256','interpretation','cuda_quality']},indent=2))

if __name__=='__main__':main()
