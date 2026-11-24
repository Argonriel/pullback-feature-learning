"""Independent real-image risk, timing, quota and bootstrap reconstruction."""
from pathlib import Path
import csv,datetime,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'realdata'
def read(p):return list(csv.DictReader((HERE/p).open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
quality=read('quality.csv');latency=read('latency.csv');cost=read('cost_quality.csv');ledger=read('deployment_ledger.csv')
max_risk=max_timing=max_ci=0.;manifest_checks=0
commit=json.loads((HERE/'TEST_COMPLETE.json').read_text())
for name,h in commit['manifest'].items():assert sha(HERE/'predictions'/name)==h;manifest_checks+=1
saved={}
for seed in (1,2,3):
 for device in ('cpu','cuda'):
    d=dict(np.load(HERE/'predictions'/f'{seed}_{device}.npz'));saved[(seed,device)]=d;y=d['labels'];names=list(d['names'])
    for j,name in enumerate(names):
        z=d['logits'][j].astype(float);a=z.max(1);loss=a+np.log(np.exp(z-a[:,None]).sum(1))-z[np.arange(len(y)),y]
        zt=d['flipped_logits'][j].astype(float);at=zt.max(1);loss_t=at+np.log(np.exp(zt-at[:,None]).sum(1))-zt[np.arange(len(y)),y]
        row=next(r for r in quality if int(r['seed'])==seed and r['device']==device and r['policy']==name)
        values=dict(original_loss=loss.mean(),original_error=np.mean(z.argmax(1)!=y),paired_loss=(loss.mean()+loss_t.mean())/2,
             paired_error=(np.mean(z.argmax(1)!=y)+np.mean(zt.argmax(1)!=y))/2)
        for key,v in values.items():
            err=abs(float(v)-float(row[key]));max_risk=max(max_risk,err);assert err<2e-7,(seed,device,name,key,err)
        mask=np.unpackbits(d['masks'],axis=1)[j,:10000]
        for begin in range(0,10000,256):
            n=min(256,10000-begin);assert mask[begin:begin+n].sum()==(0 if name=='raw' else n if name.startswith('full_') else n//4)
for row in ledger:
    n=int(row['images']);name=row['policy'];expected=[n] if name=='raw' else [2*n] if name=='full_fused' else [n,n] if name in ('full_sequential','oracle_jensen') else [n,n//4] if name=='confidence' else [n+n//4]
    assert list(map(int,row['calls'].split('|')))==expected and int(row['forward_examples'])==sum(expected)
assert len(ledger)==1920 and sum(int(r['forward_examples']) for r in ledger)==720000
for row in cost:
    seed=int(row['seed']);mode=row['mode'];name=row['policy']
    med={p:float(np.median([float(v['ms_per_batch']) for v in latency if int(v['seed'])==seed and v['mode']==mode and v['policy']==p])) for p in saved[(seed,'cpu')]['names']}
    ref=min(('full_fused','full_sequential'),key=med.get);speed=med[ref]/med[name]
    assert ref==row['strong_full'];error=max(abs(med[name]-float(row['median_ms'])),abs(speed-float(row['speedup_vs_full'])));max_timing=max(max_timing,error);assert error<1e-12
    expected=speed>=1.10 and float(row['excess_loss_vs_full'])<=.01
    assert str(expected)==row['practical_pass']
choices=json.loads((HERE/'choices.json').read_text())['choices'];draws=np.random.default_rng(2138000).integers(0,40,(5000,40));sizes=np.r_[np.full(39,256),16]
for row in read('contrasts.csv'):
    device=row['device'];key='losses' if row['world']=='original' else 'paired_losses';ref=row['contrast'].removeprefix('ridge_minus_');differences=[]
    for seed in (1,2,3):
        d=saved[(seed,device)];names=list(d['names']);control=choices[str(seed)]['strong_control'] if ref=='strong_control' else ref
        differences.append(d[key][names.index('ridge')]-d[key][names.index(control)])
    delta=np.mean(differences,axis=0);sums=np.array([delta[i:i+256].sum() for i in range(0,10000,256)])
    ci=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975]);error=max(abs(ci[0]-float(row['lower'])),abs(ci[1]-float(row['upper'])));max_ci=max(max_ci,error);assert error<1e-14
assert len(latency)==504 and sum(int(r['forward_examples']) for r in latency)==774144
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',model_device_runs=6,quality_rows=48,
  max_float64_vs_fp32_risk_error=max_risk,quality_ledger_rows=len(ledger),quality_forwards=720000,
  timing_rows=504,timing_recorded_forwards=774144,timing_warmup_forwards=331776,
  max_timing_aggregate_error=max_timing,cluster_bootstrap_checks=12,max_bootstrap_error=max_ci,
  performance_manifests_checked=manifest_checks,additional_model_forwards=0,
  qualification='Timing is descriptive for these three models; full-relative passes are not original-task predictive success.')
(ROOT/'root_audit/REALDATA_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
