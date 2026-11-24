"""Reconstruct both isolated timing summaries and charge warmup separately."""
from pathlib import Path
import csv,datetime,json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:list(csv.DictReader((ROOT/p).open(newline='')))
image=read('distillation/latency.csv');calls=read('distillation/timing_calls.csv');summary=read('distillation/serving_summary.csv')
assert len(image)==315 and len(calls)==450
for r in summary:
    med={m:float(np.median([float(v['ms_per_batch']) for v in image if v['pair']==r['pair'] and v['mode']==r['mode'] and v['method']==m])) for m in ['B','S','C','T_fused','T_sequential']}
    for key,v in med.items():assert abs(float(r[key])-v)<1e-12
    best=min(med['T_fused'],med['T_sequential']);assert abs(float(r['best_full_ms'])-best)<1e-12
    assert abs(float(r['full_over_student'])-best/med['S'])<1e-12
for r in calls:assert int(r['forward_examples'])==1024*(2 if r['method'].startswith('T_') else 1)
serving={}
for mode in ['cpu','cuda_resident','cuda_transfer']:
    rr=[r for r in summary if r['mode']==mode];ratio=np.median([float(r['S']) for r in rr])/np.median([float(r['B']) for r in rr])
    serving[mode]=dict(student_over_plain=float(ratio),within_ten_percent=bool(ratio<=1.10))
arows=read('allocation/isolated_timing.csv');aproject=read('allocation/timing_projections.csv');assert len(arows)==28
for r in aproject:
    samples=[float(v['seconds_per_update']) for v in arows if v['n_train']==r['n_train']];p=np.median(samples)*int(r['updates'])
    assert abs(float(r['projected_training_seconds'])-p)<1e-12
    for v in [v for v in arows if v['n_train']==r['n_train']]:assert abs(float(v['seconds_per_update'])-float(v['seconds'])/30)<1e-14
timedata=json.loads((ROOT/'allocation/TIMING_AUDIT.json').read_text())
ops=sum(30*(179045*int(r['n_train'])+2001144) for r in arows)
assert ops==timedata['actual_timed_nominal_operations']
out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),image_latency_rows=len(image),
    image_measured_forward_examples=sum(int(r['forward_examples']) for r in calls if r['warmup']=='False'),
    image_warmup_forward_examples=sum(int(r['forward_examples']) for r in calls if r['warmup']=='True'),
    serving_conditions=serving,allocation_timing_rows=len(arows),allocation_measured_updates=840,allocation_warmup_updates=20,
    allocation_measured_nominal_operations=ops,allocation_warmup_nominal_operations=sum(5*(179045*n+2001144) for n in [256,768,1280,2304]),
    additional_model_forwards=0,limits='Recalculates recorded isolated measurements, not an independent repeated hardware benchmark.')
(ROOT/'root_audit/TIMING_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
