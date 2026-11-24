"""Close provenance/decision audit coverage without repeating CNN replays."""
from pathlib import Path
import csv,datetime,gzip,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;B=ROOT/'distillation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fits=json.loads((B/'temperatures.json').read_text());commit=json.loads((B/'MODEL_COMMIT.json').read_text())
assert sha(B/'temperatures.json')==commit['temperatures_sha256']
labels=np.frombuffer(gzip.decompress((B/'data/train-labels-idx1-ubyte.gz').read_bytes()),np.uint8,offset=8)
ids=np.load(B/'splits.npz')['temperature']
for row in fits:
    cal=np.load(B/'models'/f"{row['pair']}_{row['arm']}_calibration.npz")
    assert np.array_equal(cal['indices'],ids) and np.array_equal(cal['labels'],labels[ids])
contrasts=list(csv.DictReader((B/'contrasts.csv').open()));original=json.loads((B/'SUCCESS_CRITERIA.json').read_text());conditions={}
for dataset in ['holdout','external']:
    vals={}
    for row in fits:
        pred=np.load(B/'predictions'/f"{dataset}_{row['pair']}_{row['arm']}.npz");y=pred['labels'];h=pred['raw'].astype(float)
        if row['arm']=='teacher':out={'B':h*row['raw']['beta'],'T':(h+pred['reflected'])/2*row['full']['beta']}
        else:out={('S' if row['arm']=='student' else 'C'):h*row['raw']['beta']}
        for key,z in out.items():vals[row['pair'],key]=np.stack([np.logaddexp.reduce(z,axis=1)-z[np.arange(len(y)),y],(z.argmax(1)!=y).astype(float)])
    means={key:np.mean([vals[pair,key] for pair in (1,2,3)],axis=0) for key in 'BTSC'};conditions[dataset]={}
    for j,metric in enumerate(['ce','error']):
        teacher=float(np.mean(means['B'][j]-means['T'][j]));student=float(np.mean(means['B'][j]-means['S'][j]));margin=.00325 if metric=='ce' else .0014
        highs={r['contrast']:float(r['upper']) for r in contrasts if r['dataset']==dataset and r['metric']==metric}
        values=dict(teacher_improves=teacher>0,half_point_gain_retained=teacher>0 and student>=teacher/2,noninferior_to_teacher=highs['S-T']<margin,noninferior_to_compute_control=highs['S-C']<margin)
        for key,value in values.items():assert value==original['conditions'][dataset][metric][key]
        conditions[dataset][metric]=values
overall=all(all(v.values()) for d in conditions.values() for v in d.values());assert overall==original['overall_success']
out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),calibration_provenance_records=9,temperature_commit_hash_matches=True,
    independently_reconstructed_preservation_success=overall,conditions=conditions,additional_model_forwards=0)
(ROOT/'root_audit/DISTILLATION_CRITERIA.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
