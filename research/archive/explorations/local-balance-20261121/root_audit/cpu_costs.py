"""Summarize the isolated equivalent CPU serving experiment."""
from pathlib import Path
import csv,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def read(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def main():
    rows=read(HERE/'cpu_serving_latency.csv');cal=read(ROOT/'validation/calibration.csv')
    names=list(dict.fromkeys(r['policy'] for r in rows))
    latency={(s,p):float(np.median([float(z['ms_per_batch']) for z in rows if int(z['seed'])==s and z['policy']==p])) for s in range(1,17) for p in names}
    full={s:min(latency[s,'full'],latency[s,'full_fused']) for s in range(1,17)}
    details=[];summary=[]
    for name in names:
        parent=name[:-6] if name.endswith('_fused') else name
        kind='new_selector' if parent.startswith('rank_') else 'iid_gate' if parent in ('radius','folded') else None
        per=[]
        for seed in range(1,17):
            c=next((r for r in cal if int(r['seed'])==seed and r['kind']==kind),None)
            cost=float(c['seconds']) if c else 0.;calls=int(c['forward_examples']) if c else 0
            ms=latency[seed,name];saving=full[seed]-ms
            entry=dict(seed=seed,policy=name,ms_per_batch=ms,fastest_full_ms=full[seed],
                speedup=full[seed]/ms,calibration_seconds=cost,calibration_forward_examples=calls,
                calibration_coordinate_only_inputs=int(c['coordinate_only_inputs']) if c else 0,
                wall_time_break_even_batches=int(np.ceil(1000*cost/saving)) if saving>0 else '',
                faster_than_full=saving>0)
            details.append(entry);per.append(entry)
        summary.append(dict(policy=name,median_model_ms=float(np.median([r['ms_per_batch'] for r in per])),
            faster_than_fastest_full_models=sum(r['faster_than_full'] for r in per),
            median_model_speedup=float(np.median([r['speedup'] for r in per])),
            median_calibration_seconds=float(np.median([r['calibration_seconds'] for r in per]))))
    for file,records in [('cpu_costs.csv',details),('cpu_cost_summary.csv',summary)]:
        with (HERE/file).open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
