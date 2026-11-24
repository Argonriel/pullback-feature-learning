"""Bounded development configurations, exclusively on previously observed models."""
from pathlib import Path
import csv,json,hashlib,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import local
ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'pilot';out.mkdir(exist_ok=False)
    spec=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_ids=[1,2,8,16],configurations=[[4,256],[8,256],[4,512],[8,512]],source_sha256=hashlib.sha256((ROOT/'local.py').read_bytes()).hexdigest(),new_model_access=False)
    (out/'spec.json').write_text(json.dumps(spec,indent=2));rows=[]
    old={int(r['seed']):r for r in csv.DictReader(open(local.PRIOR/'continuous/fresh/balance_bounds.csv')) if r['field']=='balanced'}
    for seed in spec['model_ids']:
        table=dict(np.load(local.PRIOR/f'continuous/fresh/policies/{seed}_policy.npz'))
        for sub,na in spec['configurations']:
            b=local.construct(table,sub,na);np.savez_compressed(out/f'{seed}_{sub}_{na}.npz',**b)
            row=dict(seed=seed,sub=sub,na=na,**{k:float(b[k]) for k in ['uniform_bound','integrated_bound','angular_integral','radial_integral','endpoint_integral','point_seconds','interval_seconds','interval_jet_propagations','exact_point_forwards']},old_uniform_bound=float(old[seed]['uniform_bound']),old_integrated_bound=float(old[seed]['integrated_bound']),observed_integrated_abs_b=float(old[seed]['observed_integrated_abs_balance']),C=float(old[seed]['C']))
            rows.append(row);print(json.dumps(row),flush=True)
    with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':
    with threadpool_limits(1):main()
