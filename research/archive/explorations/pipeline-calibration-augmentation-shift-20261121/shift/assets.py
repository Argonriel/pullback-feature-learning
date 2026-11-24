"""Vendor only prespecified historical inputs; this creates no fresh draws."""
import json, shutil
import numpy as np
from common import HERE,sha,jo,now
OLD=HERE.parents[1]/'calibration-symmetry-realdata-20261121'
rows=json.loads((OLD/'training/robustness_inputs.json').read_text())['models'];out=[]
for r in rows:
    if r['variant']!='legacy_raw' or r['pair']>8:continue
    i=r['pair'];theta=np.load(OLD/r['theta_path'])[r['theta_key']]
    tp=HERE/f'historical/{i}_theta.npy';np.save(tp,theta)
    row=dict(pair=i,variant='legacy_raw',theta_path=tp.relative_to(HERE).as_posix(),original=r)
    for name in ['grid','folded']:
        dest=HERE/f'historical/{i}_{name}.npz';shutil.copyfile(OLD/r[name+'_path'],dest)
        assert sha(dest)==r[name+'_sha256'];row[name+'_path']=dest.relative_to(HERE).as_posix()
    row['hashes']={row[k]:sha(HERE/row[k]) for k in ['theta_path','grid_path','folded_path']};out.append(row)
assert len(out)==8
jo(HERE/'HISTORICAL_INPUTS.json',dict(timestamp_utc=now(),selection='first eight legacy_raw pair indices, fixed without new outcomes',
    original_manifest_sha256=sha(OLD/'training/robustness_inputs.json'),models=out))
