"""Execution-only staging wrapper; frozen audit algorithms/grids are unchanged."""
from pathlib import Path
import json,hashlib,datetime,time
import numpy as np
from threadpoolctl import threadpool_limits
import independent_certificate_checks as audit
ROOT=Path(__file__).resolve().parent
expected={}
for task in ('rings','waveform'):
    folder=ROOT.parent/'certificate'/'fresh'/task
    for seed in (1,8,16):
        for suffix in ('local','distribution'):expected[(folder/f'{seed}_{suffix}.npz').resolve()]=folder/f'{seed}_provenance.json'
original=np.load;waits=[]
def staged_load(path,*args,**kwargs):
    try:resolved=Path(path).resolve()
    except TypeError:return original(path,*args,**kwargs)
    if resolved in expected:
        started=time.perf_counter()
        while not expected[resolved].exists():
            if time.perf_counter()-started>900:raise TimeoutError(f'Bound producer did not commit {expected[resolved]}')
            time.sleep(2)
        waits.append(dict(path=str(resolved),seconds=time.perf_counter()-started))
    return original(path,*args,**kwargs)

if __name__=='__main__':
    stamp=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Wait for each fixed model bound provenance before reading it, so unchanged audits can overlap producer work.',audit_source_sha256=hashlib.sha256((ROOT/'independent_certificate_checks.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'INDEPENDENT_CHECK_PROTOCOL.json').read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods_or_grids_changed=False)
    with (ROOT/'INDEPENDENT_EXECUTION.json').open('x') as f:json.dump(stamp,f,indent=2)
    np.load=staged_load
    try:
        with threadpool_limits(1):audit.run()
    finally:
        np.load=original;(ROOT/'INDEPENDENT_STAGING_WAITS.json').write_text(json.dumps(waits,indent=2))
