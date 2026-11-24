"""Internal scientific release after independent pre-fresh checks."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jo(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
assert not any((ROOT/'validation/states').glob('*.npz'))
checks=0
for folder,name,key in [('method','METHOD_FREEZE.json','source_hashes'),('task2d','TASK_FREEZE.json','files'),
                        ('certification','FREEZE.json','source_hashes'),('validation','FREEZE.json','hashes')]:
    data=json.loads((ROOT/folder/name).read_text())
    base=ROOT/folder if folder in ('method','task2d') else ROOT
    for rel,h in data[key].items():assert sha(base/rel)==h,(folder,rel);checks+=1
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
common=dict(timestamp_utc=stamp,no_fresh_states=True,verified_dependencies=checks,
            user_authorization='Continue all three recommendations after literature search',
            seed_correction='Fresh rings1161..1166; old pilot1101/1102 preserved',
            addendum_freeze_sha256=sha(ROOT/'certification/ADDENDUM_FREEZE.json'),
            certificate_addendum_freeze_sha256=sha(ROOT/'certification/ADDENDUM_FREEZE.json'))
jo(ROOT/'validation/ROOT_APPROVAL.json',dict(**common,fresh_execution_authorized=True,
     validation_freeze_sha256=sha(ROOT/'validation/FREEZE.json')))
jo(ROOT/'certification/ROOT_APPROVAL.json',dict(**common,fresh_verification_authorized=True,
     certificate_freeze_sha256=sha(ROOT/'certification/FREEZE.json')))
print(json.dumps(common))
