"""Verify all frozen source hashes and chronological release commitments."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads((ROOT/path).read_text(encoding='utf8'))
def stamp(path):return datetime.datetime.fromisoformat(read(path)['timestamp_utc'])

def run():
    checked=[]
    for name,key,base in [('transfer/FREEZE.json','file_sha256',ROOT/'transfer'),
        ('validation/freeze.json','source_hashes',ROOT),('certificate/freeze.json','source_hashes',ROOT),
        ('root_audit/certificate_freeze.json','hashes',ROOT)]:
        obj=read(name)
        for rel,digest in obj[key].items():
            assert sha(base/rel)==digest,(name,rel)
            checked.append(dict(freeze=name,path=str((base/rel).relative_to(ROOT)),sha256=digest))
    approval=read('validation/ROOT_APPROVAL.json')
    for rel,digest in approval['freezes'].items():
        assert sha(ROOT/rel)==digest
        assert stamp(rel)<stamp('validation/ROOT_APPROVAL.json')
    timeline=['validation/ROOT_APPROVAL.json','validation/training_start.json','validation/training_commit.json',
        'validation/calibration_start.json','validation/policy_commit.json','validation/score_start.json','validation/score_complete.json']
    times=[stamp(p) for p in timeline];assert all(a<=b for a,b in zip(times[:-1],times[1:]))
    (ROOT/'root_audit/freeze_checks.json').write_text(json.dumps(dict(all_passed=True,
        checked_source_entries=len(checked),sources=checked,
        chronological_records={p:str(t) for p,t in zip(timeline,times)}),indent=2),encoding='utf8')
    print(f'All {len(checked)} frozen source entries and release chronology verified.')

if __name__=='__main__':run()
