"""Verify scientific commitments and local event ordering, without rerunning fits."""
from pathlib import Path
import datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(p):return json.loads(p.read_text(encoding='utf8'))
def utc(r):return datetime.datetime.fromisoformat(r.get('utc',r.get('timestamp_utc')).replace('Z','+00:00'))
def main():
    count=0;checks=[]
    a=ROOT/'allocation';record=get(a/'SOURCE_FREEZE.json');source=a/'source'
    if not source.exists():source=Path(record['source_root'])
    for rel,h in record['source_hashes'].items():assert sha(source/rel)==h,rel;count+=1
    assert sha(a/'PROTOCOL.md')==record['protocol_sha256'] and sha(a/'PREFLIGHT.json')==record['preflight_sha256']
    approval=get(a/'ROOT_APPROVAL.json');assert approval['fresh_authorized'] and approval['source_freeze_sha256']==sha(a/'SOURCE_FREEZE.json')
    events=[record,approval,get(a/'TRAIN_START.json'),get(a/'MODEL_COMMIT.json'),get(a/'FIT_COMMIT.json'),get(a/'SCORE_START.json'),get(a/'SCORE_COMPLETE.json')]
    assert all(utc(x)<=utc(y) for x,y in zip(events,events[1:]))
    for name in ('MODEL_COMMIT.json','FIT_COMMIT.json','SCORE_COMPLETE.json'):
        for rel,h in get(a/name)['manifest'].items():assert sha(a/rel)==h,rel;count+=1
    assert get(a/'FIT_COMMIT.json')['model_commit_sha256']==sha(a/'MODEL_COMMIT.json')
    assert get(a/'SCORE_START.json')['fit_commit_sha256']==sha(a/'FIT_COMMIT.json')
    checks.append('allocation source, protocol, preflight, release, fits, outputs and chronology')
    b=ROOT/'distillation';frozen=get(b/'SOURCE_FREEZE.json')
    for rel,h in frozen['hashes'].items():assert sha(b/rel)==h,rel;count+=1
    approval=get(ROOT/'ROOT_APPROVAL.json');assert approval['distillation_freeze_sha256']==sha(b/'SOURCE_FREEZE.json')
    events=[get(b/'external/POPULATION_FREEZE.json'),frozen,approval,get(b/'TRAINING_START.json'),get(b/'MODEL_COMMIT.json'),get(b/'SCORE_START.json'),get(b/'SCORE_COMPLETE.json')]
    assert all(utc(x)<=utc(y) for x,y in zip(events,events[1:]))
    assert get(b/'MODEL_COMMIT.json')['temperatures_sha256']==sha(b/'temperatures.json')
    for name,folder in [('MODEL_COMMIT.json','models'),('SCORE_COMPLETE.json','predictions')]:
        for rel,h in get(b/name)['manifest'].items():assert sha(b/folder/rel)==h,rel;count+=1
    assert get(b/'SCORE_START.json')['model_commit']==sha(b/'MODEL_COMMIT.json')
    checks.append('distillation external boundary, source, release, model/temperature commitments and chronology')
    s=ROOT/'structured';frozen=get(s/'FREEZE.json')
    for rel,h in frozen['hashes'].items():assert sha(s/rel)==h,rel;count+=1
    approval=get(s/'ROOT_APPROVAL.json');assert approval['structured_freeze_sha256']==sha(s/'FREEZE.json') and approval['fresh_execution_authorized']
    events=[frozen,approval,get(s/'CONSTANTS.json'),get(s/'PARAMETER_COMMIT.json'),get(s/'certificate_start.json'),get(s/'CERTIFICATE_COMMIT.json'),get(s/'stress_start.json'),get(s/'STRESS_COMMIT.json')]
    assert all(utc(x)<=utc(y) for x,y in zip(events,events[1:]))
    for name,folder in [('CERTIFICATE_COMMIT.json','certificate'),('STRESS_COMMIT.json','stress')]:
        for rel,h in get(s/name)['raw_files'].items():assert sha(s/folder/rel)==h,rel;count+=1
    assert get(s/'PARAMETER_COMMIT.json')['raw_sha256']==sha(s/'parameter_raw.npz')
    checks.append('structured theory/source, historical inputs, parameter samples, moments/truth and chronology')
    result=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hash_checks=count,checks=checks,
        limitation='Local hash/event records establish internal consistency, not independent externally timestamped registration.')
    if '--record' in sys.argv:
        (ROOT/'SOURCE_INTEGRITY.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
