"""Read-only source/data commitments and scientific stage-order verification."""
from pathlib import Path
import datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
def get(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def chronology(paths):
    values=[datetime.datetime.fromisoformat(get(p)['utc']) for p in paths]
    assert all(a<=b for a,b in zip(values,values[1:])),[str(p) for p in paths]
def main():
    count=0;a=ROOT/'allocation';f=get(a/'SOURCE_FREEZE.json')
    for rel,h in f['source_hashes'].items():assert sha(a/'source'/rel)==h,rel;count+=1
    for name,field in [('PROTOCOL.md','protocol_sha256'),('PREFLIGHT.json','preflight_sha256')]:assert sha(a/name)==f[field];count+=1
    assert get(a/'ROOT_APPROVAL.json')['source_freeze_sha256']==sha(a/'SOURCE_FREEZE.json')
    stages=[a/'SOURCE_FREEZE.json',a/'ROOT_APPROVAL.json']
    for phase in ('development','confirmation'):
        d=a/phase
        stages += [d/name for name in ('TRAIN_START.json','MODEL_COMMIT.json','FIT_COMMIT.json','SCORE_START.json','SCORE_COMPLETE.json')]
        for name in ('MODEL_COMMIT.json','FIT_COMMIT.json','SCORE_COMPLETE.json'):
            for rel,h in get(d/name)['manifest'].items():assert sha(a/rel)==h,rel;count+=1
        assert get(d/'TRAIN_START.json')['source_freeze_sha256']==sha(a/'SOURCE_FREEZE.json')
        assert get(d/'FIT_COMMIT.json')['model_commit_sha256']==sha(d/'MODEL_COMMIT.json')
        assert get(d/'SCORE_START.json')['fit_commit_sha256']==sha(d/'FIT_COMMIT.json')
        if phase=='development':stages.append(a/'SELECTION_COMMIT.json')
    assert get(a/'SELECTION_COMMIT.json')['development_score_commit_sha256']==sha(a/'development/SCORE_COMPLETE.json')
    assert get(a/'confirmation/TRAIN_START.json')['selection_commit_sha256']==sha(a/'SELECTION_COMMIT.json')
    stages += [a/'ANALYSIS_COMPLETE.json',a/'TIMING_AUDIT.json'];chronology(stages)
    for name in ('SELECTION_COMMIT.json','ANALYSIS_COMPLETE.json'):
        for rel,h in get(a/name)['manifest'].items():assert sha(a/rel)==h;count+=1
    d=ROOT/'distillation'
    for rel,h in get(d/'SOURCE_FREEZE.json')['hashes'].items():assert sha(d/rel)==h,rel;count+=1
    assert get(ROOT/'ROOT_APPROVAL.json')['distillation_freeze_sha256']==sha(d/'SOURCE_FREEZE.json')
    assert get(d/'TRAINING_START.json')['approval_sha256']==sha(ROOT/'ROOT_APPROVAL.json')
    for rel,h in get(d/'MODEL_COMMIT.json')['manifest'].items():assert sha(d/'models'/rel)==h;count+=1
    assert get(d/'MODEL_COMMIT.json')['temperatures_sha256']==sha(d/'temperatures.json')
    assert get(d/'SCORE_START.json')['model_commit']==sha(d/'MODEL_COMMIT.json')
    for rel,h in get(d/'SCORE_COMPLETE.json')['manifest'].items():assert sha(d/'predictions'/rel)==h;count+=1
    chronology([d/'SOURCE_FREEZE.json',ROOT/'ROOT_APPROVAL.json',d/'TRAINING_START.json',d/'MODEL_COMMIT.json',d/'SCORE_START.json',d/'SCORE_COMPLETE.json',d/'SUCCESS_CRITERIA.json',d/'TIMING_RELEASE.json',d/'TIMING_START.json',d/'TIMING_COMPLETE.json'])
    s=ROOT/'structured'
    for rel,h in get(s/'FREEZE.json')['hashes'].items():assert sha(s/rel)==h,rel;count+=1
    assert get(s/'ROOT_APPROVAL.json')['residual_freeze_sha256']==sha(s/'FREEZE.json')
    recovery=get(s/'STRESS_RECOVERY_FREEZE.json')
    assert get(s/'STRESS_RECOVERY_APPROVAL.json')['recovery_freeze_sha256']==sha(s/'STRESS_RECOVERY_FREEZE.json')
    for rel,h in dict(recovery['hashes'],**recovery['preserved_failure']).items():assert sha(s/rel)==h;count+=1
    for folder,name in [('evidence','EVIDENCE_COMMIT.json'),('certificate','CERTIFICATE_COMMIT.json'),('stress_recovery','STRESS_RECOVERY_COMMIT.json')]:
        for rel,h in get(s/name)['raw_files'].items():assert sha(s/folder/rel)==h;count+=1
    chronology([s/'FREEZE.json',s/'ROOT_APPROVAL.json',s/'CONSTANTS.json',s/'CHECKS.json',s/'EVIDENCE_COMMIT.json',s/'CERTIFICATE_START.json',s/'CERTIFICATE_COMMIT.json',s/'STRESS_START.json',s/'STRESS_RECOVERY_APPROVAL.json',s/'STRESS_RECOVERY_START.json',s/'STRESS_RECOVERY_COMMIT.json'])
    out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hash_checks=count,
        checks=['allocation development selection before fresh confirmation','distillation frozen new population and model commitments before scoring','structured assumptions/source, evidence, verification and truth stages'],
        limits='Internal chronological/hash consistency, not an independent externally timestamped preregistration.')
    if '--record' in sys.argv:(ROOT/'SOURCE_INTEGRITY.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
