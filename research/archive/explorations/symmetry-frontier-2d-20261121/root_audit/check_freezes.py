"""Read every scientific dependency hash and preserve the final check."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for folder,file,key,root_relative in [
 ('method','METHOD_FREEZE.json','source_hashes',False),
 ('task2d','TASK_FREEZE.json','files',False),
 ('certification','FREEZE.json','source_hashes',True),
 ('certification','ADDENDUM_FREEZE.json','source_hashes',True),
 ('validation','FREEZE.json','hashes',True)]:
    data=json.loads((ROOT/folder/file).read_text());base=ROOT if root_relative else ROOT/folder
    for relative,digest in data[key].items():
        assert sha(base/relative)==digest,(folder,relative)
        checks.append(dict(freeze=folder+'/'+file,path=(base/relative).relative_to(ROOT).as_posix()))
    if folder=='method':
        for relative,digest in data['development_policy_commit_hashes'].items():
            assert sha(base/relative)==digest,relative
            checks.append(dict(freeze=folder+'/'+file,path=(base/relative).relative_to(ROOT).as_posix()))
result=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scientific_freezes=5,
            verified_hash_references=len(checks),mismatches=0,checks=checks)
(ROOT/'root_audit/SOURCE_INTEGRITY.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
