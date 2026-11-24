"""Read-only integrity checks of all manifest-covered predecessor outputs."""
from pathlib import Path
import datetime,hashlib,json,sys

ROOT=Path(__file__).resolve().parent.parent
ARCHIVE=Path('C:/Users/liwei/Documents/Research/pullback-feature-learning/explorations')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
records=[]
for name in ('budget-distillation-structured-bounds-20261121','pipeline-calibration-augmentation-shift-20261121','calibration-symmetry-realdata-20261121'):
    root=ARCHIVE/name;manifest=root/'archive_manifest.json'
    obj=json.loads(manifest.read_text(encoding='utf8'));rows=obj['files'];errors=[]
    for row in rows:
        p=root/row['path']
        if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:errors.append(row['path'])
    assert not errors,errors
    records.append(dict(archive=name,manifest_sha256=sha(manifest),files=len(rows),bytes=sum(r['bytes'] for r in rows),mismatches=errors))
out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archives=records)
if '--record' in sys.argv:
    (ROOT/'root_audit/HISTORICAL_PRESERVATION.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
