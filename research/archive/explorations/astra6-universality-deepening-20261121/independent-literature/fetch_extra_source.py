from fetch_sources import retrieve, ROOT
from pathlib import Path
import json, hashlib
row=retrieve(('thanwerdas_pennec.pdf','https://arxiv.org/pdf/2111.02990'))
out={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':[row]}
(ROOT/'EXTRA_SOURCE_HASHES.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'file':row['file'],'status':row['status']}))
