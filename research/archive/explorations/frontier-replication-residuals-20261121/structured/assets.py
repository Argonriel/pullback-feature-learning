"""Copy frozen historical assets into this new standalone study only."""
from pathlib import Path
import datetime,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/'budget-distillation-structured-bounds-20261121'/'structured'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=json.loads((OLD/'FREEZE.json').read_text())
    for name,value in old['hashes'].items():assert sha(OLD/name)==value
    for directory in ['historical','reference']:
        (HERE/directory).mkdir(exist_ok=False)
        for p in (OLD/directory).iterdir():
            if p.is_file():shutil.copy2(p,HERE/directory/p.name)
    shutil.copy2(OLD/'HISTORICAL_INPUTS.json',HERE/'HISTORICAL_INPUTS.json')
    provenance=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_branch=str(OLD),
        original_freeze_sha256=sha(OLD/'FREEZE.json'),new_model_or_selector_fit=False,
        copied={p.relative_to(HERE).as_posix():sha(p) for d in ['historical','reference'] for p in (HERE/d).iterdir() if p.is_file()})
    with (HERE/'PROVENANCE.json').open('x') as f:json.dump(provenance,f,indent=2)
if __name__=='__main__':main()
