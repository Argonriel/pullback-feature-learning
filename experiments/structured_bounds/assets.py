"""Vendor immutable historical inputs; never edits the preceding completed study."""
from pathlib import Path
import hashlib,json,shutil,datetime
HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/'pipeline-calibration-augmentation-shift-20261121'/'shift'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((OLD/'HISTORICAL_INPUTS.json').read_text())
    for directory in ('historical','reference'):
        (HERE/directory).mkdir(exist_ok=False)
        for p in (OLD/directory).iterdir():
            if p.is_file():shutil.copy2(p,HERE/directory/p.name)
    with (HERE/'HISTORICAL_INPUTS.json').open('x') as f:json.dump(manifest,f,indent=2)
    provenance=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(OLD),source_freeze_sha256=sha(OLD/'FREEZE.json'),
        prior_report_sha256=sha(OLD.parent/'REPORT.md'),new_model_or_selector_fitting=False,
        copied={p.relative_to(HERE).as_posix():sha(p) for d in ('historical','reference') for p in (HERE/d).iterdir() if p.is_file()})
    with (HERE/'PROVENANCE.json').open('x') as f:json.dump(provenance,f,indent=2)
if __name__=='__main__':main()
