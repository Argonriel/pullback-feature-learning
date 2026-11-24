"""Copy a verified manifest into a fresh research archive without overwriting."""
from pathlib import Path
import argparse, hashlib, json, shutil
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('destination');args=ap.parse_args()
    destination=Path(args.destination).resolve()
    data=json.loads((ROOT/'archive_manifest.json').read_text())
    if destination.exists():raise FileExistsError('Preserve the existing destination; choose a fresh name.')
    for entry in data['files']:
        source=ROOT/entry['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256'],entry['path']
        target=(destination/entry['path']).resolve()
        assert target.is_relative_to(destination),entry['path']
    destination.mkdir(parents=True)
    for entry in data['files']:
        source=ROOT/entry['path'];target=destination/entry['path']
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(ROOT/'archive_manifest.json',destination/'archive_manifest.json')
    failures=[]
    for entry in data['files']:
        target=destination/entry['path']
        if target.stat().st_size!=entry['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:
            failures.append(entry['path'])
    assert not failures,failures
    print(json.dumps(dict(destination=str(destination),files=data['file_count'],bytes=data['total_bytes'],mismatches=0)))
if __name__=='__main__':main()
