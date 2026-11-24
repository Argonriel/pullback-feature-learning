"""Verify immutable scientific sources and the pre-execution release hashes."""
from pathlib import Path
import argparse, datetime, hashlib, json
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record',action='store_true');args=parser.parse_args()
    freezes=[]
    for branch,name,key in [('calibration','SOURCE_FREEZE.json','source_hashes'),
                            ('augmentation','SOURCE_FREEZE.json','hashes'),('shift','FREEZE.json','hashes')]:
        path=ROOT/branch/name;document=json.loads(path.read_text(encoding='utf8'));rows=[]
        for relative,expected in document[key].items():
            source=(ROOT/branch/relative).resolve();assert source.is_relative_to(ROOT)
            actual=sha(source);assert actual==expected,(branch,relative)
            rows.append(dict(path=source.relative_to(ROOT).as_posix(),sha256=actual))
        freezes.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),verified_sources=rows))
    releases=[]
    for release,key,target in [
        ('calibration/ROOT_APPROVAL.json','source_freeze_sha256','calibration/SOURCE_FREEZE.json'),
        ('ROOT_APPROVAL.json','augmentation_source_freeze_sha256','augmentation/SOURCE_FREEZE.json'),
        ('shift/ROOT_APPROVAL.json','shift_freeze_sha256','shift/FREEZE.json')]:
        record=json.loads((ROOT/release).read_text(encoding='utf8'))
        assert record[key]==sha(ROOT/target)
        releases.append(dict(release=release,target=target,sha256=record[key]))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',
        scientific_freezes=freezes,internal_release_commitments=releases,mismatches=0)
    if args.record:(ROOT/'SOURCE_INTEGRITY.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(status='PASS',freezes=len(freezes),source_entries=sum(len(r['verified_sources']) for r in freezes),release_commitments=3,mismatches=0)))

if __name__=='__main__':main()
