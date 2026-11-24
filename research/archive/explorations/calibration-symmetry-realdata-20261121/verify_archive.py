"""Build or verify an archive-relative SHA-256 manifest without deleting files."""
from pathlib import Path
import argparse,datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');args=ap.parse_args()
    path=ROOT/'archive_manifest.json'
    if args.build:
        files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'pdf_preview' not in p.parts and p.suffix!='.pyc' and p!=path]
        entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)]
        path.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),file_count=len(entries),total_bytes=sum(r['bytes'] for r in entries),files=entries),indent=2),encoding='utf8')
    data=json.loads(path.read_text());errors=[]
    for row in data['files']:
        p=ROOT/row['path']
        if not p.exists() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:errors.append(row['path'])
    if errors:raise AssertionError(errors)
    print(json.dumps(dict(verified_files=len(data['files']),bytes=data['total_bytes'],mismatches=0)))
if __name__=='__main__':main()
