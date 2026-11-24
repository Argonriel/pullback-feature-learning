"""Manifest, verify and copy a complete study without replacing an archive.

The full external download contains tens of thousands of unused photographs.
Only the frozen evaluated subset is archived; download metadata/hash and source
taxonomy remain available. No fitted model, evaluated item or result is excluded.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil

ROOT=Path(__file__).resolve().parent
MANIFEST='archive_manifest.json'
EXCLUDED_DOWNLOAD='distillation/external/fashion-product-images-small-v1.zip'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(root):
    data=json.loads((root/MANIFEST).read_text(encoding='utf8'))
    errors=[]
    for row in data['files']:
        p=root/row['path']
        if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            errors.append(row['path'])
    if errors:raise AssertionError(errors)
    return dict(files=len(data['files']),bytes=data['total_bytes'],mismatches=0)

def build():
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if '__pycache__' in p.parts or 'pdf_preview' in p.parts or p.suffix=='.pyc':continue
        if rel in (MANIFEST,EXCLUDED_DOWNLOAD):continue
        files.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p)))
    files.sort(key=lambda r:r['path'])
    data=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        file_count=len(files),total_bytes=sum(r['bytes'] for r in files),
        excluded=['Python caches','temporary rendered previews',EXCLUDED_DOWNLOAD],files=files)
    (ROOT/MANIFEST).write_text(json.dumps(data,indent=2),encoding='utf8')
    return verify(ROOT)

def copy(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Archive destination must be new: '+str(destination))
    verify(ROOT)
    data=json.loads((ROOT/MANIFEST).read_text(encoding='utf8'))
    destination.mkdir(parents=True)
    for row in data['files']:
        target=destination/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/row['path'],target)
    shutil.copy2(ROOT/MANIFEST,destination/MANIFEST)
    return verify(destination)

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group()
    g.add_argument('--build',action='store_true');g.add_argument('--copy');g.add_argument('--verify',type=Path)
    args=ap.parse_args()
    out=build() if args.build else copy(args.copy) if args.copy else verify(args.verify or ROOT)
    print(json.dumps(out))
