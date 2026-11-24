"""Build, verify and copy a fresh study; verification is always read-only."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
MANIFEST='archive_manifest.json'
EXCLUSIONS={'distillation/external/ut-zap50k-images.zip'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(root):
    root=Path(root);obj=json.loads((root/MANIFEST).read_text(encoding='utf8'))
    errors=[]
    for r in obj['files']:
        p=root/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:errors.append(r['path'])
    assert not errors,errors
    assert len(obj['files'])==obj['file_count']
    assert sum(r['bytes'] for r in obj['files'])==obj['total_bytes']
    for rel in obj.get('preserved_empty_directories',[]):assert (root/rel).is_dir(),rel
    return dict(files=obj['file_count'],bytes=obj['total_bytes'],mismatches=0)
def build():
    rows=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if '__pycache__' in p.parts or 'pdf_preview' in p.parts or p.suffix=='.pyc':continue
        if rel==MANIFEST or rel in EXCLUSIONS:continue
        rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p)))
    rows.sort(key=lambda r:r['path'])
    empty_dirs=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_dir() and not any(p.iterdir()) and '__pycache__' not in p.parts and 'pdf_preview' not in p.parts]
    obj=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),file_count=len(rows),preserved_empty_directories=sorted(empty_dirs),
        total_bytes=sum(r['bytes'] for r in rows),excluded=['Python caches','temporary rendered previews',*sorted(EXCLUSIONS)],files=rows)
    (ROOT/MANIFEST).write_text(json.dumps(obj,indent=2),encoding='utf8')
    return verify(ROOT)
def copy(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Use a fresh archive directory: '+str(destination))
    verify(ROOT);obj=json.loads((ROOT/MANIFEST).read_text(encoding='utf8'));destination.mkdir(parents=True)
    for r in obj['files']:
        p=destination/r['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/r['path'],p)
    for rel in obj.get('preserved_empty_directories',[]):(destination/rel).mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/MANIFEST,destination/MANIFEST)
    return verify(destination)
if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group()
    group.add_argument('--build',action='store_true');group.add_argument('--verify',type=Path);group.add_argument('--copy')
    args=ap.parse_args();print(json.dumps(build() if args.build else copy(args.copy) if args.copy else verify(args.verify or ROOT)))
