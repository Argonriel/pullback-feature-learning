"""Archive this completed bounded screen while preserving every old artifact."""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime,timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_new(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):
            raise FileExistsError(target)
        return
    shutil.copy2(source,target)


def package(repo,staging,archive,previous):
    repo,staging,archive,previous=map(lambda p:Path(p).resolve(),(repo,staging,archive,previous))
    if archive.name!='geometry-condition-screen-20261121' or not archive.is_dir():
        raise ValueError('Use the existing explicitly named new study archive.')
    for name in ('REVIEW_NOISE_AGENT.md','REVIEW_CURVATURE_AGENT.md'):
        if not (staging/name).is_file():
            raise ValueError('Cross-review missing: '+name)
    old=json.loads((previous/'manifest.json').read_text())
    for name,row in old['files'].items():
        if sha(previous/name)!=row['sha256']:
            raise ValueError('Previous archive changed: '+name)
    for source in staging.rglob('*'):
        if not source.is_file() or '__pycache__' in source.parts:
            continue
        relative=source.relative_to(staging)
        destination=archive/('historical_source/'+str(relative) if source.suffix=='.py' else str(relative))
        copy_new(source,destination)
    sources=[p for p in (repo/'experiments/geometry_condition_screen_20261121').iterdir()
             if p.is_file() and p.suffix in ('.py','.md')]
    sources += [repo/'src/pullback_feature_learning'/n for n in ('core.py','data.py','__init__.py')]
    sources += [repo/n for n in ('AGENTS.md','pyproject.toml','requirements-lock.txt')]
    for source in sources:
        copy_new(source,archive/'source_snapshot'/source.relative_to(repo))
    broken=[]
    for report in archive.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
            if '://' not in link and not link.startswith('#') and not (report.parent/link.split('#')[0]).exists():
                broken.append(dict(report=report.name,target=link))
    if broken:
        raise ValueError(broken)
    source_lookup={sha(p):str(p.relative_to(archive)) for p in archive.rglob('*.py')}
    provenance=[]
    for path in archive.rglob('*.json'):
        record=json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(record,dict):
            continue
        for field in ('source_sha256','core_sha256','curvature_source_sha256'):
            if field in record and isinstance(record[field],str):
                if record[field] not in source_lookup:
                    raise ValueError(f'Unresolved source hash in {path.name}: {field}')
                provenance.append(dict(record=str(path.relative_to(archive)),field=field,
                                       resolved=source_lookup[record[field]]))
    files={str(p.relative_to(archive)).replace('\\','/'):
           dict(sha256=sha(p),bytes=p.stat().st_size)
           for p in sorted(archive.rglob('*')) if p.is_file()
           and p.name not in ('manifest.json','ARCHIVE_CHECK.json')}
    manifest=archive/'manifest.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                       files=files),indent=2),encoding='utf-8')
    for name,row in files.items():
        assert sha(archive/name)==row['sha256']
    checks=dict(status='passed',file_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),
                manifest_sha256=sha(manifest),source_provenance=provenance,
                previous_archive=str(previous),previous_unchanged_files=len(old['files']),
                previous_manifest_sha256=sha(previous/'manifest.json'),
                broken_report_links=broken,checked_utc=datetime.now(timezone.utc).isoformat())
    (archive/'ARCHIVE_CHECK.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k!='source_provenance'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('repo','staging','archive','previous'):
        p.add_argument('--'+field,required=True)
    a=p.parse_args()
    package(a.repo,a.staging,a.archive,a.previous)
