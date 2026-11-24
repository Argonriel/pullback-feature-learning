"""Package completed research outputs without replacing historical files."""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_new(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise FileExistsError(f'Refusing to replace existing artifact: {target}')
        return
    shutil.copy2(source, target)


def package(repo, staging, archive):
    repo, staging, archive = map(lambda p: Path(p).resolve(), (repo,staging,archive))
    if not archive.is_dir() or archive.name != 'learned-geometry-gate-20261121':
        raise ValueError('Expected existing explicitly named study archive.')
    names = ['REPORT.md','DATA_DICTIONARY.md','VALIDATION_CHECKLIST.md',
             'LITERATURE_AUDIT.md','THEORY_AUDIT.md','DECISION_REVIEW.md',
             'curvature_example.json','implementation_checks.json','core_numerics_checks.txt',
             'independent_training_audit.json','independent_refined_training_audit.json']
    for name in names:
        copy_new(staging/name,archive/name)
    copy_new(staging/'assemble_coverage_snapshot.py',archive/'historical_source/assemble_coverage_snapshot.py')
    for source in (staging/'figures_final').iterdir():
        if source.is_file():
            copy_new(source,archive/'figures'/source.name)
    source_files=[]
    for dirname in ('learned_geometry_gate_20261121','learned_geometry_theory_20261121'):
        source_files.extend(p for p in (repo/'experiments'/dirname).iterdir()
                            if p.is_file() and p.suffix in ('.py','.md'))
    source_files.extend(repo/'src/pullback_feature_learning'/n
                        for n in ('__init__.py','core.py','data.py'))
    source_files.extend(repo/'configs'/n for n in (
        'learned_geometry_gate_20261121.json',
        'learned_geometry_gate_coverage_20261121.json',
        'learned_geometry_gate_refinement_20261121.json'))
    source_files.extend(repo/n for n in ('AGENTS.md','requirements-lock.txt','pyproject.toml',
                                       'tests/test_numerics.py'))
    for source in source_files:
        copy_new(source,archive/'source_snapshot'/source.relative_to(repo))
    broken=[]
    for path in archive.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            if not (path.parent/target.split('#')[0]).exists():
                broken.append(dict(file=path.name,target=target))
    if broken:
        raise ValueError(broken)
    files={str(p.relative_to(archive)).replace('\\','/'):
           dict(sha256=sha(p),bytes=p.stat().st_size)
           for p in sorted(archive.rglob('*')) if p.is_file()
           and p.name not in ('manifest.json','ARCHIVE_CHECK.json')}
    manifest=archive/'manifest.json'
    if manifest.exists():
        raise FileExistsError('Manifest already exists; use a new audit rather than overwrite.')
    manifest.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        role='Integrity manifest; not an external preregistration timestamp.',files=files),
        indent=2),encoding='utf-8')
    for name,entry in files.items():
        assert sha(archive/name)==entry['sha256']
    provenance_checks=[]
    for path in archive.glob('*/provenance.json'):
        record=json.loads(path.read_text(encoding='utf-8-sig'))
        for original,expected in record.get('source_sha256',{}).items():
            candidates=[p for p in source_files if str(p)==original]
            candidates += [archive/'historical_source/assemble_coverage_snapshot.py']
            if Path(original).is_file():
                candidates.append(Path(original))
            matched=[str(p) for p in candidates if p.is_file() and sha(p)==expected]
            if not matched:
                raise ValueError(f'Missing preserved source version: {original} {expected}')
            provenance_checks.append(dict(record=str(path.relative_to(archive)),
                                          original=original,sha256=expected,resolved=matched[0]))
    status=dict(status='passed',file_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),
                manifest_sha256=sha(manifest),broken_relative_report_links=broken,
                source_provenance_checks=provenance_checks,
                checked_utc=datetime.now(timezone.utc).isoformat())
    (archive/'ARCHIVE_CHECK.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in status.items() if k!='source_provenance_checks'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True)
    parser.add_argument('--staging',required=True)
    parser.add_argument('--archive',required=True)
    args=parser.parse_args()
    package(args.repo,args.staging,args.archive)
