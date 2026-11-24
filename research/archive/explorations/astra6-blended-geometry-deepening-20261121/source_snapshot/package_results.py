"""Snapshot only this new source tree and verify the completed research package.

All writes are inside --staging. Prior archives are read and checked, never edited.
Refuses to replace an existing source_snapshot or bookkeeping record.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--staging',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True)
    args=ap.parse_args()
    stage=args.staging.resolve(); source=args.source.resolve(); previous=args.previous.resolve()
    assert stage.is_dir() and source.is_dir() and previous.is_dir()
    for name in ('source_snapshot','provenance.json','manifest.json','ARCHIVE_CHECK.json'):
        assert not (stage/name).exists(), f'Refusing overwrite: {name}'
    snapshot=stage/'source_snapshot'
    source_rows=[]
    for p in sorted(source.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts: continue
        rel=p.relative_to(source)
        dest=snapshot/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        assert sha(dest)==sha(p)
        source_rows.append({'path':rel.as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    prev_manifest=previous/'manifest.json'
    assert sha(prev_manifest)=='65482ffc1bc0e590d481e8a43b70d2fe8d9dc9374e87c3b49ee60d2c4ba99214'
    older=json.loads((previous/'provenance.json').read_text(encoding='utf-8'))['previous_archives']
    archives=[previous]+[Path(r['archive']) for r in older]
    old_rows=[]
    for archive in archives:
        mf=archive/'manifest.json'; info=json.loads(mf.read_text(encoding='utf-8'))
        table=info['files']; count=0
        for rel,entry in table.items():
            expected=entry['sha256'] if isinstance(entry,dict) else entry
            assert sha(archive/rel)==expected, f'Old archive changed: {archive/rel}'
            count+=1
        old_rows.append({'archive':str(archive),'manifest_sha256':sha(mf),'unchanged_files':count})
    read_inputs=[previous/'REPORT.md',previous/'DATA_DICTIONARY.md',
                 previous/'nonisotropic/REPORT.md',previous/'nonisotropic/SEARCH_LEDGER.md',
                 previous/'nonisotropic/validation_v1/results.json',
                 previous/'nonisotropic/bures_refined/results.json',source.parents[1]/'AGENTS.md']
    input_rows=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in read_inputs]
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),
                'user_facing_date':'2026-11-21','user_timezone':'America/Los_Angeles',
                'python':platform.python_version(),'numpy':np.__version__,
                'new_source_root':str(source),'staging_root':str(stage),
                'source_files':source_rows,'read_inputs':input_rows,'previous_archives':old_rows,
                'scope':'Only new assigned paths written; old archives read-only; no external messages or training.'}
    (stage/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    # Verify source and dependency hashes embedded in each calculation record.
    content=[p for p in stage.rglob('*') if p.is_file()]
    by_hash={}
    for p in content: by_hash.setdefault(sha(p),[]).append(p.relative_to(stage).as_posix())
    resolved=[]
    def walk(obj,path=''):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key in ('source_sha256','dependency_sha256','input_sha256'):
                    yield path+'/'+key,value
                else: yield from walk(value,path+'/'+key)
        elif isinstance(obj,list):
            for i,value in enumerate(obj): yield from walk(value,path+'/'+str(i))
    for p in stage.rglob('results.json'):
        for field,digest in walk(json.loads(p.read_text(encoding='utf-8'))):
            assert digest in by_hash, f'Unresolved embedded hash: {p}:{field}'
            resolved.append({'record':p.relative_to(stage).as_posix(),'field':field,
                             'sha256':digest,'resolved_files':by_hash[digest]})
    primary=json.loads((stage/'literature/SOURCES.json').read_text(encoding='utf-8'))
    for row in primary['sources']:
        assert sha(stage/'literature'/row['file'])==row['sha256']
    links=[]
    for p in stage.rglob('*.md'):
        for raw in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if re.match(r'^[a-zA-Z]+:',raw) or raw.startswith('#'): continue
            target=raw.split('#')[0]
            assert (p.parent/target).exists(), f'Broken local link: {p}:{raw}'
            links.append({'file':p.relative_to(stage).as_posix(),'target':raw})
    root=json.loads((stage/'validation_v1/results.json').read_text(encoding='utf-8'))
    refine=json.loads((stage/'endpoint_difference_refinement/results.json').read_text(encoding='utf-8'))
    assert root['checks']['sensitivity_fd_converges'] is False
    assert all(root['checks'][key] for key in ('normalization','area_fd_converges',
               'curvature_fd_converges','rank_controls','scaled_jacobian_converges','endpoint_residuals'))
    assert all(refine['checks'].values())
    assert all(row['scaled_singular_values'][-1]>1e-5 for row in root['endpoint_rows'])
    assert (stage/'exceptional/STATIC_THEOREM_REVIEW.md').exists()
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'files':{}}
    for p in sorted(stage.rglob('*')):
        if p.is_file():
            manifest['files'][p.relative_to(stage).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
    (stage/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for rel,row in manifest['files'].items(): assert sha(stage/rel)==row['sha256']
    checks={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
            'content_files':len(manifest['files']),
            'manifest_sha256':sha(stage/'manifest.json'),'source_snapshot_files':len(source_rows),
            'embedded_hash_resolution':resolved,'primary_snapshots_checked':len(primary['sources']),
            'local_markdown_links_checked':len(links),'previous_archives':old_rows,
            'original_fd_decay_flag_preserved':root['checks']['sensitivity_fd_converges'],
            'targeted_refinement_checks':refine['checks'],
            'bookkeeping_exclusion':'manifest.json and ARCHIVE_CHECK.json do not hash themselves',
            'limits':'Integrity checks are not a formal theorem certificate or an originality certificate.'}
    (stage/'ARCHIVE_CHECK.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps({'status':checks['status'],'content_files':checks['content_files'],
                      'source_files':len(source_rows),'source_hashes_resolved':len(resolved),
                      'primary_snapshots':len(primary['sources']),'local_links':len(links),
                      'old_archives':len(old_rows),'old_files_unchanged':sum(r['unchanged_files'] for r in old_rows)},indent=2))


if __name__=='__main__': main()
