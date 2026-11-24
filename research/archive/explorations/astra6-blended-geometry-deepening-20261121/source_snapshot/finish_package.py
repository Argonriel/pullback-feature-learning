"""Resume packaging after preserving the v1 Markdown-parser false positive.

The original package_results.py and its source snapshot stay unchanged.
This corrects link parsing by ignoring code blocks and inline code.
"""
import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from package_results import sha


def prose_only(markdown):
    lines=[]; fenced=False
    for line in markdown.splitlines():
        if line.lstrip().startswith('```'):
            fenced=not fenced; continue
        if fenced or line.startswith('    ') or line.startswith('\t'): continue
        lines.append(re.sub(r'`[^`]*`','',line))
    return '\n'.join(lines)


def walk(obj,path=''):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if key in ('source_sha256','dependency_sha256','input_sha256'): yield path+'/'+key,value
            else: yield from walk(value,path+'/'+key)
    elif isinstance(obj,list):
        for i,value in enumerate(obj): yield from walk(value,path+'/'+str(i))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--staging',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True); args=ap.parse_args()
    stage=args.staging.resolve(); source=args.source.resolve()
    assert not (stage/'manifest.json').exists() and not (stage/'ARCHIVE_CHECK.json').exists()
    # Preserve the first bookkeeping result as well as the failed parser source.
    failed=stage/'packaging_attempt_v1'; failed.mkdir(exist_ok=False)
    shutil.copy2(stage/'provenance.json',failed/'provenance.json')
    error={'status':'failed_validation_false_positive',
           'source_sha256':sha(source/'package_results.py'),
           'error':'Broken local link THEOREM.md:X',
           'cause':'The simple link regex interpreted [H_i,H_j](X) inside an indented mathematical code block as a Markdown link.',
           'correction':'Ignore indented/fenced code and inline code before checking prose links.',
           'math_or_numerical_outputs_changed':False}
    (failed/'failure.json').write_text(json.dumps(error,indent=2),encoding='utf-8')
    source_rows=[]
    for p in sorted(source.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts: continue
        rel=p.relative_to(source); dest=stage/'source_snapshot'/rel
        if dest.exists(): assert sha(dest)==sha(p)
        else:
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
        source_rows.append({'path':rel.as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    prov=json.loads((stage/'provenance.json').read_text(encoding='utf-8'))
    prov['source_files']=source_rows
    prov['packaging_followup']='See packaging_attempt_v1/failure.json; first provenance preserved.'
    prov['finalized_utc']=datetime.now(timezone.utc).isoformat()
    (stage/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
    by_hash={}
    for p in stage.rglob('*'):
        if p.is_file(): by_hash.setdefault(sha(p),[]).append(p.relative_to(stage).as_posix())
    resolved=[]
    for p in list(stage.rglob('results.json'))+[failed/'failure.json']:
        for field,digest in walk(json.loads(p.read_text(encoding='utf-8'))):
            assert digest in by_hash, f'Unresolved source hash: {p}:{field}'
            resolved.append({'record':p.relative_to(stage).as_posix(),'field':field,
                             'sha256':digest,'resolved_files':by_hash[digest]})
    primary=json.loads((stage/'literature/SOURCES.json').read_text(encoding='utf-8'))
    for row in primary['sources']: assert sha(stage/'literature'/row['file'])==row['sha256']
    links=[]
    for p in stage.rglob('*.md'):
        for raw in re.findall(r'\]\(([^)]+)\)',prose_only(p.read_text(encoding='utf-8'))):
            if re.match(r'^[a-zA-Z]+:',raw) or raw.startswith('#'): continue
            assert (p.parent/raw.split('#')[0]).exists(), f'Broken prose link: {p}:{raw}'
            links.append({'file':p.relative_to(stage).as_posix(),'target':raw})
    # Recheck the old archives at finalization, not only at the first attempt.
    for old in prov['previous_archives']:
        base=Path(old['archive']); mf=base/'manifest.json'
        assert sha(mf)==old['manifest_sha256']
        entries=json.loads(mf.read_text(encoding='utf-8'))['files']
        for rel,row in entries.items(): assert sha(base/rel)==(row['sha256'] if isinstance(row,dict) else row)
    root=json.loads((stage/'validation_v1/results.json').read_text(encoding='utf-8'))
    refine=json.loads((stage/'endpoint_difference_refinement/results.json').read_text(encoding='utf-8'))
    assert root['checks']['sensitivity_fd_converges'] is False
    assert all(root['checks'][key] for key in ('normalization','area_fd_converges',
               'curvature_fd_converges','rank_controls','scaled_jacobian_converges','endpoint_residuals'))
    assert all(refine['checks'].values())
    assert all(row['scaled_singular_values'][-1]>1e-5 for row in root['endpoint_rows'])
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'files':{}}
    for p in sorted(stage.rglob('*')):
        if p.is_file(): manifest['files'][p.relative_to(stage).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
    (stage/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for rel,row in manifest['files'].items(): assert sha(stage/rel)==row['sha256']
    check={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
           'content_files':len(manifest['files']),'manifest_sha256':sha(stage/'manifest.json'),
           'source_snapshot_files':len(source_rows),'embedded_hash_resolution':resolved,
           'primary_snapshots_checked':len(primary['sources']),'local_markdown_links_checked':len(links),
           'previous_archives':prov['previous_archives'],
           'original_fd_decay_flag_preserved':False,'targeted_refinement_checks':refine['checks'],
           'packaging_false_positive_preserved':True,
           'bookkeeping_exclusion':'manifest.json and ARCHIVE_CHECK.json do not hash themselves',
           'limits':'Integrity checks are not a formal theorem certificate or an originality certificate.'}
    (stage/'ARCHIVE_CHECK.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','content_files':len(manifest['files']),
                      'source_files':len(source_rows),'resolved_hashes':len(resolved),
                      'primary_snapshots':len(primary['sources']),'local_links':len(links),
                      'old_archives':len(prov['previous_archives']),
                      'old_files_unchanged':sum(r['unchanged_files'] for r in prov['previous_archives'])},indent=2))


if __name__=='__main__': main()
