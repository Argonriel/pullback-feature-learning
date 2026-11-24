"""Prepare or verify this isolated research package; never alters old archives.

This is integrity bookkeeping, not a mathematical or originality certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def historical_paths():
    base = Path('C:/Users/liwei/Documents/Research/pullback-feature-learning/explorations')
    old = base / 'three-candidate-exploration-20261121'
    algebra = base / 'endpoint-classification-audit-20261121'
    paths = [old / p for p in ['REPORT.md', 'DATA_DICTIONARY.md',
        'global_convergence/THEOREM.md', 'global_convergence/REVIEW.md',
        'global_convergence/validation/results.json',
        'robustness/GLOBAL_CONVERGENCE_NOVELTY_AUDIT.md']]
    paths += [algebra / p for p in ['ALGEBRA_REDUCTION.md', 'FINITEDAMPING_AUDIT.md']]
    paths += [Path('C:/Users/liwei/Documents/ChatGPT/277/pullback-feature-learning/AGENTS.md')]
    return paths, [old, algebra]


def verify_prior_manifest(root):
    manifest = read_json(root / 'manifest.json')
    failures = []
    for relative, row in manifest['files'].items():
        path = root / relative
        if not path.is_file() or digest(path) != row['sha256']:
            failures.append(relative)
    return {'archive': str(root), 'checked_files': len(manifest['files']),
            'manifest_sha256': digest(root / 'manifest.json'), 'mismatches': failures,
            'passed': not failures}


def audit(package):
    manifest = read_json(package / 'manifest.json')
    mismatches = [r for r, row in manifest['files'].items()
                  if not (package / r).is_file() or digest(package / r) != row['sha256']]
    expected = set(manifest['files'])
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*')
              if p.is_file() and p.name not in {'manifest.json', 'ARCHIVE_CHECK.json'}
              and '__pycache__' not in p.parts}
    unexpected = sorted(actual - expected)
    links = []
    for md in package.rglob('*.md'):
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', md.read_text(encoding='utf-8-sig')):
            target = target.strip('<>').split('#')[0]
            if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                continue
            if not (md.parent / target).exists():
                links.append({'file': str(md.relative_to(package)), 'target': target})
    associations = [
        ('validation_boundary_v1/results.json', 'source_snapshot/verify_boundary.py', 'source_sha256'),
        ('independent_spectral/results.json', 'source_snapshot/check_quantitative.py', 'source_sha256'),
        ('independent_boundary/exact_checks.json', 'source_snapshot/exact_controls_check.py', 'source_sha256'),
        ('independent_literature/lyapunov_counterexample_results.json', 'source_snapshot/lyapunov_counterexample.cjs', 'sourceSha256'),
    ]
    source_checks = []
    for record, source, key in associations:
        data = read_json(package / record)
        source_checks.append({'record': record, 'source': source,
                              'passed': data[key] == digest(package / source)})
    literature = read_json(package / 'independent_literature/SOURCE_MANIFEST.json')
    lit_mismatches = [row['path'] for row in literature['files']
                      if digest(package / 'independent_literature' / row['path']) != row['sha256']]
    historical = read_json(package / 'historical_input_hashes.json')
    input_mismatches = [r['path'] for r in historical['files'] if digest(Path(r['path'])) != r['sha256']]
    _, prior_roots = historical_paths()
    prior_checks = [verify_prior_manifest(root) for root in prior_roots]
    b = read_json(package / 'validation_boundary_v1/results.json')
    q = read_json(package / 'independent_spectral/results.json')
    e = read_json(package / 'independent_boundary/exact_checks.json')
    l = read_json(package / 'independent_literature/lyapunov_counterexample_results.json')
    numerical = {'boundary_record_passed': b['passed'],
                 'quantitative_record_passed': q['checks_passed'],
                 'rational_record_passed': e['status'] == 'passed',
                 'independent_obstruction_record_passed': l['pass']}
    theorem_matches = digest(package / 'THEOREM.md') == digest(package / 'independent_boundary/THEOREM_BOUNDARY.md')
    checks = {
        'date_local': '2026-11-21', 'timezone': 'America/Los_Angeles',
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'package': str(package), 'manifest_sha256': digest(package / 'manifest.json'),
        'content_files_checked': len(manifest['files']), 'content_mismatches': mismatches,
        'unexpected_content_files': unexpected, 'broken_local_markdown_links': links,
        'source_result_checks': source_checks, 'literature_manifest_mismatches': lit_mismatches,
        'historical_input_mismatches': input_mismatches, 'prior_archives': prior_checks,
        'numerical_record_checks': numerical, 'authoritative_theorem_matches_reviewed_copy': theorem_matches,
        'limits': 'Hash/link/numerical-record consistency only; no proof, originality, or continuous-time certificate.',
    }
    checks['passed'] = (not mismatches and not unexpected and not links and not lit_mismatches
                        and not input_mismatches and theorem_matches
                        and all(r['passed'] for r in source_checks)
                        and all(r['passed'] for r in prior_checks) and all(numerical.values()))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--repo-code', type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    package = args.package.resolve()
    if args.prepare:
        if args.repo_code is None:
            parser.error('--prepare requires --repo-code')
        snapshot = package / 'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for file in sorted(args.repo_code.iterdir()):
            if file.is_file() and file.suffix in {'.py', '.cjs', '.md'}:
                shutil.copy2(file, snapshot / file.name)
        paths, _ = historical_paths()
        baseline = package / 'historical_input_hashes.json'
        if not baseline.exists():
            write_json(baseline, {'scope': 'Baseline before final package preparation; old archive manifests independently checked.',
                                 'files': [{'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths]})
        write_json(package / 'provenance.json', {
            'date_local': '2026-11-21', 'timezone': 'America/Los_Angeles',
            'python_executable': sys.executable, 'python_version': platform.python_version(),
            'repository_source': str(args.repo_code.resolve()), 'research_staging': str(package),
            'planned_archive': 'C:/Users/liwei/Documents/Research/pullback-feature-learning/explorations/astra6-global-convergence-deepening-20261121',
            'historical_input_hashes': 'historical_input_hashes.json',
            'agent_settings': 'All subagents inherited the parent; no model/reasoning/speed override was requested by this chat.',
            'web_provenance_limit': 'Retained extraction records are not full publisher-file byte snapshots.',
            'data_policy': 'No training cohort, reserved sample, historical result mutation, or public communication.'})
        rows = {}
        for p in sorted(package.rglob('*')):
            if p.is_file() and p.name not in {'manifest.json', 'ARCHIVE_CHECK.json'} and '__pycache__' not in p.parts:
                rows[p.relative_to(package).as_posix()] = {'sha256': digest(p), 'bytes': p.stat().st_size}
        write_json(package / 'manifest.json', {'files': rows,
                   'exclusions': ['manifest.json', 'ARCHIVE_CHECK.json', '__pycache__']})
    check = audit(package)
    if args.prepare:
        write_json(package / 'ARCHIVE_CHECK.json', check)
    print(json.dumps(check, indent=2))
    if not check['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
