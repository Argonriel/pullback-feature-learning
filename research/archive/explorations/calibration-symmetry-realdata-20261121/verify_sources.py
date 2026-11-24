"""Read-only scientific source verification, with an optional audit record."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    specifications = [
        ('training/PILOT_FREEZE.json', 'sources', 'training'),
        ('training/FRESH_FREEZE.json', 'sources', 'training'),
        ('robustness/FREEZE.json', 'hashes', '.'),
        ('realdata/TRAINING_FREEZE.json', 'source_hashes', 'realdata'),
        ('realdata/SOURCE_FREEZE.json', 'source_hashes', 'realdata'),
    ]
    freezes = []
    for relative, key, base in specifications:
        document = json.loads((ROOT / relative).read_text(encoding='utf8'))
        rows = []
        for name, expected in document[key].items():
            path = (ROOT / base / name).resolve()
            assert path.is_relative_to(ROOT), name
            actual = digest(path)
            assert actual == expected, (relative, name, actual, expected)
            rows.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=actual))
        freezes.append(dict(freeze=relative, sha256=digest(ROOT / relative), verified_sources=rows))
    releases = []
    for release, field, target in [
        ('training/ROOT_APPROVAL.json', 'fresh_freeze_sha256', 'training/FRESH_FREEZE.json'),
        ('robustness/ROOT_APPROVAL.json', 'robustness_freeze_sha256', 'robustness/FREEZE.json'),
        ('realdata/TEST_APPROVAL.json', 'policy_commit_sha256', 'realdata/POLICY_COMMIT.json'),
    ]:
        expected = json.loads((ROOT / release).read_text(encoding='utf8'))[field]
        actual = digest(ROOT / target)
        assert actual == expected, (release, target)
        releases.append(dict(release=release, target=target, sha256=actual))
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='PASS',
                  scientific_freezes=freezes, internal_release_commitments=releases,
                  mismatches=0)
    if args.record:
        (ROOT / 'SOURCE_INTEGRITY.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(dict(status='PASS', freezes=len(freezes),
                          source_entries=sum(len(x['verified_sources']) for x in freezes),
                          release_commitments=len(releases), mismatches=0)))


if __name__ == '__main__':
    main()
