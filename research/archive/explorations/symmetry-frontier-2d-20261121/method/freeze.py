"""Create a one-time method freeze after deterministic development selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = HERE / 'METHOD_FREEZE.json'
    if target.exists():
        raise RuntimeError('The method freeze already exists; do not overwrite it.')
    choice = json.loads((HERE / 'HISTOGRAM_CHOICE.json').read_text())
    files = ['baseline.py', 'pilot.py', 'summarize.py', 'freeze.py',
             'DEVELOPMENT_PROTOCOL.md', 'THEORY.md', 'LITERATURE.md',
             'HISTOGRAM_CHOICE.json', 'development_summary.csv',
             'development_checks.csv', 'development_per_model.csv']
    record = dict(utc=datetime.now(timezone.utc).isoformat(),
        status='Frozen before any new confirmation models were accessed',
        grid={'one_invariant': [128, 512], 'two_invariants': [16, 16, 256]},
        calibration_forward_examples_each=65536, quota=[64, 256],
        histogram_choice={name: value['config'] for name, value in choice['choices'].items()},
        primary='symmetric_jensen versus frozen best histogram',
        secondary='grid_nearest on the identical calibration grid',
        historical='old128-cell histogram; one-invariant rank family only contextual',
        excluded_settings=[], failures=[],
        source_hashes={name: sha(HERE / name) for name in files},
        development_policy_commit_hashes={str(path.relative_to(HERE)): sha(path)
            for path in sorted((HERE / 'development').glob('*/policy_commit.json'))})
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
