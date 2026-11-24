"""Independent pre-release harness review, using development tables only."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import hashlib, importlib.util, json, sys
from datetime import datetime, timezone
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'validation'))
spec = importlib.util.spec_from_file_location('independent_validation_review', ROOT / 'validation' / 'study.py')
study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top(values, keys):
    order = np.lexsort((keys, -values)); mask = np.zeros(len(values), dtype=bool)
    mask[order[:64]] = True
    return mask


def main():
    choices = json.loads((ROOT / 'method' / 'HISTOGRAM_CHOICE.json').read_text())['choices']
    rows = []
    for cohort, directory in [('rings', 'rings_1'), ('waveform', 'transfer_1'), ('two_invariants', 'task2d_1')]:
        adapter = study.get_adapter(cohort)
        development = ROOT / 'method' / 'development' / directory
        grid = dict(np.load(development / 'grid.npz'))
        strong_name = choices[cohort]['chosen_method']
        legacy_name = 'folded_16x8' if cohort != 'two_invariants' else 'folded_4x4x8'
        tables = dict(grid=grid, folded=dict(np.load(development / (strong_name + '.npz'))),
                      folded_legacy=dict(np.load(development / (legacy_name + '.npz'))))
        assert list(tables['folded']['config']) == choices[cohort]['config']
        assert int(grid['forward_examples']) == 65536
        for name in ('folded', 'folded_legacy'):
            assert int(tables[name]['forward_examples']) == 65536
            assert int(tables[name]['additional_forward_examples']) == 0
        data = np.load(development / 'evaluation.npz')
        x = data['x'][:256]; keys = data['ties'][:256]; theta = grid['theta']
        first = adapter.predict(theta, x); other = adapter.predict(theta, adapter.partner(x))
        average = (first + other) / 2
        exact_j = .5 * (np.logaddexp(0., first) + np.logaddexp(0., other)) - np.logaddexp(0., average)
        for name in study.NAMES:
            actual_calls = []
            original_predict = adapter.predict
            def counted(th, inputs):
                actual_calls.append(len(inputs)); return original_predict(th, inputs)
            adapter.predict = counted
            output, mask, ledger_calls = study.deploy(adapter, theta, x, keys, name, tables)
            adapter.predict = original_predict
            expected_calls = [256] if name == 'raw' else [256, 256] if name in ('full', 'oracle_jensen') else [256, 64]
            assert actual_calls == ledger_calls == expected_calls
            if name == 'raw':
                expected_mask = np.zeros(256, dtype=bool)
            elif name == 'full':
                expected_mask = np.ones(256, dtype=bool)
            elif name == 'confidence':
                expected_mask = top(-abs(first), keys)
            elif name == 'oracle_jensen':
                expected_mask = top(exact_j, keys)
            else:
                key = strong_name if name == 'folded' else legacy_name if name == 'folded_legacy' else name
                expected_mask = data['mask_' + key][:256]
            assert np.array_equal(mask, expected_mask)
            expected = first.copy(); expected[expected_mask] = average[expected_mask]
            error = float(np.max(abs(output - expected)))
            assert error < 1e-12
            rows.append(dict(cohort=cohort, policy=name, actual_calls=actual_calls,
                             selected=int(mask.sum()), mask_changes=0, output_max_error=error))
    streams = {}
    for cohort, base in study.BASES.items():
        for kind, offset in [('training', 1000), ('initialization', 2000), ('calibration', 3000),
                             ('performance', 4000), ('ties', 5000), ('labels', 6000)]:
            streams[cohort + '_' + kind] = set(range(base + offset + 1, base + offset + 17))
    for cohort, base in [('rings', 1121000), ('waveform', 1123000), ('two_invariants', 1151000)]:
        streams[cohort + '_cert_inputs'] = set(range(base + 1, base + 17))
        streams[cohort + '_cert_ties'] = set(range(base + 1000 + 1, base + 1000 + 17))
    intersections = []
    keys = list(streams)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if streams[a] & streams[b]:
                intersections.append([a, b, sorted(streams[a] & streams[b])])
    assert not intersections
    examples_per_batch = sum(256 if name == 'raw' else 512 if name in ('full', 'oracle_jensen') else 320 for name in study.NAMES)
    expected_records = 3 * 16 * 8 * (65536 // 256)
    expected_forwards = 3 * 16 * (65536 // 256) * examples_per_batch
    assert expected_records == 98304 and expected_forwards == 35389440
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
        scope='Pre-release harness review using three development models; no fresh states read.',
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / 'validation' / 'study.py', ROOT / 'validation' / 'adapters.py',
            ROOT / 'validation' / 'PROTOCOL.md', ROOT / 'method' / 'baseline.py')},
        rows=rows, expected_records=expected_records, expected_forwards=expected_forwards,
        forward_examples_per_complete_batch=examples_per_batch,
        stream_count=len(streams), stream_seed_count=sum(map(len, streams.values())),
        pairwise_seed_intersections=intersections,
        primary_bootstrap='Three model-paired expected-loss contrasts; alpha .05/3 each, fixed 20,000 draws. Approximate bootstrap, not distribution-free.',
        source_review='Selectors never receive posterior labels; tables are committed before score. Oracle all-partner charge is explicit.',
        failure_preservation='Requested explicit failure snapshots before final source freeze; final source inspection must confirm this fix.')
    (ROOT / 'root_audit' / 'harness_review.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_hashes')}, indent=2))


if __name__ == '__main__':
    main()
