"""Independent reconstruction of all fresh outcomes, after immutable scoring.

No selector or evaluator functions from the primary harness are imported.
Only the frozen data/model adapter is used to regenerate streams and replay
six 100-step training trajectories. Metrics and masks are reconstructed here.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
from pathlib import Path
from datetime import datetime, timezone
import csv, gzip, hashlib, importlib.util, itertools, json, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
COHORTS = ('rings', 'waveform', 'two_invariants')
BASES = dict(rings=1160000, waveform=1110000, two_invariants=1140000)
NAMES = ('raw', 'full', 'confidence', 'folded_legacy', 'folded', 'symmetric_jensen', 'grid_nearest', 'oracle_jensen')
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def record(path):
    return json.loads(Path(path).read_text())


def timestamp(path):
    r = record(path)
    return datetime.fromisoformat(r.get('timestamp_utc', r.get('utc')))


def quotient(cohort, x):
    if cohort == 'rings':
        return np.linalg.norm(x, axis=1)[:, None]
    if cohort == 'waveform':
        return np.linalg.norm(x[:, 2:], axis=1)[:, None]
    return np.column_stack([np.linalg.norm(x[:, :2], axis=1), np.linalg.norm(x[:, 2:], axis=1)])


def angle(x):
    pair = x[:, :2].copy()
    flip = (pair[:, 1] < 0) | ((pair[:, 1] == 0) & (pair[:, 0] < 0))
    pair[flip] *= -1
    pair[pair == 0] = 0
    return np.mod(np.arctan2(pair[:, 1], pair[:, 0]), np.pi)


def grid_score(cohort, x, table, nearest=False):
    z = quotient(cohort, x); field = table['folded_jensen']; phi = angle(x)
    if nearest:
        indices = [np.searchsorted((table[f'axis_{j}'][1:] + table[f'axis_{j}'][:-1]) / 2,
                                  z[:, j], side='right') for j in range(z.shape[1])]
        indices.append(np.floor(phi * field.shape[-1] / np.pi).astype(int) % field.shape[-1])
        return field[tuple(indices)]
    # Uniform-coordinate interpolation is independent of the primary searchsorted
    # implementation; corner summation is retained explicitly.
    lows, fractions = [], []
    for j in range(z.shape[1]):
        axis = table[f'axis_{j}']; position = np.clip((z[:, j] - axis[0]) / (axis[-1] - axis[0]) * (len(axis) - 1), 0, len(axis) - 1)
        low = np.minimum(np.floor(position).astype(int), len(axis) - 2)
        lows.append(low); fractions.append(position - low)
    pos = phi * field.shape[-1] / np.pi - .5
    low = np.floor(pos).astype(int); lows.append(low); fractions.append(pos - low)
    answer = np.zeros(len(x))
    for corner in itertools.product((0, 1), repeat=len(lows)):
        weight = np.ones(len(x)); indices = []
        for j, bit in enumerate(corner):
            weight *= fractions[j] if bit else 1 - fractions[j]
            index = lows[j] + bit
            indices.append(index % field.shape[-1] if j == len(lows) - 1 else index)
        answer += weight * field[tuple(indices)]
    return answer


def histogram_score(cohort, x, table):
    z = quotient(cohort, x); cfg = table['config']; index = np.zeros(len(x), dtype=int)
    for j in range(z.shape[1]):
        index = index * int(cfg[j]) + np.searchsorted(table[f'edges_{j}'], z[:, j])
    abin = np.minimum((angle(x) * int(cfg[-1]) / np.pi).astype(int), int(cfg[-1]) - 1)
    return table['scores'][index * int(cfg[-1]) + abin]


def allocate(score, ties):
    order = np.lexsort((ties.reshape(-1, 256), -score.reshape(-1, 256)), axis=1)
    selected = np.zeros_like(order, dtype=bool)
    selected[np.arange(len(order))[:, None], order[:, :64]] = True
    return selected.ravel()


def posterior(cohort, x):
    z = quotient(cohort, x)
    if cohort == 'rings':
        r = z[:, 0]
        logit = (.5 ** 2 - 1.2 ** 2) / (2 * .15 ** 2)
        logit += np.log(np.i0(1.2 * r / .15 ** 2)) - np.log(np.i0(.5 * r / .15 ** 2))
        return np.exp(-np.logaddexp(0., -logit))
    if cohort == 'waveform':
        u = np.clip(z[:, 0] - .4, 0, 1)
        return u ** 3 / (u ** 3 + (1 - u) ** 3)
    return .5 + .4 * np.prod(np.tanh(8 * (z - .9)), axis=1)


def main():
    start = time.perf_counter()
    val = ROOT / 'validation'
    assert (val / 'score_complete.json').exists(), 'No audit before scoring completes.'
    assert (val / 'ROOT_APPROVAL.json').exists()
    frozen = record(val / 'FREEZE.json')
    for relative, digest in frozen['hashes'].items():
        assert sha(ROOT / relative) == digest, relative
    order_paths = [ROOT / 'method' / 'METHOD_FREEZE.json', val / 'FREEZE.json',
                   val / 'ROOT_APPROVAL.json', val / 'training_start.json', val / 'training_commit.json',
                   val / 'calibration_start.json', val / 'policy_commit.json', val / 'score_start.json', val / 'score_complete.json']
    times = [timestamp(p) for p in order_paths]
    assert all(a <= b for a, b in zip(times, times[1:]))
    manifests = [(val / 'training_commit.json', 'manifest'), (val / 'policy_commit.json', 'manifest'),
                 (val / 'score_complete.json', 'manifest')]
    checked_hashes = 0
    for path, key in manifests:
        for relative, digest in record(path)[key].items():
            assert sha(val / relative) == digest, relative
            checked_hashes += 1
    spec = importlib.util.spec_from_file_location('fresh_audit_adapters', val / 'adapters.py')
    adapters = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapters)
    metrics = {(r['cohort'], int(r['pair']), r['policy']): r for r in read_csv(val / 'per_model.csv')}
    samples = {(r['cohort'], int(r['pair'])): r for r in read_csv(val / 'samples.csv')}
    choices = record(ROOT / 'method' / 'HISTOGRAM_CHOICE.json')['choices']
    metric_rows, model_rows, training_replays = [], [], []
    training_hashes, input_hashes = [], []
    reconstructed = {}
    max_metric_error = 0.; max_output_error = 0.; mask_changes = 0
    for cohort in COHORTS:
      adapter = adapters.get_adapter(cohort)
      for pair in range(1, 17):
        state = dict(np.load(val / 'states' / f'{cohort}_{pair}.npz'))
        prediction = dict(np.load(val / 'predictions' / f'{cohort}_{pair}.npz'))
        grid = dict(np.load(val / 'policies' / f'{cohort}_{pair}_grid.npz'))
        hist = {name: dict(np.load(val / 'policies' / f'{cohort}_{pair}_{name}.npz')) for name in ('folded', 'folded_legacy')}
        assert list(prediction['names']) == list(NAMES)
        assert list(hist['folded']['config']) == choices[cohort]['config']
        assert int(grid['forward_examples']) == 65536
        assert sum(grid['forward_call_sizes']) == 65536
        for tab in hist.values():
            assert int(tab['forward_examples']) == 65536
        base = BASES[cohort]; x = prediction['x']; ties = prediction['ties']; out = prediction['outputs']
        masks = np.unpackbits(prediction['masks'], axis=1)[:, :len(x)].astype(bool)
        assert len(x) == 65536
        sample = samples[(cohort, pair)]
        regenerated = adapter.sample_inputs(base + 4000 + pair, len(x))
        assert np.array_equal(x, regenerated)
        assert ah(x) == sample['input_sha256']
        assert np.array_equal(ties, np.random.default_rng(base + 5000 + pair).random(len(x)))
        assert ah(ties) == sample['tie_sha256']
        p = posterior(cohort, x)
        assert np.max(abs(p - prediction['p'])) < 1e-12
        y = np.where(np.random.default_rng(base + 6000 + pair).random(len(x)) < p, 1., -1.)
        assert np.array_equal(y, prediction['y']) and ah(y) == sample['label_sha256']
        train_x, train_y = adapter.sample_train(base + 1000 + pair)
        assert np.array_equal(train_x, state['x']) and np.array_equal(train_y, state['y'])
        assert np.array_equal(adapter.initialize(base + 2000 + pair), state['initial'])
        assert sum(train_y == 1) == sum(train_y == -1) == 128
        training_hashes.append(ah(train_x)); input_hashes.append(ah(x))
        first = out[0]; second = 2 * out[1] - first; average = out[1]
        gaps = np.maximum(0., .5 * (np.logaddexp(0., first) + np.logaddexp(0., second)) - np.logaddexp(0., average))
        independent_masks = dict(raw=np.zeros(len(x), bool), full=np.ones(len(x), bool),
            confidence=allocate(-abs(first), ties), oracle_jensen=allocate(gaps, ties),
            symmetric_jensen=allocate(grid_score(cohort, x, grid), ties),
            grid_nearest=allocate(grid_score(cohort, x, grid, nearest=True), ties))
        for name, tab in hist.items():
            independent_masks[name] = allocate(histogram_score(cohort, x, tab), ties)
        oracle_batch = (independent_masks['oracle_jensen'] * gaps).reshape(-1, 256).mean(1)
        z = quotient(cohort, x)[:, 0]
        position = np.clip((z - .25) / 1.25, 0, 1) if cohort == 'rings' else np.clip(z - .4, 0, 1)
        alt = dict(half=np.full(len(x), .5), decreasing=1-position,
                   angular=.5+.4*np.cos(2*np.arctan2(x[:, 1], x[:, 0])))
        for index, name in enumerate(NAMES):
            mask = masks[index]; h = out[index]
            changes = int(np.sum(mask != independent_masks[name])); mask_changes += changes
            assert changes == 0, (cohort, pair, name, changes)
            quota = 0 if name == 'raw' else 256 if name == 'full' else 64
            assert np.all(mask.reshape(-1, 256).sum(1) == quota)
            expected_output = first.copy(); expected_output[mask] = average[mask]
            output_error = float(np.max(abs(h - expected_output)))
            assert output_error < 1e-10
            max_output_error = max(max_output_error, output_error)
            selected = (mask * gaps).reshape(-1, 256).mean(1)
            if name not in ('raw', 'full'):
                assert np.all(selected <= oracle_batch + 1e-12)
            values = dict(expected_loss=float(np.mean(np.logaddexp(0., h) - p*h)),
                expected_error=float(np.mean(np.where(h >= 0, 1-p, p))),
                sampled_loss=float(np.logaddexp(0., -y*h).mean()),
                sampled_error=float(np.mean(np.where(h >= 0, 1., -1.) != y)),
                selected_jensen=float(selected.mean()), oracle_fraction=float(selected.mean()/oracle_batch.mean()))
            for world, probabilities in alt.items():
                values[world + '_loss'] = float(np.mean(np.logaddexp(0., h) - probabilities*h))
            declared = metrics[(cohort, pair, name)]
            error = max(abs(value - float(declared[key])) for key, value in values.items())
            max_metric_error = max(max_metric_error, error)
            assert error < 1e-11, (cohort, pair, name, error)
            calls = 256 if name == 'raw' else 512 if name in ('full', 'oracle_jensen') else 320
            assert int(declared['forward_examples']) == 256*calls
            metric_rows.append(dict(cohort=cohort, pair=pair, policy=name, max_metric_error=error,
                mask_changes=changes, output_max_error=output_error, forward_examples=256*calls))
            reconstructed[(cohort, pair, name)] = values
        # Model prediction replay on one complete batch, independent of saved outputs.
        probe_error = float(np.max(abs(adapter.predict(state['pb'], x[:256]) - first[:256])))
        assert probe_error < 1e-12
        model_rows.append(dict(cohort=cohort, pair=pair, input_sha256=ah(x),
            training_input_sha256=ah(train_x), initial_sha256=ah(state['initial']),
            posterior_max_error=float(np.max(abs(p-prediction['p']))), first_batch_logit_error=probe_error))
        if pair in (1, 16):
            theta = state['initial'].copy(); cfg = adapter.training
            for step in range(100):
                theta += cfg['rate'] * adapter.direction(theta, train_x, train_y, method=cfg['method'], damping=cfg['damping'])
            replay_error = float(np.max(abs(theta-state['pb100'])))
            assert replay_error < 1e-12
            training_replays.append(dict(cohort=cohort, pair=pair, steps=100, max_error=replay_error))
        print('Audited', cohort, pair, flush=True)
    assert len(set(training_hashes)) == len(training_hashes) == 48
    assert len(set(input_hashes)) == len(input_hashes) == 48
    ledger_count = ledger_sum = 0; ledger_by_policy = {name: 0 for name in NAMES}
    with gzip.open(val / 'batch_ledger.csv.gz', 'rt', newline='', encoding='utf-8') as stream:
        seen = set()
        for row in csv.DictReader(stream):
            key = (row['cohort'], int(row['pair']), row['policy'], int(row['batch']))
            assert key not in seen; seen.add(key)
            name = row['policy']; calls = [int(v) for v in row['calls'].split('|')]
            expected = [256] if name == 'raw' else [256, 256] if name in ('full', 'oracle_jensen') else [256, 64]
            assert calls == expected
            assert int(row['selected']) == (0 if name == 'raw' else 256 if name == 'full' else 64)
            assert int(row['inputs']) == 256 and int(row['forward_examples']) == sum(calls)
            ledger_count += 1; ledger_sum += sum(calls); ledger_by_policy[name] += sum(calls)
    assert ledger_count == 98304 and ledger_sum == 35389440
    calibration = read_csv(val / 'calibration.csv')
    assert len(calibration) == 48
    assert all(int(r['grid_forwards']) == int(r['folded_forwards']) == 65536 and int(r['shared_legacy_extra_forwards']) == 0 for r in calibration)
    primary = []
    contrasts = read_csv(val / 'contrasts.csv') if (val / 'contrasts.csv').exists() else []
    for ci, cohort in enumerate(COHORTS):
        delta = np.array([reconstructed[(cohort, i, 'symmetric_jensen')]['expected_loss'] -
                          reconstructed[(cohort, i, 'folded')]['expected_loss'] for i in range(1, 17)])
        boot = np.random.default_rng(1171001 + ci).choice(delta, size=(20000, 16)).mean(1)
        lo, hi = np.quantile(boot, [.05/6, 1-.05/6])
        primary.append(dict(cohort=cohort, mean=float(delta.mean()), lower=float(lo), upper=float(hi), wins=int(np.sum(delta < 0))))
        if contrasts:
            row = next(r for r in contrasts if r['cohort']==cohort and r['comparison']=='symmetric_jensen_minus_folded' and r['metric']=='expected_loss')
            assert max(abs(float(row[k])-primary[-1][k]) for k in ('mean','lower','upper')) < 1e-11
    write_csv(HERE / 'fresh_metric_checks.csv', metric_rows)
    write_csv(HERE / 'fresh_seed_checks.csv', model_rows)
    write_csv(HERE / 'fresh_training_replays.csv', training_replays)
    result = dict(utc=datetime.now(timezone.utc).isoformat(), status='PASS',
        audited_source_sha256=sha(Path(__file__)), models=48, metric_rows=len(metric_rows),
        max_metric_error=max_metric_error, max_output_error=max_output_error,
        total_mask_changes=mask_changes, manifest_files_checked=checked_hashes,
        chronology={str(p.relative_to(ROOT)): t.isoformat() for p,t in zip(order_paths,times)},
        training_inputs_unique=48, performance_inputs_unique=48,
        seed_stream_replays=48, training_100_step_replays=training_replays,
        ledger_records=ledger_count, actual_forward_examples=ledger_sum, forwards_by_policy=ledger_by_policy,
        calibration_forward_examples=48*2*65536, primary_bootstrap_reconstruction=primary,
        seconds=time.perf_counter()-start,
        limits=['Ordinary floating-point audit, not directed rounding.',
                'Bootstrap intervals remain finite-replication approximations.',
                'Training direction implementation is reused for deterministic replay; risk and mask arithmetic are independent.'])
    (HERE / 'fresh_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
