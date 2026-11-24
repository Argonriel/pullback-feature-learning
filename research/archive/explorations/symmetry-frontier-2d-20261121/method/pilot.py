"""Bounded development comparison; no confirmation models are read here."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, sys, time
from datetime import datetime, timezone
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import baseline


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort_adapter(cohort, previous):
    if cohort in ('rings', 'transfer'):
        source = previous / 'validation' / 'adapters.py'
        mod = load('development_adapters_' + cohort, source)
        adapter = mod.get_adapter(cohort)
        # In-memory configuration only: all previous source files stay intact.
        adapter.rank.NR = 128
        return adapter
    task = load('development_task2d', ROOT / 'task2d' / 'task.py')
    from types import SimpleNamespace
    return SimpleNamespace(name='task2d', dimension=4, **{
        name: getattr(task, name) for name in ('coordinates', 'quotient', 'partner',
        'predict', 'sample_inputs', 'posterior', 'network_constants')})


def evaluate(cohort, pair, state_path, adapter, destination):
    start = time.perf_counter()
    state = dict(np.load(state_path))
    theta = state['pb'] if 'pb' in state else state['theta']
    offset = {'rings': 0, 'transfer': 100, 'task2d': 200}[cohort]
    streams = dict(inputs=1191000 + offset + pair, ties=1192000 + offset + pair,
                   histogram=1193000 + offset + pair, cdf=1194000 + offset + pair,
                   action=1195000 + offset + pair)
    folder = destination / f'{cohort}_{pair}'
    folder.mkdir(parents=True, exist_ok=False)
    grid = baseline.fit_grid(adapter, theta)
    assert int(grid['forward_examples']) == 65536
    np.savez_compressed(folder / 'grid.npz', **grid)
    calx = adapter.sample_inputs(streams['histogram'], 32768)
    calh = adapter.predict(theta, calx)
    calp = adapter.predict(theta, adapter.partner(calx))
    ndim = baseline.quotient_matrix(adapter, calx).shape[1]
    histograms = {}
    for config in baseline.histogram_configs(ndim):
        key = 'folded_' + 'x'.join(map(str, config))
        tab = baseline.fit_folded(adapter, theta, calx, config, prepared=(calh, calp))
        histograms[key] = tab
        np.savez_compressed(folder / (key + '.npz'), **tab)
    rank = None
    if cohort != 'task2d':
        calr = adapter.rank.sample_radii(streams['cdf'], adapter.rank.CDF_N)
        a = np.interp(calr, grid['radii'], grid['gain_profile'])
        prepared = dict(theta=theta, radii=grid['radii'], h=grid['h'],
            gain_profile=grid['gain_profile'], cdf_cal_radii=calr, cdf_a=np.sort(a),
            width_scale=np.array(max(a.std(), 1e-8)), cal_seed=np.array(streams['cdf']),
            forward_examples=np.array(65536), coordinate_examples=np.array(len(calr)),
            prepare_seconds=grid['fit_seconds'])
        rank = adapter.rank.calibrate(theta, streams['cdf'], tau=1., prepared=prepared)
        np.savez_compressed(folder / 'rank.npz', **rank)
    (folder / 'policy_commit.json').write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(), cohort=cohort, pair=pair,
        state=str(state_path.resolve()), state_sha256=sha(state_path), streams=streams,
        policies={path.name: sha(path) for path in folder.glob('*.npz')},
        calibration_forward_examples_each_policy=65536,
        unique_shared_calibration_forward_examples=131072,
        extra_rank_coordinate_examples=0 if rank is None else adapter.rank.CDF_N), indent=2))
    # All target logits below are evaluator calls, after the policy commit.
    x = adapter.sample_inputs(streams['inputs'], 32768)
    ties = np.random.default_rng(streams['ties']).random(len(x))
    action = np.random.default_rng(streams['action']).random(len(x))
    first = adapter.predict(theta, x)
    other = adapter.predict(theta, adapter.partner(x))
    true_j = baseline.jensen(first, other)
    posterior = adapter.posterior(x)
    loss = lambda h: np.logaddexp(0., h) - posterior * h
    raw_loss = loss(first)
    masks = dict(symmetric_jensen=baseline.select(adapter, x, grid, ties),
                 grid_nearest=baseline.select_nearest(adapter, x, grid, ties))
    for name, table in histograms.items():
        masks[name] = baseline.select_folded(adapter, x, table, ties)
    if rank is not None:
        for name in ('rank_balanced', 'rank_even'):
            masks[name] = adapter.rank.select(x, first, rank, name, ties, action)
    masks['confidence'] = baseline.allocate(-np.abs(first), ties)
    masks['oracle_jensen'] = baseline.allocate(true_j, ties)
    rows, checks, batch_rows = [], [], []
    flips = np.random.default_rng(streams['action'] + 3000).random(len(x)) < .5
    flipped = x.copy(); flipped[flips] = adapter.partner(flipped[flips])
    for name, mask in masks.items():
        output = first.copy(); output[mask] = .5 * (first[mask] + other[mask])
        selected_j = (mask * true_j).reshape(-1, 256).mean(axis=1)
        observed_gain = (raw_loss - loss(output)).reshape(-1, 256).mean(axis=1)
        oracle_mean = float((masks['oracle_jensen'] * true_j).mean())
        row = dict(cohort=cohort, pair=pair, method=name, risk=float(loss(output).mean()),
            raw_risk=float(raw_loss.mean()), gain=float(observed_gain.mean()),
            selected_j=float(selected_j.mean()), oracle_selected_j=oracle_mean,
            oracle_fraction=float(selected_j.mean() / oracle_mean) if oracle_mean else 1.,
            gain_minus_selected_j=float((observed_gain - selected_j).mean()),
            input_examples=len(x), quota_min=int(mask.reshape(-1, 256).sum(1).min()),
            quota_max=int(mask.reshape(-1, 256).sum(1).max()),
            deploy_forward_examples_per_batch=512 if name == 'oracle_jensen' else 320)
        rows.append(row)
        for batch in range(len(selected_j)):
            batch_rows.append(dict(cohort=cohort, pair=pair, method=name, batch=batch,
                                   selected_j=float(selected_j[batch]), gain=float(observed_gain[batch])))
        if name == 'symmetric_jensen':
            flipped_mask = baseline.select(adapter, flipped, grid, ties)
            score_error = float(np.max(abs(baseline.scores(adapter, x, grid) - baseline.scores(adapter, flipped, grid))))
        elif name == 'grid_nearest':
            flipped_mask = baseline.select_nearest(adapter, flipped, grid, ties)
            score_error = float(np.max(abs(baseline.nearest_scores(adapter, x, grid) - baseline.nearest_scores(adapter, flipped, grid))))
        elif name in histograms:
            flipped_mask = baseline.select_folded(adapter, flipped, histograms[name], ties)
            score_error = float(np.max(abs(baseline.folded_scores(adapter, x, histograms[name]) - baseline.folded_scores(adapter, flipped, histograms[name]))))
        else:
            continue
        assert np.array_equal(mask, flipped_mask)
        assert score_error == 0.
        calls = []
        def predict(a):
            calls.append(len(a)); return adapter.predict(theta, a)
        table = histograms[name] if name in histograms else grid
        deployed, actual_mask = baseline.deploy(adapter, theta, x[:256], table, ties[:256],
            name='folded' if name in histograms else name, predict=predict)
        assert calls == [256, 64]
        assert np.array_equal(actual_mask, mask[:256])
        assert np.allclose(deployed, output[:256], rtol=0, atol=1e-12)
        checks.append(dict(cohort=cohort, pair=pair, method=name,
                          individual_flip_mask_changes=int(np.sum(mask != flipped_mask)),
                          score_max_error=score_error, first_calls=calls[0], second_calls=calls[1],
                          deploy_max_error=float(np.max(abs(deployed - output[:256])))))
    rows.extend([dict(cohort=cohort, pair=pair, method='raw', risk=float(raw_loss.mean()),
        raw_risk=float(raw_loss.mean()), gain=0., selected_j=0., oracle_selected_j=oracle_mean,
        oracle_fraction=0., gain_minus_selected_j=0., input_examples=len(x), quota_min=0,
        quota_max=0, deploy_forward_examples_per_batch=256),
        dict(cohort=cohort, pair=pair, method='full', risk=float(loss(.5 * (first + other)).mean()),
        raw_risk=float(raw_loss.mean()), gain=float((raw_loss - loss(.5 * (first + other))).mean()),
        selected_j=float(true_j.mean()), oracle_selected_j=oracle_mean,
        oracle_fraction=float(true_j.mean() / oracle_mean),
        gain_minus_selected_j=float((raw_loss - loss(.5 * (first + other)) - true_j).mean()),
        input_examples=len(x), quota_min=256, quota_max=256, deploy_forward_examples_per_batch=512)])
    write_csv(folder / 'results.csv', rows)
    write_csv(folder / 'checks.csv', checks)
    write_csv(folder / 'batches.csv', batch_rows)
    np.savez_compressed(folder / 'evaluation.npz', x=x, ties=ties, action=action,
        first=first, partner=other, posterior=posterior,
        **{'mask_' + key: value for key, value in masks.items()})
    (folder / 'complete.json').write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - start,
        evaluation_forward_examples=65536, additional_actual_deploy_check_examples=320 * len(checks),
        exceptions=[], files={p.name: sha(p) for p in folder.iterdir() if p.is_file()}), indent=2))
    print(cohort, pair, 'done', round(time.perf_counter() - start, 2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=['rings', 'transfer', 'task2d'], required=True)
    parser.add_argument('--previous', type=Path, default=ROOT.parent / 'rank-certificate-transfer-20261121')
    parser.add_argument('--state-pattern', default=None)
    parser.add_argument('--pairs', default=None)
    args = parser.parse_args()
    destination = HERE / 'development'
    destination.mkdir(exist_ok=True)
    adapter = cohort_adapter(args.cohort, args.previous)
    pairs = range(1, 17) if args.pairs is None else [int(p) for p in args.pairs.split(',')]
    for pair in pairs:
        state = Path(args.state_pattern.format(pair=pair)) if args.state_pattern else args.previous / 'validation' / 'states' / f'{args.cohort}_{pair}.npz'
        evaluate(args.cohort, pair, state, adapter, destination)


if __name__ == '__main__':
    main()
