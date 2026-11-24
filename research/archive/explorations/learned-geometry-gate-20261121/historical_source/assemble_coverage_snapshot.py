"""Assemble explicit component runs without overwriting their frozen records."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from runner import digest, utcnow, write_csv, write_json


def identity(row):
    damping = row['damping']
    return (row['task'], row['seed'], row['arm'],
            None if damping in (None, '') else float(damping), float(row['step_size']))


def assemble(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data, runs, seedbank, components, configs = {}, {}, {}, [], []
    overlap = []
    for name in paths:
        root = Path(name)
        config = json.loads((root / 'config.json').read_text())
        configs.append(config)
        prov = json.loads((root / 'provenance.json').read_text())
        if 'finished_utc' not in prov:
            raise ValueError(f'Unfinished input: {root}')
        components.append(dict(path=str(root.resolve()),
                               provenance_sha256=digest(root / 'provenance.json'),
                               original_max_steps=config['max_steps']))
        for state in json.loads((root / 'actual_seeds.json').read_text()):
            key = (state['task'], state['id'])
            if key in seedbank:
                assert seedbank[key] == state
            seedbank[key] = state
        with np.load(root / 'data.npz') as source:
            for key in source.files:
                if key in data:
                    assert np.array_equal(data[key], source[key])
                data[key] = source[key].copy()
        groups = {}
        with np.load(root / 'states.npz') as states:
            for row in json.loads((root / 'snapshots.json').read_text()):
                groups.setdefault(identity(row), []).append((row, states[row['state_key']].copy()))
        with (root / 'summaries.csv').open(newline='') as f:
            summaries = {identity(row): row for row in csv.DictReader(f)}
        with (root / 'trajectory.csv').open(newline='') as f:
            traces = list(csv.DictReader(f))
        for key, rows in groups.items():
            if key in runs:
                before = {float(r['fraction']):(r,t) for r,t in runs[key][0]
                          if r['status'] in ('initial','target')}
                for row, theta in rows:
                    q = float(row['fraction'])
                    if q in before:
                        old, oldtheta = before[q]
                        err = float(np.max(np.abs(oldtheta-theta)))
                        overlap.append(dict(run=str(key), fraction=q, max_parameter_error=err))
                        assert np.array_equal(oldtheta,theta), (key,q,err)
            runs[key] = (rows, summaries[key], [r for r in traces if identity(r)==key])
    rows, states, summaries, traces = [], {}, [], []
    for group, summary, trace in runs.values():
        summaries.append(summary)
        traces.extend(trace)
        for row, theta in group:
            key = f's{len(states):06d}'
            states[key] = theta
            rows.append(dict(row, state_key=key))
    config = configs[0].copy()
    config['assembled_components'] = components
    config['max_steps'] = max(c['max_steps'] for c in configs)
    write_json(output/'config.json', config)
    write_json(output/'actual_seeds.json', list(seedbank.values()))
    write_json(output/'snapshots.json', rows)
    write_csv(output/'snapshots.csv', rows)
    write_csv(output/'summaries.csv', summaries)
    write_csv(output/'trajectory.csv', traces)
    np.savez_compressed(output/'states.npz', **states)
    np.savez_compressed(output/'data.npz', **data)
    write_json(output/'overlap_checks.json', overlap)
    provenance = dict(assembled_utc=utcnow(), finished_utc=utcnow(), components=components,
                      role='Explicit supplemental composite; original component runs remain frozen.',
                      run_count=len(summaries), state_count=len(states),
                      completed_runs=sum(r['status']=='complete' for r in summaries),
                      source_sha256={str(Path(__file__).resolve()):digest(__file__)})
    provenance['output_sha256'] = {p.name:digest(p) for p in output.iterdir()}
    write_json(output/'provenance.json', provenance)
    print(json.dumps(dict(runs=len(summaries), snapshots=len(states), exact_overlaps=len(overlap))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assemble(args.runs, args.output)
