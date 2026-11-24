"""Frozen fresh PB model generation, guarded by the root's approval record."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(STUDY / 'source'))
sys.path.insert(0, str(STUDY / 'legacy' / 'budgeted-rotation-final'))
from pullback_feature_learning import core
from generator import rings

CONFIG = dict(n_models=16, train_seed_base=921000, init_seed_base=922000,
              steps=500, prefix_steps=100, rate=8.0, damping=0.1,
              n_train=256, n_parameters=80)

def sha(v):
    return hashlib.sha256(v).hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write_json(path, obj):
    with path.open('x', encoding='utf8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def verify_freeze():
    freeze = json.loads((HERE / 'freeze.json').read_text(encoding='utf8'))
    approval = json.loads((HERE / 'ROOT_APPROVAL.json').read_text(encoding='utf8'))
    assert approval['fresh_training_authorized'] is True
    assert approval['validation_freeze_sha256'] == sha((HERE / 'freeze.json').read_bytes())
    for path, digest in freeze['source_hashes'].items():
        assert sha((STUDY / path).read_bytes()) == digest, path
    assert freeze['training'] == CONFIG

def train():
    verify_freeze()
    folder = HERE / 'states'
    folder.mkdir(exist_ok=False)
    records, failures = [], []
    for seed in range(1, CONFIG['n_models'] + 1):
        started = time.perf_counter()
        x, y = rings(CONFIG['train_seed_base'] + seed)
        initial = core.initialize_parameters(CONFIG['init_seed_base'] + seed)
        assert initial.shape == (80,)
        theta = initial.copy()
        losses = []
        try:
            for step in range(1, 501):
                direction = core.direction(theta, x, y, 'pullback', damping=.1)[0]
                theta = theta + 8. * direction
                if not np.isfinite(theta).all():
                    raise ArithmeticError(f'Nonfinite parameters at step {step}')
                if step == 100:
                    prefix = theta.copy()
                if step in (1, 100, 500):
                    h = core.representation(theta, x)[:, 0]
                    losses.append((step, float(np.logaddexp(0., -y*h).mean())))
            path = folder / f'{seed}.npz'
            np.savez_compressed(path, x=x, y=y, initial=initial, prefix=prefix,
                                pb=theta, losses=np.array(losses))
            record = dict(seed=seed, train_seed=921000+seed, init_seed=922000+seed,
                          training_seconds=time.perf_counter()-started,
                          x_sha256=sha(x.tobytes()), y_sha256=sha(y.tobytes()),
                          initial_sha256=sha(initial.tobytes()), prefix_sha256=sha(prefix.tobytes()),
                          pb_sha256=sha(theta.tobytes()), file_sha256=sha(path.read_bytes()))
            records.append(record)
            print(f'Trained fresh PB model {seed}/16', flush=True)
        except Exception as e:
            failure = dict(seed=seed, exception=type(e).__name__, message=str(e),
                           last_step=step, elapsed_seconds=time.perf_counter()-started)
            failures.append(failure)
            np.savez_compressed(folder / f'{seed}_failure.npz', x=x, y=y,
                                initial=initial, last_theta=theta)
            print(f'FAILED model {seed}: {failure}', flush=True)
    with (HERE / 'training.csv').open('x', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, list(records[0]) if records else ['seed'])
        w.writeheader(); w.writerows(records)
    write_json(HERE / 'training_commit.json', dict(timestamp_utc=now(), training=CONFIG,
        failures=failures, n_successful=len(records),
        approval_sha256=sha((HERE / 'ROOT_APPROVAL.json').read_bytes()),
        states={p.name:sha(p.read_bytes()) for p in folder.glob('*.npz')}))
    if failures:
        raise RuntimeError('Training failures retained; do not silently replace or omit seeds.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['train'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        train()
