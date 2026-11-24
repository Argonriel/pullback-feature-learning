"""CPU-only quality replay of fixed GPU-study weights; no training or timing."""
from pathlib import Path
import csv, datetime, hashlib, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
GPU = STUDY / 'gpu'
sys.dont_write_bytecode = True
sys.path.insert(0, str(GPU))
import benchmark as base
import optimized_serving as serving

POLICIES = ['full_fused', 'folded_fused', 'confidence_radial']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open('x', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def json_write(path, obj):
    with path.open('x', encoding='utf8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def main():
    base.verify()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    before = {str(p.relative_to(STUDY)): sha(p) for p in
              [Path(__file__), GPU/'benchmark.py', GPU/'optimized_serving.py',
               GPU/'training_commit.json', GPU/'policy_commit.json',
               GPU/'optimized/freeze_v2.json']}
    for folder, commit in [('models', 'training_commit.json'),
                           ('policies', 'policy_commit.json')]:
        for name, digest in json.loads((GPU/commit).read_text())['files'].items():
            assert sha(GPU/folder/name) == digest
    json_write(HERE/'cpu_quality_replay_start.json', dict(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_and_commit_hashes=before, policies=POLICIES, n_models=9,
        n_inputs_per_model=16384, batch=256, cpu_threads=1,
        scope='Fixed-weight CPU quality replay; no training, calibration, tuning or timing.'))
    folder = HERE/'cpu_quality_predictions'
    folder.mkdir(exist_ok=False)
    rows, calls_rows = [], []
    for width in [64, 256, 1024]:
        for seed in [1, 2, 3]:
            data = dict(np.load(GPU/f'predictions/w{width}_s{seed}.npz'))
            n = len(data['x'])
            assert n == 16384
            net = base.load_model(torch, width, seed, 'cpu')
            table = serving.prepare_tables(torch, base.tables(torch, width, seed, 'cpu'))
            x = torch.as_tensor(data['x'], dtype=torch.float32)
            ties = torch.as_tensor(data['ties'], dtype=torch.float64)
            cuda_masks = np.unpackbits(data['masks'], axis=1, count=n).astype(bool)
            outputs, masks = [], []
            with torch.inference_mode():
                for name in POLICIES:
                    output = np.empty(n, dtype=np.float32)
                    mask = np.zeros(n, dtype=bool)
                    for batch in range(n//256):
                        sl = slice(256*batch, 256*(batch+1))
                        calls = []
                        out, indices = serving.deploy(torch, net, x[sl],
                            None if name == 'full_fused' else ties[sl], table, name, calls)
                        assert calls == base.expected_calls(name)
                        if indices is None:
                            mask[sl] = True
                        else:
                            ix = indices.numpy()
                            assert len(np.unique(ix)) == 64 and np.all((ix >= 0) & (ix < 256))
                            mask[256*batch+ix] = True
                        wanted = 256 if name == 'full_fused' else 64
                        assert mask[sl].sum() == wanted
                        output[sl] = out.numpy()
                        calls_rows.append(dict(width=width, seed=seed, policy=name,
                            batch=batch, calls=json.dumps(calls), selected=int(mask[sl].sum()),
                            forward_examples=sum(calls)))
                    assert np.isfinite(output).all()
                    j = data['names'].tolist().index(name)
                    h, gh = output.astype(float), data['outputs'][j].astype(float)
                    p, y = data['posterior'], data['labels']
                    risk = float(np.mean(np.logaddexp(0, h)-p*h))
                    gpu_risk = float(np.mean(np.logaddexp(0, gh)-p*gh))
                    row = dict(width=width, seed=seed, policy=name,
                        n_inputs=n, cpu_expected_risk=risk, cuda_expected_risk=gpu_risk,
                        expected_risk_difference=risk-gpu_risk,
                        cpu_sampled_label_risk=float(np.mean(np.logaddexp(0, h)-y*h)),
                        cuda_sampled_label_risk=float(np.mean(np.logaddexp(0, gh)-y*gh)),
                        cpu_expected_error=float(np.mean(np.where(h >= 0, 1-p, p))),
                        cuda_expected_error=float(np.mean(np.where(gh >= 0, 1-p, p))),
                        max_logit_error=float(np.max(abs(h-gh))),
                        mean_absolute_logit_error=float(np.mean(abs(h-gh))),
                        mask_differences=int(np.sum(mask != cuda_masks[j])),
                        classification_differences=int(np.sum((h >= 0) != (gh >= 0))),
                        forward_examples=int(n+mask.sum()),
                        cpu_output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                        cpu_mask_sha256=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest())
                    rows.append(row)
                    outputs.append(output)
                    masks.append(np.packbits(mask))
            np.savez_compressed(folder/f'w{width}_s{seed}.npz', names=np.array(POLICIES),
                outputs=np.array(outputs), masks=np.array(masks))
            print(f'CPU quality replay width {width}, seed {seed}', flush=True)
    for rel, digest in before.items():
        assert sha(STUDY/rel) == digest
    csv_write(HERE/'cpu_quality_checks.csv', rows)
    csv_write(HERE/'cpu_quality_calls.csv', calls_rows)
    summaries = []
    for width in [64, 256, 1024]:
        for name in POLICIES:
            chosen = [r for r in rows if r['width'] == width and r['policy'] == name]
            summaries.append(dict(width=width, policy=name, n_models=3,
                mean_cpu_expected_risk=float(np.mean([r['cpu_expected_risk'] for r in chosen])),
                mean_cuda_expected_risk=float(np.mean([r['cuda_expected_risk'] for r in chosen])),
                mean_risk_difference=float(np.mean([r['expected_risk_difference'] for r in chosen])),
                max_absolute_risk_difference=max(abs(r['expected_risk_difference']) for r in chosen)))
    result = dict(status='completed', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
        n_quality_rows=len(rows), n_batch_call_records=len(calls_rows),
        actual_network_invocations=sum(len(json.loads(r['calls'])) for r in calls_rows),
        actual_forward_examples=sum(r['forward_examples'] for r in calls_rows),
        max_logit_error=max(r['max_logit_error'] for r in rows),
        max_absolute_expected_risk_difference=max(abs(r['expected_risk_difference']) for r in rows),
        total_mask_differences=sum(r['mask_differences'] for r in rows),
        total_classification_differences=sum(r['classification_differences'] for r in rows),
        all_batch_calls_match=True, frozen_sources_unchanged=True, summaries=summaries,
        input_prediction_hashes={p.name:sha(p) for p in (GPU/'predictions').glob('*.npz')},
        output_hashes={p.name:sha(p) for p in folder.glob('*.npz')},
        interpretation='Direct CPU quality for three requested policies on the complete saved test inputs. '
        'Small cross-device FP32 differences are measured, not assumed absent. '
        'This adds fixed-model verification, not independent model replications or a new timing experiment.')
    json_write(HERE/'cpu_quality_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
