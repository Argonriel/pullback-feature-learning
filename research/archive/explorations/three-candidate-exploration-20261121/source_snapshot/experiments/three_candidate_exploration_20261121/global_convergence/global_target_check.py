"""Large-target consistency check of the classified global-convergence claim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

DEPENDENCY = Path(__file__).resolve().parents[2] / 'endpoint_classification_audit_20261121' / 'verify_finite_damping.py'
spec = importlib.util.spec_from_file_location('finite_damping_reference', DEPENDENCY)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def solve_prediction(feasible, normal, damping):
    x = feasible.copy()
    for iteration in range(100):
        residual = reference.coords(normal, x + damping * reference.logm(x) / 2 - np.eye(6))
        if np.linalg.norm(residual) < 1e-12:
            return x, iteration, float(np.linalg.norm(residual))
        jac = np.array([[np.trace(a @ (b + damping * np.linalg.solve(x, b) / 2))
                         for b in normal] for a in normal])
        step = np.einsum('a,aij->ij', np.linalg.solve(jac, residual), normal)
        fraction = 1.0
        while fraction > 1e-13:
            candidate = x - fraction * step
            if np.linalg.eigvalsh(candidate).min() > 0:
                next_residual = reference.coords(normal, candidate + damping * reference.logm(candidate) / 2 - np.eye(6))
                if np.linalg.norm(next_residual) < np.linalg.norm(residual):
                    break
            fraction /= 2
        x = candidate
    raise RuntimeError('Prediction solve failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    b, normal = reference.basis()
    rng = np.random.default_rng(20261122)
    feasible = np.zeros((6, 6))
    for sl, eigenvalues in ((slice(0, 2), [.4, 3.0]), (slice(2, 5), [.3, 1.2, 4.0]), (slice(5, 6), [.8])):
        rotation, _ = np.linalg.qr(rng.normal(size=(len(eigenvalues), len(eigenvalues))))
        feasible[sl, sl] = (rotation * eigenvalues) @ rotation.T
    target = reference.coords(b, feasible - np.eye(6))
    q, _ = np.linalg.qr(rng.normal(size=(len(b), len(b))))
    weights = [np.eye(len(b)), (q * np.linspace(.7, 2.1, len(b))) @ q.T]
    damping, horizon = 4.0, 300.0
    predicted, iterations, invariant_residual = solve_prediction(feasible, normal, damping)
    records, endpoints = [], {}
    for wi, weight in enumerate(weights):
        for dt in (.025, .0125):
            x, record = reference.integrate(b, normal, target, weight, damping, dt, horizon)
            record.update(weight_index=wi, prediction_error=float(np.linalg.norm(x - predicted)))
            records.append(record)
            endpoints[f'w{wi}_dt{dt}'] = x
    fine_gap = np.linalg.norm(endpoints['w0_dt0.0125'] - endpoints['w1_dt0.0125'])
    refinements = [float(np.linalg.norm(endpoints[f'w{i}_dt0.025'] - endpoints[f'w{i}_dt0.0125'])) for i in range(2)]
    result = {'scope': 'One large-target check; proof, not this case, establishes global convergence.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dependency_sha256': hashlib.sha256(DEPENDENCY.read_bytes()).hexdigest(),
              'dependency_name': DEPENDENCY.name, 'damping': damping, 'horizon': horizon,
              'target_displacement_norm': float(np.linalg.norm(target)),
              'strict_feasible_minimum_eigenvalue': float(np.linalg.eigvalsh(feasible).min()),
              'euclidean_interpolant_minimum_eigenvalue': float(np.linalg.eigvalsh(np.eye(6) + np.einsum('a,aij->ij', target, b)).min()),
              'prediction_minimum_eigenvalue': float(np.linalg.eigvalsh(predicted).min()),
              'prediction_steps': iterations, 'prediction_invariant_residual': invariant_residual,
              'fine_weight_gap': float(fine_gap), 'refinement_errors': refinements,
              'records': records}
    result['passed'] = bool(fine_gap < 2e-8 and max(refinements) < 2e-7
                            and all(r['measurement_residual'] < 1e-8 and r['prediction_error'] < 2e-7 for r in records))
    np.savez(args.output / 'matrices.npz', basis=b, normal=normal, feasible=feasible,
             target=target, weights=np.array(weights), predicted=predicted, **endpoints)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit('Inspect preserved validation outputs.')


if __name__ == '__main__':
    main()
