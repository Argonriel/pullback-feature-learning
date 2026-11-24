"""Known entropy-Hessian control for a fixed three-by-three endpoint example.

This is a numerical control, not a proposed new optimizer. Original metric
sources were reviewed before fixing these cases. All outputs are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


U = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
V = np.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])
A = np.array([U, V]) / np.sqrt(2)


def logarithmic_kernel(eigenvalues):
    mean = (eigenvalues[:, None] + eigenvalues[None, :]) / 2
    ratio = (eigenvalues[:, None] - eigenvalues[None, :]) / (2 * mean)
    small = np.abs(ratio) < 1e-5
    factor = np.ones_like(ratio)
    factor[small] += ratio[small]**2 / 3 + ratio[small]**4 / 5
    factor[~small] = np.arctanh(ratio[~small]) / ratio[~small]
    return factor / mean


def project_null(x):
    return x - np.einsum("k,kij->ij", np.einsum("kij,ij->k", A, x), A)


def matrix_log(x):
    eigenvalues, vectors = np.linalg.eigh(x)
    return (vectors * np.log(eigenvalues)) @ vectors.T


def integrate(metric, weight, step, damping=.2, amplitude=.25):
    x = np.eye(3)
    target = np.full(2, np.sqrt(2) * amplitude)
    initial_invariant = project_null(x + damping / 2 * matrix_log(x))
    min_eigenvalue = 1.
    max_invariant_error = 0.

    def rhs(state):
        residual = np.einsum("kij,ij->k", A, state) - target
        gradient = np.einsum("k,kij->ij", np.array([1., weight]) * residual, A)
        eigenvalues, vectors = np.linalg.eigh(state)
        if metric == "pullback":
            kernel = 1 / (eigenvalues[:, None] + eigenvalues[None, :])
        elif metric == "entropy":
            kernel = logarithmic_kernel(eigenvalues) / 2
        else:
            raise ValueError(metric)
        return -vectors @ ((vectors.T @ gradient @ vectors) / (1 + damping * kernel)) @ vectors.T

    for count in range(1, int(100 / step) + 1):
        k1 = rhs(x)
        k2 = rhs(x + step * k1 / 2)
        k3 = rhs(x + step * k2 / 2)
        k4 = rhs(x + step * k3)
        x += step * (k1 + 2*k2 + 2*k3 + k4) / 6
        min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(x)[0]))
        invariant_error = np.linalg.norm(project_null(x + damping / 2 * matrix_log(x)) - initial_invariant)
        max_invariant_error = max(max_invariant_error, float(invariant_error))
        residual = float(np.max(np.abs(np.einsum("kij,ij->k", A, x) - target)))
        if residual < 1e-13:
            break
    return {"metric": metric, "weight": weight, "step": step, "damping": damping,
            "raw_matrix_amplitude": amplitude, "target_measured_values": target.tolist(),
            "matrix": x.tolist(), "steps": count, "time": count*step,
            "residual": residual, "minimum_eigenvalue": min_eigenvalue,
            "entropy_invariant_max_drift": max_invariant_error}


def static_checks():
    z = np.array([[.2, .3, -.1], [.3, -.1, .2], [-.1, .2, .15]])
    tangent = np.array([[.1, -.2, .4], [-.2, .3, .1], [.4, .1, -.1]])
    commutator = lambda first, second: first @ second - second @ first
    double = commutator(z, commutator(z, tangent))
    rows = []
    for epsilon in [.2, .1, .05, .025]:
        x = np.eye(3) + epsilon*z
        eig, q = np.linalg.eigh(x)
        r_kernel = logarithmic_kernel(eig)/2 - 1/(eig[:,None]+eig[None,:])
        exact = q @ (r_kernel * (q.T @ tangent @ q)) @ q.T
        quadratic = epsilon**2 * double / 24
        cubic = quadratic - epsilon**3 * (z @ double + double @ z) / 16
        rows.append({"epsilon": epsilon,
                     "minimum_operator_eigenvalue": float(r_kernel.min()),
                     "after_quadratic_error": float(np.linalg.norm(exact - quadratic)),
                     "after_cubic_error": float(np.linalg.norm(exact - cubic))})
    # Frechet log action checked against a finite difference of the matrix log.
    x = np.eye(3) + z
    eig, q = np.linalg.eigh(x)
    action = q @ (logarithmic_kernel(eig) * (q.T @ tangent @ q)) @ q.T
    h = 1e-5
    fd = (matrix_log(x+h*tangent)-matrix_log(x-h*tangent))/(2*h)
    return {"series_checks": rows, "log_derivative_finite_difference_error": float(np.linalg.norm(action-fd))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [integrate(metric, weight, step)
               for metric in ["pullback", "entropy"] for weight in [1., 2.]
               for step in [.02, .01]]
    contrasts = []
    for metric in ["pullback", "entropy"]:
        data = {(r["weight"],r["step"]):np.array(r["matrix"]) for r in records if r["metric"] == metric}
        contrast = data[2.,.01]-data[1.,.01]
        contrasts.append({"metric": metric, "weight2_minus_weight1": contrast.tolist(),
                          "null_contrast_norm":float(np.linalg.norm(project_null(contrast))),
                          "max_endpoint_refinement_norm":max(float(np.linalg.norm(data[w,.02]-data[w,.01])) for w in [1.,2.])})
    static = static_checks()
    checks = {"residual":max(r["residual"] for r in records)<1e-13,
              "spd":min(r["minimum_eigenvalue"] for r in records)>.5,
              "entropy_endpoint_neutral":contrasts[1]["null_contrast_norm"]<1e-9,
              "pullback_endpoint_sensitive":contrasts[0]["null_contrast_norm"]>1e-6,
              "log_derivative":static["log_derivative_finite_difference_error"]<1e-8}
    output = {"status":"passed" if all(checks.values()) else "failed", "checks":checks,
              "records":records,"contrasts":contrasts,"static_checks":static,
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope":"Known entropy-Hessian control; no novelty or prediction-benefit claim."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"checks":checks,"contrasts":contrasts,"static":static},indent=2))
    assert all(checks.values())


if __name__ == "__main__":
    main()
