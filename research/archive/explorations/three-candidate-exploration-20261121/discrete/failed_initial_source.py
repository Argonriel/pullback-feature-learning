"""Bounded small-target audit of actual balanced-factor pullback steps.

Usage: python factor_step_cancellation.py --output NEW_DIRECTORY
Reports and generated results belong outside the repository. The target and
parameter cases are fixed; this is not a parameter search.
"""

import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np


ETA = 0.5
BASIS = np.array([[[1.0, 0], [0, -1.0]], [[0, 1.0], [1.0, 0]]]) / np.sqrt(2)


def project_gradient(matrix, weights, degree):
    coords = np.einsum("aij,ij->a", BASIS, matrix)
    if degree == 1:
        coords -= 1
    return np.einsum("a,aij->ij", weights * coords, BASIS)


def jet_endpoint(lam, weights, degree=4):
    """Differentiate the discrete map algebraically via truncated power series."""
    x = np.zeros((degree + 1, 2, 2))
    x[0] = np.eye(2)
    d = 2 + lam
    for step in range(3000):
        g, u = np.zeros_like(x), np.zeros_like(x)
        for j in range(1, degree + 1):
            g[j] = project_gradient(x[j], np.array(weights), j)
            interaction = sum((x[k] @ u[j-k] + u[j-k] @ x[k]
                               for k in range(1, j)), np.zeros((2, 2)))
            u[j] = (g[j] - interaction) / d
        nxt = x - ETA * (g - lam * u)
        for j in range(2, degree + 1):
            for a in range(1, j):
                for c in range(1, j-a+1):
                    b = j-a-c
                    nxt[j] += ETA**2 * u[a] @ x[b] @ u[c]
        change = np.max(np.abs(nxt - x))
        x = nxt
        if change < 2e-15 and np.max(np.abs(g)) < 2e-14:
            return {"steps": step+1, "coefficients": x.tolist(),
                    "scalar_coefficients": [float(np.trace(z)/2) for z in x],
                    "final_change": float(change),
                    "gradient_coeff_max": float(np.max(np.abs(g)))}
    raise RuntimeError("Taylor jets failed to converge")


def actual_endpoint(lam_text, weights_text, epsilon_text):
    """Independent 80-digit direct map in spin coordinates; no Taylor truncation."""
    mp.mp.dps = 80
    lam, eta, eps = mp.mpf(lam_text), mp.mpf("0.5"), mp.mpf(epsilon_text)
    weights = list(map(mp.mpf, weights_text))
    a, v = mp.mpf(1), [mp.mpf(0), mp.mpf(0)]
    min_eigen = mp.mpf(1)
    min_factor_eigen = mp.mpf(1)
    drift_sum = mp.mpf(0)
    for step in range(3000):
        g = [weights[i]*(v[i]-eps) for i in range(2)]
        residual = mp.sqrt(sum((z-eps)**2 for z in v))
        if residual < mp.mpf("1e-65"):
            break
        d = 2*a+lam
        u0 = -sum(v[i]*g[i] for i in range(2))/(d*d-2*sum(z*z for z in v))
        u = [(g[i]-2*u0*v[i])/d for i in range(2)]
        uu = sum(z*z for z in u)
        vu = sum(v[i]*u[i] for i in range(2))
        scalar = a*(u0*u0+uu/2)+u0*vu
        vector = [2*a*u0*u[i]+(u0*u0-uu/2)*v[i]+vu*u[i] for i in range(2)]
        factor_min = 1-eta*(u0+mp.sqrt(uu/2))
        min_factor_eigen = min(min_factor_eigen, factor_min)
        if factor_min <= 0:
            raise RuntimeError("Step crossed the positive congruence branch")
        factor_det = (1-eta*u0)**2-eta**2*uu/2
        drift_sum += eta**2*scalar+lam*(mp.log(factor_det)/2+eta*u0)
        a += eta*lam*u0+eta**2*scalar
        v = [v[i]-eta*(g[i]-lam*u[i])+eta**2*vector[i] for i in range(2)]
        min_eigen = min(min_eigen, a-mp.sqrt(sum(z*z for z in v)/2))
    else:
        raise RuntimeError("Direct map failed to converge")
    invariant_change = a+lam*mp.log(a*a-sum(z*z for z in v)/2)/4-1
    continuous = mp.findroot(lambda z: z+lam*mp.log(z*z-eps*eps)/4-1, 1+eps*eps)
    return {"steps": step, "scalar_endpoint": str(a),
            "vector_endpoint": [str(z) for z in v], "residual": str(residual),
            "min_eigenvalue": str(min_eigen),
            "min_factor_eigenvalue": str(min_factor_eigen),
            "invariant_change": str(invariant_change),
            "summed_exact_drift": str(drift_sum),
            "drift_identity_error": str(abs(invariant_change-drift_sum)),
            "continuous_scalar_endpoint": str(continuous),
            "scalar_error_vs_continuous": str(a-continuous)}


def factor_coordinate_check():
    """Independent full 8-coordinate factor-Jacobian solve, away from isotropy."""
    x = np.array([[1.12, .07], [.07, .94]])
    ev, vec = np.linalg.eigh(x)
    w1 = (vec*np.sqrt(ev)) @ vec.T
    w2 = w1.T.copy()
    g = .13*BASIS[0]-.09*BASIS[1]
    lam = 2.0
    eye4 = np.eye(4).reshape(4, 2, 2)
    jac = np.stack([(w2@e).ravel() for e in eye4] +
                   [(e@w1).ravel() for e in eye4], axis=1)
    p = -np.linalg.solve(jac.T@jac+lam*np.eye(8), jac.T@g.ravel())
    lyap = np.stack([(x@e+e@x+lam*e).ravel() for e in eye4], axis=1)
    u = np.linalg.solve(lyap, g.ravel()).reshape(2, 2)
    induced = (np.eye(2)-ETA*u)@x@(np.eye(2)-ETA*u)
    direct = (w2+ETA*p[4:].reshape(2, 2))@(w1+ETA*p[:4].reshape(2, 2))
    return {"direction_error": float(np.max(np.abs(p-np.concatenate([
                (-w1@u).ravel(), (-u@w2).ravel()])))),
            "product_map_error": float(np.max(np.abs(induced-direct)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results = {"protocol": "fixed 3 damping values x 3 target amplitudes x 2 equal-trace weights",
               "factor_coordinate_check": factor_coordinate_check(), "cases": []}
    for lam_text in ["0.5", "2", "4"]:
        lam = float(lam_text)
        jets = [jet_endpoint(lam, w) for w in [(1, 1), (.5, 1.5)]]
        jet_difference = [b-a for a, b in zip(jets[0]["scalar_coefficients"],
                                              jets[1]["scalar_coefficients"])]
        m, prefactor = 2/(2+lam), ETA*(1-lam/2)/(2*(2+lam)**2)
        coefficient = prefactor*(.5/(2-ETA*m*.5)+1.5/(2-ETA*m*1.5)-2/(2-ETA*m))
        case = {"lambda": lam, "eta": ETA, "weights": [[1, 1], [.5, 1.5]],
                "quadratic_coefficient_formula": coefficient,
                "jet_scalar_difference": jet_difference, "jets": jets,
                "direct": []}
        for eps in ["0.1", "0.05", "0.025"]:
            runs = [actual_endpoint(lam_text, w, eps) for w in [("1", "1"), ("0.5", "1.5")]]
            delta = mp.mpf(runs[1]["scalar_endpoint"])-mp.mpf(runs[0]["scalar_endpoint"])
            case["direct"].append({"epsilon": eps, "runs": runs, "scalar_difference": str(delta),
                                   "difference_over_epsilon2": str(delta/mp.mpf(eps)**2),
                                   "difference_over_epsilon4": str(delta/mp.mpf(eps)**4)})
        results["cases"].append(case)
    (out/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"factor_coordinate_check": results["factor_coordinate_check"], "cases": [
        {"lambda": c["lambda"], "quadratic_formula": c["quadratic_coefficient_formula"],
         "jet_difference": c["jet_scalar_difference"], "direct": [
             {key: z[key] for key in ["epsilon", "scalar_difference", "difference_over_epsilon2", "difference_over_epsilon4"]}
             for z in c["direct"]]} for c in results["cases"]]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
