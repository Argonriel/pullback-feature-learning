"""One 3x3 fixed-loss endpoint submersion check, plus exact area-map controls.

Standalone NumPy; no legacy imports, training, random sampling, or reserved data.
Use --output NEW_DIRECTORY. Output includes source hash and environment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np


def sym_basis(n):
    out = [np.diag(np.eye(n)[i]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            e = np.zeros((n, n)); e[i, j] = e[j, i] = 1 / np.sqrt(2)
            out.append(e)
    return np.array(out)


def pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def skewvec(x):
    return np.array([x[i, j] for i, j in pairs(len(x))])


def lyap(c, rhs):
    vals, u = np.linalg.eigh(c)
    return u @ ((u.T @ rhs @ u) / (vals[:, None] + vals[None, :])) @ u.T


def area(c, d):
    g = lyap(c, np.outer(d, d))
    return (c @ g - g @ c) / 2, g


def area_jac(c, d):
    j, g = area(c, d)
    cols = []
    for e in sym_basis(len(c)):
        z = lyap(c, -e @ g - g @ e)
        cols.append(skewvec((e @ g + c @ z - z @ c - g @ e) / 2))
    return np.array(cols).T


def mob(x, u, lam):
    val, v = np.linalg.eigh(x)
    sums = val[:, None] + val[None, :]
    return v @ ((sums / (sums + lam)) * (v.T @ u @ v)) @ v.T


def mob_and_derivatives(x, u, directions, lam):
    val, v = np.linalg.eigh(x)
    den = val[:, None] + val[None, :] + lam
    def resolvent(w):
        return v @ ((v.T @ w @ v) / den) @ v.T
    ru = resolvent(u)
    return u - lam * ru, np.array([
        lam * resolvent(t @ ru + ru @ t) for t in directions])


def frame(x, lam):
    n = len(x)
    f = np.array([mob(x, np.diag(e), lam) for e in np.eye(n)])
    a = np.array([np.diag(fi) for fi in f]).T
    h = np.einsum('kij,kl->lij', f, np.linalg.inv(a))
    return h, a


def dframe(x, t, lam):
    h, a = frame(x, lam)
    df = np.array([mob_and_derivatives(x, np.diag(e), [t], lam)[1][0]
                   for e in np.eye(len(x))])
    da = np.array([np.diag(fi) for fi in df]).T
    inv = np.linalg.inv(a)
    return np.einsum('kij,kl->lij', df, inv) - np.einsum('kij,kl->lij', h, da @ inv)


def curvature_matrix(x, lam):
    h, a = frame(x, lam)
    return np.array([skewvec(dframe(x, h[i], lam)[j] - dframe(x, h[j], lam)[i])
                     for i, j in pairs(len(x))]).T


def exterior(p):
    pp = pairs(len(p))
    return np.array([[p[i, k] * p[j, l] - p[i, l] * p[j, k]
                      for k, l in pp] for i, j in pp])


def run(x0, target, b, dbs, lam, step, duration=42):
    state = np.zeros((1 + len(dbs), len(x0), len(x0))); state[0] = x0
    def rhs(w):
        x, ts = w[0], w[1:]
        residual = np.diag(x) - target
        u = np.diag(b @ residual)
        mu, dmu = mob_and_derivatives(x, u, ts, lam)
        out = [-mu]
        for t, dm, db in zip(ts, dmu, dbs):
            out.append(-dm - mob(x, np.diag(b @ np.diag(t) + db @ residual), lam))
        return np.array(out)
    count = int(np.ceil(duration / step)); step = duration / count
    mineig = float(np.linalg.eigvalsh(x0)[0])
    for _ in range(count):
        k1 = rhs(state); k2 = rhs(state + step * k1 / 2)
        k3 = rhs(state + step * k2 / 2); k4 = rhs(state + step * k3)
        state += step * (k1 + 2*k2 + 2*k3 + k4) / 6
        state = (state + state.transpose(0, 2, 1)) / 2
        mineig = min(mineig, float(np.linalg.eigvalsh(state[0])[0]))
    jac = np.array([skewvec(t) for t in state[1:]]).T
    return {'endpoint': state[0].tolist(), 'endpoint_jacobian': jac.tolist(),
            'residual': float(np.linalg.norm(np.diag(state[0]) - target)),
            'sensitivity_diagonal_residual': float(np.linalg.norm([np.diag(t) for t in state[1:]])),
            'minimum_sampled_eigenvalue': mineig, 'step': step, 'duration': duration}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    x0 = np.array([[1., .2, -.15], [.2, 2., .3], [-.15, .3, 3.]])
    lam = 2.; c = np.diag([1., 2., 4.]); d = np.ones(3)
    h, a = frame(x0, lam); vals, u = np.linalg.eigh(a)
    p = (u * np.sqrt(vals)) @ u.T; pinv = np.linalg.inv(p)
    v = p @ d; b = pinv @ c @ pinv
    k = curvature_matrix(x0, lam); ja = area_jac(c, d)
    _, s, vt = np.linalg.svd(ja, full_matrices=False)
    e_dirs = np.einsum('rk,kij->rij', vt, sym_basis(3))
    dbs = np.array([pinv @ e @ pinv for e in e_dirs])
    pred = k @ exterior(p) @ ja @ vt.T
    fd_rows = []
    for delta in (1e-3, 5e-4, 2.5e-4):
        fd = np.array([skewvec(area(c + delta*e, d)[0] - area(c - delta*e, d)[0])/(2*delta)
                       for e in sym_basis(3)]).T
        fd_rows.append({'delta': delta, 'error': float(np.linalg.norm(fd-ja))})
    # Direct orthonormal-Sym operator representation independently checks normalization.
    basis = sym_basis(3)
    kk = np.array([[np.sum(e*(x0@f+f@x0)) for f in basis] for e in basis])
    mm = np.linalg.solve(kk + lam*np.eye(6), kk)
    direct = np.einsum('kij,kl->lij', basis, mm[:, :3])
    direct_a = np.array([np.diag(f) for f in direct]).T
    normalized_error = np.linalg.norm(np.array([np.diag(hi) for hi in h])-np.eye(3))
    curvature_fd = []
    for delta in (1e-3, 5e-4, 2.5e-4):
        cols=[]
        for i,j in pairs(3):
            diff = (frame(x0+delta*h[i],lam)[0][j]-frame(x0-delta*h[i],lam)[0][j]
                    -frame(x0+delta*h[j],lam)[0][i]+frame(x0-delta*h[j],lam)[0][i])/(2*delta)
            cols.append(skewvec(diff))
        curvature_fd.append({'delta':delta,'error':float(np.linalg.norm(np.array(cols).T-k))})
    rows=[]
    for eps in (.08, .04, .02):
        rr=run(x0,np.diag(x0)+eps*v,b,dbs,lam,.02)
        actual=np.array(rr['endpoint_jacobian'])/eps**2
        rr.update(epsilon=eps, scaled_jacobian=actual.tolist(),
                  scaled_error=float(np.linalg.norm(actual-pred)),
                  scaled_singular_values=np.linalg.svd(actual,compute_uv=False).tolist())
        rows.append(rr)
    refined=run(x0,np.diag(x0)+.02*v,b,dbs,lam,.01)
    # A single central difference of actual endpoints checks the tangent integration.
    endpoint_fd=[]
    for delta in (0.002,0.001):
        plus=run(x0,np.diag(x0)+.04*v,b+delta*dbs[0],[],lam,.02)
        minus=run(x0,np.diag(x0)+.04*v,b-delta*dbs[0],[],lam,.02)
        fd=skewvec(np.array(plus['endpoint'])-np.array(minus['endpoint']))/(2*delta)
        endpoint_fd.append({'delta':delta,'derivative':fd.tolist(),
                            'error':float(np.linalg.norm(fd-np.array(rows[1]['endpoint_jacobian'])[:,0]))})
    controls=[]
    for label,cc,dd,expected in [('scalar_weight',np.eye(3),np.ones(3),2),
                              ('two_active_modes',c,np.array([1.,1.,0.]),3),
                              ('one_active_mode',c,np.array([1.,0.,0.]),2)]:
        jj=area_jac(cc,dd); sv=np.linalg.svd(jj,compute_uv=False)
        controls.append({'label':label,'C':cc.tolist(),'d':dd.tolist(),
                         'singular_values':sv.tolist(),'rank':int(np.sum(sv>1e-10)),
                         'expected_rank':expected})
    result={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'numpy':np.__version__,
            'initial_state':x0.tolist(),'damping':lam,'C':c.tolist(),'d':d.tolist(),
            'P':p.tolist(),'weight':b.tolist(),'target_direction':v.tolist(),
            'weight_directions':dbs.tolist(),'area_directions':e_dirs.tolist(),
            'observed_mobility':a.tolist(),'observed_eigenvalues':vals.tolist(),
            'direct_operator_discrepancy':float(np.linalg.norm(direct_a-a)),
            'normalized_frame_error':float(normalized_error),'curvature':k.tolist(),
            'curvature_determinant':float(np.linalg.det(k)),
            'curvature_finite_differences':curvature_fd,
            'area_jacobian':ja.tolist(),'area_jacobian_singular_values':s.tolist(),
            'area_finite_differences':fd_rows,'predicted_scaled_endpoint_jacobian':pred.tolist(),
            'predicted_singular_values':np.linalg.svd(pred,compute_uv=False).tolist(),
            'endpoint_rows':rows,'refined_endpoint':refined,'endpoint_finite_differences':endpoint_fd,
            'rank_controls':controls}
    result['checks']={
        'normalization':normalized_error<1e-12 and result['direct_operator_discrepancy']<1e-12,
        'area_fd_converges':fd_rows[-1]['error']<fd_rows[0]['error']/10,
        'curvature_fd_converges':curvature_fd[-1]['error']<curvature_fd[0]['error']/10,
        'rank_controls':all(r['rank']==r['expected_rank'] for r in controls),
        'scaled_jacobian_converges':rows[-1]['scaled_error']<rows[0]['scaled_error']/3,
        'refinement_error':float(np.linalg.norm(np.array(refined['endpoint_jacobian'])-
                                              np.array(rows[-1]['endpoint_jacobian']))),
        'endpoint_residuals':max(r['residual'] for r in rows)<1e-11,
        'sensitivity_fd_converges':endpoint_fd[-1]['error']<endpoint_fd[0]['error']*.4}
    (args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'checks':result['checks'],'predicted_singular_values':result['predicted_singular_values'],
                      'scaled_errors':[r['scaled_error'] for r in rows],
                      'scaled_singular_values':[r['scaled_singular_values'] for r in rows]},indent=2))


if __name__=='__main__': main()
