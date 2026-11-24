"""Independent 18-parameter replay of the fixed three-by-three witness."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def integrate(weight, step):
    identity = np.eye(3)
    state = np.tile(identity.ravel(), 2)
    sensing = np.zeros((2, 3, 3))
    sensing[0, 0, 1] = sensing[0, 1, 0] = 1/np.sqrt(2)
    sensing[1, 0, 2] = sensing[1, 2, 0] = 1/np.sqrt(2)
    target = np.full(2, np.sqrt(2)*.25)
    damping = .2

    def rhs(z):
        first, second = z[:9].reshape(3,3), z[9:].reshape(3,3)
        product = second @ first
        residual = np.einsum("kij,ij->k", sensing, product)-target
        gradient = np.einsum("k,kij->ij", residual*np.array([1.,weight]), sensing)
        jacobian = np.hstack((np.kron(second,identity), np.kron(identity,first.T)))
        return -np.linalg.solve(jacobian.T@jacobian+damping*np.eye(18),jacobian.T@gradient.ravel())

    balance = symmetry = 0.
    min_eigenvalue = 1.
    for count in range(1,int(100/step)+1):
        k1=rhs(state)
        k2=rhs(state+step*k1/2)
        k3=rhs(state+step*k2/2)
        k4=rhs(state+step*k3)
        state += step*(k1+2*k2+2*k3+k4)/6
        first, second=state[:9].reshape(3,3),state[9:].reshape(3,3)
        product=second@first
        balance=max(balance,float(np.linalg.norm(first@first.T-second.T@second)))
        symmetry=max(symmetry,float(np.linalg.norm(product-product.T)))
        min_eigenvalue=min(min_eigenvalue,float(np.linalg.eigvalsh((product+product.T)/2)[0]))
        residual=float(np.max(np.abs(np.einsum("kij,ij->k",sensing,product)-target)))
        if residual<1e-13:
            break
    return {"weight":weight,"step":step,"matrix":product.tolist(),"steps":count,
            "residual":residual,"balance_error":balance,"symmetry_error":symmetry,
            "minimum_eigenvalue":min_eigenvalue}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records=[integrate(weight,step) for weight in [1.,2.] for step in [.01,.005]]
    control=json.loads(args.control.read_text(encoding="utf-8"))
    differences=[]
    for weight in [1.,2.]:
        reduced=next(r for r in control["records"] if r["metric"]=="pullback" and r["weight"]==weight and r["step"]==.01)
        direct=next(r for r in records if r["weight"]==weight and r["step"]==.005)
        differences.append(float(np.linalg.norm(np.array(reduced["matrix"])-direct["matrix"])))
    checks={"endpoint_agreement":max(differences)<1e-9,
            "interpolation":max(r["residual"] for r in records)<1e-13,
            "balance":max(r["balance_error"] for r in records)<1e-10,
            "symmetry":max(r["symmetry_error"] for r in records)<1e-10,
            "spd":min(r["minimum_eigenvalue"] for r in records)>.5}
    output={"status":"passed" if all(checks.values()) else "failed","checks":checks,
            "records":records,"cross_implementation_endpoint_errors":differences,
            "control_data_sha256":hashlib.sha256(args.control.read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"checks":checks,"endpoint_errors":differences},indent=2))
    assert all(checks.values())


if __name__=="__main__":
    main()
