"""Check saved paths and measure which predictions change at equal nominal time."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core

ROOT=Path(__file__).resolve().parent
PRIOR=next(p for p in (ROOT.parent/"budgeted-rotation", ROOT.parent/"budgeted-rotation-final")
           if (p/"verify_results.py").is_file())
sys.dont_write_bytecode=True
sys.path.insert(0,str(PRIOR))
from verify_results import draw
from generator import rings


def read(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def record(path, rows):
    with path.open("w",encoding="utf8",newline="") as stream:
        w=csv.DictWriter(stream, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def sigmoid(h):
    return np.exp(-np.logaddexp(0.,-h))


def main():
    clock=ROOT/"clock_results"
    records=read(clock/"by_seed.csv")
    rows={(int(r["pair"]),float(r["rate"]),int(r["gd_steps"])):r for r in records}
    curvature={(int(r["pair"]),float(r["rate"])):r for r in read(ROOT/"curvature_results"/"by_seed.csv")}
    audit={"checkpoint_hashes":0,"training_losses":0,"gd_updates_replayed":0,
           "max_training_ce_error":0.,"max_gd_replay_error":0.,"max_gap_identity_error":0.,
           "max_gate_quota_error":0,"forks_checked":0,"early_traces_checked":0}
    function_rows=[]
    with np.load(clock/"training_paths.npz") as paths, threadpool_limits(1):
        for pair in range(1,33):
            x,y=rings(121000+pair)
            test=draw(141000+pair,8192)
            prob={}
            first=None
            for rate in (.5,1.,2.):
                suffix=f"p{pair}_r{int(10*rate)}"
                path=paths["weights_"+suffix]
                loss=paths["loss_"+suffix]
                assert len(path)==len(loss)==1+int(1600/rate)
                assert np.isfinite(path).all() and np.isfinite(loss).all()
                if first is None:
                    first=path[0].copy()
                else:
                    np.testing.assert_array_equal(first,path[0])
                early=curvature[pair,rate]
                assert np.count_nonzero(np.diff(loss[:26])>1e-12)==int(early["upward_steps"])
                np.testing.assert_allclose(loss[[0,1,2,25]],
                    list(map(float,[early["initial_loss"],early["first_loss"],early["second_loss"],early["loss_after_25"]])),
                    atol=1e-12,rtol=1e-12)
                audit["early_traces_checked"]+=1
                for (sid,lr,k),row in rows.items():
                    if (pair,rate)!=(sid,lr):
                        continue
                    theta=path[k]
                    assert hashlib.sha256(theta.tobytes()).hexdigest()==row["state_sha256"]
                    audit["checkpoint_hashes"]+=1
                    assert float(row["tau"])==k*rate
                    gap=float(row["folded_loss"])-float(row["confidence_loss"])
                    resid=max(abs(gap-float(row["gap"])), abs(gap-float(row["orientation_term"])+float(row["invariant_advantage"])))
                    assert resid<1e-12
                    audit["max_gap_identity_error"]=max(audit["max_gap_identity_error"],resid)
                    for field in ("folded_extra","confidence_extra","confidence_swapped_extra"):
                        assert int(row[field])==32768//4
                    single=(np.logaddexp(0.,-y*core.representation(theta,x)[:,0])).mean()
                    err=max(abs(single-loss[k]),abs(single-float(row["train_loss"])))
                    assert err<1e-12
                    audit["max_training_ce_error"]=max(audit["max_training_ce_error"],err)
                    audit["training_losses"]+=1
                    if k<len(path)-1:
                        updated=theta-rate*core.parameter_gradient(theta,x,y)
                        dev=np.max(abs(updated-path[k+1]))
                        assert dev<1e-12
                        audit["max_gd_replay_error"]=max(audit["max_gd_replay_error"],dev)
                        audit["gd_updates_replayed"]+=1
                    if float(row["tau"]) in (800,1600):
                        prob[rate,int(float(row["tau"]))]=sigmoid(core.representation(theta,test)[:,0])
            audit["forks_checked"]+=1
            for tau in (800,1600):
                for left,right in ((.5,1.),(1.,2.),(.5,2.)):
                    a,b=prob[left,tau],prob[right,tau]
                    function_rows.append(dict(pair=pair,tau=tau,low_rate=left,high_rate=right,
                        mean_absolute_probability_difference=np.abs(a-b).mean(),
                        class_disagreement=np.mean((a>.5)!=(b>.5))))
    audit={k:float(v) if isinstance(v,np.floating) else v for k,v in audit.items()}
    (ROOT/"independent_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    record(ROOT/"matched_time_predictions.csv",function_rows)
    print(audit)


if __name__=="__main__":
    main()
