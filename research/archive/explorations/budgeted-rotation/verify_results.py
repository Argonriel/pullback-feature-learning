"""Audit records, reconstruct routing and integrate the available geometric gain."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning.core import representation
from gate import deploy, Gate

ROOT = Path(__file__).resolve().parent
POLICIES = ("raw","full","random","radius","folded","boundary")


def read(path):
    with path.open(encoding="utf-8",newline="") as file:
        return list(csv.DictReader(file))


def draw(seed,n):
    rng=np.random.default_rng(seed)
    r=rng.choice([.5,1.2],n)
    t=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.column_stack((np.cos(t),np.sin(t))) + rng.normal(0,.15,(n,2))


def radius_and_bin(x,edges):
    r=np.sqrt(np.sum(x*x,axis=1))
    rb=np.searchsorted(edges,r)
    theta=np.mod(np.arctan2(x[:,1],x[:,0]), np.pi)
    return r, rb, 8*rb+np.minimum((8*theta/np.pi).astype(int),7)


def cutoff(values,priorities):
    # Select by threshold then include exactly the needed items within the tied cell.
    val=np.sort(values)[-64]
    keep=values>val
    ties=np.flatnonzero(values==val)
    take=64-int(np.count_nonzero(keep))
    keep[ties[np.argsort(priorities[ties],kind="stable")[:take]]]=True
    assert np.count_nonzero(keep)==64
    return keep


def sha(data):
    return hashlib.sha256(data.tobytes()).hexdigest()


def independent_batch_checks(out,mrows,batches,prov,states,gates):
    worst=0.
    gate_error=0.
    deploy_error=0.
    audited_runs=0
    report={}
    for pair in (1,8,16):
        cal=draw(123000+pair,8192)
        eval_x=draw(124000+pair,32768)
        tiebreak=np.random.default_rng(125000+pair).random(32768)
        for model in ("gd","pb","hybrid"):
            key=f"{pair}_{model}"
            record,training=mrows[pair,model],prov[pair,model]
            theta=states[key]
            assert sha(theta)==training["state_hash"] and sha(cal)==training["cal_hash"] and sha(eval_x)==record["eval_hash"]
            fn=lambda x: representation(theta,x)[:,0]
            h1,h2=fn(cal),fn(-cal)
            # Closed-form label-free Jensen score independent of the runner's formula.
            p=1/(1+np.exp(-.5*(h1+h2)))
            gains=.5*np.log1p(4*p*(1-p)*np.sinh((h1-h2)/4)**2)
            edges=np.quantile(np.sqrt(np.sum(cal**2,axis=1)),np.arange(1,16)/16)
            _,rad,fold=radius_and_bin(cal,edges)
            rscore=np.array([gains[rad==i].mean() for i in range(16)])
            fscore=np.array([gains[fold==i].mean() if np.any(fold==i) else rscore[i//8] for i in range(128)])
            for field,tmp in (("edges",edges),("radius_scores",rscore),("folded_scores",fscore)):
                gate_error=max(gate_error,float(np.max(np.abs(tmp-gates[key+"_"+field]))))
            g=Gate(edges,rscore,fscore)
            for batch in (1,64,128):
                sl=slice((batch-1)*256,batch*256)
                x,t=eval_x[sl],tiebreak[sl]
                h1,h2=fn(x),fn(-x)
                m=(h1+h2)/2
                r,rad,fold=radius_and_bin(x,edges)
                a=((.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(r*1.2/.15**2))-np.log(np.i0(r*.5/.15**2)))
                label_p=1/(1+np.exp(-a))
                raw=.5*(np.logaddexp(0,h1)+np.logaddexp(0,h2))-label_p*m
                full=np.logaddexp(0,m)-label_p*m
                losses={"raw":raw,"full":full,"random":.75*raw+.25*full}
                for name,score in (("radius",rscore[rad]),("folded",fscore[fold]),("boundary",-np.abs(r-.86419687750821))):
                    mask=cutoff(score,t)
                    losses[name]=np.where(mask,full,raw)
                    if name=="boundary":
                        continue
                    calls=[]
                    def counted(points):
                        calls.append(points.copy())
                        return fn(points)
                    ans,keep=deploy(x,counted,g,name,t,quota=64)
                    assert [len(p) for p in calls]==[256,64] and np.array_equal(keep,mask)
                    assert np.array_equal(calls[0],x) and np.array_equal(calls[1],-x[mask])
                    deploy_error=max(deploy_error,float(np.max(np.abs(ans-np.where(mask,m,h1)))))
                    audited_runs+=1
                original=batches[pair,model,batch]
                worst=max(worst,max(abs(float(original[name+"_loss"])-float(loss.mean())) for name,loss in losses.items()))
    assert gate_error < 1e-10 and worst < 1e-10 and deploy_error < 1e-10
    return dict(reconstructed_calibrations=9,reconstructed_batches=27,actual_deployment_paths=audited_runs,
                maximum_calibration_discrepancy=gate_error,maximum_batch_discrepancy=worst,
                maximum_deployment_discrepancy=deploy_error)


def load_polar(path):
    spec=importlib.util.spec_from_file_location("prior_ring_polar",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polar_benchmark(out,mrows,batches,states,polar):
    results=[]
    for pair in range(1,17):
        state=states[f"{pair}_pb"]
        exact=polar.evaluate(state)
        # On an even uniform angular grid, an antipodal view is a half-period shift.
        last=None
        for size in (96,192,384,768,1536,3072):
            r,w0,w1,unit=polar.polar_grid(size,size)
            h=polar.logits_on_circles(state,r,unit)
            hm=.5*(h+np.roll(h,size//2,axis=1))
            full=polar.integrate(hm,r,w0,w1)
            now=full["population_loss"]
            if last is not None and abs(now-last) < 3e-7+1e-6*now:
                break
            last=now
        else:
            raise ArithmeticError("Projected model integral failed grid refinement")
        available=exact["population_loss"]-now
        assert abs(exact["radial_excess"]-full["radial_excess"])<3e-6
        assert -3e-6<=available<=exact["angular_excess"]+3e-6
        mc=[float(batches[pair,"pb",i]["raw_loss"])-float(batches[pair,"pb",i]["full_loss"]) for i in range(1,129)]
        se=np.std(mc,ddof=1)/np.sqrt(len(mc))
        z=abs(np.mean(mc)-available)/se
        assert z<6, (pair,z)
        rec=mrows[pair,"pb"]
        folded=float(rec["raw_loss"])-float(rec["folded_loss"])
        results.append(dict(pair=pair,population_angular_excess=exact["angular_excess"],
                       population_antipodal_gain=available,unremoved_angular_excess=full["angular_excess"],
                       estimated_folded_gain=folded,MC_full_gain=np.mean(mc),MC_full_gain_se=se,
                       MC_z_score=float(z),folded_below_population_angular_ceiling=bool(folded<=exact["angular_excess"])))
    return results


def main(polar_path):
    out=ROOT/"results"
    mrows={(int(r["pair"]),r["model"]):r for r in read(out/"models.csv")}
    prov={(int(r["pair"]),r["model"]):r for r in read(out/"training_and_calibration.csv")}
    batches={(int(r["pair"]),r["model"],int(r["batch"])):r for r in read(out/"batches.csv")}
    assert len(mrows)==48 and len(batches)==6144 and set(prov)==set(mrows)
    max_mean=max_random=0.
    for (pair,model),r in mrows.items():
        block=[batches[pair,model,i] for i in range(1,129)]
        for p in POLICIES:
            field=p+"_loss"
            max_mean=max(max_mean,abs(float(r[field])-np.mean([float(b[field]) for b in block])))
        for b in block:
            lo,hi=float(b["full_loss"]),float(b["raw_loss"])
            max_random=max(max_random,abs(float(b["random_loss"])-(.75*hi+.25*lo)))
            assert all(lo-1e-12 <= float(b[n+"_loss"]) <= hi+1e-12 for n in ("random","radius","folded","boundary"))
    assert max_mean<1e-12 and max_random<1e-12
    audit=dict(models=48,batches=6144,max_mean_discrepancy=max_mean,max_random_expectation_discrepancy=max_random)
    with np.load(out/"models.npz") as states, np.load(out/"calibrated_gates.npz") as gates, threadpool_limits(1):
        audit.update(independent_batch_checks(out,mrows,batches,prov,states,gates))
        audit["pb_population_ceiling"]=polar_benchmark(out,mrows,batches,states,load_polar(polar_path))
    summary=audit["pb_population_ceiling"]
    audit["pb_means"]={f:np.mean([r[f] for r in summary]) for f in ("population_angular_excess","population_antipodal_gain","unremoved_angular_excess","estimated_folded_gain")}
    audit["pb_max_full_gain_in_MC_SE"]=max(r["MC_z_score"] for r in summary)
    (out/"independent_audit.json").write_text(json.dumps(audit,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({k:v for k,v in audit.items() if k!="pb_population_ceiling"},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--polar",type=Path,default=ROOT.parent/"rings-mechanism"/"polar.py")
    main(ap.parse_args().polar)
