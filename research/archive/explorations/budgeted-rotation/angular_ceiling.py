"""Connect the antipodal gain to the preceding exact radial/angular risk decomposition."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits


def main(folder, mechanism):
    spec = importlib.util.spec_from_file_location("rings_polar", mechanism/"polar.py")
    polar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(polar)
    fields=("raw", "full", "radial", "angular", "angular_after")
    rows=[]
    with np.load(folder/"models.npz") as bank, threadpool_limits(1):
        for key in bank.files:
            previous=None
            for nr,na in polar.GRIDS:
                r,w0,w1,u=polar.polar_grid(nr,na)
                h=polar.logits_on_circles(bank[key],r,u)
                base=polar.integrate(h,r,w0,w1)
                mean=.5*(h+np.roll(h,na//2,axis=1))
                augmented=polar.integrate(mean,r,w0,w1)
                v=np.array([base["population_loss"],augmented["population_loss"],
                            base["radial_excess"],base["angular_excess"],augmented["angular_excess"]])
                if previous is not None and np.all(abs(v-previous)<2e-7+2e-6*abs(v)):
                    break
                previous=v
            else:
                raise ArithmeticError(f"Grid failed to converge: {key}")
            gain=v[0]-v[1]
            assert abs(v[2]-augmented["radial_excess"])<1e-10 and abs(gain-(v[3]-v[4]))<1e-10
            assert -1e-12<=gain<=v[3]+1e-12
            p,m=key.split("_")
            rows.append(dict(pair=int(p), model=m, **dict(zip(fields,map(float,v))),
                             two_view_gain=gain, radial_nodes=nr, angular_nodes=na))
    with (folder/"angular_ceiling.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out={}
    for m in ("gd","pb","hybrid"):
        group=[r for r in rows if r["model"]==m]
        amount=np.mean([r["angular"] for r in group])
        gain=np.mean([r["two_view_gain"] for r in group])
        out[m]=dict(angular=amount,two_view_gain=gain,fraction_of_angular_loss_addressable=gain/amount)
    (folder/"angular_ceiling.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(out)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mechanism",type=Path,required=True)
    ap.add_argument("--results",type=Path,default=Path(__file__).resolve().parent/"results")
    args=ap.parse_args()
    main(args.results,args.mechanism)
