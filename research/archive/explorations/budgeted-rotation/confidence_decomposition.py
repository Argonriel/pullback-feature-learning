"""Post-hoc arithmetic decomposition on the frozen confidence supplement."""
import csv
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning.core import representation
from verify_results import cutoff, draw, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "supplemental_results"
FIELDS = ("raw_to_confidence", "symmetric_gain", "orientation_term", "raw_to_full",
          "unused_symmetric_gain", "confidence_gain_over_full", "mask_disagreement")


def save(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        wr=csv.DictWriter(f,list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def choose(h, tie):
    return np.concatenate([cutoff(-abs(s), t) for s,t in zip(h.reshape(-1,256),tie.reshape(-1,256))])


def main():
    with (OUT/"models.csv").open(newline="", encoding="utf-8") as f:
        archive={(int(r["pair"]),r["model"]):r for r in csv.DictReader(f)}
    rows=[]
    with np.load(ROOT/"results"/"models.npz") as models, threadpool_limits(1):
        for pair in range(1,17):
            x=draw(131000+pair,32768)
            tie=np.random.default_rng(132000+pair).random(len(x))
            r=np.linalg.norm(x,axis=1)
            odds=(.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(1.2*r/.15**2))-np.log(np.i0(.5*r/.15**2))
            p=np.exp(-np.logaddexp(0,-odds))
            for name in ("gd","pb","hybrid"):
                old=archive[pair,name]
                assert old["eval_hash"]==sha(x) and int(old["tie_seed"])==132000+pair
                v=models[f"{pair}_{name}"]
                a=representation(v,x)[:,0]
                b=representation(v,-x)[:,0]
                m=(a+b)/2
                ma,mb=choose(a,tie).astype(int),choose(b,tie).astype(int)
                assert np.all(ma.reshape(-1,256).sum(axis=1)==64) and np.all(mb.reshape(-1,256).sum(axis=1)==64)
                ce=lambda h: np.logaddexp(0,h)-p*h
                la,lb,lm=ce(a),ce(b),ce(m)
                raw=(la+lb)/2
                chosen=(np.where(ma,lm,la)+np.where(mb,lm,lb))/2
                d=(np.logaddexp(0,a)+np.logaddexp(0,b))/2-np.logaddexp(0,m)
                sym=(ma+mb)*d/2
                signed=(ma-mb)*(la-lb)/4
                max_error=float(np.max(abs((raw-chosen)-(sym+signed))))
                assert max_error<1e-11
                assert max(abs(raw.mean()-float(old["raw_loss"])),abs(chosen.mean()-float(old["confidence_loss"])),
                           abs(lm.mean()-float(old["full_loss"])),abs((ma!=mb).mean()-float(old["confidence_orientation_mask_disagreement"])))<1e-12
                values={
                    "raw_to_confidence":raw-chosen, "symmetric_gain":sym, "orientation_term":signed,
                    "raw_to_full":d, "unused_symmetric_gain":d-sym, "confidence_gain_over_full":lm-chosen,
                    "mask_disagreement":ma!=mb}
                assert abs((lm-chosen).mean()-(signed-(d-sym)).mean())<1e-12
                rows.append(dict(pair=pair,model=name, **{k:float(v.mean()) for k,v in values.items()},
                                 max_pointwise_identity_error=max_error))
    save(OUT/"confidence_terms_models.csv",rows)
    groups=[]
    for name in ("gd","pb","hybrid"):
        items=[r for r in rows if r["model"]==name]
        groups.append(dict(model=name, **{k:float(np.mean([r[k] for r in items])) for k in FIELDS},
                           positive_signed_pairs=sum(r["orientation_term"]>0 for r in items),
                           beats_full_pairs=sum(r["confidence_gain_over_full"]>0 for r in items)))
    save(OUT/"confidence_terms_summary.csv",groups)
    for g in groups:
        print(g)
    print("Max pointwise residual:",max(r["max_pointwise_identity_error"] for r in rows))


if __name__=="__main__":
    main()
