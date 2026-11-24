"""Display the inputs routed by the calibrated gates of the first paired draw."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gate import Gate, allocate
from run import sample_inputs, DECISION_RADIUS

def main():
    out=Path(__file__).resolve().parent/"results"
    x=sample_inputs(124001,32768)[:4096]
    tie=np.random.default_rng(125001).random(len(x)).reshape(-1,256)
    fig,axes=plt.subplots(1,3,figsize=(9.3,3.5),layout="constrained")
    with np.load(out/"calibrated_gates.npz") as tables:
        for ax,m,label in zip(axes,("gd","pb","hybrid"),("GD","Pullback","Pullback → GD")):
            gate=Gate(*(tables["1_"+m+"_"+v] for v in ("edges","radius_scores","folded_scores")))
            chosen=allocate(gate.priority(x,"folded").reshape(-1,256),tie,64).ravel()
            assert chosen.sum()==1024
            ax.scatter(x[~chosen,0],x[~chosen,1],s=2,color="#aab3bd",alpha=.45,edgecolor="none",rasterized=True)
            ax.scatter(x[chosen,0],x[chosen,1],s=3,color="#007b74",alpha=.7,edgecolor="none",rasterized=True)
            ax.add_patch(plt.Circle((0,0),DECISION_RADIUS,fill=False,linestyle="--",lw=1,color="#a25532"))
            ax.set(aspect="equal", xticks=[-1,0,1],yticks=[-1,0,1],xlim=(-1.67,1.67),ylim=(-1.67,1.67),title=label)
            ax.spines[["right","top"]].set_visible(False)
    fig.suptitle("Where each network spends its second views",fontsize=15)
    fig.supxlabel("Teal: selected inputs (25%)     Gray: single view     Dashed circle: Bayes decision boundary",fontsize=9)
    for suffix in ("png","svg"):
        fig.savefig(out/("routing_map."+suffix),dpi=180)
    plt.close(fig)

if __name__=="__main__":
    main()
