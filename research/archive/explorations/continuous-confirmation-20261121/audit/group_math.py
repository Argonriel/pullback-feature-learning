"""Independent exact-quota identity and a genuinely continuous exact example."""
from pathlib import Path
import itertools,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    rng=np.random.default_rng(991101)
    u=rng.uniform(-1,1,(20480,4));q=.25+(u-u.mean(1,keepdims=True))/6
    assert np.min(q)>=0 and np.max(q)<=.5 and np.max(abs(q.sum(1)-1))<1e-14
    coins=rng.random(len(u));selected=(coins[:,None]>np.cumsum(q,axis=1)).sum(1)
    mask=np.eye(4,dtype=int)[selected];assert np.all(mask.reshape(-1,256).sum(1)==64)
    values=np.array([-.91,-.4,.1,.2,.78]);weights=np.array([.1,.2,.3,.15,.25]);average=values@weights
    errors=[]
    for v in values:
        actual=0.
        for inds in itertools.product(range(5),repeat=3):
            others=values[list(inds)];prob=np.prod(weights[list(inds)])
            actual+=prob*(.25+(v-np.mean(np.r_[v,others]))/6)
        expected=.25+(v-average)/8;errors.append(abs(actual-expected))
    # Smooth continuous model: R uniform on [.5,1.5], angle uniform on [0,2pi).
    # h=1+2R cos(phi); antipodal average=1. u=-cos(3phi), E[u d|R]=0 exactly.
    r=.5+(np.arange(512)+.5)/512;phi=2*np.pi*(np.arange(2048)+.5)/2048
    h=1+2*r[:,None]*np.cos(phi);m=np.ones_like(h);u=-np.cos(3*phi);d=h-m
    marginal=.25+u/8;c=np.logaddexp(0,h)-np.logaddexp(0,m)
    b=(marginal*d).mean(1);C=float((marginal*c).mean());random_C=float(c.mean()/4)
    worlds={'constant_half':np.full(512,.5),'increasing':np.clip(r-.5,0,1),'decreasing':np.clip(1.5-r,0,1),'step':(r>1).astype(float),'oscillating':.5+.5*np.sin(30*r)}
    gains={name:float((marginal*(c-p[:,None]*d)).mean()) for name,p in worlds.items()}
    assert max(abs(v-C) for v in gains.values())<1e-12 and np.max(abs(b))<1e-12
    assert C>random_C
    result={'status':'PASS','range_min':float(q.min()),'range_max':float(q.max()),'max_sum_error':float(np.max(abs(q.sum(1)-1))),
      'exact_64_batches':len(mask)//64,'max_enumerated_marginal_error':max(errors),
      'continuous_analytic_example':{'model':'h(r,phi)=1+2*r*cos(phi)','law':'R uniform [.5,1.5], angle uniform','u':'-cos(3phi)',
       'mean_gain':C,'uniform_quarter_gain':random_C,'increment_over_uniform':C-random_C,'max_abs_conditional_b':float(np.max(abs(b))),
       'hidden_world_gains':gains,'scope':'Exact analytic existence example, separate from trained neural-network experiments.'}}
    (ROOT/'audit/group_mathematics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
