"""Check sharp one-record sensitivities; no model/data files or fresh seeds."""
from itertools import product
from pathlib import Path
from datetime import datetime,timezone
import json,math
import numpy as np
HERE=Path(__file__).resolve().parent
def selected_sum(score,weights,keys,k):
    # Stable index as the final fallback for exactly equal score and key.
    index=np.arange(len(score));order=np.lexsort((index,keys,-np.asarray(score)))
    return float(np.sum(np.asarray(weights)[order[:k]]))
def main():
    failures=[];count=0;maximum=0.;alphabet=list(product((0.,1.),repeat=2))
    # Score and weight may be arbitrarily related; every binary pair is tested.
    for records in product(alphabet,repeat=4):
      s=np.array(records)[:,0];w=np.array(records)[:,1]
      for keys in (np.zeros(4),np.arange(4.),np.arange(4.)[::-1]):
       for k in (1,2,3):
        before=selected_sum(s,w,keys,k)
        for i in range(4):
         for sn,wn in alphabet:
          for key in (0.,.5,3.):
           ss=s.copy();ww=w.copy();kk=keys.copy();ss[i]=sn;ww[i]=wn;kk[i]=key
           change=abs(selected_sum(ss,ww,kk,k)-before);maximum=max(maximum,change);count+=1
           if change>1.:failures.append(dict(stage='exhaustive',change=change))
    rng=np.random.default_rng(1209001);relative_max=0.;random_max=0.;random_count=4096
    for _ in range(random_count):
        sa=rng.integers(0,5,256).astype(float);sb=rng.integers(0,5,256).astype(float)
        weights=rng.choice([0.,.25,.5,1.],256);keys=rng.integers(0,3,256).astype(float)
        olda=selected_sum(sa,weights,keys,64);oldb=selected_sum(sb,weights,keys,64)
        i=int(rng.integers(256));sa[i]=rng.integers(0,5);sb[i]=rng.integers(0,5)
        weights[i]=rng.choice([0.,.25,.5,1.]);keys[i]=rng.integers(0,3)
        newa=selected_sum(sa,weights,keys,64);newb=selected_sum(sb,weights,keys,64)
        absolute=abs(newa-olda);relative=abs((newa-newb)-(olda-oldb))
        random_max=max(random_max,absolute);relative_max=max(relative_max,relative)
        if absolute>1. or relative>2.:failures.append(dict(stage='random256',absolute=absolute,relative=relative))
    # Sharpness at the actual quota: 63 fixed winners, one contested place,
    # and 191 fixed losers. The changing record has weight 1 throughout.
    weights=np.r_[np.ones(1),np.zeros(255)];keys=np.arange(256.)
    sa=np.r_[0.,1.,np.full(63,3.),np.full(191,-1.)]
    sb=sa.copy();sb[0]=2.
    old=selected_sum(sa,weights,keys,64)-selected_sum(sb,weights,keys,64)
    sa[0]=2.;sb[0]=0.
    new=selected_sum(sa,weights,keys,64)-selected_sum(sb,weights,keys,64)
    assert old==-1 and new==1
    delta=.05/864
    penalties={str(n):dict(absolute_c1=math.sqrt(math.log(1/delta)/(2*256*n)),
                          relative_J1=math.sqrt(2*math.log(1/delta)/(256*n))) for n in (128,512,2048)}
    out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='PASS' if not failures else 'FAIL',
      independent_seed=1209001,exhaustive_single_record_replacements=count,exhaustive_max_change=maximum,
      random_full_size_replacements=random_count,random_absolute_max=random_max,random_relative_max=relative_max,
      sharp_relative_actual_quota=dict(before=old,after=new,difference=new-old),
      delta=delta,penalties=penalties,failures=failures,fresh_models_read=0)
    (HERE/'bounded_difference.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert not failures
if __name__=='__main__':main()
