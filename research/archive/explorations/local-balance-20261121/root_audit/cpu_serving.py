"""Equivalent fused CPU serving for selectors that use coordinates alone."""
from pathlib import Path
import csv, datetime, hashlib, importlib.util, json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'validation'))
import study as validation
NAMES=['raw','full','confidence','radius','folded','rank_balanced','rank_even','rank_unbalanced',
       'full_fused','radius_fused','folded_fused','rank_balanced_fused','rank_even_fused','rank_unbalanced_fused']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def freeze():
    with (HERE/'cpu_serving_freeze.json').open('x',encoding='utf8') as out:
        json.dump(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            code_sha256=digest(Path(__file__)),names=NAMES,warmups=3,repeats=7,batches=16,
            fresh_models_accessed=False,reason='Input-only allocation can precede the first network call; count one fused320 call versus full512, with identical masks and outputs. No policy change.'),out,indent=2)

def serve(theta,x,tables,name,ties,group_uniform,actions,new,calls=None):
    def pred(z):
        if calls is not None:calls.append(len(z))
        return validation.pred(theta,z)
    if name=='full_fused':
        h=pred(np.concatenate([x,-x]));return (h[:len(x)]+h[len(x):])/2,np.ones(len(x),bool)
    if name.endswith('_fused'):
        parent=name[:-6]
        if parent.startswith('rank_'):
            mask=new.select(x,None,tables[1],parent,ties,actions)
        else:
            mask=validation.ref.allocate(tables[2].priority(x,parent),ties)
        assert np.all(mask.reshape(-1,256).sum(1)==64)
        h=pred(np.concatenate([x,-x[mask]]));out=h[:len(x)].copy()
        out[mask]=(out[mask]+h[len(x):])/2
        return out,mask
    return validation.deploy(x,pred,tables,name,ties,group_uniform,actions,new)

def run():
    f=json.loads((HERE/'cpu_serving_freeze.json').read_text());assert digest(Path(__file__))==f['code_sha256']
    validation.verify();new=validation.new_module();records=[];checks=[]
    with threadpool_limits(1):
        for seed in range(1,17):
            theta=np.load(ROOT/'validation/states'/f'{seed}.npz')['pb'];tables=validation.load_tables(seed)
            x=validation.ref.sample(926000+seed,65536)
            tie=np.random.default_rng(927000+seed).random(len(x))
            group=np.random.default_rng(928000+seed).random(len(x)//4)
            action=np.random.default_rng(928100+seed).random(len(x))
            stored=dict(np.load(ROOT/'validation/predictions'/f'{seed}.npz'))
            for name in NAMES:
                parent=name[:-6] if name.endswith('_fused') else name
                index=stored['names'].tolist().index(parent)
                expected_mask=np.unpackbits(stored['masks'][index])[:len(x)].astype(bool)
                out,mask=serve(theta,x,tables,name,tie,group,action,new)
                error=float(np.max(abs(out-stored['outputs'][index])))
                assert np.array_equal(mask,expected_mask) and error<1e-11,(seed,name,error)
                calls=[];serve(theta,x[:256],tables,name,tie[:256],group[:64],action[:256],new,calls)
                desired=[512] if name=='full_fused' else ([320] if name.endswith('_fused') else ([256] if name=='raw' else ([256,256] if name=='full' else [256,64])))
                assert calls==desired,(name,calls)
                checks.append(dict(seed=seed,policy=name,mask_equal=True,max_logit_error=error,calls=json.dumps(calls)))
            rng=np.random.default_rng(949100+seed)
            for repeat in range(10):
                for name in rng.permutation(NAMES):
                    start=time.perf_counter_ns()
                    for batch in range(16):
                        sl=slice(batch*256,(batch+1)*256);gs=slice(batch*64,(batch+1)*64)
                        serve(theta,x[sl],tables,str(name),tie[sl],group[gs],action[sl],new)
                    ms=(time.perf_counter_ns()-start)/1e6/16
                    if repeat>=3:records.append(dict(seed=seed,policy=str(name),repeat=repeat-3,ms_per_batch=ms))
            print(f'Independent equivalent CPU serving {seed}/16',flush=True)
    for name,rows in [('cpu_serving_equivalence.csv',checks),('cpu_serving_latency.csv',records)]:
        with (HERE/name).open('x',newline='',encoding='utf8') as out:
            writer=csv.DictWriter(out,list(rows[0]));writer.writeheader();writer.writerows(rows)
    (HERE/'cpu_serving_complete.json').write_text(json.dumps(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        all_masks_identical=True,max_logit_error=max(r['max_logit_error'] for r in checks),n_checks=len(checks),n_timings=len(records)),indent=2),encoding='utf8')

if __name__=='__main__':freeze() if '--freeze' in sys.argv else run()
