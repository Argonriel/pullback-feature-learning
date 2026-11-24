"""Portable replay of the original old-only checks, using archived inputs only."""
from pathlib import Path
import sys,json
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e

def main():
    history=json.loads((ROOT/'old_pilot'/'historical_defaults.json').read_text());curves=np.load(ROOT/'old_pilot'/'historical_gain_curves.npz')
    for method in e.CONFIG['models']:
        assert curves[method].shape==(16,64)
        np.testing.assert_allclose(curves[method].mean(0),history['mean_gain_curves'][method],rtol=0,atol=1e-15)
        assert curves[method].mean(0).argmax()+1==e.CONFIG['default_index'][method]
    bankpath=ROOT.parent/'legacy'/'budgeted-rotation-final'/'results'/'models.npz'
    assert e.sha(bankpath.read_bytes())==history['old_model_bank_sha256']
    bank=np.load(bankpath);records=[]
    for seed in (1,2):
        rng=np.random.default_rng(734000+seed);r=np.linalg.norm(e.sample(734000+seed,512),axis=1);phase=rng.uniform(0,2*np.pi,512)
        angles=phase[:,None]+np.arange(128)*2*np.pi/128;x=(r[:,None,None]*np.stack([np.cos(angles),np.sin(angles)],axis=-1)).reshape(-1,2)
        for method in e.CONFIG['models']:
            h=e.predict(bank[f'{seed}_{method}'],x).reshape(512,128)
            for budget in e.CONFIG['budgets']:
                n=budget//128;t=e.fit(h[:n],r[:n],phase[:n],method);records.append(dict(seed=seed,method=method,budget=budget,min_radius_count=int(t['radius_counts'].min()),min_folded_independent_radii=int(t['cell_independent_radii'].min()),global_angle=float(t['global_index'])*360/128,radial_unique_angles=len(np.unique(t['radial_index'])),minimum_gain=float(t['curves'].min())))
    old=e.ci(ROOT/'old_pilot'/'checks.csv');assert len(old)==len(records)
    for a,b in zip(old,records):
        for key,val in b.items():
            if key=='method':assert a[key]==val
            else:assert abs(float(a[key])-float(val))<1e-13
    print(json.dumps(dict(historical_default_indices=e.CONFIG['default_index'],historical_model_curves=48,old_pilot_rows_reproduced=len(records),old_model_hash_pass=True,external_dependencies=False)))
if __name__=='__main__':
    with threadpool_limits(1):main()
