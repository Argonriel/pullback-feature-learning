"""Reconstruct every reported confidence bound from retained observations."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run():
    checks=[];used=[]
    for mode,folder in [('known','certificates'),('empirical','certificates_empirical')]:
        path=HERE/folder
        with (path/'per_model.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
        assert len(rows)==32
        for row in rows:
            cohort=row['cohort'];seed=int(row['seed']);n=int(row['n']);used.append((mode,cohort,int(row['input_seed'])))
            obs=dict(np.load(path/f'{cohort}_{seed}_observations.npz'))
            W=obs['W_lower'];B=obs['B'];Z=obs['Z'];assert len(Z)==n
            assert np.array_equal(Z,W-B/2)
            m=float(np.sum(Z)/n);variance=float(np.sum((Z-m)**2)/(n-1))
            delta=.05/(32 if mode=='known' else 64)
            span=float(row['range_upper'])-float(row['range_lower'])
            err=(2*variance*np.log(2/delta)/n)**.5+7*span*np.log(2/delta)/(3*(n-1))
            gain=m-err;assert abs(gain-float(row['gain_lower']))<1e-13
            assert abs(variance-float(row['sample_variance']))<1e-13
            assert (gain>0)==(row['positive']=='True')
            assert np.all(obs['W_upper']>=W-1e-12)
            assert np.max(B)<=float(row['uniform_bound'])+1e-9
            task='rings' if cohort=='rings' else 'waveform'
            bound_root=ROOT/'certificate'/('fresh' if mode=='known' else 'fresh_empirical')/task
            assert sha(bound_root/f'{seed}_local.npz')==row['local_sha256']
            assert sha(bound_root/f'{seed}_distribution.npz')==row['distribution_sha256']
            if mode=='empirical':
                d=dict(np.load(bound_root/f'{seed}_distribution.npz'))
                assert len(d['independent_verification_radii'])==1048576
                assert float(d['alpha'])==delta
                expected=(1013000 if cohort=='rings' else 1053000)+seed
                assert int(d['verification_seed'])==expected
                assert expected!=int(row['input_seed'])
            checks.append(dict(mode=mode,cohort=cohort,seed=seed,gain_lower=gain,all_passed=True))
    assert len(used)==len(set(used))
    (HERE/'certificate_checks.json').write_text(json.dumps(dict(models_per_family=32,
        families=2,family_confidence=.95,joint_two_family_confidence_not_claimed=True,
        checks=checks,all_passed=True),indent=2),encoding='utf8')
    print('Reconstructed all64 certificate rows and verified source/sample identities.')

if __name__=='__main__':run()
