"""Independent reconstruction of fresh outcomes, quota logs and chronology."""
from pathlib import Path
import csv, datetime, gzip, hashlib, json, sys
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;V=ROOT/'validation'
sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,str(ROOT/'legacy/budgeted-rotation-final'))
from pullback_feature_learning import core
from generator import rings

def sha(b):return hashlib.sha256(b).hexdigest()
def read_csv(p):
    with p.open(newline='',encoding='utf8') as stream:return list(csv.DictReader(stream))
def time_of(d):
    return datetime.datetime.fromisoformat(d.get('timestamp_utc',d.get('timestamp')))
def fresh_inputs(seed,n):
    rng=np.random.default_rng(926000+seed)
    rad=rng.choice(np.array([.5,1.2]),size=n)
    angle=rng.uniform(low=0,high=2*np.pi,size=n)
    return rad[:,None]*np.stack([np.cos(angle),np.sin(angle)],axis=1)+rng.normal(0,.15,size=(n,2))

def main():
    freezing=json.loads((V/'freeze.json').read_text())
    for p,digest in freezing['source_hashes'].items():assert sha((ROOT/p).read_bytes())==digest,p
    approval=json.loads((V/'ROOT_APPROVAL.json').read_text())
    training=json.loads((V/'training_commit.json').read_text())
    calibration=json.loads((V/'policy_commit.json').read_text())
    scoring=json.loads((V/'score_start.json').read_text())
    sequence=[json.loads((ROOT/'selection/FREEZE.json').read_text()),freezing,approval,training,calibration,scoring]
    assert all(time_of(a)<=time_of(b) for a,b in zip(sequence[:-1],sequence[1:]))
    assert not training['failures'] and training['n_successful']==16
    for name,digest in training['states'].items():assert sha((V/'states'/name).read_bytes())==digest
    for name,digest in calibration['manifest'].items():assert sha((V/name).read_bytes())==digest
    rows=read_csv(V/'per_seed.csv');index={(int(r['seed']),r['policy']):r for r in rows}
    samples={int(r['seed']):r for r in read_csv(V/'samples.csv')}
    old=dict(np.load(ROOT/'selection/development/models.npz'))
    old_hashes={sha(t.tobytes()) for t in old.values()}
    model_hashes=[];risk_error=0.;sample_risk_error=0.;checked=0;training_errors=[]
    for seed in range(1,17):
        saved=dict(np.load(V/'states'/f'{seed}.npz'))
        xtrain,ytrain=rings(921000+seed)
        assert np.array_equal(xtrain,saved['x']) and np.array_equal(ytrain,saved['y'])
        assert np.array_equal(core.initialize_parameters(922000+seed),saved['initial'])
        model_hashes.append(sha(saved['pb'].tobytes()));assert model_hashes[-1] not in old_hashes
        if seed in (1,8,16):
            state=saved['initial'].copy()
            for step in range(500):state+=8*core.direction(state,xtrain,ytrain,'pullback',damping=.1)[0]
            error=float(np.max(abs(state-saved['pb'])));assert error<1e-11
            training_errors.append(dict(seed=seed,max_parameter_error=error))
        x=fresh_inputs(seed,65536);assert sha(x.tobytes())==samples[seed]['x_sha256']
        radius=np.sqrt((x*x).sum(axis=1))
        logit=(.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(1.2*radius/.15**2))-np.log(np.i0(.5*radius/.15**2))
        prob=1/(1+np.exp(-logit))
        labels=(np.random.default_rng(928200+seed).random(len(x))<prob).astype(float)
        assert sha(labels.tobytes())==samples[seed]['labels_sha256']
        result=dict(np.load(V/'predictions'/f'{seed}.npz'))
        for j,name in enumerate(result['names'].tolist()):
            h=result['outputs'][j];mask=np.unpackbits(result['masks'][j])[:len(x)].astype(bool)
            expected=0 if name=='raw' else 256 if name in ('full','full_radial') else 64
            assert np.all(mask.reshape(-1,256).sum(1)==expected)
            rec=index[seed,name]
            assert sha(h.tobytes())==rec['output_sha256'] and sha(np.packbits(mask).tobytes())==rec['mask_sha256']
            er=float(np.mean(np.logaddexp(0,h)-prob*h))
            sr=float(np.mean(np.logaddexp(0,h)-labels*h))
            risk_error=max(risk_error,abs(er-float(rec['expected_risk'])))
            sample_risk_error=max(sample_risk_error,abs(sr-float(rec['sampled_label_risk'])))
            checked+=1
    assert len(set(model_hashes))==16 and risk_error<1e-12 and sample_risk_error<1e-12
    count=0;total=0
    with gzip.open(V/'deployment_audit.csv.gz','rt',newline='',encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            name=row['policy'];calls=json.loads(row['calls'])
            want=[256] if name=='raw' else [256,256] if name in ('full','full_radial') else [256,64]
            assert calls==want
            count+=1;total+=sum(calls)
    assert count==16*len(freezing['policies'])*256
    result=dict(chronology_pass=True,all_frozen_hashes_pass=True,all_calibration_hashes_pass=True,
        unique_fresh_models=16,overlap_with_64_development_models=0,training_replays=training_errors,
        risk_rows_reconstructed=checked,max_expected_risk_error=risk_error,max_sampled_risk_error=sample_risk_error,
        call_audits_checked=count,deployment_forward_examples=total,all_per_batch_quotas_pass=True)
    (HERE/'validation_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':
    with threadpool_limits(1):main()
