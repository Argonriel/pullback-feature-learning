"""Independent verification of the fresh-cohort empirical confirmation."""
from pathlib import Path
import sys,csv,json,hashlib,datetime
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'source'))
from pullback_feature_learning import core
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf8')))
def sha(x):return hashlib.sha256(x).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
errors={}
def same(a,b,name):
    e=float(np.max(abs(np.asarray(a)-np.asarray(b)),initial=0));assert e<3e-11,(name,e);errors[name]=max(errors.get(name,0),e)
def sample(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);phi=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.c_[np.cos(phi),np.sin(phi)]+rng.normal(0,.15,(n,2))
def rotate(z,a):return np.c_[z[:,0]*np.cos(a)-z[:,1]*np.sin(a),z[:,0]*np.sin(a)+z[:,1]*np.cos(a)]
def top(s,tie):
    mask=np.zeros(len(s),bool)
    for i in range(0,len(s),256):mask[i+np.lexsort((tie[i:i+256],-s[i:i+256]))[:64]]=True
    return mask
def predict(t,x):return core.representation(t,x)[:,0]
def main():
    freeze=read('confirmation/freeze.json')
    for p,h in freeze['hashes'].items():assert sha((ROOT/p).read_bytes())==h,p
    for p,h in read('confirmation/training_commit.json')['state_files'].items():assert sha((ROOT/'confirmation/states'/p).read_bytes())==h
    for p,h in read('confirmation/calibration/policy_commit.json')['manifest'].items():assert sha((ROOT/'confirmation/calibration'/p).read_bytes())==h
    chain=['confirmation/freeze.json','confirmation/training_commit.json','confirmation/calibration/policy_commit.json','confirmation/results/score_start.json','confirmation/results/score_complete.json']
    times=[datetime.datetime.fromisoformat(read(p)['timestamp']) for p in chain];assert times==sorted(times)
    records=rows('confirmation/results/per_seed.csv');idx={(r['geometry'],int(r['pair']),r['model'],int(r['budget']),r['policy']):r for r in records}
    for row in rows('confirmation/results/deployment_audits.csv'):
        p=row['policy'];assert json.loads(row['actual_calls'])==([256] if p=='raw' else [256,256] if p.startswith('full_') else [256,64]);assert float(row['conjugated_radius_error'])<2e-12
    for geom in ('original','ellipse'):
     for pair in range(1,17):
      for method in ('gd','pb','hybrid'):
       for budget in (8192,65536):
        assert len({idx[geom,pair,method,budget,p]['mask_hash'] for p in ('confidence_pi','confidence_global','confidence_radial')})==1
    for geom in ('original','ellipse'):
     for pair in (1,16):
      g=pair+(16 if geom=='ellipse' else 0);state=np.load(ROOT/f'confirmation/states/{geom}_{pair}.npz');A=state['A'];inv=np.linalg.inv(A)
      z=sample(826000+g,65536);x=z@A.T;zz=x@inv.T;rad=np.linalg.norm(z,axis=1)
      logodds=(.25-1.44)/.045+np.log(np.i0(1.2*rad/.0225))-np.log(np.i0(.5*rad/.0225));p=np.exp(-np.logaddexp(0,-logodds));ties=np.random.default_rng(827000+g).random(len(z));y=np.random.default_rng(828000+g).random(len(z))<p
      saved=np.load(ROOT/f'confirmation/results/test_inputs/{geom}_{pair}.npz');same(z,saved['z'],'test_inputs');same(p,saved['p'],'posterior');assert np.array_equal(y,saved['y'])
      for method in ('gd','pb','hybrid'):
       t=state[method];first=predict(t,x);raw=np.load(ROOT/f'confirmation/calibration/{geom}_{pair}_{method}_orbit.npz')
       same(raw['phases'],np.random.default_rng(824000+g).uniform(0,2*np.pi,512),'independent_phase');same(raw['radii'],np.linalg.norm(sample(823000+g,512),axis=1),'calibration_radii')
       for budget in (8192,65536):
        tab=dict(np.load(ROOT/f'confirmation/calibration/policies/{geom}_{pair}_{method}_{budget}.npz'));tab.update(np.load(ROOT/f'confirmation/calibration/policies/{geom}_{pair}_{method}_{budget}_iid.npz'))
        h=raw['h'][:budget//128];curves=np.array([(.5*(np.logaddexp(0,h)+np.logaddexp(0,np.roll(h,-j,1)))-np.logaddexp(0,(h+np.roll(h,-j,1))/2)).mean(1) for j in range(1,65)]).T
        same(curves.mean(0),tab['global_curve'],'calibration_global');bins=np.searchsorted(tab['edges'],raw['radii'][:len(h)],side='right')
        radial=np.array([curves[bins==b].mean(0) if (bins==b).sum()>=4 else curves.mean(0) for b in range(16)]);same(radial,tab['curves'],'calibration_radial')
        rb=np.searchsorted(tab['edges'],np.linalg.norm(zz,axis=1),side='right')
        for policy in ('confidence_pi','confidence_global','confidence_radial','legacy_iid_folded_pi','joint','raw','full_pi','full_radial'):
            if policy=='raw':mask=np.zeros(len(z),bool);a=np.zeros(len(z))
            elif policy=='legacy_iid_folded_pi':
                b=np.searchsorted(tab['iid_edges'],np.linalg.norm(zz,axis=1));cz=np.where(((zz[:,1]<0)|((zz[:,1]==0)&(zz[:,0]<0)))[:,None],-zz,zz);ab=np.minimum((8*(np.arctan2(cz[:,1],cz[:,0])%np.pi)/np.pi).astype(int),7);mask=top(tab['iid_folded_scores'][8*b+ab],ties);a=np.full(len(z),np.pi)
            else:
                ix=np.full(len(z),64) if policy.endswith('_pi') else np.full(len(z),int(tab['global_index'])) if policy.endswith('_global') else tab['radial_index'][rb];a=ix*2*np.pi/128
                if policy.startswith('full_'):mask=np.ones(len(z),bool)
                else:mask=top(-abs(first) if policy.startswith('confidence_') else tab['curves'][rb,ix-1],ties)
            out=first.copy();out[mask]=(first[mask]+predict(t,rotate(zz[mask],a[mask])@A.T))/2;record=idx[geom,pair,method,budget,policy]
            assert sha(np.packbits(mask).tobytes())==record['mask_hash'];same((np.logaddexp(0,out)-p*out).mean(),float(record['expected_risk']),'expected_risk');same((np.logaddexp(0,out)-y*out).mean(),float(record['sampled_risk']),'sampled_risk')
    rng=np.random.default_rng(991103);draw=rng.integers(0,16,(20000,16));contrasts=[]
    for geom in ('original','ellipse'):
        diff=np.array([float(idx[geom,s,'pb',8192,'confidence_radial']['expected_risk'])-float(idx[geom,s,'pb',8192,'confidence_pi']['expected_risk']) for s in range(1,17)])
        contrasts.append(dict(geometry=geom,mean_difference=float(diff.mean()),interval_975=np.quantile(diff[draw].mean(1),[.0125,.9875]).tolist(),wins=int((diff<0).sum())))
    result=dict(status='PASS',rows=len(records),independent_full_test_replays=192,checks=['All source/state/calibration hashes and chronology','All deployment call counts and confidence mask identities','Independent inputs, posterior, calibration curves, conjugated transformations, masks and test risks for four representative seed/geometry groups'],max_errors=errors,independent_primary_bootstrap=contrasts)
    (ROOT/'audit/confirmation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
