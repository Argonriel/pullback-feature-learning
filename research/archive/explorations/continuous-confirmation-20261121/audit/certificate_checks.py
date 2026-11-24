"""Independent partition, quota and risk reconstruction for fresh certificates."""
from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'source'))
from pullback_feature_learning import core
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadjson(p):return json.loads((ROOT/p).read_text())
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf8')))
def sample(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);a=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0,.15,(n,2))
def lookup(x,z):
    y=x.copy();y[(y[:,1]<0)|((y[:,1]==0)&(y[:,0]<0))]*=-1;r=np.linalg.norm(y,axis=1);a=np.arctan2(y[:,1],y[:,0]);nr=int(z['root_nr']);na=int(z['root_na']);idx=np.minimum((r/3*nr).astype(int),nr-1)*na+np.minimum((a/np.pi*na).astype(int),na-1)
    if 'left' in z:
        active=z['left'][idx]>=0
        while active.any():
            nodes=idx[active];value=np.where(z['axis'][nodes]==0,r[active],a[active]);idx[active]=np.where(value<z['threshold'][nodes],z['left'][nodes],z['right'][nodes]);active=z['left'][idx]>=0
    return idx,r<=3
def select(v,t):
    mask=np.zeros(len(v),bool)
    for k in range(0,len(v),256):
        order=k+np.lexsort((t[k:k+256],-v[k:k+256]))[:64];mask[order[v[order]>1e-14]]=True
    return mask
def main():
    fr=loadjson('certificate/freeze.json');commit=loadjson('certificate/fresh/calibration/selector_commit.json')
    for p,h in fr['source_hashes'].items():assert sha(ROOT/'certificate'/p)==h
    for p,h in commit['npz_sha256'].items():assert sha(ROOT/'certificate/fresh/calibration'/p)==h
    raw=rows('certificate/fresh/results/by_model.csv');index={(int(r['pair']),float(r['epsilon']),r['method']):r for r in raw};maxerror=0.;maxviol=-np.inf
    methods={'base_independent':('base','independent'),'base_joint':('base','joint'),'uniform_independent':('uniform','independent'),'uniform_joint':('uniform','joint'),'adaptive_joint':('adaptive','joint')}
    for pair in range(1,17):
        t=np.load(ROOT/f'confirmation/states/original_{pair}.npz')['pb'];x=sample(8651000+pair,32768);ties=np.random.default_rng(8652000+pair).random(len(x));a=core.representation(t,x)[:,0];b=core.representation(t,-x)[:,0];m=(a+b)/2;rad=np.linalg.norm(x,axis=1);cos=x[:,0]/rad
        tables={name:dict(np.load(ROOT/f'certificate/fresh/calibration/{pair}_{name}.npz')) for name in ('base','uniform','adaptive')};lookups={}
        for name,z in tables.items():
            ids,inside=lookup(x,z);lookups[name]=(ids,inside);assert np.array_equal(ids,lookup(-x,z)[0]);bounds=z['intervals'][ids]
            flip=(x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0));aa=np.where(flip,b,a);bb=np.where(flip,a,b);violation=np.c_[bounds[:,0]-aa,aa-bounds[:,1],bounds[:,2]-bb,bb-bounds[:,3]][inside].max();maxviol=max(maxviol,float(violation));assert violation<1e-9
            if name=='adaptive':
                assert len(z['cells'])*2==65536 and z['leaf'].sum()==20480
                parents=np.flatnonzero(z['left']>=0);assert len(parents)==12288
                left=z['cells'][z['left'][parents]];right=z['cells'][z['right'][parents]];expect_left=z['cells'][parents].copy();expect_right=expect_left.copy();dims=z['axis'][parents]
                expect_left[np.arange(len(parents)),np.where(dims==0,1,3)]=z['threshold'][parents]
                expect_right[np.arange(len(parents)),np.where(dims==0,0,2)]=z['threshold'][parents]
                assert np.array_equal(left,expect_left) and np.array_equal(right,expect_right)
            samegrid=z['joint']-z['independent'];assert samegrid.min()>-1e-10
        J=(np.logaddexp(0,a)+np.logaddexp(0,b))/2-np.logaddexp(0,m)
        for ei,e in enumerate((0.,.1,.25,.5)):
            gain=J-e*cos*(a-b)/4
            for name in [*methods,'pointwise_oracle']:
                if name=='pointwise_oracle':score=J-e*abs(a-b)/4
                else:
                    grid,kind=methods[name];z=tables[grid];ids,inside=lookups[grid];score=np.where(inside,z[kind][ei,ids],-e*(2*abs(x[:,0])+2*float(z['residual_bound']))/4)
                mask=select(score,ties);assert mask.reshape(-1,256).sum(1).max()<=64;assert np.min(gain[mask],initial=0)>-1e-9;assert np.max(score-gain)<1e-8
                row=index[pair,e,name];error=abs(np.mean(gain*mask)-float(row['mean_gain']));maxerror=max(maxerror,error);assert error<3e-12;assert mask.mean()==float(row['selected_fraction'])
    result=dict(status='PASS',independently_reconstructed_rows=len(raw),max_gain_error=maxerror,max_logit_enclosure_violation=maxviol,checked_adaptive_partitions=16,checked_split_nodes=16*12288,charged_propagations_per_adaptive_model=65536)
    (ROOT/'audit/certificate.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
