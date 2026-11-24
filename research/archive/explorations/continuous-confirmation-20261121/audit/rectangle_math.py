"""Independently minimize certificate edges by grids and golden search."""
from pathlib import Path
import importlib.util,json
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bounds',ROOT/'certificate/bounds.py');bounds=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounds)
def f(a,b,e):return .5*(np.logaddexp(0,a)+np.logaddexp(0,b))-np.logaddexp(0,(a+b)/2)-e*abs(a-b)/4
def main():
    rng=np.random.default_rng(991102);p=rng.uniform(-20,20,(1500,4));al=np.minimum(p[:,0],p[:,1]);au=np.maximum(p[:,0],p[:,1]);bl=np.minimum(p[:,2],p[:,3]);bu=np.maximum(p[:,2],p[:,3])
    fixed=np.concatenate([al,au,bl,bu]);lo=np.concatenate([bl,bl,al,al]);hi=np.concatenate([bu,bu,au,au]);rows=[]
    for e in (0.,.01,.1,.25,.5,.9):
        grid=lo[:,None]+(hi-lo)[:,None]*np.linspace(0,1,2049);values=f(fixed[:,None],grid,e);ix=values.argmin(1);inds=np.arange(len(fixed))
        l=grid[inds,np.maximum(ix-1,0)];r=grid[inds,np.minimum(ix+1,2048)];phi=(np.sqrt(5)-1)/2
        # Independent derivative-free search, not the cubic implementation.
        for _ in range(65):
            u=r-phi*(r-l);v=l+phi*(r-l);left=f(fixed,u,e)<=f(fixed,v,e)
            r=np.where(left,v,r);l=np.where(left,l,u)
        edge=np.minimum(values.min(1),f(fixed,(l+r)/2,e)).reshape(4,-1).min(0)
        edge=np.where(np.maximum(al,bl)<=np.minimum(au,bu),np.minimum(edge,0),edge)
        joint,diag=bounds.joint_bound(al,au,bl,bu,e);base=bounds.independent_bound(al,au,bl,bu,e)
        assert np.max(joint-edge)<1e-8 and np.max(edge-joint)<1e-6,(e,np.max(joint-edge),np.max(edge-joint))
        assert np.min(joint-base)>-1e-10
        rows.append(dict(epsilon=e,max_upper_violation=float(np.max(joint-edge)),max_gap_to_independent_min=float(np.max(edge-joint)),mean_tightening=float(np.mean(joint-base)),**diag))
    result={'status':'PASS','rectangles':len(al),'independent_method':'2049-point edge scan plus 65 golden-section refinements; all corners/diagonal retained','results':rows}
    (ROOT/'audit/rectangle_mathematics.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
