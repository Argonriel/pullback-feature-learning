"""Numerical spotchecks of the local rank enclosure; not formal interval proof."""
from pathlib import Path
import csv, json, sys
import numpy as np
from threadpoolctl import threadpool_limits
import local_rank as lr
import known_cdf as kc

HERE=Path(__file__).resolve().parent


def direct_moment(table,dist,qt,r,task,na=8192):
    phi=2*np.pi*(np.arange(na)+.371)/na
    x=r*np.c_[np.cos(phi),np.sin(phi)] if task=='rings' else lr.waveform.coordinates(r,phi)
    partner=-x if task=='rings' else lr.waveform.partner(x)
    net=lr.task_core(task)
    d=(net.representation(table['theta'],x)[:,0]-net.representation(table['theta'],partner)[:,0])/2
    edges=table['radii'];parent=int(np.clip(np.searchsorted(edges,r,side='right')-1,0,len(edges)-2))
    t=(r-edges[parent])/(edges[parent+1]-edges[parent]);location=np.interp(r,edges,table['gain_profile'])
    q=np.zeros(na)
    for anchor,aw in [(parent,1-t),(parent+1,t)]:
        for k,phase,mix in lr.weighted_pair(table,anchor):
            score=location+float(table['width'])*((k*phi/(2*np.pi)+phase)%1)
            lo,hi=kc.q_bracket(dist,score,qt);q+=aw*mix*(lo+hi)/2
    return float(np.mean(q*d)),2*na


def check(table,result,dist,task,seed):
    qt=kc.q_table();rng=np.random.default_rng(seed);grid=result['r']
    active=np.flatnonzero(~result['exact_even']);take=rng.choice(active,min(24,len(active)),replace=False)
    radii=grid[take]+np.diff(grid)[take]*rng.uniform(.1,.9,len(take))
    lo,hi=lr.pointwise_bound(result,radii);values=[];forwards=0;derivative_violations=0
    for index,r in enumerate(radii):
        value,n=direct_moment(table,dist,qt,r,task);values.append(value);forwards+=n
        step=min(1e-5,np.diff(grid)[take[index]]/100)
        left,n=direct_moment(table,dist,qt,r-step,task);forwards+=n
        right,n=direct_moment(table,dist,qt,r+step,task);forwards+=n
        derivative=(right-left)/(2*step);band=take[index]
        derivative_violations+=int(derivative<result['derivative_lo'][band]-1e-4 or derivative>result['derivative_hi'][band]+1e-4)
    values=np.array(values);slack=np.minimum(values-lo,hi-values)
    assert np.all(slack>=-1e-8),(task,slack.min())
    assert derivative_violations==0,(task,derivative_violations)
    # Actual differentiable network values at independently chosen points inside
    # polar boxes; all five jet components get finite-difference spotchecks.
    n=96;r0=rng.uniform(grid[0],grid[-1]-.002,n);r1=r0+.002
    p0=rng.uniform(-np.pi,np.pi,n);p1=p0+.004
    enclosure=lr.difference_jets(table['theta'],r0,r1,p0,p1,task)
    r=rng.uniform(r0,r1);phi=rng.uniform(p0,p1);eps=2e-5
    def dvalue(rr,pp):
        xx=rr[:,None]*np.c_[np.cos(pp),np.sin(pp)] if task=='rings' else lr.waveform.coordinates(rr,pp)
        yy=-xx if task=='rings' else lr.waveform.partner(xx);net=lr.task_core(task)
        return (net.representation(table['theta'],xx)[:,0]-net.representation(table['theta'],yy)[:,0])/2
    center=dvalue(r,phi);rl=dvalue(r-eps,phi);rh=dvalue(r+eps,phi)
    pl=dvalue(r,phi-eps);ph=dvalue(r,phi+eps)
    exact=[center,(rh-rl)/(2*eps),(rh-2*center+rl)/eps**2,
           (ph-pl)/(2*eps),(ph-2*center+pl)/eps**2]
    jet_violations=[]
    for dim,(v,(a,b)) in enumerate(zip(exact,enclosure)):
        tolerance=1e-3 if dim in [2,4] else 1e-7
        jet_violations.append(int(np.sum((v<a-tolerance)|(v>b+tolerance))))
    assert sum(jet_violations)==0,(task,jet_violations)
    # Interior CDF probabilities must be enclosed by coarser and finer known-law
    # partitions; this is a consistency check, not a substitute for the proof.
    score=np.linspace(float(table['gain_profile'].min()),float(table['gain_profile'].max()+table['width']),1025)
    fl,fh=kc.bracket(dist,score);assert np.all(fl<=fh+1e-12)
    waveform_error=None
    if task=='waveform':
        # Load the task's independent exact polynomial-integral evaluator.
        import importlib.util
        spec=importlib.util.spec_from_file_location('transfer_rank_check',HERE.parent/'transfer/rank_policy.py')
        other=importlib.util.module_from_spec(spec);spec.loader.exec_module(other)
        f=other.population_cdf(table,score)
        assert np.all(f>=fl-1e-10) and np.all(f<=fh+1e-10)
        waveform_error=float(np.max(np.maximum(fl-f,f-fh)))
    return dict(task=task,seed=seed,moment_spots=len(radii),minimum_moment_slack=float(slack.min()),
        derivative_spots=len(radii),derivative_violations=derivative_violations,
        interval_jet_spots=n,jet_violations=jet_violations,
        maximum_CDF_bracket_width=float((fh-fl).max()),
        waveform_exact_CDF_max_excess=waveform_error,
        ordinary_verification_forwards=forwards+10*n,
        interval_jet_propagations=2*n,all_checks_pass=True)


def main():
    rows=[]
    for task,seed,na in [('rings',1,256),('rings',8,256),('rings',16,256),('waveform',1,512)]:
        table_path=(HERE.parent.parent/'local-balance-20261121/validation/policies'/f'{seed}_new.npz'
                    if task=='rings' else HERE.parent/'transfer/pilot_policy/1_pb.npz')
        table=dict(np.load(table_path))
        result=dict(np.load(HERE/f'pilot/{task}{seed}_v2_sub8_na{na}_inner64.npz'))
        dist=kc.distribution(table,task,131072)
        rows.append(check(table,result,dist,task,9914000+seed))
        print(rows[-1],flush=True)
    (HERE/'pilot/checks.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(1):main()
