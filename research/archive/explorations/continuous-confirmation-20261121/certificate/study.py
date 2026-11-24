"""Tighter rectangle certificates and budget-accounted adaptive polar cells."""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'source'))
from pullback_feature_learning import core
from bounds import independent_bound,joint_bound,enclose,uniform_cells,lookup_nodes,jensen,objective
CONFIG=dict(epsilons=[0.,.1,.25,.5],primary_epsilon=.1,n_models=16,n_test=32768,n_mass=16384,
    base_grid=[128,128],uniform_grid=[128,256],adaptive_root_grid=[64,128],adaptive_rounds=24,adaptive_splits_per_round=512,
    adaptive_budget=65536,base_budget=32768,uniform_budget=65536,quota=64,batch=256,
    old_mass_seed=8640000,old_test_seed=8641000,old_tie_seed=8642000,
    fresh_mass_seed=8650000,fresh_test_seed=8651000,fresh_tie_seed=8652000,bootstrap_seed=8653000,
    primary_method='adaptive_joint',primary_min_coverage=.02,primary_min_oracle_gain_fraction=.5,
    additional_success='Positive paired gain improvement over base_independent with pointwise95%model-bootstrap lower limit above0',
    numerical_padding='1e-10*(1+max absolute endpoint); stationary residual >1e-7 triggers original-bound fallback',
    old_models='Previously inspected historical PB500',fresh_models='Locked original-ring PB500 states from confirmation; no ellipse claims')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def csvout(p,rows):
    with Path(p).open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def sig(x):return np.exp(-np.logaddexp(0,-x))
def soft(x):return np.logaddexp(0,x)

def sample(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);a=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0,.15,(n,2))

def model(cohort,pair):
    if cohort=='old':return np.load(ROOT.parent/'legacy/budgeted-rotation-final/results/models.npz')[f'{pair}_pb']
    return np.load(ROOT.parent/f'confirmation/states/original_{pair}.npz')['pb']

def mass(cells,radii):
    return (np.searchsorted(radii,cells[:,1],side='right')-np.searchsorted(radii,cells[:,0],side='left'))/len(radii)*(cells[:,3]-cells[:,2])/np.pi

def priority(cells,intervals,lower,radii):
    al,au,bl,bu=intervals.T
    jmax=np.maximum(jensen(al,bu),jensen(au,bl));dmin=np.maximum.reduce([al-bu,bl-au,np.zeros(len(al))])
    upper=jmax-CONFIG['primary_epsilon']*dmin/4
    return mass(cells,radii)*np.maximum(upper,0)*(lower<=1e-14)

def all_bounds(intervals):
    rows=[];diagnostics=[]
    for epsilon in CONFIG['epsilons']:
        value,meta=joint_bound(*intervals.T,epsilon);rows.append(value);diagnostics.append(dict(epsilon=epsilon,**meta))
    independent=np.array([independent_bound(*intervals.T,epsilon) for epsilon in CONFIG['epsilons']])
    return np.array(rows),independent,diagnostics

def adaptive(theta,radii):
    cells=uniform_cells(*CONFIG['adaptive_root_grid']);intervals=enclose(theta,cells,core);rootn=len(cells)
    lower,meta=joint_bound(*intervals.T,CONFIG['primary_epsilon']);cubic=meta['cubic_solves'];fallback=meta['fallback_cells'];maxres=meta['max_stationary_residual']
    left=np.full(rootn,-1,dtype=int);right=left.copy();axis=left.copy();threshold=np.zeros(rootn);leaf=np.ones(rootn,bool)
    priorities=priority(cells,intervals,lower,radii);splits=[]
    for round_id in range(CONFIG['adaptive_rounds']):
        ids=np.flatnonzero(leaf);chosen=ids[np.argsort(-priorities[ids],kind='stable')[:CONFIG['adaptive_splits_per_round']]]
        old=cells[chosen];dim=((old[:,1]-old[:,0])<.5*(old[:,0]+old[:,1])*(old[:,3]-old[:,2])).astype(int)
        mid=np.where(dim==0,.5*(old[:,0]+old[:,1]),.5*(old[:,2]+old[:,3]));low=old.copy();high=old.copy()
        low[np.arange(len(chosen)),np.where(dim==0,1,3)]=mid;high[np.arange(len(chosen)),np.where(dim==0,0,2)]=mid
        children=np.concatenate([low,high]);n=len(cells);k=len(chosen)
        left[chosen]=np.arange(n,n+k);right[chosen]=np.arange(n+k,n+2*k);axis[chosen]=dim;threshold[chosen]=mid;leaf[chosen]=False
        new_intervals=enclose(theta,children,core);new_lower,meta=joint_bound(*new_intervals.T,CONFIG['primary_epsilon'])
        cubic+=meta['cubic_solves'];fallback+=meta['fallback_cells'];maxres=max(maxres,meta['max_stationary_residual'])
        cells=np.concatenate([cells,children]);intervals=np.concatenate([intervals,new_intervals]);lower=np.r_[lower,new_lower]
        left=np.r_[left,np.full(2*k,-1,int)];right=np.r_[right,np.full(2*k,-1,int)];axis=np.r_[axis,np.full(2*k,-1,int)];threshold=np.r_[threshold,np.zeros(2*k)];leaf=np.r_[leaf,np.ones(2*k,bool)]
        priorities=np.r_[priorities,priority(children,new_intervals,new_lower,radii)]
        splits.append(dict(round=round_id+1,split_cells=k,positive_priority_splits=int((priorities[chosen]>0).sum()),nodes=len(cells),leaves=int(leaf.sum()),propagations=2*len(cells)))
    assert 2*len(cells)==CONFIG['adaptive_budget'] and leaf.sum()==20480
    return (dict(cells=cells,intervals=intervals,left=left,right=right,axis=axis,threshold=threshold,leaf=leaf,
        root_nr=CONFIG['adaptive_root_grid'][0],root_na=CONFIG['adaptive_root_grid'][1]),splits,
        dict(cubic_solves=cubic,fallback_cells=fallback,max_stationary_residual=maxres))

def freeze():
    if (ROOT/'freeze.json').exists():raise FileExistsError('Already frozen')
    protocol='''# Frozen certificate confirmation

The externally specified epsilon bound is not estimated from unlabeled scores. Fixed epsilon values are0,.1,.25,.5. The original half-turn posterior mixture and all16locked fresh original-ring PB500models are used for confirmation. Historical16PBmodels are development checks, not new model replications. No fresh model states or test outcomes were inspected before this freeze.

Compare original128x128 independent-Jensen/separation bound at32768interval propagations; tight joint rectangle minimum on that same grid; uniform128x256 with independent and joint bounds at65536propagations; adaptive joint at65536propagations. Adaptive starts64x128cells, then24batches of512binarysplits. Initial bound work counts16384propagations; eachsplit adds4, so total65536. Parents are charged;20480finalleaves are not misreported as32768freeleaves. All evaluated32768nodes are saved. Uniform controls use32768cells. Per-policy setup requirements are distinguished from shared physical study computations.

Primary adaptive priorities are estimated input mass times a conservative upper potential at epsilon.1, only for currently uncertifiedcells. Input mass uses16384independent unlabeled radii and the known uniform-orientation law. Split the larger radial versus mean-radius arc diameter; refine512highest-priority leaves at a time. No posterior or held-out gain affects partitions. The algorithm is fixed, not selected from confirmation outcomes.

Every executable gate ranks stored positive lower bounds, selecting atmost64of256. Unused budget is allowed. A pointwise lower-bound oracle queriesboth fresh logits beforeallocation and is charged2calls/input. First-view and chosen partner calls are explicitly counted; all calibration interval, polynomial, timing and lookup costs are recorded. Scores use both orientations only for offline expected-risk evaluation. The batchmask must be invariant under flipping any one input with tiesfixed.

Prespecified practicalsuccess at epsilon.1 requires cohortmean selectedfraction>=.02, gain>=.5times the positive two-call pointwise oracle gain, and a strictlypositive lowerendpoint of a pointwise paired95%model-bootstrap interval for gain improvementover originalbase_independent. Report all three evenif any fails. Success is a cohortaverage, not a permodel guarantee. Equal-cost uniform_joint is a mandatory control. No changedthreshold, extragrid or extraepsilon after outcomes.

Joint rectangle minima enumerate corners, a diagonalcandidate and every validpositive rootof both edge cubic equations. No non-diagonal interior minimumexists. Subtract1e-10times(1+maxabsoluteendpoint); stationaryresidual>1e-7or nearlyreal ambiguousroots fallbackto the originalbound. Preservefallbackcounts. This numerically checked floating-pointimplementation is not a formal directed-roundingproof system. Modelintervals are conservative residual-tanh enclosures; tighter objectives do not remove their dependencyoverestimation.

Freeze source hashes, thencalibrate partitionsandcommit them before evaluation. Use independent test32768inputs/model, matched across methods and epsilon. Bootstrap complete16modelpairs; policycounts/batchesare not extra trainedmodels. Retain everymethod/model/epsilon, numericalfailure and oldpilot. No original researchfilesare modified.
'''
    (ROOT/'PROTOCOL.md').write_text(protocol,encoding='utf8')
    dump(ROOT/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,source_hashes={p.name:sha(p) for p in (ROOT/'study.py',ROOT/'bounds.py',ROOT/'THEORY.md')},
        protocol_sha256=sha(ROOT/'PROTOCOL.md'),core_sha256=sha(core.__file__),fresh_outcomes_inspected=False))

def calibrate(cohort):
    fr=json.loads((ROOT/'freeze.json').read_text());assert fr['config']==CONFIG
    for name,h in fr['source_hashes'].items():assert sha(ROOT/name)==h
    out=ROOT/cohort/'calibration';out.mkdir(parents=True,exist_ok=False);records=[];splits=[];numerics=[]
    for pair in range(1,17):
        theta=model(cohort,pair);radii=np.sort(np.linalg.norm(sample(CONFIG[cohort+'_mass_seed']+pair,CONFIG['n_mass']),axis=1))
        residual=sum(.5*np.abs(b[0]).sum() for _,b,_ in core.unpack(theta))
        for gridname,grid in (('base',CONFIG['base_grid']),('uniform',CONFIG['uniform_grid'])):
            start=time.perf_counter();cells=uniform_cells(*grid);intervals=enclose(theta,cells,core);boundtime=time.perf_counter()-start
            start=time.perf_counter();joint,independent,diag=all_bounds(intervals);jointtime=time.perf_counter()-start
            np.savez_compressed(out/f'{pair}_{gridname}.npz',cells=cells,intervals=intervals,joint=joint,independent=independent,root_nr=grid[0],root_na=grid[1],residual_bound=residual,theta=theta)
            records.append(dict(pair=pair,grid=gridname,nodes=len(cells),leaves=len(cells),interval_propagations=2*len(cells),interval_seconds=boundtime,objective_seconds=jointtime,
                total_setup_seconds=boundtime+jointtime,coordinate_only_calibration_inputs=0,model_hash=hashlib.sha256(theta.tobytes()).hexdigest()))
            numerics.extend(dict(pair=pair,grid=gridname,stage='final_objectives',**r) for r in diag)
        start=time.perf_counter();table,trace,meta=adaptive(theta,radii);adapt_time=time.perf_counter()-start
        start=time.perf_counter();joint,independent,diag=all_bounds(table['intervals']);final_time=time.perf_counter()-start
        np.savez_compressed(out/f'{pair}_adaptive.npz',**table,joint=joint,independent=independent,residual_bound=residual,theta=theta,mass_radii=radii)
        records.append(dict(pair=pair,grid='adaptive',nodes=len(table['cells']),leaves=int(table['leaf'].sum()),interval_propagations=2*len(table['cells']),interval_seconds=adapt_time,objective_seconds=final_time,
            total_setup_seconds=adapt_time+final_time,coordinate_only_calibration_inputs=CONFIG['n_mass'],model_hash=hashlib.sha256(theta.tobytes()).hexdigest()))
        splits.extend(dict(pair=pair,**r) for r in trace);numerics.append(dict(pair=pair,grid='adaptive',stage='partition_priority',epsilon=.1,**meta));numerics.extend(dict(pair=pair,grid='adaptive',stage='final_objectives',**r) for r in diag)
        print('calibrate',cohort,pair,flush=True)
    csvout(out/'costs.csv',records);csvout(out/'adaptive_trace.csv',splits);csvout(out/'root_diagnostics.csv',numerics)
    dump(out/'selector_commit.json',dict(timestamp_utc=now(),freeze_sha256=sha(ROOT/'freeze.json'),source_hashes=fr['source_hashes'],npz_sha256={p.name:sha(p) for p in out.glob('*.npz')},heldout_evaluation_started=False))

def score_values(x,table,epsilon,kind):
    idx,inside=lookup_nodes(x,table);row=CONFIG['epsilons'].index(epsilon);value=table[kind][row,idx]
    return np.where(inside,value,-epsilon*(2*abs(x[:,0])+2*float(table['residual_bound']))/4)

def allocate(scores,ties):
    score=scores.reshape(-1,256);tie=ties.reshape(-1,256);order=np.lexsort((tie,-score),axis=1)[:,:64];mask=np.zeros_like(score,bool)
    np.put_along_axis(mask,order,np.take_along_axis(score,order,axis=1)>1e-14,axis=1)
    return mask.ravel()

def deploy(theta,x,table,epsilon,kind,ties,predict=None):
    predict=(lambda z:core.representation(theta,z)[:,0]) if predict is None else predict
    first=predict(x);mask=allocate(score_values(x,table,epsilon,kind),ties);result=first.copy()
    if mask.any():result[mask]=.5*(first[mask]+predict(-x[mask]))
    return result,mask

def evaluate(cohort):
    cal=ROOT/cohort/'calibration';commit=json.loads((cal/'selector_commit.json').read_text())
    for p,h in commit['npz_sha256'].items():assert sha(cal/p)==h
    out=ROOT/cohort/'results';out.mkdir(exist_ok=False);dump(out/'score_start.json',dict(timestamp_utc=now(),selector_commit_sha256=sha(cal/'selector_commit.json')))
    rows=[];batchrows=[];audits=[];timing=[];bound_checks=[]
    for pair in range(1,17):
        theta=model(cohort,pair);x=sample(CONFIG[cohort+'_test_seed']+pair,CONFIG['n_test']);ties=np.random.default_rng(CONFIG[cohort+'_tie_seed']+pair).random(len(x))
        a=core.representation(theta,x)[:,0];b=core.representation(theta,-x)[:,0];m=.5*(a+b);r=np.linalg.norm(x,axis=1);cos=x[:,0]/r
        p0=sig((.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(1.2*r/.15**2))-np.log(np.i0(.5*r/.15**2)))
        tables={k:np.load(cal/f'{pair}_{k}.npz') for k in ('base','uniform','adaptive')}
        methods={'base_independent':('base','independent'),'base_joint':('base','joint'),'uniform_independent':('uniform','independent'),'uniform_joint':('uniform','joint'),'adaptive_joint':('adaptive','joint')}
        canonical=np.where(((x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0))),-1.,1.);aa=np.where(canonical==1,a,b);bb=np.where(canonical==1,b,a)
        for name,t in tables.items():
            idx,inside=lookup_nodes(x,t);assert np.array_equal(idx,lookup_nodes(-x,t)[0]);bounds=t['intervals'][idx]
            violation=np.max(np.c_[bounds[:,0]-aa,aa-bounds[:,1],bounds[:,2]-bb,bb-bounds[:,3]][inside]);assert violation<1e-9
            bound_checks.append(dict(pair=pair,grid=name,max_logit_enclosure_violation=float(violation),outside_grid=int((~inside).sum())))
        for epsilon in CONFIG['epsilons']:
            p=(1-epsilon)*p0+epsilon*(1+cos)/2;q=(1-epsilon)*p0+epsilon*(1-cos)/2
            raw=.5*(soft(a)-p*a+soft(b)-q*b);gain=jensen(a,b)-(p-q)*(a-b)/4
            allmethods=list(methods)+['pointwise_oracle']
            for name in allmethods:
                if name=='pointwise_oracle':scores=objective(a,b,epsilon);calls_per_input=2.
                else:
                    grid,kind=methods[name];scores=score_values(x,tables[grid],epsilon,kind);assert np.min(gain-scores)>-1e-8
                mask=allocate(scores,ties);improve=mask*gain
                if name!='pointwise_oracle':calls_per_input=1+mask.mean()
                assert np.min(gain[mask],initial=0)>-1e-8
                rows.append(dict(pair=pair,epsilon=epsilon,method=name,mean_gain=float(improve.mean()),mean_risk=float((raw-improve).mean()),selected_fraction=float(mask.mean()),
                    calls_per_input=float(calls_per_input),mean_lower_bound=float((mask*scores).mean()),harmful_selected=int(np.sum(mask&(gain< -1e-10))),
                    max_bound_violation=float(np.max(scores-gain)),input_hash=hashlib.sha256(x.tobytes()).hexdigest()))
                counts=mask.reshape(-1,256).sum(1);assert counts.max()<=64
                for k,v in enumerate(improve.reshape(-1,256).mean(1)):batchrows.append(dict(pair=pair,epsilon=epsilon,method=name,batch=k,gain=float(v),selected=int(counts[k])))
                if name!='pointwise_oracle':
                    grid,kind=methods[name];calls=[]
                    def counted(z):calls.append(len(z));return core.representation(theta,z)[:,0]
                    result,selected=deploy(theta,x[:256],tables[grid],epsilon,kind,ties[:256],counted);expected=np.where(selected,m[:256],a[:256])
                    assert np.array_equal(selected,mask[:256]) and np.max(abs(result-expected))<1e-11 and calls==([256,int(selected.sum())] if selected.any() else [256])
                    audits.append(dict(pair=pair,epsilon=epsilon,method=name,calls=calls,selected=int(selected.sum()),output_error=float(np.max(abs(result-expected)))))
                    # Fixed small benchmark includes all lookup and allocation work.
                    for rep in range(3):
                        start=time.perf_counter_ns()
                        for k in range(8):deploy(theta,x[k*256:(k+1)*256],tables[grid],epsilon,kind,ties[k*256:(k+1)*256])
                        if rep:timing.append(dict(pair=pair,epsilon=epsilon,method=name,repeat=rep,ms_per_batch=(time.perf_counter_ns()-start)/8/1e6))
        print('evaluate',cohort,pair,flush=True)
    csvout(out/'by_model.csv',rows);csvout(out/'by_batch.csv',batchrows);csvout(out/'latency.csv',timing);csvout(out/'enclosure_checks.csv',bound_checks);dump(out/'deployment_checks.json',audits)
    dump(out/'score_complete.json',dict(timestamp_utc=now(),csv_sha256={p.name:sha(p) for p in out.glob('*.csv')}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','calibrate','evaluate']);ap.add_argument('--cohort',choices=['old','fresh']);args=ap.parse_args()
    with threadpool_limits(1):
        if args.action=='freeze':freeze()
        else:globals()[args.action](args.cohort)
