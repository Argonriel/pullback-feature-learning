"""Prospective calibration-budget study with historical defaults and raw orbits.

Each stage saves immutable artifacts before the next. Population/test outcomes
are inaccessible to historical fitting, training, and new calibration choices.
"""
import argparse,csv,datetime,hashlib,importlib.util,json,platform,sys
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path.insert(0,str(STUDY/'source'))
sys.path.insert(0,str(STUDY/'legacy'))
from pullback_feature_learning import core
from rotation_selectors import orbit_logits,radii_sample,phi,rotate
MODELS=('gd','pb','hybrid')
BUDGETS=(0,128,512,2048,8192,65536)
RADII=(0,1,4,16,64,512)
CONFIG=dict(pairs=16,calibration_repeats=20,train_seed_base=721000,init_seed_base=722000,
    calibration_seed_formula='72300000 + 100*pair + repeat (repeat1..20)',test_seed_base=724000,
    budgets=list(BUDGETS),radii=list(RADII),n_angles=128,n_candidates=64,
    primary='none',secondary='shrink16',shrinkage_prior_radius_count=16,
    historical_cal_seed_base=603000,historical_models=16,historical_radii_per_model=512,
    historical_new_model_setup_forward_cost=0,
    training=dict(gd_steps=500,gd_rate=.5,pb_steps=500,pb_rate=8.,damping=.1,
        hybrid_pb_steps=100,hybrid_gd_steps=400,hybrid_gd_rate=2.),
    iid_n=32768,bootstrap_repeats=20000,bootstrap_seed=729001,
    quadrature_levels=[[192,512],[384,1024],[768,2048],[1536,4096]],
    quadrature_atol=2e-7,quadrature_rtol=2e-6,
    recovery='Ratio of cohort mean incremental gain to mean none/65536 minus default gain; only positive denominators.',
    threshold='First tested budget with recovery>=.9; undefined when denominator<=0; explicit probabilities at all budgets.',
    full_reference='Finite512-radius no-shrinkage calibration, NOT a population oracle.',
    uncertainty='Resample16model-pair clusters after averaging20calibration draws. Report within-model calibration SD separately.',
    policy_rule='Argmax64scores; exact ties choose smallest positive index. Complete uniform orbit cancels signed label term.',
    leakage='All20complete pools generated for research reuse; every budget choice uses only its permitted prefix.',
    no_new_outcome_tuning=True)


def sha(v):return hashlib.sha256(v).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def csvout(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def csvin(path):return list(csv.DictReader(path.open(encoding='utf8')))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def radius_gain_curves(h):
    convex=phi(h)
    return np.column_stack([(convex-phi(.5*(h+np.roll(h,-j,axis=1)))).mean(1) for j in range(1,65)])


def choose_from_prefix(radius_scores,prior,pseudo=0):
    """This function is passed ONLY the allowed radius prefix."""
    if len(radius_scores)==0:return int(prior.argmax()),prior.copy()
    score=(radius_scores.sum(0)+pseudo*prior)/(len(radius_scores)+pseudo)
    return int(score.argmax()),score


def history(retrospective_source):
    out=HERE/'history';out.mkdir(exist_ok=False)
    bank=np.load(STUDY/'legacy/budgeted-rotation-final/results/models.npz')
    curves=np.zeros((3,16,64));rows=[]
    for pair in range(1,17):
        r,phase=radii_sample(603000+pair,512)
        for mi,model in enumerate(MODELS):
            start=perf_counter();h=orbit_logits(bank[f'{pair}_{model}'],r,phase)
            curves[mi,pair-1]=radius_gain_curves(h).mean(0)
            rows.append(dict(pair=pair,model=model,cal_seed=603000+pair,
                state_hash=sha(bank[f'{pair}_{model}'].tobytes()),radius_phase_hash=sha(r.tobytes()+phase.tobytes()),
                forward_examples=65536,elapsed_seconds=perf_counter()-start))
    prior=curves.mean(1);indices=prior.argmax(1)
    np.savez_compressed(out/'historical_curves.npz',curves=curves,prior=prior,indices=indices)
    csvout(out/'historical_calibration.csv',rows)
    # Explicit retrospective check on already-inspected611/612 models, never called fresh confirmation.
    rr=csvin(retrospective_source/'risks.csv');cc=csvin(retrospective_source/'population_gain_curves.csv')
    risk=np.empty((3,16));angular=np.empty_like(risk);full=np.empty_like(risk);gains=np.empty((3,16,64));retro=[]
    for mi,model in enumerate(MODELS):
        for pair in range(1,17):
            q=[z for z in rr if z['model']==model and int(z['pair'])==pair]
            risk[mi,pair-1]=float(next(z['risk'] for z in q if z['policy']=='raw'))
            angular[mi,pair-1]=float(q[0]['angular'])
            full[mi,pair-1]=float(next(z['risk'] for z in q if z['policy']=='exact_global'))
            gains[mi,pair-1]=[float(z['gain']) for z in cc if z['model']==model and int(z['pair'])==pair]
        d=gains[mi,:,indices[mi]]
        retro.append(dict(model=model,status='RETROSPECTIVE_ALREADY_INSPECTED_MODELS',
            historical_angle_degrees=(indices[mi]+1)*360/128,
            default_risk=float((risk[mi]-d).mean()),raw_risk=float(risk[mi].mean()),
            default_angular_fraction=float(d.mean()/angular[mi].mean()),
            pi_angular_fraction=float(gains[mi,:,-1].mean()/angular[mi].mean()),
            full_angular_fraction=float((risk[mi]-full[mi]).mean()/angular[mi].mean()),
            full_beats_default=int(np.sum(full[mi]<risk[mi]-d-1e-12))))
    np.savez_compressed(out/'retrospective_inputs.npz',risk=risk,angular=angular,full=full,gains=gains)
    csvout(out/'retrospective_pilot.csv',retro)
    dump(out/'historical_commit.json',dict(timestamp=now(),historical_curves_hash=sha((out/'historical_curves.npz').read_bytes()),
        old_model_bank_hash=sha((STUDY/'legacy/budgeted-rotation-final/results/models.npz').read_bytes()),
        retrospective_source_hashes={p.name:sha(p.read_bytes()) for p in (retrospective_source/'risks.csv',retrospective_source/'population_gain_curves.csv')},
        defaults={m:dict(index=int(indices[i])+1,degrees=float((indices[i]+1)*360/128),historical_forward_examples=16*65536) for i,m in enumerate(MODELS)},
        retrospective_not_used_to_choose_defaults=True))
    print(retro,flush=True)


def freeze():
    path=HERE/'freeze.json';assert not path.exists()
    dump(path,dict(timestamp=now(),config=CONFIG,
        code_hashes={str(p.relative_to(STUDY)):sha(p.read_bytes()) for p in
            (Path(__file__),STUDY/'legacy/rotation_selectors.py',STUDY/'legacy/budgeted-rotation-final/generator.py',
             STUDY/'legacy/rings-mechanism-20261120/polar.py',STUDY/'source/pullback_feature_learning/core.py')},
        historical_commit_hash=sha((HERE/'history/historical_commit.json').read_bytes()),
        historical_curves_hash=sha((HERE/'history/historical_curves.npz').read_bytes()),
        fresh_training_not_started=True))


def check_freeze():
    f=json.loads((HERE/'freeze.json').read_text())
    assert f['config']==CONFIG
    for p,h in f['code_hashes'].items():assert sha((STUDY/p).read_bytes())==h,p
    assert sha((HERE/'history/historical_curves.npz').read_bytes())==f['historical_curves_hash']


def advance(t,x,y,method,rate,n):
    for _ in range(n):t=t+rate*core.direction(t,x,y,method,damping=.1)[0]
    assert np.isfinite(t).all();return t


def train():
    check_freeze();out=HERE/'fresh';out.mkdir(exist_ok=False);(out/'training').mkdir()
    rings=module('generator',STUDY/'legacy/budgeted-rotation-final/generator.py').rings
    bank={};records=[]
    for pair in range(1,17):
        start=perf_counter();x,y=rings(721000+pair);initial=core.initialize_parameters(722000+pair)
        prefix=advance(initial,x,y,'pullback',8,100)
        models=dict(gd=advance(initial,x,y,'gd',.5,500),pb=advance(prefix,x,y,'pullback',8,400),
            hybrid=advance(prefix,x,y,'gd',2,400))
        elapsed=perf_counter()-start
        path=out/f'training/{pair}_training_states.npz'
        np.savez_compressed(path,x=x,y=y,initial=initial,prefix=prefix,**models)
        bank.update({f'{pair}_{k}':v for k,v in dict(initial=initial,prefix=prefix,**models).items()})
        records.append(dict(pair=pair,train_seed=721000+pair,init_seed=722000+pair,
            train_hash=sha(x.tobytes()+y.tobytes()),initial_hash=sha(initial.tobytes()),
            state_file_hash=sha(path.read_bytes()),three_models_training_seconds=elapsed))
        print('train',pair,round(elapsed,3),flush=True)
    np.savez_compressed(out/'models.npz',**bank);csvout(out/'training.csv',records)
    dump(out/'training_commit.json',dict(timestamp=now(),model_bank_hash=sha((out/'models.npz').read_bytes()),
        state_files={p.name:sha(p.read_bytes()) for p in (out/'training').glob('*.npz')}))


def calibrate():
    check_freeze();out=HERE/'fresh';(out/'orbits').mkdir(exist_ok=False)
    bank=np.load(out/'models.npz');historical=np.load(HERE/'history/historical_curves.npz');prior=historical['prior']
    scores=np.zeros((16,3,20,5,64));choices=[];costs=[];pools=[]
    for pair in range(1,17):
        full=np.empty((3,20,512,128));rs=np.empty((20,512));phases=np.empty_like(rs)
        for repeat in range(1,21):
            seed=72300000+100*pair+repeat;r,phase=radii_sample(seed,512)
            rs[repeat-1]=r;phases[repeat-1]=phase
            pools.append(dict(pair=pair,repeat=repeat,cal_seed=seed,
                pool_hash=sha(r.tobytes()+phase.tobytes())))
            for mi,model in enumerate(MODELS):
                start=perf_counter();h=orbit_logits(bank[f'{pair}_{model}'],r,phase)
                forward_time=perf_counter()-start
                full[mi,repeat-1]=h
                start=perf_counter();radial=radius_gain_curves(h);score_time=perf_counter()-start
                for bi,(budget,n) in enumerate(zip(BUDGETS,RADII)):
                    for rule,pseudo in [('none',0),('shrink16',16)]:
                        index,curve=choose_from_prefix(radial[:n],prior[mi],pseudo)
                        if n and rule=='none':scores[pair-1,mi,repeat-1,bi-1]=curve
                        choices.append(dict(pair=pair,model=model,repeat=repeat,cal_seed=seed,
                            rule=rule,budget=budget,n_radii=n,candidate_index=index+1,
                            angle_degrees=(index+1)*360/128,calibration_score=float(curve[index]),
                            permitted_prefix_hash=sha(h[:n].tobytes()),
                            new_model_forward_budget=budget))
                costs.append(dict(pair=pair,model=model,repeat=repeat,forward_examples=65536,
                    full_forward_seconds=forward_time,all_candidate_score_seconds=score_time,
                    full_logit_hash=sha(h.tobytes())))
        np.savez_compressed(out/f'orbits/{pair}_calibration_orbits.npz',h=full,radii=rs,phases=phases)
        print('calibrate',pair,flush=True)
    np.savez_compressed(out/'prefix_score_curves.npz',scores=scores)
    csvout(out/'decisions.csv',choices);csvout(out/'calibration_pools.csv',pools);csvout(out/'calibration_costs.csv',costs)
    paths=[out/'decisions.csv',out/'prefix_score_curves.npz',out/'calibration_pools.csv',out/'calibration_costs.csv',*sorted((out/'orbits').glob('*.npz'))]
    dump(out/'policy_commit.json',dict(timestamp=now(),freeze_hash=sha((HERE/'freeze.json').read_bytes()),
        model_bank_hash=sha((out/'models.npz').read_bytes()),
        artifacts={str(p.relative_to(out)):sha(p.read_bytes()) for p in paths},
        no_new_population_risk_or_test_labels_accessed=True))


def population(theta):
    polar=module('polar',STUDY/'legacy/rings-mechanism-20261120/polar.py')
    previous=None
    for nr,na in CONFIG['quadrature_levels']:
        r,w0,w1,u=polar.polar_grid(nr,na);h=polar.logits_on_circles(theta,r,u)
        base=polar.integrate(h,r,w0,w1);convex=phi(h);mass=w0+w1
        gains=np.array([mass@(convex-phi(.5*(h+np.roll(h,-j*na//128,1)))).mean(1) for j in range(1,65)])
        vector=np.r_[base['population_loss'],base['angular_excess'],gains]
        if previous is not None:
            error=float(np.max(abs(vector-previous)))
            if error<=CONFIG['quadrature_atol']+CONFIG['quadrature_rtol']*max(base['population_loss'],gains.max()):break
        previous=vector
    else:raise ArithmeticError('Population integration failed to converge')
    assert gains.min()>-1e-10 and gains.max()<=base['angular_excess']+1e-9
    return gains,dict(**base,nr=nr,na=na,grid_error=error)


def evaluate():
    check_freeze();out=HERE/'fresh';commit=json.loads((out/'policy_commit.json').read_text())
    assert sha((out/'models.npz').read_bytes())==commit['model_bank_hash']
    for p,h in commit['artifacts'].items():assert sha((out/p).read_bytes())==h
    bank=np.load(out/'models.npz');popg=np.empty((16,3,64));rows=[];iid=[]
    polar=module('polar_eval',STUDY/'legacy/rings-mechanism-20261120/polar.py')
    for pair in range(1,17):
        rng=np.random.default_rng(724000+pair);n=32768;y=rng.integers(0,2,n)
        rr=np.where(y==0,.5,1.2);angle=rng.uniform(0,2*np.pi,n)
        x=rr[:,None]*np.c_[np.cos(angle),np.sin(angle)]+rng.normal(0,.15,(n,2))
        post=np.exp(-np.logaddexp(0,-polar.bayes_logit(np.linalg.norm(x,axis=1))))
        rand=rng.integers(0,128,n)*2*np.pi/128
        for mi,model in enumerate(MODELS):
            theta=bank[f'{pair}_{model}'];g,base=population(theta);popg[pair-1,mi]=g
            random_gain=(2*g.sum()-g[-1])/128
            rows.append(dict(pair=pair,model=model,**base,random_gain=random_gain,
                population_best_candidate_DIAGNOSTIC_ONLY=int(g.argmax())+1))
            h=core.representation(theta,x)[:,0]
            raw=np.logaddexp(0,h)-y*h;expected_raw=np.logaddexp(0,h)-post*h
            for j in range(0,66):
                if j==0:m=h;kind='raw'
                elif j==65:m=.5*(h+core.representation(theta,rotate(x,rand))[:,0]);kind='random_alpha'
                else:m=.5*(h+core.representation(theta,rotate(x,j*2*np.pi/128))[:,0]);kind='candidate'
                sampled=np.logaddexp(0,m)-y*m;expected=np.logaddexp(0,m)-post*m
                iid.append(dict(pair=pair,model=model,kind=kind,candidate_index=j,
                    sampled_risk=float(sampled.mean()),expected_risk=float(expected.mean()),
                    sampled_gain=float((raw-sampled).mean()),expected_gain=float((expected_raw-expected).mean()),
                    sampled_gain_se=float((raw-sampled).std(ddof=1)/np.sqrt(n)),
                    expected_gain_se=float((expected_raw-expected).std(ddof=1)/np.sqrt(n)),
                    test_seed=724000+pair,input_hash=sha(x.tobytes()),label_hash=sha(y.tobytes()),n=n))
            print('evaluate',pair,model,'grid',base['nr'],base['na'],base['grid_error'],flush=True)
    np.savez_compressed(out/'population_gain_curves.npz',gains=popg)
    csvout(out/'population_models.csv',rows);csvout(out/'iid_candidate_risks.csv',iid)
    dump(out/'evaluation_commit.json',dict(timestamp=now(),policy_commit_hash=sha((out/'policy_commit.json').read_bytes()),
        artifacts={p.name:sha(p.read_bytes()) for p in (out/'population_gain_curves.npz',out/'population_models.csv',out/'iid_candidate_risks.csv')},
        python=platform.python_version(),numpy=np.__version__,threads=threadpool_info()))


def benchmark():
    out=HERE/'fresh';bank=np.load(out/'models.npz');prior=np.load(HERE/'history/historical_curves.npz')['prior'];records=[]
    for pair in range(1,17):
        r,phase=radii_sample(72300000+100*pair+1,512)
        for mi,model in enumerate(MODELS):
            theta=bank[f'{pair}_{model}']
            for budget,n in zip(BUDGETS,RADII):
                start=perf_counter()
                if n:
                    h=orbit_logits(theta,r[:n],phase[:n]);scores=radius_gain_curves(h)
                else:scores=np.empty((0,64))
                choice,curve=choose_from_prefix(scores,prior[mi])
                records.append(dict(pair=pair,model=model,budget=budget,radii=n,
                    seconds=perf_counter()-start,candidate_index=choice+1,
                    budget_executed_forward_examples=budget))
    csvout(out/'standalone_budget_timing.csv',records)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['history','freeze','train','calibrate','evaluate','benchmark'])
    ap.add_argument('--retrospective-source',type=Path,default=HERE.parents[1]/'predictive-symmetry-final/direction2/fresh_results')
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.stage=='history':history(args.retrospective_source)
        else:globals()[args.stage]()
