"""Independent confirmation of fixed-quota angle policies on rings and ellipses.

All routing/table algorithms are imported unchanged from the previous frozen
factorial. Ellipse observations x=A z use the known conjugate action A R A^-1.
"""
from pathlib import Path
import sys,argparse,csv,gzip,hashlib,importlib.util,json,datetime,time,platform
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(STUDY/'source'))
from pullback_feature_learning import core
spec=importlib.util.spec_from_file_location('previous_fixed_quota',STUDY/'reference/experiment.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
CONFIG=dict(geometries={'original':[[1.,0.],[0.,1.]],'ellipse':[[1.5,0.],[0.,.7]]},
    n_pairs_per_geometry=16,models=['gd','pb','hybrid'],budgets=[8192,65536],
    train_seed_base=821000,init_seed_base=822000,orbit_radius_seed_base=823000,
    orbit_phase_seed_base=824000,iid_gate_seed_base=825000,test_seed_base=826000,
    tie_seed_base=827000,label_seed_base=828000,bootstrap_seed=829000,
    n_test=65536,batch=256,quota=64,orbit_angles=128,radius_bins=16,min_radius_count=4,
    calibration_radii={8192:64,65536:512},
    training={'gd_steps':500,'gd_rate':.5,'pb_steps':500,'pb_rate':8.,'damping':.1,
        'hybrid_pb_steps':100,'hybrid_gd_steps':400,'hybrid_gd_rate':2.},
    primary_method='pb',primary_budget=8192,
    primary_contrast=['confidence_radial','confidence_pi'],
    primary_family='Two geometries; report ordinary95% and Bonferroni97.5% bootstrap intervals.',
    bootstrap_repeats=20000,timing_repetitions=8,timing_batches=16,
    latent_information='All policies know A and may access z=A^-1 x; model itself trains on observed x.',
    risk_labels='Analytic latent-radius posterior only in evaluation; independent sampled Bernoulli labels also scored.',
    no_retuning=True,
    practical_cost_targets={'calibration_overhead_relative_to_deployment':[.1,.01],
        'forward_break_even_reference':'full_pi','latency_break_even_reference':'full_pi'})
POLICIES=['confidence_pi','confidence_global','confidence_radial','legacy_iid_folded_pi',
    'joint','raw','full_pi','full_radial']


def sha(v):return hashlib.sha256(v).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,v):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def csvout(p,rows):
    with p.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def csvin(p):return list(csv.DictReader(p.open(encoding='utf8')))
def predict(t,x):return core.representation(t,x)[:,0]
def gid(geometry,pair):return pair+(0 if geometry=='original' else 16)


def freeze():
    files=[Path(__file__),HERE/'PROTOCOL.md',STUDY/'reference/experiment.py',
        STUDY/'source/pullback_feature_learning/core.py',STUDY/'legacy/budgeted-rotation-final/gate.py',
        STUDY/'legacy/budgeted-rotation-final/generator.py']
    dump(HERE/'freeze.json',dict(timestamp=now(),config=CONFIG,policies=POLICIES,
        hashes={str(p.relative_to(STUDY)):sha(p.read_bytes()) for p in files},
        frozen_before_new_training=True))


def check_freeze():
    f=json.loads((HERE/'freeze.json').read_text())
    # JSON object keys turn the two integer budget keys into strings.
    assert f['config']==json.loads(json.dumps(CONFIG))
    for name,digest in f['hashes'].items():assert sha((STUDY/name).read_bytes())==digest,name


def advance(t,x,y,method,rate,n):
    for _ in range(n):t=t+rate*core.direction(t,x,y,method,damping=.1)[0]
    if not np.isfinite(t).all():raise ArithmeticError('Nonfinite trained state; do not silently drop it.')
    return t


def train():
    check_freeze();folder=HERE/'states';folder.mkdir(exist_ok=False)
    gs=importlib.util.spec_from_file_location('rings_generator',STUDY/'legacy/budgeted-rotation-final/generator.py')
    gen=importlib.util.module_from_spec(gs);gs.loader.exec_module(gen)
    records=[]
    for geometry,av in CONFIG['geometries'].items():
        A=np.array(av)
        for pair in range(1,17):
            index=gid(geometry,pair);start=time.perf_counter()
            z,y=gen.rings(821000+index);x=z@A.T;initial=core.initialize_parameters(822000+index)
            prefix=advance(initial,x,y,'pullback',8,100)
            models=dict(gd=advance(initial,x,y,'gd',.5,500),pb=advance(prefix,x,y,'pullback',8,400),
                hybrid=advance(prefix,x,y,'gd',2,400))
            elapsed=time.perf_counter()-start;path=folder/f'{geometry}_{pair}.npz'
            np.savez_compressed(path,x=x,z=z,y=y,initial=initial,prefix=prefix,A=A,**models)
            records.append(dict(geometry=geometry,pair=pair,global_id=index,train_seed=821000+index,
                init_seed=822000+index,z_hash=sha(z.tobytes()),x_hash=sha(x.tobytes()),y_hash=sha(y.tobytes()),
                initial_hash=sha(initial.tobytes()),pb100_hash=sha(prefix.tobytes()),state_file_hash=sha(path.read_bytes()),
                all_three_training_seconds=elapsed))
            print('trained',geometry,pair,flush=True)
    csvout(HERE/'training.csv',records)
    dump(HERE/'training_commit.json',dict(timestamp=now(),state_files={p.name:sha(p.read_bytes()) for p in folder.glob('*.npz')},
        frozen_protocol_hash=sha((HERE/'freeze.json').read_bytes())))


def calibrate():
    check_freeze();folder=HERE/'calibration';folder.mkdir(exist_ok=False)
    (folder/'policies').mkdir();records=[];manifest={}
    for geometry,av in CONFIG['geometries'].items():
        A=np.array(av)
        for pair in range(1,17):
            index=gid(geometry,pair);state=np.load(HERE/f'states/{geometry}_{pair}.npz')
            r=np.linalg.norm(ref.sample(823000+index,512),axis=1)
            phase=np.random.default_rng(824000+index).uniform(0,2*np.pi,512)
            angle=phase[:,None]+2*np.pi*np.arange(128)[None,:]/128
            z=(r[:,None,None]*np.stack((np.cos(angle),np.sin(angle)),axis=-1)).reshape(-1,2)
            x=z@A.T;independent=ref.sample(825000+index,32768)
            for model in CONFIG['models']:
                theta=state[model];start=time.perf_counter()
                h0=predict(theta,x[:8192]);first=time.perf_counter()
                h1=predict(theta,x[8192:]);last=time.perf_counter();h=np.r_[h0,h1].reshape(512,128)
                raw=folder/f'{geometry}_{pair}_{model}_orbit.npz'
                np.savez_compressed(raw,h=h,radii=r,phases=phase,theta=theta,A=A)
                manifest[raw.name]=sha(raw.read_bytes())
                for budget in CONFIG['budgets']:
                    n=budget//128;fitstart=time.perf_counter();table=ref.fit(h[:n],r[:n],phase[:n],model)
                    fitsec=time.perf_counter()-fitstart
                    dest=folder/f'policies/{geometry}_{pair}_{model}_{budget}.npz'
                    np.savez_compressed(dest,**table);manifest[str(dest.relative_to(folder))]=sha(dest.read_bytes())
                    records.append(dict(geometry=geometry,pair=pair,model=model,budget=budget,kind='orbit',
                        calibration_seconds=(first-start if budget==8192 else last-start)+fitsec,
                        forward_examples=budget,independent_inputs=n,orbit_nodes=128,
                        radius_seed=823000+index,phase_seed=824000+index,
                        min_radius_count=int(table['radius_counts'].min()),
                        min_folded_radius_count=int(table['cell_independent_radii'].min()),
                        policy_hash=manifest[str(dest.relative_to(folder))]))
                    calls=[]
                    def latent_predict(q):calls.append(len(q));return predict(theta,q@A.T)
                    gate_start=time.perf_counter();gate=ref.Gate.fit(independent[:budget//2],latent_predict)
                    gate_elapsed=time.perf_counter()-gate_start
                    assert calls==[budget//2,budget//2]
                    dest=folder/f'policies/{geometry}_{pair}_{model}_{budget}_iid.npz'
                    np.savez_compressed(dest,iid_edges=gate.edges,iid_radius_scores=gate.radius_scores,
                        iid_folded_scores=gate.folded_scores)
                    manifest[str(dest.relative_to(folder))]=sha(dest.read_bytes())
                    records.append(dict(geometry=geometry,pair=pair,model=model,budget=budget,kind='legacy_iid',
                        calibration_seconds=gate_elapsed,forward_examples=budget,independent_inputs=budget//2,
                        orbit_nodes=2,radius_seed=825000+index,phase_seed='',min_radius_count='',
                        min_folded_radius_count='',policy_hash=manifest[str(dest.relative_to(folder))]))
            print('calibrated',geometry,pair,flush=True)
    csvout(folder/'costs.csv',records)
    dump(folder/'policy_commit.json',dict(timestamp=now(),manifest=manifest,
        cost_hash=sha((folder/'costs.csv').read_bytes()),training_commit_hash=sha((HERE/'training_commit.json').read_bytes()),
        test_data_and_outcomes_not_accessed=True))


def deploy(x,pred,table,policy,tie,A,Ainv):
    first=pred(x)
    if policy=='raw':return first,np.zeros(len(x),dtype=bool),np.zeros(len(x))
    z=x@Ainv.T
    mask,alpha=ref.choose(z,first,table,policy,tie)
    out=first.copy()
    if mask.any():out[mask]=.5*(first[mask]+pred(ref.rotate(z[mask],alpha[mask])@A.T))
    return out,mask,alpha


def score():
    check_freeze();cal=HERE/'calibration';commit=json.loads((cal/'policy_commit.json').read_text())
    for name,digest in commit['manifest'].items():assert sha((cal/name).read_bytes())==digest
    out=HERE/'results';out.mkdir(exist_ok=False);(out/'test_inputs').mkdir()
    dump(out/'score_start.json',dict(timestamp=now(),policy_commit_hash=sha((cal/'policy_commit.json').read_bytes())))
    rows=[];batchrows=[];audits=[];timings=[];samples=[]
    for geometry,av in CONFIG['geometries'].items():
        A=np.array(av);Ainv=np.linalg.inv(A)
        for pair in range(1,17):
            index=gid(geometry,pair);state=np.load(HERE/f'states/{geometry}_{pair}.npz')
            z=ref.sample(826000+index,65536);x=z@A.T;p=ref.posterior(z)
            tie=np.random.default_rng(827000+index).random(len(z))
            y=(np.random.default_rng(828000+index).random(len(z))<p).astype(np.uint8)
            np.savez_compressed(out/f'test_inputs/{geometry}_{pair}.npz',z=z,p=p,y=y,tie=tie,A=A)
            samples.append(dict(geometry=geometry,pair=pair,global_id=index,test_seed=826000+index,
                tie_seed=827000+index,label_seed=828000+index,z_hash=sha(z.tobytes()),x_hash=sha(x.tobytes()),
                p_hash=sha(p.tobytes()),y_hash=sha(y.tobytes()),tie_hash=sha(tie.tobytes()),n=len(z)))
            for model in CONFIG['models']:
                theta=state[model];pred=lambda q:predict(theta,q)
                for budget in CONFIG['budgets']:
                    table=dict(np.load(cal/f'policies/{geometry}_{pair}_{model}_{budget}.npz'))
                    table.update(dict(np.load(cal/f'policies/{geometry}_{pair}_{model}_{budget}_iid.npz')))
                    batch=[dict(geometry=geometry,pair=pair,model=model,budget=budget,batch=b) for b in range(len(z)//256)]
                    masks={}
                    for policy in POLICIES:
                        start=time.perf_counter();output,mask,alpha=deploy(x,pred,table,policy,tie,A,Ainv)
                        elapsed=time.perf_counter()-start
                        expected=ref.nll(output,p);sampled=ref.nll(output,y)
                        eh=expected.reshape(-1,256).mean(1);sh=sampled.reshape(-1,256).mean(1)
                        for b in range(len(batch)):
                            batch[b][policy+'_expected']=eh[b];batch[b][policy+'_sampled']=sh[b]
                        mh=sha(np.packbits(mask).tobytes());masks[policy]=mh
                        rows.append(dict(geometry=geometry,pair=pair,model=model,budget=budget,policy=policy,
                            expected_risk=float(expected.mean()),sampled_risk=float(sampled.mean()),
                            first_evaluations=len(x),second_evaluations=int(mask.sum()),
                            deployed_calls_per_input=1+mask.mean(),mask_hash=mh,alpha_hash=sha(alpha.tobytes()),
                            output_hash=sha(output.tobytes()),test_evaluation_seconds=elapsed))
                        calls=[]
                        def counted(q):calls.append(len(q));return pred(q)
                        check,selected,aa=deploy(x[:256],counted,table,policy,tie[:256],A,Ainv)
                        required=[256] if policy=='raw' else ([256,256] if policy.startswith('full_') else [256,64])
                        error=float(np.max(abs(check-output[:256])));assert error<2e-12
                        assert calls==required and np.array_equal(selected,mask[:256])
                        aug=ref.rotate((x[:256]@Ainv.T)[selected],aa[selected])@A.T
                        radius_error=float(np.max(abs(np.linalg.norm(aug@Ainv.T,axis=1)-np.linalg.norm(z[:256][selected],axis=1)))) if selected.any() else 0.
                        assert radius_error<2e-12
                        audits.append(dict(geometry=geometry,pair=pair,model=model,budget=budget,policy=policy,
                            actual_calls=json.dumps(calls),selected=int(selected.sum()),max_output_error=error,
                            conjugated_radius_error=radius_error))
                    assert len({masks[n] for n in ('confidence_pi','confidence_global','confidence_radial')})==1
                    batchrows.extend(batch)
                    rng=np.random.default_rng(830000+index+budget)
                    for rep in range(8):
                        for policy in rng.permutation(POLICIES):
                            start=time.perf_counter_ns()
                            for j in range(16):deploy(x[j*256:(j+1)*256],pred,table,str(policy),tie[j*256:(j+1)*256],A,Ainv)
                            duration=(time.perf_counter_ns()-start)/16/1e6
                            if rep:timings.append(dict(geometry=geometry,pair=pair,model=model,budget=budget,
                                policy=str(policy),repeat=rep,ms_per_batch256=duration))
            print('scored',geometry,pair,flush=True)
    csvout(out/'per_seed.csv',rows);csvout(out/'samples.csv',samples);csvout(out/'deployment_audits.csv',audits)
    csvout(out/'latency.csv',timings)
    with gzip.open(out/'per_batch.csv.gz','wt',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(batchrows[0]));w.writeheader();w.writerows(batchrows)
    dump(out/'score_complete.json',dict(timestamp=now(),outcome_hash=sha((out/'per_seed.csv').read_bytes()),
        policy_immutability=all(sha((cal/p).read_bytes())==h for p,h in commit['manifest'].items()),
        n_seed_rows=len(rows),n_deployment_audits=len(audits),max_output_error=max(r['max_output_error'] for r in audits),
        max_conjugated_radius_error=max(r['conjugated_radius_error'] for r in audits),
        python=platform.python_version(),numpy=np.__version__,threads=threadpool_info()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','train','calibrate','score']);args=ap.parse_args()
    with threadpool_limits(limits=1):globals()[args.stage]()
