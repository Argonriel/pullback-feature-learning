"""Prospective calibration, exact-quota deployment and independent risk scoring."""
from pathlib import Path
import argparse, csv, datetime, gzip, hashlib, importlib.util, json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(STUDY / 'source'))
sys.path.insert(0, str(STUDY / 'legacy' / 'budgeted-rotation-final'))
from pullback_feature_learning import core
from gate import Gate
import train

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

old = module('previous_continuous', STUDY / 'reference' / 'continuous_policy.py')
ref = module('previous_angle', STUDY / 'reference' / 'experiment.py')

CONFIG = dict(n_models=16, test_n=65536, batch=256, quota=64,
    new_cal_seed_base=923000, angle_radius_seed_base=923100, angle_phase_seed_base=924000,
    gate_seed_base=925000, test_seed_base=926000, tie_seed_base=927000,
    group_seed_base=928000, action_seed_base=928100, label_seed_base=928200,
    bootstrap_seed=928999, bootstrap_repeats=20000,
    continuous_forwards=129*512, gate_inputs=33024, angle_radii=64, angle_nodes=128,
    timing_warmup=2, timing_repeats=7, timing_batches=16,
    hidden_worlds={'half':.5, 'increasing_midpoint':.85, 'increasing_scale':.15},
    no_retuning=True)
CONTROLS = ['group_balanced', 'group_unbalanced', 'group_even', 'group_uniform',
    'group_confidence', 'confidence', 'radius', 'folded', 'confidence_radial',
    'raw', 'full', 'full_radial']

def new_module():
    return module('richer_selector', STUDY / 'selection' / 'policy.py')

def sha(v):
    return hashlib.sha256(v).hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def jo(path, obj):
    with path.open('x', encoding='utf8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def co(path, rows):
    with path.open('x', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

def ci(path):
    with path.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))

def freeze():
    new = new_module()
    files = [Path(__file__), HERE/'train.py', HERE/'PROTOCOL.md',
        STUDY/'selection'/'policy.py', STUDY/'selection'/'FREEZE.json',
        STUDY/'selection'/'FROZEN_CHOICE.json',
        STUDY/'reference'/'continuous_policy.py', STUDY/'reference'/'experiment.py',
        STUDY/'source'/'pullback_feature_learning'/'core.py',
        STUDY/'legacy'/'budgeted-rotation-final'/'gate.py',
        STUDY/'legacy'/'budgeted-rotation-final'/'generator.py']
    names = CONTROLS + list(new.POLICIES)
    assert len(names) == len(set(names))
    jo(HERE/'freeze.json', dict(timestamp_utc=now(), config=CONFIG, training=train.CONFIG,
        policies=names, timing_policies=names+['full_fused'], new_policies=list(new.POLICIES),
        primary_contrasts=[['rank_balanced', 'confidence'],
                           ['rank_balanced', 'rank_even']],
        source_hashes={str(p.relative_to(STUDY)):sha(p.read_bytes()) for p in files},
        fresh_training_started=False, fresh_test_outcomes_accessed=False))

def verify():
    frozen = json.loads((HERE/'freeze.json').read_text(encoding='utf8'))
    assert frozen['config'] == CONFIG
    for name, digest in frozen['source_hashes'].items():
        assert sha((STUDY/name).read_bytes()) == digest, name
    return frozen

def pred(theta, x):
    return core.representation(theta, x)[:, 0]

def measured_fit(function):
    original = core.representation
    calls = []
    def counted(theta, x):
        calls.append(len(x))
        return original(theta, x)
    core.representation = counted
    start = time.perf_counter()
    try:
        result = function()
    finally:
        elapsed = time.perf_counter() - start
        core.representation = original
    return result, elapsed, calls

def calibrate():
    frozen = verify(); new = new_module()
    training_commit = json.loads((HERE/'training_commit.json').read_text())
    assert training_commit['n_successful'] == 16 and not training_commit['failures']
    folder = HERE/'policies'; folder.mkdir(exist_ok=False)
    records, manifest, failures = [], {}, []
    for seed in range(1, 17):
        state_path = HERE/'states'/f'{seed}.npz'
        assert sha(state_path.read_bytes()) == training_commit['states'][state_path.name]
        theta = np.load(state_path)['pb']
        try:
            table, elapsed, calls = measured_fit(lambda: old.fit(theta))
            assert sum(calls) == CONFIG['continuous_forwards']
            continuous_path = folder/f'{seed}_continuous.npz'
            np.savez_compressed(continuous_path, **table)
            records.append(dict(seed=seed, kind='previous_continuous',
                forward_examples=sum(calls), forward_invocations=len(calls),
                seconds=elapsed, independent_inputs=0, coordinate_only_inputs=0, calibration_seed=0))
            table, elapsed, calls = measured_fit(lambda: new.calibrate(theta, 923000+seed))
            assert sum(calls) == 66048
            np.savez_compressed(folder/f'{seed}_new.npz', **table)
            records.append(dict(seed=seed, kind='new_selector', forward_examples=sum(calls),
                forward_invocations=len(calls), seconds=elapsed, independent_inputs=0,
                coordinate_only_inputs=131072, calibration_seed=923000+seed))
            x = ref.sample(925000+seed, 33024)
            gate, elapsed, calls = measured_fit(lambda: Gate.fit(x, lambda z:pred(theta,z)))
            assert calls == [33024, 33024]
            np.savez_compressed(folder/f'{seed}_gate.npz', edges=gate.edges,
                radius_scores=gate.radius_scores, folded_scores=gate.folded_scores)
            records.append(dict(seed=seed, kind='iid_gate', forward_examples=sum(calls),
                forward_invocations=len(calls), seconds=elapsed, independent_inputs=33024,
                coordinate_only_inputs=0, calibration_seed=925000+seed))
            radii = np.linalg.norm(ref.sample(923100+seed, 64), axis=1)
            phase = np.random.default_rng(924000+seed).uniform(0., 2*np.pi, 64)
            angles = phase[:,None] + 2*np.pi*np.arange(128)[None,:]/128
            orbit = (radii[:,None,None]*np.stack((np.cos(angles),np.sin(angles)),axis=-1)).reshape(-1,2)
            def angle_fit():
                h = pred(theta, orbit).reshape(64,128)
                table = ref.fit(h, radii, phase, 'pb')
                return dict(**table, calibration_h=h, calibration_radii=radii, calibration_phases=phase)
            table, elapsed, calls = measured_fit(angle_fit)
            assert calls == [8192]
            np.savez_compressed(folder/f'{seed}_angles.npz', **table)
            records.append(dict(seed=seed, kind='radius_angles', forward_examples=sum(calls),
                forward_invocations=len(calls), seconds=elapsed, independent_inputs=64,
                coordinate_only_inputs=0, calibration_seed=923100+seed))
            print(f'Calibrated fresh model {seed}/16', flush=True)
        except Exception as e:
            failures.append(dict(seed=seed, exception=type(e).__name__, message=str(e)))
            jo(HERE/f'calibration_failure_{seed}.json', failures[-1])
            raise
    manifest = {str(p.relative_to(HERE)):sha(p.read_bytes()) for p in folder.glob('*.npz')}
    co(HERE/'calibration.csv', records)
    jo(HERE/'policy_commit.json', dict(timestamp_utc=now(), manifest=manifest, failures=failures,
        freeze_sha256=sha((HERE/'freeze.json').read_bytes()),
        training_commit_sha256=sha((HERE/'training_commit.json').read_bytes()),
        calibration_sha256=sha((HERE/'calibration.csv').read_bytes()),
        test_outcomes_accessed=False, policies=frozen['policies']))

def load_tables(seed):
    continuous = dict(np.load(HERE/'policies'/f'{seed}_continuous.npz'))
    new_table = dict(np.load(HERE/'policies'/f'{seed}_new.npz'))
    g = dict(np.load(HERE/'policies'/f'{seed}_gate.npz'))
    gate = Gate(g['edges'],g['radius_scores'],g['folded_scores'])
    angles = dict(np.load(HERE/'policies'/f'{seed}_angles.npz'))
    return continuous, new_table, gate, angles

def deploy(x, predict, tables, name, ties, group_uniform, action_keys, new):
    continuous, new_table, gate, angles = tables
    if name == 'full_fused':
        both = predict(np.concatenate((x,-x),axis=0))
        return .5*(both[:len(x)]+both[len(x):]), np.ones(len(x),dtype=bool)
    first = predict(x)
    if name in new.POLICIES:
        mask = np.asarray(new.select(x, first, new_table, name, ties, action_keys),dtype=bool)
        second_x = -x[mask]
    elif name in ('confidence_radial','full_radial'):
        mask, alpha = ref.choose(x, first, angles, name, ties)
        second_x = ref.rotate(x[mask],alpha[mask])
    elif name in ('raw','full','confidence','radius','folded'):
        if name == 'raw':
            mask = np.zeros(len(x),dtype=bool)
        elif name == 'full':
            mask = np.ones(len(x),dtype=bool)
        else:
            priority = -np.abs(first) if name=='confidence' else gate.priority(x,name)
            mask = ref.allocate(priority,ties)
        second_x = -x[mask]
    else:
        mask = old.selection(x, first, continuous, gate, name, ties, group_uniform)
        second_x = -x[mask]
    assert mask.shape == (len(x),)
    expected = 0 if name=='raw' else (256 if name in ('full','full_radial') else 64)
    assert np.all(mask.reshape(-1,256).sum(axis=1)==expected), name
    result = first.copy()
    if mask.any():
        result[mask] = .5*(first[mask]+predict(second_x))
    return result, mask

def score():
    frozen = verify(); new = new_module()
    commit = json.loads((HERE/'policy_commit.json').read_text())
    for path, digest in commit['manifest'].items():
        assert sha((HERE/path).read_bytes()) == digest, path
    jo(HERE/'score_start.json',dict(timestamp_utc=now(),
        policy_commit_sha256=sha((HERE/'policy_commit.json').read_bytes())))
    folder = HERE/'predictions'; folder.mkdir(exist_ok=False)
    rows, audit, samples, batch_rows, timings, timing_audits = [], [], [], [], [], []
    for seed in range(1,17):
        theta = np.load(HERE/'states'/f'{seed}.npz')['pb']
        tables = load_tables(seed); names = frozen['policies']
        x = ref.sample(926000+seed,65536)
        p = ref.posterior(x)
        ties = np.random.default_rng(927000+seed).random(len(x))
        group_uniform = np.random.default_rng(928000+seed).random(len(x)//4)
        action_keys = np.random.default_rng(928100+seed).random(len(x))
        y = (np.random.default_rng(928200+seed).random(len(x))<p).astype(float)
        r = np.linalg.norm(x,axis=1)
        increasing = np.exp(-np.logaddexp(0., -(r-.85)/.15))
        worlds = dict(half=np.full(len(x),.5), increasing=increasing, decreasing=1-increasing)
        samples.append(dict(seed=seed,x_sha256=sha(x.tobytes()),ties_sha256=sha(ties.tobytes()),
            group_uniform_sha256=sha(group_uniform.tobytes()),action_keys_sha256=sha(action_keys.tobytes()),
            labels_sha256=sha(y.tobytes()),n_inputs=len(x),n_positive_labels=int(y.sum())))
        outputs, masks = [], []
        for name in names:
            output = np.empty(len(x)); selected = np.zeros(len(x),dtype=bool)
            started = time.perf_counter()
            for batch in range(len(x)//256):
                sl = slice(256*batch,256*(batch+1)); gs = slice(64*batch,64*(batch+1))
                calls = []
                def counted(z):
                    calls.append(len(z)); return pred(theta,z)
                out, mask = deploy(x[sl],counted,tables,name,ties[sl],group_uniform[gs],action_keys[sl],new)
                expected = [256] if name=='raw' else ([256,256] if name in ('full','full_radial') else [256,64])
                assert calls == expected,(seed,name,batch,calls)
                assert np.isfinite(out).all()
                output[sl] = out; selected[sl] = mask
                audit.append(dict(seed=seed,policy=name,batch=batch,calls=json.dumps(calls),
                    selected=int(mask.sum()),mask_sha256=sha(np.packbits(mask).tobytes())))
                batch_rows.append(dict(seed=seed,policy=name,batch=batch,
                    expected_risk=float(ref.nll(out,p[sl]).mean()),sampled_label_risk=float(ref.nll(out,y[sl]).mean())))
            row = dict(seed=seed,policy=name,expected_risk=float(ref.nll(output,p).mean()),
                sampled_label_risk=float(ref.nll(output,y).mean()),
                expected_classification_error=float(np.where(output>=0,1-p,p).mean()),
                sampled_classification_error=float(((output>=0)!=y).mean()),
                evaluation_seconds=time.perf_counter()-started,first_forwards=len(x),
                second_forwards=int(selected.sum()),calls_per_input=(len(x)+selected.sum())/len(x),
                mask_sha256=sha(np.packbits(selected).tobytes()),output_sha256=sha(output.tobytes()))
            for world, posterior in worlds.items():
                row[f'{world}_risk'] = float(ref.nll(output,posterior).mean())
            rows.append(row); outputs.append(output); masks.append(np.packbits(selected))
        np.savez_compressed(folder/f'{seed}.npz',names=np.array(names),outputs=np.array(outputs),masks=np.array(masks))
        fused_calls=[]
        def counted_fused(z):
            fused_calls.append(len(z));return pred(theta,z)
        fused_output,fused_mask=deploy(x[:256],counted_fused,tables,'full_fused',ties[:256],group_uniform[:64],action_keys[:256],new)
        fused_error=float(np.max(abs(fused_output-outputs[names.index('full')][:256])))
        assert fused_calls==[512] and fused_mask.all() and fused_error<1e-12
        timing_audits.append(dict(seed=seed,policy='full_fused',calls=json.dumps(fused_calls),
            selected=int(fused_mask.sum()),max_output_error=fused_error))
        timing_rng = np.random.default_rng(928300+seed)
        for repeat in range(CONFIG['timing_warmup']+CONFIG['timing_repeats']):
            for name in timing_rng.permutation(frozen['timing_policies']):
                started = time.perf_counter_ns()
                for batch in range(CONFIG['timing_batches']):
                    sl = slice(256*batch,256*(batch+1)); gs = slice(64*batch,64*(batch+1))
                    deploy(x[sl],lambda z:pred(theta,z),tables,str(name),ties[sl],group_uniform[gs],action_keys[sl],new)
                elapsed = (time.perf_counter_ns()-started)/1e6/CONFIG['timing_batches']
                if repeat>=CONFIG['timing_warmup']:
                    timings.append(dict(seed=seed,policy=str(name),repeat=repeat-CONFIG['timing_warmup'],ms_per_batch=elapsed))
        print(f'Scored fresh model {seed}/16',flush=True)
    co(HERE/'per_seed.csv',rows); co(HERE/'samples.csv',samples); co(HERE/'latency.csv',timings)
    co(HERE/'timing_reference_audit.csv',timing_audits)
    for filename, records in [('deployment_audit.csv.gz',audit),('per_batch.csv.gz',batch_rows)]:
        with gzip.open(HERE/filename,'xt',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)
    jo(HERE/'score_complete.json',dict(timestamp_utc=now(),n_models=16,n_policy_results=len(rows),
        n_batch_audits=len(audit),all_calls_and_quotas_pass=True,
        policy_commit_sha256=sha((HERE/'policy_commit.json').read_bytes()),
        outcomes_sha256=sha((HERE/'per_seed.csv').read_bytes()),threadpools=threadpool_info()))

def analyze():
    frozen = verify(); rows = ci(HERE/'per_seed.csv')
    index = {(r['policy'],int(r['seed'])):r for r in rows}
    rng=np.random.default_rng(928999);draw=rng.integers(0,16,(20000,16))
    outcomes=['expected_risk','sampled_label_risk','expected_classification_error',
              'sampled_classification_error','half_risk','increasing_risk','decreasing_risk']
    summary=[];contrasts=[]
    primary = [tuple(x) for x in frozen['primary_contrasts']]
    comparisons = primary + [(p,b) for p in frozen['new_policies']
        for b in ('group_balanced','group_even','folded','radius','confidence_radial','raw','full')
        if (p,b) not in primary]
    for outcome in outcomes:
        vectors={name:np.array([float(index[name,s][outcome]) for s in range(1,17)]) for name in frozen['policies']}
        for name,v in vectors.items():
            boot=v[draw].mean(axis=1)
            summary.append(dict(policy=name,outcome=outcome,mean=float(v.mean()),
                standard_deviation_across_models=float(v.std(ddof=1)),
                low95=float(np.quantile(boot,.025)),high95=float(np.quantile(boot,.975))))
        for candidate,baseline in comparisons:
            delta=vectors[candidate]-vectors[baseline];boot=delta[draw].mean(axis=1)
            contrasts.append(dict(policy=candidate,baseline=baseline,outcome=outcome,
                primary=outcome=='expected_risk' and (candidate,baseline) in primary,
                delta_risk=float(delta.mean()),low95=float(np.quantile(boot,.025)),
                high95=float(np.quantile(boot,.975)),low97_5=float(np.quantile(boot,.0125)),
                high97_5=float(np.quantile(boot,.9875)),wins=int((delta<0).sum()),n_models=16))
    co(HERE/'summary.csv',summary);co(HERE/'contrasts.csv',contrasts)
    latency=ci(HERE/'latency.csv');costs=ci(HERE/'calibration.csv');cost_rows=[]
    names=frozen['policies']
    for name in names:
        model_ms=np.array([np.median([float(z['ms_per_batch']) for z in latency if z['policy']==name and int(z['seed'])==s]) for s in range(1,17)])
        kind='new_selector' if name in frozen['new_policies'] else ('previous_continuous' if name in ('group_balanced','group_unbalanced','group_even') else ('iid_gate' if name in ('radius','folded') else ('radius_angles' if name in ('confidence_radial','full_radial') else None)))
        cal_seconds=np.array([next(float(z['seconds']) for z in costs if int(z['seed'])==s and z['kind']==kind) if kind else 0. for s in range(1,17)])
        cal_forwards=np.array([next(float(z['forward_examples']) for z in costs if int(z['seed'])==s and z['kind']==kind) if kind else 0. for s in range(1,17)])
        full_ms=np.array([min(np.median([float(z['ms_per_batch']) for z in latency if z['policy']==reference and int(z['seed'])==s]) for reference in ('full','full_fused')) for s in range(1,17)])
        saved=full_ms-model_ms
        for i in range(16):
            denominator=512-(256 if name=='raw' else (512 if name in ('full','full_radial') else 320))
            cost_rows.append(dict(seed=i+1,policy=name,resident_ms_per_batch=float(model_ms[i]),
                fastest_equivalent_full_ms=float(full_ms[i]),
                calibration_seconds=float(cal_seconds[i]),calibration_forwards=int(cal_forwards[i]),
                faster_than_full=bool(saved[i]>0),latency_break_even_batches=(int(np.ceil(1000*cal_seconds[i]/saved[i])) if saved[i]>0 else ''),
                forwards_break_even_batches=(int(np.ceil(cal_forwards[i]/denominator)) if denominator>0 else ''),
                deployment_examples_for_10pct_calibration_overhead=int(np.ceil(10*cal_seconds[i]*1000/model_ms[i]))*256))
    co(HERE/'costs.csv',cost_rows)
    jo(HERE/'analysis_complete.json',dict(timestamp_utc=now(),primary_contrasts=frozen['primary_contrasts'],
        bootstrap_unit='paired trained model',n_models=16,bootstrap_draws=20000))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','calibrate','score','analyze'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.stage]()
