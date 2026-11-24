"""Freeze and run the bounded global-rank certificate verification."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,traceback
import numpy as np
from threadpoolctl import threadpool_limits
import local_rank as local
import known_cdf as cdf

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
CONFIG=dict(sub=8,angular_cells={'rings':256,'waveform':512},inner=64,
    cdf_bins=131072,beta_nodes=262145,n_models=16,
    empirical_n=1048576,empirical_alpha=.05/64,
    empirical_seed_base={'rings':1013000,'waveform':1053000})


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)


def freeze():
    files=[HERE/name for name in ['known_cdf.py','local_rank.py','experiment.py','checks.py','THEORY.md','PROTOCOL.md']]
    files += [ROOT/'reference/local_jets.py',ROOT/'reference/rank_policy.py',
              ROOT/'source/pullback_feature_learning/core.py',ROOT/'transfer/task.py']
    assert not(ROOT/'validation/states').exists()
    write_json(HERE/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        pilot_hashes={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'pilot').glob('*') if p.is_file()},
        fresh_models_accessed=False,fresh_test_outcomes_accessed=False,
        fresh_state_directory_absent=True))


def verify():
    obj=json.loads((HERE/'freeze.json').read_text());assert obj['config']==CONFIG
    for relative,digest in obj['source_hashes'].items():assert sha(ROOT/relative)==digest,relative
    return obj


def run(task,kind):
    verify();approval=json.loads((ROOT/'validation/ROOT_APPROVAL.json').read_text())
    assert approval['fresh_training_authorized'] is True
    commit_path=ROOT/'validation/policy_commit.json';commit=json.loads(commit_path.read_text())
    parent=HERE/('fresh' if kind=='known' else 'fresh_empirical');parent.mkdir(exist_ok=True)
    out=parent/task;out.mkdir(exist_ok=False)
    write_json(out/'START.json',dict(timestamp_utc=now(),freeze_sha256=sha(HERE/'freeze.json'),
        policy_commit_sha256=sha(commit_path),task=task,kind=kind))
    rows=[];failures=[]
    for seed in range(1,17):
        cohort='rings' if task=='rings' else 'transfer'
        path=ROOT/'validation/policies'/f'{cohort}_{seed}_pb_rank.npz'
        relative=path.relative_to(ROOT/'validation').as_posix()
        assert relative in commit['manifest'] and sha(path)==commit['manifest'][relative]
        table=dict(np.load(path));dist=None
        if kind=='empirical':
            draw_seed=CONFIG['empirical_seed_base'][task]+seed
            radii=(cdf.policy.sample_radii(draw_seed,CONFIG['empirical_n']) if task=='rings'
                   else local.waveform.sample_radii(draw_seed,CONFIG['empirical_n']))
            dist=cdf.empirical_distribution(table,radii,CONFIG['empirical_alpha'])
            dist['verification_seed']=np.array(draw_seed)
        try:
            value,dist=local.construct(table,task,sub=CONFIG['sub'],na=CONFIG['angular_cells'][task],
                inner=CONFIG['inner'],cdf_bins=CONFIG['cdf_bins'],distribution=dist)
            np.savez_compressed(out/f'{seed}_local.npz',**value)
            np.savez_compressed(out/f'{seed}_distribution.npz',**dist)
            row=dict(seed=seed,task=task,kind=kind,
                **{key:float(value[key]) for key in ['integrated_bound','uniform_bound','integrated_endpoint',
                    'integrated_angular','integrated_radial','interval_jet_propagations',
                    'coordinate_cdf_queries','node_seconds','total_seconds']})
            rows.append(row)
            write_json(out/f'{seed}_provenance.json',dict(timestamp_utc=now(),input_policy_sha256=sha(path),
                local_sha256=sha(out/f'{seed}_local.npz'),distribution_sha256=sha(out/f'{seed}_distribution.npz'),
                config=CONFIG,task=task,kind=kind,seed=seed))
            print(f'{task} {kind} {seed}/16: B={row["integrated_bound"]:.8g}, {row["total_seconds"]:.1f}s',flush=True)
        except Exception:
            failure=dict(seed=seed,task=task,kind=kind,timestamp_utc=now(),traceback=traceback.format_exc())
            failures.append(failure);write_json(out/f'{seed}_FAILURE.json',failure)
            print(failure['traceback'],flush=True)
    if rows:
        with (out/'summary.csv').open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'COMPLETE.json',dict(timestamp_utc=now(),task=task,kind=kind,successes=len(rows),
        failures=failures,all_models_retained=True,source_freeze_sha256=sha(HERE/'freeze.json'),
        interval_jet_propagations=sum(int(r['interval_jet_propagations']) for r in rows),
        coordinate_cdf_queries=sum(int(r['coordinate_cdf_queries']) for r in rows)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run'])
    parser.add_argument('--task',choices=['rings','waveform']);parser.add_argument('--kind',choices=['known','empirical'])
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.stage=='freeze':freeze()
        else:
            assert args.task and args.kind
            run(args.task,args.kind)
