"""Prospective independent-batch verification. No labels or posterior access."""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.util, json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits
from bounds import BATCH, QUOTA, LOOKS, jensen_gap, jensen_bound, rings_tail_bound, empirical_bernstein_lower

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CONFIG=dict(n_pairs=16,cohorts=['rings','waveform','two_invariants'],batch=256,quota=64,
    looks=[128,512,2048],primary_batches=512,family_alpha=.05,family_events=432,
    clips={'rings':.25,'waveform':.5,'two_invariants':.5},
    input_seeds={'rings':1121000,'waveform':1123000,'two_invariants':1151000},
    tie_seeds={'rings':1122000,'waveform':1124000,'two_invariants':1152000},
    d_radii={'rings':3.,'waveform':1.,'two_invariants':1.4},rings_cutoff=3.,
    max_predict_chunk=8192,policy_names=['symmetric_jensen','folded'],no_labels=True,
    source_policy='Frozen before fresh model generation; no adaptive additional sampling.')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
def jo(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def co(path,rows):
    with Path(path).open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def freeze():
    assert not any((ROOT/'validation'/'states').glob('*.npz')), 'Fresh states already exist'
    dependencies=[Path(__file__),HERE/'bounds.py',HERE/'PROTOCOL.md',HERE/'LITERATURE_AND_PROOF.md',
        HERE/'check_bounds.py',HERE/'bounds_checks.json',HERE/'development_old32'/'choice.json',
        ROOT/'validation'/'adapters.py',ROOT/'method'/'baseline.py']
    # Include all source dependencies reached by the portable adapter.
    dependencies += list((ROOT/'source').rglob('*.py'))
    dependencies += list((ROOT/'legacy').rglob('*.py'))
    dependencies += list((ROOT/'reference').rglob('*.py'))
    dependencies += [ROOT/'task2d'/'task.py']
    jo(HERE/'FREEZE.json',dict(timestamp_utc=now(),config=CONFIG,
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies},
        no_fresh_state_files=not any((ROOT/'validation'/'states').glob('*.npz')),
        fresh_outcomes_accessed=False))


def verify():
    frozen=json.loads((HERE/'FREEZE.json').read_text())
    assert frozen['config']==CONFIG
    for relative,digest in frozen['source_hashes'].items():
        assert sha(ROOT/relative)==digest,relative
    return frozen


def get_all_strings(obj):
    if isinstance(obj,str):return [obj]
    if isinstance(obj,dict):return [v for item in obj.values() for v in get_all_strings(item)]
    if isinstance(obj,list):return [v for item in obj for v in get_all_strings(item)]
    return []


def run():
    verify()
    approval=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
    assert approval['certificate_freeze_sha256']==sha(HERE/'FREEZE.json')
    assert approval['fresh_verification_authorized'] is True
    commit_path=ROOT/'validation'/'policy_commit.json'
    training_path=ROOT/'validation'/'training_commit.json'
    committed=get_all_strings(json.loads(commit_path.read_text()))
    trained=get_all_strings(json.loads(training_path.read_text()))
    adapters=load('fresh_certificate_adapters',ROOT/'validation'/'adapters.py')
    policy=load('fresh_certificate_policy',ROOT/'method'/'baseline.py')
    out=HERE/'fresh';out.mkdir(exist_ok=False)
    jo(HERE/'run_start.json',dict(timestamp_utc=now(),freeze_sha256=sha(HERE/'FREEZE.json'),
        approval_sha256=sha(HERE/'ROOT_APPROVAL.json'),policy_commit_sha256=sha(commit_path)))
    rows=[];costs=[];audits=[];ledger=[];manifest={};start_all=time.perf_counter()
    for cohort in CONFIG['cohorts']:
        adapter=adapters.get_adapter(cohort)
        for pair in range(1,17):
            start=time.perf_counter()
            state_path=ROOT/'validation'/'states'/f'{cohort}_{pair}.npz'
            grid_path=ROOT/'validation'/'policies'/f'{cohort}_{pair}_grid.npz'
            folded_path=ROOT/'validation'/'policies'/f'{cohort}_{pair}_folded.npz'
            assert sha(state_path) in trained,state_path
            assert sha(grid_path) in committed and sha(folded_path) in committed
            theta=np.load(state_path)['pb'];grid=dict(np.load(grid_path));folded=dict(np.load(folded_path))
            input_seed=CONFIG['input_seeds'][cohort]+pair;tie_seed=CONFIG['tie_seeds'][cohort]+pair
            x=adapter.sample_inputs(input_seed,max(LOOKS)*BATCH)
            ties=np.random.default_rng(tie_seed).random(len(x))
            masks=[policy.select(adapter,x,grid,ties),policy.select_folded(adapter,x,folded,ties)]
            tx=adapter.partner(x)
            transformed=[policy.select(adapter,tx,grid,ties),policy.select_folded(adapter,tx,folded,ties)]
            for mask,other in zip(masks,transformed):
                assert np.array_equal(mask,other),'Half-turn mask identity failed'
                assert np.all(mask.reshape(-1,BATCH).sum(1)==QUOTA)
            # Also check a subset of independently transformed coordinates.
            mixed=np.where((np.arange(len(x))%3==0)[:,None],tx,x)
            assert np.array_equal(masks[0],policy.select(adapter,mixed,grid,ties))
            assert np.array_equal(masks[1],policy.select_folded(adapter,mixed,folded,ties))
            union=masks[0]|masks[1];selected=np.flatnonzero(union)
            first=np.zeros(len(x));second=np.zeros(len(x));calls=[]
            for offset in range(0,len(selected),CONFIG['max_predict_chunk']):
                ids=selected[offset:offset+CONFIG['max_predict_chunk']]
                first[ids]=adapter.predict(theta,x[ids]);calls.append(len(ids))
                second[ids]=adapter.predict(theta,tx[ids]);calls.append(len(ids))
            gaps=jensen_gap(first,second);clip=CONFIG['clips'][cohort]
            radius=np.linalg.norm(x,axis=1) if cohort=='rings' else None
            region=radius<=CONFIG['rings_cutoff'] if cohort=='rings' else np.ones(len(x),bool)
            L,C=adapter.network_constants(theta);R=CONFIG['d_radii'][cohort]
            d_bound=min(R+C,R*L);Jmax=jensen_bound(d_bound)
            tail=rings_tail_bound(C,CONFIG['rings_cutoff']) if cohort=='rings' else 0.
            assert np.all(gaps[region]<=Jmax+1e-12)
            batch={}
            for name,mask in zip(CONFIG['policy_names'],masks):
                batch[name+'_clipped']=(mask*np.minimum(gaps,clip)).reshape(-1,BATCH).mean(1)
                batch[name+'_unclipped']=(mask*gaps).reshape(-1,BATCH).mean(1)
            difference=masks[0].astype(float)-masks[1]
            batch['relative_truncated']=(difference*gaps*region).reshape(-1,BATCH).mean(1)
            batch['relative_unclipped']=(difference*gaps).reshape(-1,BATCH).mean(1)
            batch['union_count']=union.reshape(-1,BATCH).sum(1)
            batch['forward_examples']=2*batch['union_count']
            batch['outside_count']=(~region).reshape(-1,BATCH).sum(1)
            for b in range(max(LOOKS)):
                ledger.append(dict(cohort=cohort,pair=pair,batch=b,inputs=BATCH,candidate_selected=64,
                    folded_selected=64,union_selected=int(batch['union_count'][b]),
                    forward_examples=int(batch['forward_examples'][b]),outside_region=int(batch['outside_count'][b])))
            path=out/f'{cohort}_{pair}_batches.npz';np.savez_compressed(path,**batch)
            manifest[path.name]=sha(path)
            for look in LOOKS:
                for name in CONFIG['policy_names']+['candidate_minus_folded']:
                    relative=name=='candidate_minus_folded'
                    values=batch['relative_truncated' if relative else name+'_clipped'][:look]
                    width=Jmax/2 if relative else clip/4
                    bound=empirical_bernstein_lower(values,width,.05/432)
                    bound['lower']-=tail if relative else 0.
                    bound.update(cohort=cohort,pair=pair,claim=name,clip=0. if relative else clip,
                        tail_subtraction=tail if relative else 0.,
                        full_2048_unclipped_mean=float(batch['relative_unclipped' if relative else name+'_unclipped'].mean()),
                        actual_shared_forward_examples=int(batch['forward_examples'][:look].sum()),
                        standalone_absolute_forward_examples=0 if relative else 128*look,
                        coordinate_inputs=BATCH*look,positive=bool(bound['lower']>0.))
                    rows.append(bound)
            identity_error=0.
            p=.5+.4*np.tanh(np.square(x).sum(1)-1.)
            loss=lambda h:np.logaddexp(0.,h)-p*h
            average=.5*(first+second)
            for mask in masks:
                out_h=np.where(mask,average,first);out_t=np.where(mask,average,second)
                actual=.5*(loss(first)-loss(out_h)+loss(second)-loss(out_t))
                identity_error=max(identity_error,float(np.max(abs(actual-mask*gaps))))
            assert identity_error<1e-12
            assert sum(calls)==2*union.sum()==batch['forward_examples'].sum()
            assert all(64<=count<=128 for count in batch['union_count'])
            elapsed=time.perf_counter()-start
            costs.append(dict(cohort=cohort,pair=pair,input_seed=input_seed,tie_seed=tie_seed,
                independent_batches=max(LOOKS),coordinate_inputs=len(x),actual_forward_examples=sum(calls),
                actual_forward_invocations=len(calls),seconds=elapsed,C=C,L=L,d_bound=d_bound,
                gap_bound=Jmax,tail_upper=tail,input_sha256=array_sha(x),tie_sha256=array_sha(ties),
                model_sha256=sha(state_path),grid_sha256=sha(grid_path),folded_sha256=sha(folded_path)))
            audits.append(dict(cohort=cohort,pair=pair,full_T_masks_equal=True,mixed_T_masks_equal=True,
                quota_exact=True,actual_forward_count_exact=True,antithetic_identity_max_error=identity_error,
                candidate_mask_sha256=array_sha(masks[0]),folded_mask_sha256=array_sha(masks[1]),
                max_observed_gap=float(gaps.max()),deterministic_region_gap_bound=Jmax))
            print(f'Certified {cohort} pair {pair}/16; {sum(calls)} actual forwards',flush=True)
    co(HERE/'per_model_bounds.csv',rows);co(HERE/'costs.csv',costs);co(HERE/'audits.csv',audits)
    co(HERE/'batch_ledger.csv',ledger)
    jo(HERE/'completion.json',dict(timestamp_utc=now(),seconds=time.perf_counter()-start_all,
        n_models=len(costs),n_bounds=len(rows),n_batch_units=len(ledger),manifest=manifest,
        policy_commit_sha256=sha(commit_path),freeze_sha256=sha(HERE/'FREEZE.json'),
        outputs={p.name:sha(p) for p in [HERE/'per_model_bounds.csv',HERE/'costs.csv',HERE/'audits.csv',HERE/'batch_ledger.csv']}))


def analyze():
    verify()
    with (HERE/'per_model_bounds.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    summary=[];thresholds=[]
    for cohort in CONFIG['cohorts']:
        for name in CONFIG['policy_names']+['candidate_minus_folded']:
            group=[r for r in rows if r['cohort']==cohort and r['claim']==name]
            full={int(r['pair']):float(r['lower']) for r in group if int(r['n_batches'])==2048}
            for look in LOOKS:
                current=[r for r in group if int(r['n_batches'])==look]
                lower=np.array([float(r['lower']) for r in current])
                retention=[float(r['lower'])/full[int(r['pair'])] for r in current if full[int(r['pair'])]>0]
                recovery=[float(r['lower'])/float(r['full_2048_unclipped_mean']) for r in current if float(r['full_2048_unclipped_mean'])>0]
                summary.append(dict(cohort=cohort,claim=name,n_batches=look,n_models=len(current),
                    positive=int(np.sum(lower>0)),mean_lower=float(lower.mean()),min_lower=float(lower.min()),
                    max_lower=float(lower.max()),mean_unclipped_gain=float(np.mean([float(r['full_2048_unclipped_mean']) for r in current])),
                    mean_actual_shared_forwards=float(np.mean([int(r['actual_shared_forward_examples']) for r in current])),
                    mean_2048_lower_bound_retention=float(np.mean(retention)) if retention else '',
                    valid_retention_denominators=len(retention),
                    mean_lower_to_empirical_full_gain=float(np.mean(recovery)) if recovery else '',
                    valid_empirical_gain_denominators=len(recovery)))
            for pair in range(1,17):
                eligible=[int(r['n_batches']) for r in group if int(r['pair'])==pair and float(r['lower'])>0]
                thresholds.append(dict(cohort=cohort,claim=name,pair=pair,
                    first_positive_batches=min(eligible) if eligible else 'unreached'))
    co(HERE/'summary.csv',summary);co(HERE/'thresholds.csv',thresholds)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','analyze'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):globals()[args.stage]()
