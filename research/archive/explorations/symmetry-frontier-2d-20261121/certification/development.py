"""Old-only clipping development; no fresh model or risk access."""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.util, json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits
from bounds import BATCH, QUOTA, LOOKS, CLIPS, jensen_gap, jensen_bound, rings_tail_bound, statistics, empirical_bernstein_lower

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(old_root, output):
    output.mkdir(exist_ok=False)
    old = load('old_certificate_development_adapter', old_root/'validation'/'adapters.py')
    seed_bases = {'rings': 1101000, 'transfer': 1102000}
    # Six clipping candidates are exploratory. Fresh confirmation uses only frozen clips.
    delta = .05/(32*3*3)
    records = []; costs = []; audits = []
    start_all = time.perf_counter()
    for cohort in ('rings', 'transfer'):
        adapter = old.get_adapter(cohort)
        for pair in range(1, 17):
            start = time.perf_counter()
            state_path = old_root/'validation'/'states'/f'{cohort}_{pair}.npz'
            theta = np.load(state_path)['pb']
            table_path = old_root/'validation'/'policies'/f'{cohort}_{pair}_pb_rank.npz'
            folded_path = old_root/'validation'/'policies'/f'{cohort}_{pair}_pb_folded.npz'
            rank = dict(np.load(table_path)); folded = dict(np.load(folded_path))
            seed = seed_bases[cohort]+pair
            x = adapter.sample_inputs(seed, max(LOOKS)*BATCH)
            ties = np.random.default_rng(seed+100).random(len(x))
            keys = np.random.default_rng(seed+200).random(len(x))
            zero = np.zeros(len(x))
            candidate = adapter.rank.select(x,zero,rank,'rank_even',ties,keys)
            b = np.searchsorted(folded['edges'],adapter.quotient(x))
            baseline = old.allocate(folded['scores'][8*b+old.folded_bins(adapter,x)],ties)
            tx = adapter.partner(x)
            candidate_t = adapter.rank.select(tx,zero,rank,'rank_even',ties,keys)
            bt = np.searchsorted(folded['edges'],adapter.quotient(tx))
            baseline_t = old.allocate(folded['scores'][8*bt+old.folded_bins(adapter,tx)],ties)
            assert np.array_equal(candidate,candidate_t) and np.array_equal(baseline,baseline_t)
            union = candidate | baseline
            selected = np.flatnonzero(union)
            gaps = np.zeros(len(x)); calls = []
            for offset in range(0,len(selected),8192):
                ids = selected[offset:offset+8192]
                a = adapter.predict(theta,x[ids]); calls.append(len(ids))
                b = adapter.predict(theta,tx[ids]); calls.append(len(ids))
                gaps[ids] = jensen_gap(a,b)
            L,C = adapter.network_constants(theta)
            cutoff = 3. if cohort=='rings' else 1.
            region = adapter.quotient(x)<=cutoff if cohort=='rings' else np.ones(len(x),bool)
            d_bound = cutoff+C
            tail = rings_tail_bound(C,cutoff) if cohort=='rings' else 0.
            batch = statistics(gaps,[candidate,baseline],region)
            batch['calls_per_batch'] = 2*batch['union_selected']
            np.savez_compressed(output/f'{cohort}_{pair}_batches.npz',**batch)
            for look in LOOKS:
                for policy, index in [('rank_even',0),('folded',1)]:
                    for clip in CLIPS:
                        result = empirical_bernstein_lower(batch[f'policy{index}_clip_{clip:g}'][:look],clip*QUOTA/BATCH,delta)
                        result.update(cohort=cohort,pair=pair,policy=policy,clip=clip,
                            full_same_sample_gain=float(batch[f'policy{index}_unclipped'].mean()),
                            individual_forwards=2*QUOTA*look,
                            shared_forwards=int(batch['calls_per_batch'][:look].sum()))
                        records.append(result)
                result = empirical_bernstein_lower(batch['relative_truncated'][:look],2*QUOTA/BATCH*jensen_bound(d_bound),delta)
                result.update(cohort=cohort,pair=pair,policy='rank_even_minus_folded',clip=0.,
                    full_same_sample_gain=float(batch['relative_unclipped'].mean()),
                    individual_forwards=-1,shared_forwards=int(batch['calls_per_batch'][:look].sum()))
                result['lower'] -= tail
                records.append(result)
            costs.append(dict(cohort=cohort,pair=pair,seed=seed,independent_batches=max(LOOKS),
                coordinate_inputs=len(x),actual_forward_examples=sum(calls),actual_forward_invocations=len(calls),
                seconds=time.perf_counter()-start,model_sha256=sha(state_path),rank_sha256=sha(table_path),
                folded_sha256=sha(folded_path),x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                C=C,L=L,jensen_upper=jensen_bound(d_bound),tail_upper=tail))
            audits.append(dict(cohort=cohort,pair=pair,candidate_T_mask_equal=True,folded_T_mask_equal=True,
                quota_min=int(candidate.reshape(-1,BATCH).sum(1).min()),quota_max=int(candidate.reshape(-1,BATCH).sum(1).max()),
                exact_forward_count=bool(sum(calls)==2*union.sum())))
            print(f'Old-only verification {cohort} {pair}/16 completed',flush=True)
    for filename,rows in [('clipping_grid.csv',records),('costs.csv',costs),('audits.csv',audits)]:
        with (output/filename).open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    chosen = {}
    for cohort in ('rings','transfer'):
        values = []
        for clip in CLIPS:
            rows=[row for row in records if row['cohort']==cohort and row['clip']==clip and row['n_batches']==512]
            values.append((float(np.mean([row['lower'] for row in rows])),clip))
        chosen[cohort]=max(values,key=lambda item:(item[0],-item[1]))[1]
    with (output/'choice.json').open('x',encoding='utf8') as f:
        json.dump(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rule='Maximize mean lower bound at 512 batches across old 16 models and both old even policies; smallest clip breaks exact ties.',
            selected_clips=chosen,development_only=True,fresh_outcomes_accessed=False,
            old_root=str(old_root),seconds=time.perf_counter()-start_all,
            delta_per_claim=delta,source_hashes={p.name:sha(p) for p in (Path(__file__),HERE/'bounds.py')}),f,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.old_root,args.output)
