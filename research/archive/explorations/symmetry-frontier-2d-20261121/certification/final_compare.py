"""Pre-fresh additive McDiarmid comparison with a joint 864-event family."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,math
import numpy as np
from bounds import BATCH,LOOKS,empirical_bernstein_lower,jensen_bound,rings_tail_bound

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FAMILY_EVENTS=864
DELTA=.05/FAMILY_EVENTS
CLIPS={'rings':.25,'waveform':.5,'two_invariants':.5}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jo(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def co(path,rows):
    with Path(path).open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
def ci(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def combine(values,range_width,sensitivity_sum,tail=0.):
    """sensitivity_sum bounds replacement of one input in a batch SUM."""
    eb=empirical_bernstein_lower(values,range_width,DELTA)
    penalty=sensitivity_sum*math.sqrt(math.log(1./DELTA)/(2*len(values)*BATCH))
    eb_lower=eb['lower']-tail
    mcd_lower=eb['mean']-penalty-tail
    return dict(mean=eb['mean'],variance=eb['variance'],n_batches=len(values),
        delta=DELTA,family_events=FAMILY_EVENTS,range_width=range_width,
        replacement_sensitivity_sum=sensitivity_sum,tail_subtraction=tail,
        empirical_bernstein_lower=eb_lower,bounded_difference_lower=mcd_lower,
        empirical_bernstein_variance_penalty=eb['stochastic_penalty'],
        empirical_bernstein_range_penalty=eb['range_penalty'],bounded_difference_penalty=penalty,
        lower=max(eb_lower,mcd_lower),positive=bool(max(eb_lower,mcd_lower)>0))

def rows_for(data,cohort,pair,gap_bound,tail,candidate='symmetric_jensen',old=False):
    clip=CLIPS[cohort];rows=[]
    for look in LOOKS:
        for index,name in enumerate([candidate,'folded','candidate_minus_folded']):
            relative=index==2
            if relative:
                values=data['relative_truncated'][:look];full=data['relative_unclipped']
            elif old:
                values=data[f'policy{index}_clip_{clip:g}'][:look];full=data[f'policy{index}_unclipped']
            else:
                values=data[name+'_clipped'][:look];full=data[name+'_unclipped']
            result=combine(values,gap_bound/2 if relative else clip/4,
                2*gap_bound if relative else clip,tail if relative else 0.)
            calls=data['calls_per_batch' if old else 'forward_examples'][:look]
            result.update(cohort=cohort,pair=pair,claim=name,clip=0. if relative else clip,
                full_2048_unclipped_mean=float(full.mean()),actual_shared_forward_examples=int(calls.sum()),
                standalone_absolute_forward_examples=0 if relative else 128*look,coordinate_inputs=BATCH*look)
            rows.append(result)
    return rows

def summarize(rows):
    summaries=[];thresholds=[]
    for cohort in sorted({r['cohort'] for r in rows}):
        for name in sorted({r['claim'] for r in rows if r['cohort']==cohort}):
            group=[r for r in rows if r['cohort']==cohort and r['claim']==name]
            full={r['pair']:r['lower'] for r in group if r['n_batches']==2048}
            for look in LOOKS:
                current=[r for r in group if r['n_batches']==look]
                lower=np.array([r['lower'] for r in current])
                retention=[r['lower']/full[r['pair']] for r in current if full[r['pair']]>0]
                recovery=[r['lower']/r['full_2048_unclipped_mean'] for r in current if r['full_2048_unclipped_mean']>0]
                summaries.append(dict(cohort=cohort,claim=name,n_batches=look,n_models=len(current),
                    positive=int(np.sum(lower>0)),empirical_bernstein_positive=sum(r['empirical_bernstein_lower']>0 for r in current),
                    bounded_difference_positive=sum(r['bounded_difference_lower']>0 for r in current),
                    mean_lower=float(lower.mean()),min_lower=float(lower.min()),max_lower=float(lower.max()),
                    mean_empirical_bernstein_lower=float(np.mean([r['empirical_bernstein_lower'] for r in current])),
                    mean_bounded_difference_lower=float(np.mean([r['bounded_difference_lower'] for r in current])),
                    mean_unclipped_gain=float(np.mean([r['full_2048_unclipped_mean'] for r in current])),
                    mean_actual_shared_forwards=float(np.mean([r['actual_shared_forward_examples'] for r in current])),
                    mean_2048_lower_bound_retention=float(np.mean(retention)) if retention else '',
                    valid_retention_denominators=len(retention),
                    mean_lower_to_empirical_full_gain=float(np.mean(recovery)) if recovery else '',
                    valid_empirical_gain_denominators=len(recovery)))
            for pair in range(1,17):
                for method in ['empirical_bernstein_lower','bounded_difference_lower','lower']:
                    eligible=[r['n_batches'] for r in group if r['pair']==pair and r[method]>0]
                    thresholds.append(dict(cohort=cohort,claim=name,pair=pair,method=method,
                        first_positive_batches=min(eligible) if eligible else 'unreached'))
    return summaries,thresholds

def development():
    folder=HERE/'development_old32';costs=ci(folder/'costs.csv');rows=[]
    for cost in costs:
        old_cohort=cost['cohort'];cohort='waveform' if old_cohort=='transfer' else 'rings';pair=int(cost['pair'])
        data=np.load(folder/f'{old_cohort}_{pair}_batches.npz')
        R=3. if cohort=='rings' else 1.;C=float(cost['C']);L=float(cost['L'])
        Jmax=jensen_bound(min(R+C,R*L));tail=rings_tail_bound(C) if cohort=='rings' else 0.
        rows+=rows_for(data,cohort,pair,Jmax,tail,candidate='rank_even',old=True)
    summary,thresholds=summarize(rows)
    co(folder/'addendum_864_per_model.csv',rows);co(folder/'addendum_864_summary.csv',summary)
    co(folder/'addendum_864_thresholds.csv',thresholds)
    print(json.dumps(summary,indent=2))

def checks():
    rng=np.random.default_rng(1109002);maximum=0.;maximum_relative=0.
    for _ in range(10000):
        score=rng.integers(0,8,256);other_score=rng.integers(0,8,256);ties=rng.random(256)
        weight=rng.choice([0.,.5,1.],256)
        pick=lambda s,t,w:float(w[np.lexsort((t,-s))[:64]].sum())
        before=pick(score,ties,weight);relative_before=before-pick(other_score,ties,weight)
        index=int(rng.integers(256));score[index]=rng.integers(0,8);other_score[index]=rng.integers(0,8)
        ties[index]=rng.random();weight[index]=rng.choice([0.,.5,1.])
        after=pick(score,ties,weight);relative_after=after-pick(other_score,ties,weight)
        maximum=max(maximum,abs(after-before));maximum_relative=max(maximum_relative,abs(relative_after-relative_before))
    assert maximum<=1. and maximum_relative<=2.
    jo(HERE/'addendum_checks.json',dict(seed=1109002,replacements=10000,
        maximum_absolute_sum_change=maximum,absolute_bound=1.,maximum_relative_sum_change=maximum_relative,
        relative_bound=2.,passed=True,qualification='Numerical audit supplements the deterministic proof.'))

def freeze():
    assert not any((ROOT/'validation'/'states').glob('*.npz')),'Fresh states already exist'
    files=[Path(__file__),HERE/'ADDENDUM.md',HERE/'addendum_checks.json',HERE/'FREEZE.json',
        HERE/'development_old32'/'addendum_864_summary.csv']
    jo(HERE/'ADDENDUM_FREEZE.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_freeze_sha256=sha(HERE/'FREEZE.json'),final_family_events=864,delta_per_bound=DELTA,
        primary_batches_unchanged=512,fresh_states_absent=True,fresh_outcomes_accessed=False,
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))

def fresh():
    frozen=json.loads((HERE/'ADDENDUM_FREEZE.json').read_text())
    for path,digest in frozen['source_hashes'].items():assert sha(ROOT/path)==digest,path
    approval=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
    assert approval['certificate_addendum_freeze_sha256']==sha(HERE/'ADDENDUM_FREEZE.json')
    assert approval['fresh_verification_authorized'] is True
    rows=[]
    for cost in ci(HERE/'costs.csv'):
        cohort=cost['cohort'];pair=int(cost['pair']);data=np.load(HERE/'fresh'/f'{cohort}_{pair}_batches.npz')
        rows+=rows_for(data,cohort,pair,float(cost['gap_bound']),float(cost['tail_upper']))
    assert len(rows)==432
    summary,thresholds=summarize(rows)
    co(HERE/'final_per_model_bounds.csv',rows);co(HERE/'final_summary.csv',summary);co(HERE/'final_thresholds.csv',thresholds)
    jo(HERE/'FINAL_CERTIFICATE_COMMIT.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        family_events=864,alpha=.05,delta=DELTA,maximum_of_two_simultaneous_bounds=True,
        original_432_event_outputs='Preserved diagnostic; superseded for primary reporting.',
        addendum_freeze_sha256=sha(HERE/'ADDENDUM_FREEZE.json'),
        original_completion_sha256=sha(HERE/'completion.json'),
        outputs={p.name:sha(p) for p in [HERE/'final_per_model_bounds.csv',HERE/'final_summary.csv',HERE/'final_thresholds.csv']}))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','checks','freeze','fresh'])
    args=parser.parse_args();globals()[args.stage]()
