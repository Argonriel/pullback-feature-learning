"""Timing-only strong references: pi is exactly -x in both geometries.

This preserves all scientific policies and original timings. It removes avoidable
coordinate/bin/rotation work from fixed-pi deployment references and checks
identical masks, predictions and network-call counts before timing them.
"""
import argparse,json,math,time
import numpy as np
from threadpoolctl import threadpool_limits
from study import HERE,CONFIG,POLICIES,ref,core,predict,deploy,gid,csvin,csvout,dump,sha,now

NAMES=['confidence_radial','fast_confidence_pi','fast_full_pi','fast_legacy_iid_folded_pi',
    'original_confidence_pi','original_full_pi','original_legacy_iid_folded_pi']


def fast_deploy(x,pred,table,policy,tie,Ainv):
    h=pred(x)
    if policy=='full_pi':return .5*(h+pred(-x)),np.ones(len(x),bool)
    if policy=='confidence_pi':priority=-abs(h)
    elif policy=='legacy_iid_folded_pi':
        gate=ref.Gate(table['iid_edges'],table['iid_radius_scores'],table['iid_folded_scores'])
        priority=gate.priority(x@Ainv.T,'folded')
    else:raise ValueError(policy)
    mask=ref.allocate(priority,tie);out=h.copy()
    out[mask]=.5*(h[mask]+pred(-x[mask]))
    return out,mask


def freeze():
    dump(HERE/'timing_controls_freeze.json',dict(timestamp=now(),source_hash=sha(__import__('pathlib').Path(__file__).read_bytes()),
        primary_freeze_hash=sha((HERE/'freeze.json').read_bytes()),
        reason='Fixed-pi references need no radius bins or conjugated coordinate transform; preserve frozen main timings and add stronger equivalent references.',
        candidate='Unchanged confidence_radial deployment',test_input_seed_base=831000,
        tie_seed_base=832000,repetitions=8,batches_per_repeat=16,variants=NAMES,
        timing_only_no_risk_selection=True,all_primary_policy_settings_unchanged=True))


def run():
    f=json.loads((HERE/'timing_controls_freeze.json').read_text())
    assert f['source_hash']==sha(__import__('pathlib').Path(__file__).read_bytes())
    records=[];audits=[]
    for geometry,av in CONFIG['geometries'].items():
        A=np.array(av);Ainv=np.linalg.inv(A)
        for pair in range(1,17):
            index=gid(geometry,pair);state=np.load(HERE/f'states/{geometry}_{pair}.npz')
            z=ref.sample(831000+index,4096);x=z@A.T;tie=np.random.default_rng(832000+index).random(len(x))
            for model in CONFIG['models']:
                theta=state[model];pred=lambda q:predict(theta,q)
                for budget in CONFIG['budgets']:
                    table=dict(np.load(HERE/f'calibration/policies/{geometry}_{pair}_{model}_{budget}.npz'))
                    table.update(dict(np.load(HERE/f'calibration/policies/{geometry}_{pair}_{model}_{budget}_iid.npz')))
                    for policy in ('confidence_pi','full_pi','legacy_iid_folded_pi'):
                        original,mask,alpha=deploy(x[:256],pred,table,policy,tie[:256],A,Ainv)
                        calls=[]
                        def counted(q):calls.append(len(q));return pred(q)
                        fast,selected=fast_deploy(x[:256],counted,table,policy,tie[:256],Ainv)
                        error=float(np.max(abs(original-fast)))
                        assert error<2e-12 and np.array_equal(mask,selected)
                        assert calls==([256,256] if policy=='full_pi' else [256,64])
                        audits.append(dict(geometry=geometry,pair=pair,model=model,budget=budget,policy=policy,
                            max_equivalence_error=error,identical_mask=1,calls=json.dumps(calls)))
                    rng=np.random.default_rng(833000+index+budget)
                    for rep in range(8):
                        for variant in rng.permutation(NAMES):
                            start=time.perf_counter_ns()
                            for j in range(16):
                                block=x[j*256:(j+1)*256];ties=tie[j*256:(j+1)*256]
                                if variant.startswith('fast_'):
                                    fast_deploy(block,pred,table,str(variant)[5:],ties,Ainv)
                                else:
                                    policy='confidence_radial' if variant=='confidence_radial' else str(variant)[9:]
                                    deploy(block,pred,table,policy,ties,A,Ainv)
                            elapsed=(time.perf_counter_ns()-start)/16/1e6
                            if rep:records.append(dict(geometry=geometry,pair=pair,model=model,budget=budget,
                                variant=str(variant),repeat=rep,ms_per_batch256=elapsed))
            print('strong timing controls',geometry,pair,flush=True)
    csvout(HERE/'results/strong_timing_raw.csv',records);csvout(HERE/'results/strong_timing_equivalence.csv',audits)
    cal=csvin(HERE/'calibration/costs.csv');rows=[];byseed=[]
    draws=np.random.default_rng(834000).integers(0,16,(20000,16))
    for geometry in CONFIG['geometries']:
        for model in CONFIG['models']:
            for budget in CONFIG['budgets']:
                for reference in ('fast_confidence_pi','fast_full_pi','fast_legacy_iid_folded_pi'):
                    group=[]
                    for pair in range(1,17):
                        def median(name):return float(np.median([r['ms_per_batch256'] for r in records if r['geometry']==geometry and r['pair']==pair and r['model']==model and r['budget']==budget and r['variant']==name]))
                        candidate=median('confidence_radial');baseline=median(reference)
                        setup=1000*float(next(r['calibration_seconds'] for r in cal if r['geometry']==geometry and int(r['pair'])==pair and r['model']==model and int(r['budget'])==budget and r['kind']=='orbit'))
                        refsetup=0 if reference!='fast_legacy_iid_folded_pi' else 1000*float(next(r['calibration_seconds'] for r in cal if r['geometry']==geometry and int(r['pair'])==pair and r['model']==model and int(r['budget'])==budget and r['kind']=='legacy_iid'))
                        saving=baseline-candidate
                        crossing=256*math.ceil(max(setup-refsetup,0)/saving) if saving>0 else None
                        entry=dict(geometry=geometry,pair=pair,model=model,budget=budget,reference=reference,
                            candidate_ms=candidate,reference_ms=baseline,ms_saved_per_batch=saving,
                            calibration_ms=setup,reference_calibration_ms=refsetup,
                            finite_latency_break_even=int(crossing is not None),latency_break_even_inputs='' if crossing is None else crossing,
                            runtime_10pct_overhead_inputs=256*math.ceil(setup/(.1*candidate)),
                            runtime_1pct_overhead_inputs=256*math.ceil(setup/(.01*candidate)))
                        group.append(entry);byseed.append(entry)
                    finite=[r['latency_break_even_inputs'] for r in group if r['finite_latency_break_even']]
                    savings=np.array([r['ms_saved_per_batch'] for r in group]);lo,hi=np.quantile(savings[draws].mean(1),[.025,.975])
                    rows.append(dict(geometry=geometry,model=model,budget=budget,reference=reference,
                        median_candidate_ms=float(np.median([r['candidate_ms'] for r in group])),
                        median_reference_ms=float(np.median([r['reference_ms'] for r in group])),
                        mean_ms_saved_per_batch=savings.mean(),saving_ci95_low=lo,saving_ci95_high=hi,
                        finite_latency_break_even_models=len(finite),no_finite_latency_break_even_models=16-len(finite),
                        median_break_even_inputs_conditional_on_finite=float(np.median(finite)) if finite else '',
                        median_runtime_10pct_overhead_inputs=float(np.median([r['runtime_10pct_overhead_inputs'] for r in group])),
                        median_runtime_1pct_overhead_inputs=float(np.median([r['runtime_1pct_overhead_inputs'] for r in group]))))
    csvout(HERE/'results/strong_timing_summary.csv',rows);csvout(HERE/'results/strong_timing_amortization_by_seed.csv',byseed)
    dump(HERE/'results/strong_timing_complete.json',dict(timestamp=now(),source_hash=f['source_hash'],
        max_equivalence_error=max(r['max_equivalence_error'] for r in audits),
        n_equivalence_checks=len(audits),risk_outcomes_or_learned_policies_changed=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);a=ap.parse_args()
    with threadpool_limits(limits=1):globals()[a.stage]()
