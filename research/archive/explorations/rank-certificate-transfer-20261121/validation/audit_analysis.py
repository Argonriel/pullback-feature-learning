"""Independent multiplicity-weight bootstrap and saved-prediction outcome audit."""
from pathlib import Path
import csv,datetime,hashlib,json,sys
import numpy as np
from threadpoolctl import threadpool_limits
from adapters import get_adapter,worlds

HERE=Path(__file__).resolve().parent
def read(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    frozen=json.loads((HERE/'freeze.json').read_text());cfg=frozen['config']
    recorded=read(HERE/'per_model.csv');lookup={(r['cohort'],int(r['pair']),r['method'],r['policy']):r for r in recorded}
    reconstructed={};replays=[]
    for cohort,seeds in cfg['cohorts'].items():
        adapter=get_adapter(cohort)
        for pair in range(1,17):
            x=adapter.sample_inputs(seeds['test']+pair,65536);p=adapter.posterior(x)
            labels=(np.random.default_rng(seeds['labels']+pair).random(len(x))<p)
            alternate=worlds(adapter,x)
            for method in adapter.methods:
                saved=np.load(HERE/'predictions'/f'{cohort}_{pair}_{method}.npz')
                for name,h in zip(saved['names'].tolist(),saved['outputs']):
                    soft=np.logaddexp(0.,h);positive=h>=0
                    # Algebraically distinct loss form from the main mixture-of-softplus implementation.
                    values=dict(expected_risk=float(np.mean(soft-p*h)),
                        sampled_label_risk=float(np.logaddexp(0.,np.where(labels,-h,h)).mean()),
                        expected_error=float((np.sum(p[~positive])+np.sum(1-p[positive]))/len(h)),
                        sampled_error=float(np.count_nonzero(positive!=labels)/len(h)))
                    for world,posterior in alternate.items():values[world+'_risk']=float(np.mean(soft-posterior*h))
                    key=(cohort,pair,method,name);reconstructed[key]=values
                    error=max(abs(v-float(lookup[key][field])) for field,v in values.items())
                    assert error<2e-14
                    replays.append(dict(cohort=cohort,pair=pair,method=method,policy=name,max_outcome_error=error))
    # Rebuild bootstrap averages using counts, rather than selecting/resumming data rows.
    rng=np.random.default_rng(cfg['bootstrap_seed']);weights={}
    for cohort in cfg['cohorts']:
        sampled=rng.integers(0,16,(cfg['bootstrap_repeats'],16))
        weights[cohort]=np.array([np.bincount(row,minlength=16) for row in sampled],dtype=float)/16
    checks=[]
    for row in read(HERE/'contrasts.csv'):
        cohort,method,name,reference,outcome=(row[k] for k in ('cohort','method','policy','reference','outcome'))
        delta=np.array([reconstructed[cohort,i,method,name][outcome]-reconstructed[cohort,i,method,reference][outcome] for i in range(1,17)])
        boot=weights[cohort]@delta
        calculated=dict(delta_risk=float(np.sum(delta)/16),wins=int(np.count_nonzero(delta<0)),
            low95=float(np.percentile(boot,2.5)),high95=float(np.percentile(boot,97.5)),
            low98_75=float(np.percentile(boot,.625)),high98_75=float(np.percentile(boot,99.375)))
        error=max(abs(value-float(row[field])) for field,value in calculated.items() if field!='wins')
        assert error<3e-14 and calculated['wins']==int(row['wins'])
        checks.append(dict(cohort=cohort,method=method,policy=name,reference=reference,
            outcome=outcome,primary=row['primary'],max_bootstrap_stat_error=error,wins_match=True))
    summary_checks=[]
    for row in read(HERE/'summary.csv'):
        cohort,method,name,outcome=(row[k] for k in ('cohort','method','policy','outcome'))
        vector=np.array([reconstructed[cohort,i,method,name][outcome] for i in range(1,17)])
        boot=weights[cohort]@vector
        error=max(abs(float(row['mean'])-float(vector.mean())),
            abs(float(row['low95'])-float(np.percentile(boot,2.5))),
            abs(float(row['high95'])-float(np.percentile(boot,97.5))))
        assert error<3e-14
        summary_checks.append(dict(cohort=cohort,method=method,policy=name,outcome=outcome,max_error=error))
    calibration=read(HERE/'calibration.csv');costs=[]
    for cohort in cfg['cohorts']:
        for kind in ('rank','folded','angles'):
            selected=[r for r in calibration if r['cohort']==cohort and r['kind']==kind]
            assert len(selected)==16
            count=int(selected[0]['forward_examples'])
            assert all(int(r['forward_examples'])==count for r in selected)
            costs.append(dict(cohort=cohort,kind=kind,n_models=16,forwards_per_model=count,
                total_forwards=sum(int(r['forward_examples']) for r in selected),
                coordinate_only_per_model=int(selected[0]['coordinate_only_inputs']),
                total_coordinate_only=sum(int(r['coordinate_only_inputs']) for r in selected),
                mean_calibration_seconds=float(np.mean([float(r['seconds']) for r in selected])),
                median_calibration_seconds=float(np.median([float(r['seconds']) for r in selected]))))
    gains=[]
    for cohort in cfg['cohorts']:
        adapter=get_adapter(cohort)
        for method in adapter.methods:
            names=frozen['policies'] if method=='pb' else ['raw','full']
            for name in names:
                for outcome in ('expected_risk','half_risk','increasing_risk','decreasing_risk','threshold_risk'):
                    gain=np.array([reconstructed[cohort,i,method,'raw'][outcome]-reconstructed[cohort,i,method,name][outcome] for i in range(1,17)])
                    boot=weights[cohort]@gain
                    gains.append(dict(cohort=cohort,method=method,policy=name,world=outcome,
                        mean_gain=float(gain.mean()),low95=float(np.percentile(boot,2.5)),
                        high95=float(np.percentile(boot,97.5)),positive_models=int((gain>0).sum()),n_models=16))
    write(HERE/'outcome_audit.csv',replays);write(HERE/'bootstrap_audit.csv',checks)
    write(HERE/'summary_audit.csv',summary_checks);write(HERE/'calibration_costs.csv',costs)
    write(HERE/'world_gains.csv',gains)
    result=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        n_outcome_rows=len(replays),n_contrasts=len(checks),n_summary_rows=len(summary_checks),
        maximum_outcome_error=max(r['max_outcome_error'] for r in replays),
        maximum_bootstrap_error=max(r['max_bootstrap_stat_error'] for r in checks),
        maximum_summary_error=max(r['max_error'] for r in summary_checks),
        all_contrast_wins_match=True,bootstrap_method='Multiplicity-weight averages from independently reconstructed saved-prediction risks.',
        primary_family_size=4,primary_interval=.9875,total_calibration_forwards=sum(r['total_forwards'] for r in costs),
        total_coordinate_only_draws=sum(r['total_coordinate_only'] for r in costs),
        source_hash=digest(Path(__file__)),outcome_hash=digest(HERE/'per_model.csv'),
        contrast_hash=digest(HERE/'contrasts.csv'),summary_hash=digest(HERE/'summary.csv'))
    with (HERE/'analysis_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
