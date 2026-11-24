"""Audit exact finite balance, paired-input identity, matching and retained failures."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import study as s
from audit_failure_traces import trace
from threadpoolctl import threadpool_limits

def main():
    bank=np.load(ROOT/'states.npz');rows=s.csvin(ROOT/'outcomes.csv');checks=[];errors=[];traces=[];failurestates={}
    for seed in range(1,17):
        xr=bank[f'radial_{seed}_x'];xa=bank[f'angular_{seed}_x']
        assert np.array_equal(xr,xa)
        assert np.array_equal(bank[f'radial_{seed}_initial'],bank[f'angular_{seed}_initial'])
        for task in ('radial','angular'):
            y=bank[f'{task}_{seed}_y'];assert np.sum(y==1)==128 and np.sum(y==-1)==128 and np.array_equal(y[128:],-y[:128])
            checks.append(dict(seed=seed,task=task,positive=int(np.sum(y==1)),negative=int(np.sum(y==-1)),pair_labels_opposite=True,x_sha256=s.sha(xr.tobytes())))
    for r in rows:
        if r['metric']!='fit':continue
        task=r['task'];seed=r['seed'];pre=bank[r['pre_key']];x=bank[f'{task}_{seed}_x'];y=bank[f'{task}_{seed}_y']
        if r['status']=='ok':
            end=bank[r['end_key']];errors.append(abs(s.loss(pre,x,y)-s.loss(end,x,y)-float(r['requested_drop'])))
            assert s.sha(end.tobytes())==r['end_sha256']
        else:
            trace(pre,x,y,r['method'],float(r['requested_drop']),dict(cohort='balanced',task=task,seed=seed,stage='initial',method=r['method'],experiment='single',step=1,drop=r['requested_drop']),traces,failurestates)
    freeze=json.loads((ROOT/'freeze.json').read_text());commit=json.loads((ROOT/'prediction_commit.json').read_text());score=json.loads((ROOT/'score_start.json').read_text())
    assert freeze['timestamp_utc']<commit['timestamp_utc']<score['timestamp_utc']
    assert freeze['script_sha256']==s.sha((ROOT/'balanced.py').read_bytes())
    assert commit['predictions_sha256']==s.sha((ROOT/'predictions.csv').read_bytes())
    s.csvout(ROOT/'balance_checks.csv',checks);s.csvout(ROOT/'failed_root_traces.csv',traces);np.savez_compressed(ROOT/'failed_root_states.npz',**failurestates)
    s.jsonout(ROOT/'numerical_audit.json',dict(all32_task_samples_exactly_balanced=True,all16_task_pairs_share_inputs_and_initialization=True,all_training_pairs_have_opposite_labels=True,max_recomputed_loss_match_error=max(errors),successful_roots=len(errors),failed_rays=len(failurestates),failed_trial_rows=len(traces),freeze_before_commit_before_score=True,hashes_pass=True,important_scope='256 training observations comprise128 independent antithetic pairs; angular substantial-drop inference has only2 jointly reachable seed pairs.'))
    note='\n\nOnly two angular substantial-drop pairs were jointly reachable. Their percentile intervals are descriptive of these two successes and do not establish robust transfer. The exact-balance primary radial fitting comparison shows no reproducible second-order improvement; the iid follow-up improvement must not be generalized across sampling designs.\n'
    with (ROOT/'BALANCED_CONTROL.md').open('a',encoding='utf8') as f:f.write(note)
    manifest={str(p.relative_to(ROOT)):s.sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    s.jsonout(ROOT/'manifest.json',dict(timestamp_utc=s.now(),sha256=manifest))
if __name__=='__main__':
    with threadpool_limits(1):main()
