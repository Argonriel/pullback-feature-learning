"""Reconstruct and retain every failed bounded ray's training-only search trace."""
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import study as s

def trace(t,x,y,method,drop,meta,rows,states):
    d=s.core.direction(t,x,y,method,.1)[0];sl=-s.core.parameter_gradient(t,x,y)@d;alpha=drop/sl/8;initial=s.loss(t,x,y)
    key='_'.join(str(meta[k]) for k in meta);states[key]=t.copy()
    for j in range(160):
        norm=float(np.linalg.norm(alpha*d));cap=norm>s.CONFIG['root_max_norm']
        val=s.loss(t+alpha*d,x,y) if not cap else ''
        rows.append(dict(**meta,failed_state_key=key,failed_state_sha256=s.sha(t.tobytes()),attempt=j+1,alpha=alpha,parameter_displacement=norm,start_loss=initial,target_loss=initial-drop,candidate_loss=val,above_cap=int(cap)))
        if cap:return
        assert val>initial-drop
        alpha*=s.CONFIG['root_expansion']
    raise AssertionError('Unexpected non-cap failure')
def main():
    rows=[];states={};maps=[]
    for cohort in ('old','new'):
        bank=np.load(ROOT/cohort/'states.npz')
        cases=[r for r in s.csvin(ROOT/cohort/'outcomes.csv') if r['metric']=='fit']
        for r in cases:
            if r['status']=='ok' or r['experiment']!='single':continue
            t=bank[r['pre_key']];seed=r['seed'];task=r['task'];x=bank[f'{task}_{seed}_x'];y=bank[f'{task}_{seed}_y']
            trace(t,x,y,r['method'],float(r['requested_drop']),dict(cohort=cohort,task=task,seed=seed,stage=r['stage'],method=r['method'],experiment='single',step=1,drop=r['requested_drop']),rows,states)
        keys=sorted(set((r['task'],r['seed'],r['stage'],r['method'],r['pre_key']) for r in cases if r['experiment']=='path' and r['status']!='ok'))
        for task,seed,stage,method,prekey in keys:
            t=bank[prekey];x=bank[f'{task}_{seed}_x'];y=bank[f'{task}_{seed}_y']
            for step in range(1,9):
                d=s.core.direction(t,x,y,method,.1)[0];end,info=s.match(t,x,y,d,.005)
                if info['status']!='ok':
                    trace(t,x,y,method,.005,dict(cohort=cohort,task=task,seed=seed,stage=stage,method=method,experiment='path_step',step=step,drop=.005),rows,states);maps.append(dict(cohort=cohort,task=task,seed=seed,stage=stage,method=method,first_failed_step=step));break
                t=end
    folder=ROOT/'second_order_followup';bank=np.load(folder/'states.npz')
    for r in s.csvin(folder/'outcomes.csv'):
        if r['metric']!='fit' or r['status']=='ok':continue
        task=r['task'];seed=r['seed'];trace(bank[r['pre_key']],bank[f'{task}_{seed}_x'],bank[f'{task}_{seed}_y'],r['method'],float(r['requested_drop']),dict(cohort='third',task=task,seed=seed,stage='initial',method=r['method'],experiment='single',step=1,drop=r['requested_drop']),rows,states)
    s.csvout(ROOT/'failed_root_traces.csv',rows);s.csvout(ROOT/'first_failed_path_steps.csv',maps);np.savez_compressed(ROOT/'failed_root_states.npz',**states)
    s.jsonout(ROOT/'failure_trace_check.json',dict(timestamp_utc=s.now(),failed_ray_count=len(states),attempt_rows=len(rows),all_failed_sampled_points_above_target=True,terminal_reason='next geometric trial exceeds norm4; no claim of global infeasibility',traces_sha256=s.sha((ROOT/'failed_root_traces.csv').read_bytes()),states_sha256=s.sha((ROOT/'failed_root_states.npz').read_bytes())))
if __name__=='__main__':
    with threadpool_limits(1):main()
