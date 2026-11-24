"""Prior sixteen-model check; no fresh model or test access."""
from pathlib import Path
import argparse,time
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,components,co,jo,sha,predict_chunks
from bounds import LOOKS,gap,clip_from_grid,moment_bounds

def main(old_root):
    output=HERE/'development_old16';output.mkdir(exist_ok=False)
    task,policy,adapter=components();rows=[];costs=[];start_all=time.perf_counter()
    for pair in range(1,17):
        start=time.perf_counter();state=old_root/'validation'/'states'/f'two_invariants_{pair}.npz'
        theta=np.load(state)['pb'];grid0=dict(np.load(old_root/'validation'/'policies'/f'two_invariants_{pair}_grid.npz'))
        folded0=dict(np.load(old_root/'validation'/'policies'/f'two_invariants_{pair}_folded.npz'))
        x=task.sample_inputs(2081000+pair,2048*256);ties=np.random.default_rng(2082000+pair).random(len(x))
        h,calls1=predict_chunks(task,theta,x);ht,calls2=predict_chunks(task,theta,task.partner(x));L,C=task.network_constants(theta)
        for beta in (1.,.5,.25):
            grid=dict(grid0);g=gap(beta*grid0['h'],beta*np.roll(grid0['h'],grid0['h'].shape[-1]//2,axis=-1))
            k=g.shape[-1]//2;grid['folded_jensen']=.5*(g[...,:k]+g[...,k:])
            folded=policy.fit_folded(adapter,theta,folded0['calibration_x'],config=folded0['config'],
                prepared=(beta*folded0['calibration_first'],beta*folded0['calibration_partner']))
            clip=clip_from_grid(grid);J=gap(beta*h,beta*ht);absd=.5*beta*abs(h-ht);D=beta*min(1.4+C,1.4*L)
            masks=[policy.select(adapter,x,grid,ties),policy.select_folded(adapter,x,folded,ties)]
            batch={}
            for name,mask in zip(('symmetric_jensen','folded'),masks):
                A=(mask*np.minimum(J,clip)).reshape(-1,256).mean(1)
                B=(mask*absd).reshape(-1,256).mean(1)
                raw=(mask*J).reshape(-1,256).mean(1);fixed=(mask*np.minimum(J,.5)).reshape(-1,256).mean(1)
                batch[name+'_A']=A;batch[name+'_B']=B;batch[name+'_unclipped']=raw;batch[name+'_fixed_half']=fixed
                for n in LOOKS:
                    row=moment_bounds(A[:n],B[:n],clip,D)
                    row.update(pair=pair,beta=beta,policy=name,mean_unclipped=float(raw.mean()),
                        clipped_retention=float(A.mean()/raw.mean()) if raw.mean()>0 else 1.,
                        fixed_half_retention=float(fixed.mean()/raw.mean()) if raw.mean()>0 else 1.)
                    rows.append(row)
            np.savez_compressed(output/f'{pair}_beta{beta:g}_batches.npz',**batch)
        costs.append(dict(pair=pair,input_seed=2081000+pair,tie_seed=2082000+pair,
            actual_forward_examples=sum(calls1+calls2),actual_forward_invocations=len(calls1+calls2),
            seconds=time.perf_counter()-start,source_state_sha256=sha(state),
            temperature_diagnostics='Fixed beta 1,.5,.25; reuse same raw network logits; no fitted temperature.'))
        print(f'Old development pair {pair}/16',flush=True)
    co(output/'per_model.csv',rows);co(output/'costs.csv',costs)
    summary=[]
    for beta in (1.,.5,.25):
        for name in ('symmetric_jensen','folded'):
            group=[r for r in rows if r['beta']==beta and r['policy']==name and r['n_batches']==2048]
            summary.append(dict(beta=beta,policy=name,mean_clip=float(np.mean([r['c'] for r in group])),
                mean_retention=float(np.mean([r['clipped_retention'] for r in group])),
                positive_at_zero=sum(r['A_lower']>0 for r in group),
                mean_critical_epsilon=float(np.mean([r['epsilon_critical'] for r in group]))))
    co(output/'summary.csv',summary)
    jo(output/'completion.json',dict(old_root=str(old_root),seconds=time.perf_counter()-start_all,
        fixed_clip_rule='min(.5,max(1e-6,quantile95(grid_J)))',no_selection_on_fresh=True,
        sources={p.name:sha(p) for p in [Path(__file__),HERE/'bounds.py',HERE/'common.py']}))
    print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-root',required=True,type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a.old_root)
