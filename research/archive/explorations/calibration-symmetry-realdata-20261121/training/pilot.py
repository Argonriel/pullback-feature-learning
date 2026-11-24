"""Fixed 72-trajectory labeled-validation pilot, retaining all outcomes."""
import argparse,csv,hashlib,json,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import model
HERE=Path(__file__).resolve().parent
GRID={optimizer:[dict(optimizer=optimizer,rate=rate,lam=lam,damping=.1) for rate in rates for lam in (0.,.001,.01)]
      for optimizer,rates in [('pb',(2.,4.,8.)),('gd',(.1,.3,1.))]}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def jo(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)
def name(cfg):return f"{cfg['optimizer']}_rate{cfg['rate']:g}_lam{cfg['lam']:g}"
def csvwrite(path,rows):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)

def freeze():
    paths=['model.py','pilot.py','PROTOCOL.md','LITERATURE.md','reference/task.py','reference/core4d.py','reference/TRAINING_CHOICE.json']
    jo(HERE/'PILOT_FREEZE.json',dict(timestamp_utc=now(),grid=GRID,pilot_pairs=4,trajectories=72,
        train_seeds=list(range(2091001,2091005)),init_seeds=list(range(2092001,2092005)),validation_seeds=list(range(2093001,2093005)),
        training_n=256,validation_n=2048,checkpoints=model.CHECKPOINTS,
        rule='Within each trajectory choose minimum sampled-validation NLL, ties earlier checkpoint. Within optimizer choose minimum four-seed mean of those selected NLLs; exact ties lower mean selected step, then lower lambda, then lower rate. Failed trajectory makes recipe ineligible.',
        sources={p:sha(HERE/p) for p in paths},fresh_read=False))
def verify():
    frozen=json.loads((HERE/'PILOT_FREEZE.json').read_text())
    for p,h in frozen['sources'].items():assert sha(HERE/p)==h,p
    return frozen
def run():
    frozen=verify();out=HERE/'pilot';out.mkdir(exist_ok=False);jo(out/'START.json',dict(timestamp_utc=now(),freeze_sha256=sha(HERE/'PILOT_FREEZE.json')))
    rows=[];checkpoints=[]
    with threadpool_limits(limits=1):
      for pair in range(1,5):
        x,y=model.task.sample_train(2091000+pair);initial=model.task.initialize(2092000+pair)
        xv,yv=model.labeled_inputs(2093000+pair,2048)
        np.savez_compressed(out/f'{pair}_inputs.npz',x=x,y=y,initial=initial,xv=xv,yv=yv)
        for optimizer in ('pb','gd'):
          for cfg in GRID[optimizer]:
            label=name(cfg);row=dict(pair=pair,recipe=label,**cfg,status='ok')
            try:
                states,trace,index,seconds=model.trajectory(initial,x,y,xv,yv,**cfg)
                np.savez_compressed(out/f'{pair}_{label}.npz',states=states,selected_index=index,checkpoints=model.CHECKPOINTS)
                row.update(selected_step=trace[index]['step'],selected_validation_nll=trace[index]['validation_nll'],seconds=seconds)
                checkpoints.extend([dict(pair=pair,recipe=label,**r) for r in trace])
            except Exception as exc:
                row.update(status='failed',error=repr(exc));(out/f'{pair}_{label}_failure.txt').write_text(traceback.format_exc())
            rows.append(row);print(json.dumps(row),flush=True)
            (out/'progress.json').write_text(json.dumps(rows,indent=2))
    csvwrite(out/'trajectories.csv',rows);csvwrite(out/'checkpoints.csv',checkpoints)
    summaries=[];choices={}
    for optimizer in ('pb','gd'):
      for cfg in GRID[optimizer]:
        rs=[r for r in rows if r['recipe']==name(cfg)];good=len(rs)==4 and all(r['status']=='ok' for r in rs)
        summaries.append(dict(recipe=name(cfg),**cfg,eligible=good,mean_selected_validation_nll=float(np.mean([r['selected_validation_nll'] for r in rs])) if good else None,
          mean_selected_step=float(np.mean([r['selected_step'] for r in rs])) if good else None))
      eligible=[s for s in summaries if s['optimizer']==optimizer and s['eligible']]
      assert eligible,f'No eligible {optimizer} recipe'
      winner=min(eligible,key=lambda s:(s['mean_selected_validation_nll'],s['mean_selected_step'],s['lam'],s['rate']))
      choices[optimizer]={k:winner[k] for k in ('optimizer','rate','lam','damping')}
    csvwrite(out/'recipe_summary.csv',summaries)
    jo(HERE/'TRAINING_CHOICE.json',dict(timestamp_utc=now(),choices=choices,summaries=summaries,rule=frozen['rule'],pilot_freeze_sha256=sha(HERE/'PILOT_FREEZE.json'),trajectory_sha256=sha(out/'trajectories.csv')))
    jo(out/'COMPLETE.json',dict(timestamp_utc=now(),trajectories=len(rows),checkpoint_rows=len(checkpoints),failures=sum(r['status']!='ok' for r in rows)))
    print('CHOICE',json.dumps(choices),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args();globals()[args.stage]()
