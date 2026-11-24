"""Fixed four-seed, three-recipe development; never opens fresh streams."""
import csv, hashlib, json, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import task
HERE=Path(__file__).resolve().parent
RECIPES=[('pb8_500',8.,500),('pb4_500',4.,500),('pb8_1000',8.,1000)]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()

def freeze():
    target=HERE/'TASK_LAW_FREEZE.json'
    if target.exists(): raise FileExistsError(target)
    spec=dict(timestamp_utc=now(),task='r1,r2 independent U[.4,1.4], phi independent uniform; x=(r1 cos phi,r1 sin phi,r2 cos 2phi,r2 sin 2phi)',
      posterior='0.5+0.4*tanh(8*(r1-.9))*tanh(8*(r2-.9))',architecture='exact copied 144-parameter 4D two-block residual network',
      training='256 inputs; exactly 128 labels per class via rejection, class-conditional law preserved',
      pilot_train_seeds=list(range(1131001,1131005)),pilot_init_seeds=list(range(1132001,1132005)),pilot_eval_seeds=list(range(1133001,1133005)),
      pilot_eval_n=32768,recipes=[dict(name=n,rate=r,steps=s,method='pb',damping=.1) for n,r,s in RECIPES],
      selection='minimum four-seed mean known-posterior expected log loss; exact tie prefers listed order; a failed recipe is ineligible; retain every failure',
      no_task_retuning=True,fresh_train_reserved=list(range(1141001,1141017)),fresh_init_reserved=list(range(1142001,1142017)),
      calibration_grid=dict(shape=[16,16,256],quotient_axes='linspace(.4,1.4,16) each',phase_offset=.5,forwards=65536),
      sources={f:sha(HERE/f) for f in ('task.py','core4d.py','pilot.py','LITERATURE.md')})
    target.write_text(json.dumps(spec,indent=2)+'\n')
    print('TASK LAW FROZEN',target,flush=True)

def run():
    frozen=json.loads((HERE/'TASK_LAW_FREEZE.json').read_text())
    for f,digest in frozen['sources'].items(): assert sha(HERE/f)==digest,(f,'changed since freeze')
    out=HERE/'pilot_results'; out.mkdir(exist_ok=False)
    (out/'START.json').write_text(json.dumps(dict(timestamp_utc=now(),freeze_hash=sha(HERE/'TASK_LAW_FREEZE.json')),indent=2))
    rows=[]
    with threadpool_limits(limits=1):
      for seed in range(1,5):
        x,y,meta=task.sample_train_with_metadata(1131000+seed); initial=task.initialize(1132000+seed)
        xe,ye,p=task.sample_test(1133000+seed,32768)
        np.savez_compressed(out/f'{seed}_inputs.npz',x=x,y=y,initial=initial,eval_x=xe,eval_y=ye,eval_p=p)
        for name,rate,steps in RECIPES:
          row=dict(seed=seed,recipe=name,rate=rate,steps=steps,status='ok',**meta)
          try:
            theta,training=task.train(initial,x,y,steps=steps,rate=rate,method='pb',damping=.1)
            h=task.predict(theta,xe); hp=task.predict(theta,task.partner(xe)); m=(h+hp)/2
            row.update(train_loss=task.loss(theta,x,y),expected_loss=float(np.mean(np.logaddexp(0,h)-p*h)),
              sampled_loss=float(np.logaddexp(0,-ye*h).mean()),full_expected_loss=float(np.mean(np.logaddexp(0,m)-p*m)),
              classification_error=float(np.mean(np.where(h>=0,1-p,p))),full_classification_error=float(np.mean(np.where(m>=0,1-p,p))),
              bayes_loss=float(np.mean(-p*np.log(p)-(1-p)*np.log1p(-p))),bayes_error=float(np.mean(np.minimum(p,1-p))),
              seconds=training['seconds'])
            np.savez_compressed(out/f'{seed}_{name}.npz',theta=theta,loss_trace=np.asarray(training['loss_trace']))
          except Exception as exc:
            row.update(status='failed',error=repr(exc)); (out/f'{seed}_{name}_failure.txt').write_text(traceback.format_exc())
          rows.append(row)
          (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
          print(json.dumps(row),flush=True)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'rows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summaries=[]
    for name,rate,steps in RECIPES:
        selected=[r for r in rows if r['recipe']==name]
        summaries.append(dict(name=name,eligible=all(r['status']=='ok' for r in selected),mean_expected_loss=float(np.mean([r.get('expected_loss',float('inf')) for r in selected])),config=dict(steps=steps,method='pb',rate=rate,damping=.1)))
    eligible=[s for s in summaries if s['eligible']]
    choice=min(eligible,key=lambda s:s['mean_expected_loss']) if eligible else None
    payload=dict(timestamp_utc=now(),choice=choice['name'] if choice else None,config=choice['config'] if choice else None,
      criterion=frozen['selection'],summaries=summaries,task_law_freeze_sha256=sha(HERE/'TASK_LAW_FREEZE.json'),pilot_rows_sha256=sha(out/'rows.csv'))
    (HERE/'TRAINING_CHOICE.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(timestamp_utc=now(),cases=len(rows),failures=sum(r['status']!='ok' for r in rows)),indent=2))
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['freeze','run']); args=parser.parse_args()
    freeze() if args.action=='freeze' else run()
