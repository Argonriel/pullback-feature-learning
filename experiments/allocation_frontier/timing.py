"""Frozen complete-procedure benchmark; run only in root's exclusive window."""
import argparse,json,platform,time
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
import runner,model

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exclusive-confirmed',action='store_true');args=parser.parse_args()
    assert args.exclusive_confirmed
    runner.verify();runner.verify_commit('development/MODEL_COMMIT.json');runner.verify_commit('development/FIT_COMMIT.json');runner.verify_commit('confirmation/SCORE_COMPLETE.json');runner.verify_commit('ANALYSIS_COMPLETE.json')
    procs=runner.procedures('confirmation');split=dict(np.load(runner.OUT/'development/splits/1.npz'));states=dict(np.load(runner.OUT/'development/states/1.npz'))
    expected={p['name']:json.loads((runner.OUT/'development/temperatures'/f"1_{p['name']}.json").read_text())['fits'] for p in procs}
    rng=np.random.default_rng(2419100);rows=[];checks=[]
    def fit(proc):
        n=proc['n'];x=split[f'x_{n}'];y=split[f'y_{n}'];cn=proc['calibration'];cx=split['calibration_x'][:cn];cy=split['calibration_y'][:cn];marks=runner.checkpoints(proc['steps'])
        start=time.perf_counter();theta=split['initial'].copy();stored={};losses=[]
        for step in range(1,proc['steps']+1):
            theta+=(4*n/256)*model.task.direction(theta,x,y,method='pb',damping=.1*n/256)
            if not np.isfinite(theta).all():raise FloatingPointError('timing nonfinite')
            if step in marks:stored[step]=theta.copy();losses.append(model.task.loss(theta,x,y))
        training_end=time.perf_counter();raw=model.task.predict(theta,cx);partner=model.task.predict(theta,model.task.partner(cx));full=.5*(raw+partner)
        fits={pipeline:model.temperature(h,cy) for pipeline,h in [('raw',raw),('full',full)]};end=time.perf_counter()
        return theta,fits,training_end-start,end-training_end,end-start
    with threadpool_limits(limits=1):
        pools=threadpool_info()
        for block in range(4):
            order=rng.permutation(len(procs))
            for position,index in enumerate(order):
                proc=procs[index];theta,fits,training,calibration,total=fit(proc);name=proc['name']
                error=float(np.max(abs(theta-states[name])));beta_error=max(abs(fits[p]['beta']-expected[name][p]['beta']) for p in ('raw','full'))
                assert error<=1e-12 and beta_error<=1e-12
                checks.append(dict(block=block,procedure=name,theta_error=error,beta_error=beta_error))
                rows.append(dict(block=block,role='warmup' if block==0 else 'measured',order=position,procedure=name,n_train=proc['n'],n_calibration=proc['calibration'],
                  training_seconds=training,calibration_seconds=calibration,total_fit_seconds=total,**runner.work(proc['n'],proc['steps']),training_diagnostic_forwards=4*proc['n'],calibration_forwards=2*proc['calibration']))
                print('Benchmarked',block,name,flush=True)
    runner.cw(runner.OUT/'complete_fit_timing.csv',rows)
    summaries=[]
    for proc in procs:
        rs=[r for r in rows if r['procedure']==proc['name'] and r['role']=='measured'];values=np.array([r['total_fit_seconds'] for r in rs])
        summaries.append(dict(procedure=proc['name'],measured_fits=3,median_total_seconds=float(np.median(values)),minimum_total_seconds=float(values.min()),maximum_total_seconds=float(values.max()),
          median_training_seconds=float(np.median([r['training_seconds'] for r in rs])),median_calibration_seconds=float(np.median([r['calibration_seconds'] for r in rs]))))
    runner.cw(runner.OUT/'timing_summary.csv',summaries)
    runner.jo(runner.OUT/'TIMING_AUDIT.json',dict(utc=runner.now(),exclusive_window_confirmed_by_root=True,source_sha256=runner.sha(runner.SOURCE/'timing.py'),
      platform=platform.platform(),processor=platform.processor(),numpy=np.__version__,threadpools=pools,config=runner.CONFIG['timing'],checks=checks,
      warmup_fits=len(procs),measured_fits=3*len(procs),total_fits=4*len(procs),new_unique_labels=0,
      warmup_nominal_operations=sum(r['nominal_arithmetic_flops'] for r in rows if r['role']=='warmup'),measured_nominal_operations=sum(r['nominal_arithmetic_flops'] for r in rows if r['role']=='measured'),
      measured_calibration_forwards=sum(r['calibration_forwards'] for r in rows if r['role']=='measured'),warmup_calibration_forwards=sum(r['calibration_forwards'] for r in rows if r['role']=='warmup'),
      selection_commit_sha256=runner.sha(runner.OUT/'SELECTION_COMMIT.json'),confirmation_score_commit_sha256=runner.sha(runner.OUT/'confirmation/SCORE_COMPLETE.json'),summaries=summaries,
      limitation='Three repeated complete fits on one fixed development pair are descriptive workload timings, not a distribution over independent fitting datasets or an equal-wall-time optimization. Data generation/loading, result I/O, test scoring and verification are excluded.'))

if __name__=='__main__':main()
