"""Isolated development-state PB timing, never used to select schedules."""
import argparse,json,platform,time
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
import runner,model

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exclusive-confirmed',action='store_true');args=parser.parse_args()
    assert args.exclusive_confirmed,'Root must grant an exclusive CPU window before execution'
    runner.verify();runner.verify_commit('FIT_COMMIT.json');runner.verify_commit('SCORE_COMPLETE.json')
    old=np.load(runner.SOURCE/'reference/DEVELOPMENT_STATE.npz');rows=[];check=[]
    prepared={n:(np.tile(old['x'],(n//256,1)),np.tile(old['y'],n//256)) for n in runner.NS}
    def block(n,updates):
        x,y=prepared[n];theta=old['selected_pb'].copy();start=time.perf_counter()
        for _ in range(updates):
            theta+=(4*n/256)*model.task.direction(theta,x,y,method='pb',damping=.1*n/256)
            assert np.isfinite(theta).all()
        return time.perf_counter()-start,theta
    with threadpool_limits(limits=1):
        pools=threadpool_info()
        for n in runner.NS:block(n,5)
        for repeat in range(7):
            order=runner.NS[repeat%4:]+runner.NS[:repeat%4]
            reference=None
            for position,n in enumerate(order):
                seconds,theta=block(n,30)
                if reference is None:reference=theta
                check.append(float(np.max(abs(theta-reference))))
                rows.append(dict(repeat=repeat,n_train=n,order=position,updates=30,seconds=seconds,seconds_per_update=seconds/30,nominal_arithmetic_flops_per_update=179045*n+2001144))
    runner.cw(runner.OUT/'isolated_timing.csv',rows)
    projected=[]
    for proc in runner.PROCEDURES:
        seconds=np.array([r['seconds_per_update'] for r in rows if r['n_train']==proc['n']])
        projected.append(dict(procedure=proc['name'],n_train=proc['n'],updates=proc['step'],median_seconds_per_update=float(np.median(seconds)),
          projected_training_seconds=float(np.median(seconds)*proc['step']),min_projected_seconds=float(seconds.min()*proc['step']),max_projected_seconds=float(seconds.max()*proc['step'])))
    runner.cw(runner.OUT/'timing_projections.csv',projected)
    baseline=next(r for r in projected if r['procedure']=='fixed_2304')['projected_training_seconds']
    summary='After all predictive results were committed, an exclusive CPU window measured seven 30-update blocks per n with one BLAS thread, following five warm-up updates. Each block resets to the same old development state and duplicates the same 256 training examples. Median block times project to '+', '.join(f"{r['procedure']} {r['projected_training_seconds']:.3f} s ({r['projected_training_seconds']/baseline:.3f} times baseline)" for r in projected if r['procedure'] in ('fixed_2304','flop_256','flop_768','flop_1280','examples_256'))+'. These are development-workload projections, not timed complete prospective fits; they exclude diagnostics, calibration, validation, data generation and I/O, and do not alter any schedule.'
    runner.jo(runner.OUT/'TIMING_AUDIT.json',dict(utc=runner.now(),exclusive_window_confirmed_by_root=True,source_sha256=runner.sha(runner.SOURCE/'timing.py'),
      fit_commit_sha256=runner.sha(runner.OUT/'FIT_COMMIT.json'),score_commit_sha256=runner.sha(runner.OUT/'SCORE_COMPLETE.json'),
      development_state_sha256=runner.sha(runner.SOURCE/'reference/DEVELOPMENT_STATE.npz'),platform=platform.platform(),processor=platform.processor(),numpy=np.__version__,threadpools=pools,
      repeats=7,updates_per_block=30,warmup_per_n=5,maximum_normalization_end_state_difference=max(check),measured_updates=4*7*30,warmup_updates=4*5,total_updates=4*(5+7*30),
      actual_timed_nominal_operations=sum(r['updates']*r['nominal_arithmetic_flops_per_update'] for r in rows),
      warmup_nominal_operations=sum(5*(179045*n+2001144) for n in runner.NS),summary=summary))
    print(summary)

if __name__=='__main__':main()
