"""Portable replay of the frozen checks using preserved development inputs."""
from pathlib import Path
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
import checks
import known_cdf as kc

HERE=Path(__file__).resolve().parent


def main(output):
    records=[]
    for task,seed,na in [('rings',1,256),('rings',8,256),('rings',16,256),('waveform',1,512)]:
        path=(HERE.parent/f'development/rings_{seed}_rank.npz' if task=='rings'
              else HERE.parent/'transfer/pilot_policy/1_pb.npz')
        table=dict(np.load(path))
        result=dict(np.load(HERE/f'pilot/{task}{seed}_v2_sub8_na{na}_inner64.npz'))
        dist=kc.distribution(table,task,131072)
        records.append(checks.check(table,result,dist,task,9914000+seed))
    with Path(output).open('x',encoding='utf8') as f:json.dump(records,f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(1):main(a.output)
