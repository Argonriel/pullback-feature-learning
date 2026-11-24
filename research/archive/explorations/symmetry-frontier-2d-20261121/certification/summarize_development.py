"""Recompute the old clipping pilot with the final 432-event error allocation."""
from pathlib import Path
import csv,json
import numpy as np
from bounds import LOOKS,empirical_bernstein_lower
HERE=Path(__file__).resolve().parent

def main():
    folder=HERE/'development_old32';rows=[]
    for cohort,clip in [('rings',.25),('transfer',.5)]:
        for pair in range(1,17):
            data=np.load(folder/f'{cohort}_{pair}_batches.npz')
            for index,name in enumerate(['rank_even','folded']):
                for look in LOOKS:
                    bound=empirical_bernstein_lower(data[f'policy{index}_clip_{clip:g}'][:look],clip/4,.05/432)
                    bound.update(cohort=cohort,pair=pair,policy=name,clip=clip,
                        full_gain=float(data[f'policy{index}_unclipped'].mean()),
                        standalone_forwards=128*look,shared_forwards=int(data['calls_per_batch'][:look].sum()))
                    rows.append(bound)
    summary=[]
    for cohort in ['rings','transfer']:
        for name in ['rank_even','folded']:
            group=[r for r in rows if r['cohort']==cohort and r['policy']==name]
            full=np.mean([r['lower'] for r in group if r['n_batches']==2048])
            for look in LOOKS:
                current=[r for r in group if r['n_batches']==look]
                mean=np.mean([r['lower'] for r in current])
                summary.append(dict(cohort=cohort,policy=name,batches=look,
                    mean_lower=float(mean),positive=sum(r['lower']>0 for r in current),
                    ratio_of_mean_lowers=float(mean/full),
                    mean_unclipped_gain=float(np.mean([r['full_gain'] for r in current])),
                    standalone_forwards=128*look,
                    mean_shared_forwards=float(np.mean([r['shared_forwards'] for r in current]))))
    for name,values in [('final_family_per_model.csv',rows),('final_family_summary.csv',summary)]:
        with (folder/name).open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,list(values[0]));writer.writeheader();writer.writerows(values)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
