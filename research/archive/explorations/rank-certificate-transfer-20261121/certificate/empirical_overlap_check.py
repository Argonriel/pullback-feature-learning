"""Descriptive, all-model overlap check of independent empirical CDF bands.

This diagnostic is specified after fresh enclosure work began. It does not
choose models, policies, numerical settings or confidence levels. A finite
grid check is not a proof of the uniform DKW event.
"""
from pathlib import Path
import csv,datetime,json
import numpy as np
import known_cdf as kc

HERE=Path(__file__).resolve().parent


def main():
    rows=[]
    for task in ['rings','waveform']:
        for seed in range(1,17):
            known=dict(np.load(HERE/f'fresh/{task}/{seed}_distribution.npz'))
            empirical=dict(np.load(HERE/f'fresh_empirical/{task}/{seed}_distribution.npz'))
            assert len(empirical['independent_verification_radii'])==1048576
            expected=(1013000 if task=='rings' else 1053000)+seed
            assert int(empirical['verification_seed'])==expected
            assert float(empirical['alpha'])==.05/64
            for name in ['score_F','location_G']:
                grid=np.linspace(float(known['amin'].min()),
                    float(known['amax'].max())+(float(known['width']) if name=='score_F' else 0),1025)
                if name=='score_F':
                    kl,kh=kc.bracket(known,grid);el,eh=kc.bracket(empirical,grid)
                else:
                    kpad=float(known['tail'])+2e-12;epad=float(empirical['tail'])+2e-12
                    kl=np.maximum(0,kc.cdf_a(known,grid,'hi')-kpad)
                    kh=np.minimum(1,kc.cdf_a(known,grid,'lo')+kpad)
                    ec=kc.cdf_a(empirical,grid,'lo')
                    el=np.maximum(0,ec-epad);eh=np.minimum(1,ec+epad)
                disjoint=np.maximum(el-kh,kl-eh)
                containment=np.minimum(kl-el,eh-kh)
                rows.append(dict(task=task,seed=seed,quantity=name,grid_points=len(grid),
                    dkw_epsilon=float(empirical['tail']),
                    maximum_disjoint_excess=float(disjoint.max()),
                    minimum_containment_margin=float(containment.min()),
                    known_enclosure_max_width=float((kh-kl).max()),
                    disjoint_grid_points=int(np.sum(disjoint>1e-10)),
                    known_enclosure_outside_empirical_points=int(np.sum(containment < -1e-10)),
                    verification_seed=expected))
    with (HERE/'empirical_overlap_check.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    out=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        models=32,quantity_model_rows=len(rows),
        disjoint_points=sum(r['disjoint_grid_points'] for r in rows),
        known_enclosure_outside_empirical_points=sum(r['known_enclosure_outside_empirical_points'] for r in rows),
        maximum_disjoint_excess=max(r['maximum_disjoint_excess'] for r in rows),
        minimum_containment_margin=min(r['minimum_containment_margin'] for r in rows),
        ordinary_forward_examples=0,interval_jet_propagations=0,
        score_CDF_query_inputs=2*32*1025,location_CDF_query_inputs=3*32*1025,
        scope='Finite-grid descriptive overlap check; not a proof of the uniform confidence event. '
              'No model or setting is changed or excluded by this diagnostic.')
    (HERE/'empirical_overlap_check.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
