"""Resolve the finite-grid location-CDF overlaps without changing any policy.

The earlier coarse radial partition can give a wide location-CDF enclosure
near profile extrema or flat pieces. Direct segment thresholding resolves only
those flagged diagnostic points. It does not select models or alter bands.
"""
from pathlib import Path
import csv,datetime,json
import numpy as np
import known_cdf as kc

HERE=Path(__file__).resolve().parent


def direct_location_cdf(table,z,task):
    edges=table['radii'];a=table['gain_profile'];z=np.asarray(z)
    edgecdf,tail=kc.radial_cdf(edges,task)
    out=np.where(z>=a[0],edgecdf[0],0.)+np.where(z>=a[-1],1-edgecdf[-1],0.)
    for index in range(len(edges)-1):
        if a[index+1]==a[index]:
            out+=np.where(z>=a[index],edgecdf[index+1]-edgecdf[index],0.)
        else:
            fraction=np.clip((z-a[index])/(a[index+1]-a[index]),0,1)
            r=edges[index]+fraction*(edges[index+1]-edges[index])
            middle,_=kc.radial_cdf(r,task)
            out+=(middle-edgecdf[index] if a[index+1]>a[index] else edgecdf[index+1]-middle)
    return np.clip(out,0,1),tail+2e-12


def main():
    records=[]
    for task in ['rings','waveform']:
        for seed in range(1,17):
            known=dict(np.load(HERE/f'fresh/{task}/{seed}_distribution.npz'))
            empirical=dict(np.load(HERE/f'fresh_empirical/{task}/{seed}_distribution.npz'))
            z=np.linspace(float(known['amin'].min()),float(known['amax'].max()),1025)
            kpad=float(known['tail'])+2e-12;epad=float(empirical['tail'])+2e-12
            kl=np.maximum(0,kc.cdf_a(known,z,'hi')-kpad)
            kh=np.minimum(1,kc.cdf_a(known,z,'lo')+kpad)
            ec=kc.cdf_a(empirical,z,'lo');el=np.maximum(0,ec-epad);eh=np.minimum(1,ec+epad)
            flagged=np.flatnonzero(np.minimum(kl-el,eh-kh)<-1e-10)
            if not len(flagged):continue
            cohort='rings' if task=='rings' else 'transfer'
            table=dict(np.load(HERE.parent/f'validation/policies/{cohort}_{seed}_pb_rank.npz'))
            true,error=direct_location_cdf(table,z[flagged],task)
            for j,index in enumerate(flagged):
                low=max(0,float(true[j]-error));high=min(1,float(true[j]+error))
                margin=min(low-el[index],eh[index]-high)
                records.append(dict(task=task,seed=seed,grid_index=int(index),location=float(z[index]),
                    initial_known_lower=float(kl[index]),initial_known_upper=float(kh[index]),
                    empirical_lower=float(el[index]),empirical_upper=float(eh[index]),
                    direct_known_lower=low,direct_known_upper=high,contained_margin=float(margin),
                    contained=bool(margin>=-1e-10)))
    with (HERE/'empirical_atom_check.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)
    result=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        flagged_points=len(records),all_resolved_inside_empirical_band=all(r['contained'] for r in records),
        minimum_containment_margin=min(r['contained_margin'] for r in records),
        ordinary_forward_examples=0,interval_jet_propagations=0,
        interpretation='The eight wide known-law location-CDF brackets came from radial discretization near profile extrema or flat pieces. '
        'Direct segment-threshold radial-CDF evaluation checks the actual location law at those points. '
        'This remains a finite-grid diagnostic, not a uniform confidence-event proof. No policy, bound or model is changed.')
    (HERE/'empirical_atom_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
