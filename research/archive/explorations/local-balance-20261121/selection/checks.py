"""Old-only direct numerical checks before the algorithm freeze."""
from pathlib import Path
import sys,json,hashlib,csv,datetime
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import policy as p
from develop import sample

def run():
    out=ROOT/'numerical_checks';out.mkdir(exist_ok=False);rows=[];summary={}
    for key in ('121_1','121_16','611_7','721_8','821_1','821_16'):
        base=dict(np.load(ROOT/'development_v2'/'bases'/f'{key}.npz'))
        table={**base,**dict(np.load(ROOT/'development_v2'/'actions'/f'{key}_tau1.npz'))}
        for radius_index in (0,1,21,53,100,128):
            r=table['radii'][radius_index];phi=2*np.pi*(np.arange(p.NA)+.5)/p.NA;x=r*np.c_[np.cos(phi),np.sin(phi)]
            h=table['h'][radius_index];partner=np.roll(h,p.NA//2);d=(h-partner)/2;A=np.logaddexp(0,h)-np.logaddexp(0,(h+partner)/2)
            for k in p.FREQUENCIES:
                for shift in (0,1,63,117,255,256,383,511):
                    phase=(shift+.25)/p.NA;angle=p.action_angle(x,np.full(p.NA,k));u=(k*angle/(2*np.pi)+phase)%1
                    # r=0 has no meaningful angle, and d=A=0 exactly.
                    q=p.rank_probability(p.score_cdf(table,table['gain_profile'][radius_index]+float(table['width'])*u))
                    discrete_u=((k*np.arange(p.NA)+k//2+shift)%p.NA+(.75 if k%2 else .25))/p.NA
                    qm=p.rank_probability(p.score_cdf(table,table['gain_profile'][radius_index]+float(table['width'])*discrete_u));ix=(k-1)*p.NA+shift
                    rows.append(dict(model=key,radius_index=radius_index,k=k,shift=shift,coordinate_C_error=abs(float(q@A/p.NA-table['action_objectives'][radius_index,ix])),coordinate_b_error=abs(float(q@d/p.NA-table['action_moments'][radius_index,ix])),modular_C_error=abs(float(qm@A/p.NA-table['action_objectives'][radius_index,ix])),modular_b_error=abs(float(qm@d/p.NA-table['action_moments'][radius_index,ix]))))
        x=sample(957001,16384);keys=np.random.default_rng(957002).random(len(x));ties=np.random.default_rng(957003).random(len(x));h=p.core.representation(table['theta'],x)[:,0]
        sx=p.scores(x,table,'rank_even',keys);sn=p.scores(-x,table,'rank_even',keys)
        assert np.array_equal(sx,sn),(key,'even score symmetry')
        m=p.select(x,h,table,'rank_even',ties,keys);mn=p.select(-x,h,table,'rank_even',ties,keys);assert np.array_equal(m,mn)
        for name in p.POLICIES:assert np.all(p.select(x,h,table,name,ties,keys).reshape(-1,256).sum(1)==64)
        assert np.all(p.conditional_bound(table,np.linspace(0,4,513),'rank_even')['total']==0)
    with (out/'direct_fft.csv').open('x',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    for col in ('coordinate_C_error','coordinate_b_error','modular_C_error','modular_b_error'):
        summary[col+'_max']=max(r[col] for r in rows);assert summary[col+'_max']<1e-12
    nodes,weights=np.polynomial.legendre.leggauss(256);integral=float(weights@p.rank_probability((nodes+1)/2)/2)
    assert abs(integral-.25)<1e-12
    summary.update(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),direct_actions=len(rows),old_models=6,rank_probability_integral=integral,rank_probability_lipschitz=p.Q_LIPSCHITZ,even_scores_bitwise_equal=True,even_masks_equal=True,all_quotas_64=True,even_bounds_zero=True,source_sha256=hashlib.sha256((ROOT/'policy.py').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):run()
